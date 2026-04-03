"""
可视化不同 SEG Token (Global / Local / Detail) 在 voxel cross-attention 中
关注的 3D 点云空间区域。

将 attention weights 从体素空间通过三线性插值映射回每个点，
然后在点云上着色可视化。

用法 (集成到 eval 流程):
  python visualize_attention.py \
      --cfg-path  path/to/eval_config.yaml \
      --checkpoint path/to/checkpoint.pth \
      --output-dir vis_output/ \
      --max-samples 20

也支持从已保存的 npz 离线可视化:
  python visualize_attention.py --load-npz vis_output/sample_000/ --output-dir vis_offline/
"""
import os
import random
import logging
import argparse
from os import path
from typing import List
os.environ["WANDB_API_KEY"] = "fd06e756ef66a77184ae576ae979b7586146c10f"
from torch.utils.data import DataLoader, Dataset
import json
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from accelerate import Accelerator
import torch.distributed as dist
from trainer import Trainer
from common.config import LLMConfig
from common.registry import registry
from common.logger import setup_logger
from common.utils import now, get_rank, update_cfg_for_dist
"""
可视化不同 SEG Token (Global / Local / Detail) 在 voxel cross-attention 中
关注的 3D 点云空间区域。

用法:
  python visualize_attention.py \
      --cfg-path  path/to/eval_config.yaml \
      --checkpoint path/to/checkpoint.pth \
      --output-dir vis_output/ \
      --max-samples 20

离线可视化 (从已保存的 npz):
  python visualize_attention.py --load-npz vis_output/ --output-dir vis_offline/
"""

import os
import sys
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────
# 1. 提取 cross-attention weights (修改版 forward)
# ─────────────────────────────────────────────────────────

def forward_with_attn_weights(voxel_module, pred_tokens, multiscale_voxels):
    """
    与 VoxelTokenMultiScaleAblation.forward 相同逻辑，
    额外返回每个 branch 的 cross-attention weights。
    """
    enhanced_list = []
    attn_weights_list = []
    token_names = ["Global", "Local", "Detail"]
    for j in range(3):
        token_j = pred_tokens[:, j:j+1, :]
        voxel_feat = voxel_module.voxel_projs[j](multiscale_voxels[j])

        if voxel_module.share_attn:
            context, attn_w = voxel_module.shared_cross_attn(
                query=token_j, key=voxel_feat, value=voxel_feat,
                need_weights=True, average_attn_weights=False,
            )
            gate = torch.sigmoid(voxel_module.shared_gate(context))
        else:
            context, attn_w = voxel_module.cross_attns[j](
                query=token_j, key=voxel_feat, value=voxel_feat,
                need_weights=True, average_attn_weights=False,
            )
            gate = torch.sigmoid(voxel_module.gates[j](context))

        # ---- 诊断: gate 值和 per-head attention 分布 ----
        gate_val = gate.mean().item()
        # attn_w: (B, num_heads, 1, K) when average_attn_weights=False
        attn_per_head = attn_w[0, :, 0, :]  # (num_heads, K)
        head_max_attn = attn_per_head.max(dim=1).values  # 每个 head 的最大 attention
        logging.info(
            f"    {token_names[j]}: gate_mean={gate_val:.4f}, "
            f"per_head_max_attn={head_max_attn.cpu().tolist()}"
        )
        print(            f"    {token_names[j]}: gate_mean={gate_val:.4f}, "
            f"per_head_max_attn={head_max_attn.cpu().tolist()}")
        # 取所有 head 中空间选择性最强的那个 head
        head_entropy = -(attn_per_head * (attn_per_head + 1e-10).log()).sum(dim=1)
        best_head = head_entropy.argmin().item()
        attn_best = attn_per_head[best_head:best_head+1, :].unsqueeze(0)  # (1, 1, K)
        logging.info(
            f"    → best_head={best_head}, entropy={head_entropy[best_head]:.4f} "
            f"(uniform={np.log(attn_per_head.shape[1]):.4f})"
        )
        print(            f"    → best_head={best_head}, entropy={head_entropy[best_head]:.4f} "
            f"(uniform={np.log(attn_per_head.shape[1]):.4f})")
        context = voxel_module.drop(context)
        enhanced_list.append(token_j + gate * context)
        attn_weights_list.append(attn_best)  # 用最具空间选择性的 head

    enhanced = torch.cat(enhanced_list, dim=1)
    return enhanced, attn_weights_list


# ─────────────────────────────────────────────────────────
# 2. 体素 attention → 点云 attention (三线性插值)
# ─────────────────────────────────────────────────────────

def voxel_attn_to_point_attn(attn_3d, points):
    """
    将 3D 体素空间的 attention map 通过三线性插值映射到每个点上。

    关键坐标对应关系:
      - pointcloud_to_voxel 用: voxel[..., x, y, z]
        即 volume 的 dim0=x, dim1=y, dim2=z
      - grid_sample 对 5D input (B,C,D,H,W) 的 grid 最后一维是 (W方向, H方向, D方向)
        即 grid[..., 0] 索引 W(=dim4), grid[..., 1] 索引 H(=dim3), grid[..., 2] 索引 D(=dim2)
      - 所以 volume(D=x, H=y, W=z) 对应 grid 需要传 (z, y, x)

      坐标范围:
        - 体素空间: pointcloud_to_voxel clamp 到 [-0.5, 0.5]
        - pc_norm 后的点云: ~[-1, 1]
        - 映射: grid = clamp(point * 2, -1, 1)
          点在 [-0.5, 0.5] 范围内 → grid [-1, 1] (精确采样)
          点在 [-1, -0.5) 或 (0.5, 1] → clamp 到体素边界

    Args:
        attn_3d: (R, R, R) numpy array, 对应 (x, y, z) 轴
        points:  (N, 3) numpy array, [x, y, z], pc_norm 后 ~[-1, 1]

    Returns:
        point_attn: (N,) numpy array
    """
    vol = torch.from_numpy(attn_3d).float().unsqueeze(0).unsqueeze(0)  # (1,1,D=x,H=y,W=z)
    pts = torch.from_numpy(points).float()  # (N, 3) = (x, y, z)

    # 体素空间覆盖 [-0.5, 0.5], 映射到 grid_sample 的 [-1, 1]
    # 点云 pc_norm 后在 [-1, 1], 先 clamp 到体素有效范围 [-0.5, 0.5]
    pts_clamped = pts.clamp(-0.5, 0.5)
    grid_coords = pts_clamped * 2.0  # [-0.5, 0.5] → [-1, 1]

    # grid_sample 需要 (W, H, D) 顺序 = (z, y, x)
    grid = torch.stack([grid_coords[:, 2], grid_coords[:, 1], grid_coords[:, 0]], dim=-1)
    grid = grid.unsqueeze(0).unsqueeze(1).unsqueeze(1)  # (1, 1, 1, N, 3)

    sampled = F.grid_sample(
        vol, grid, mode="bilinear", align_corners=True, padding_mode="border"
    )
    return sampled.squeeze().numpy()  # (N,)


# ─────────────────────────────────────────────────────────
# 3. 从模型 + 样本提取 attention
# ─────────────────────────────────────────────────────────

@torch.no_grad()
def extract_attention_for_sample(model, samples):
    """
    对单个 batch 样本执行 generate，然后提取 voxel cross-attention weights，
    并映射回点云空间。
    """
    assert model.enable_voxel_token, "模型未启用 enable_voxel_token"

    device = next(model.parameters()).device

    # ---- 第一步: 用 model.generate() 完成推理 ----
    gen_output = model.generate(samples, num_beams=1, max_length=256)
    output_text = gen_output.get("text", [])
    output_ids = gen_output.get("output_ids", None)
    pred_masks_scores = gen_output.get("masks_scores", [])

    questions = samples["question"]
    if isinstance(questions, str):
        questions = [questions]
    bs = len(questions)
    points = samples["points"]
    shape_id = samples.get("shape_id", [None] * bs)

    if output_ids is None:
        return []

    # ---- 第二步: 重新跑 LLM forward 拿 hidden_states ----
    new_questions = []
    for q in questions:
        if model.llm_model_type in ("phi3.5", "phi4"):
            new_questions.append(model.prepare_input_phi(q))
        else:
            new_questions.append(model.prepare_input_qwen(q))

    text_input_tokens = model.llm_tokenizer(
        new_questions, return_tensors="pt", padding="longest",
        truncation=True, max_length=model.max_txt_len, add_special_tokens=False,
    ).to(device)

    gen_texts = []
    for t in output_text:
        if model.llm_model_type in ("phi3.5", "phi4"):
            gen_texts.append(model.prepare_response_phi(t))
        else:
            gen_texts.append(model.prepare_response_qwen(t))

    text_output_tokens = model.llm_tokenizer(
        gen_texts, return_tensors="pt", padding="longest",
        truncation=True, max_length=model.max_output_txt_len, add_special_tokens=False,
    ).to(device)

    llm_tokens, _ = model.concat_text_input_output(
        text_input_tokens.input_ids, text_input_tokens.attention_mask,
        text_output_tokens.input_ids, text_output_tokens.attention_mask,
    )

    inputs_embeds = model.llm_model.get_input_embeddings()(llm_tokens["input_ids"])
    attention_mask = llm_tokens["attention_mask"]

    inputs_llm, atts_llm, _ = model.encode_point(points)
    inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
    attention_mask = torch.cat([atts_llm, attention_mask], dim=1)

    with model.maybe_autocast(model.mix_precision):
        llm_out = model.llm_model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            return_dict=True, output_hidden_states=True,
        )

    image_token_length = inputs_llm.shape[1]
    last_hidden_states = llm_out.hidden_states[-1][:, image_token_length:, :]
    combined_ids = llm_tokens["input_ids"]

    # ---- 第三步: 提取 voxel 特征 + attention weights (float32) ----
    points2bs = torch.stack(points)

    model.vqvae.float()
    model.voxel_token_multiscale.float()
    multiscale_voxels = model._get_voxel_multiscale_tokens(points2bs.float())
    model.vqvae.bfloat16()

    # ---- 逐样本提取 attention ----
    seq_len = last_hidden_states.shape[1]

    def last_occurrence_mask(ids, token_id):
        mask = (ids == token_id)
        idx = mask.nonzero(as_tuple=True)[0]
        out = torch.zeros_like(mask, dtype=torch.bool)
        if len(idx) > 0:
            out[idx[-1]] = True
        return out

    zero_vec = torch.zeros(1, model.hidden_dim, device=device)

    results = []
    for i in range(bs):
        cur_ids = combined_ids[i][:seq_len]
        seg_masks = [last_occurrence_mask(cur_ids, tid) for tid in model.seg_token_ids]

        if all(m.sum() == 0 for m in seg_masks):
            results.append(None)
            continue

        pred_tokens = []
        for j in range(3):
            if seg_masks[j].sum() == 0:
                pred_tokens.append(zero_vec)
            else:
                pred_tokens.append(last_hidden_states[i][seg_masks[j]])
        pred_tokens = torch.stack(pred_tokens, dim=0).squeeze(1).unsqueeze(0)

        ms_voxels_i = [v[i:i+1] for v in multiscale_voxels]

        _, attn_weights_list = forward_with_attn_weights(
            model.voxel_token_multiscale,
            pred_tokens.float(),
            [v.float() for v in ms_voxels_i],
        )

        R_base = model.voxel_spatial_res
        pts_np = points[i].float().cpu().numpy()

        # ---- 诊断: 点云坐标范围 ----
        logging.info(
            f"  Sample {i}: pts range=[{pts_np.min():.3f}, {pts_np.max():.3f}], "
            f"pts in [-0.5,0.5]: {(np.abs(pts_np) <= 0.5).all(axis=1).mean()*100:.1f}%"
        )
        print(            f"  Sample {i}: pts range=[{pts_np.min():.3f}, {pts_np.max():.3f}], "
            f"pts in [-0.5,0.5]: {(np.abs(pts_np) <= 0.5).all(axis=1).mean()*100:.1f}%")
        attn_3d_list = []
        point_attn_list = []
        voxel_res_list = []

        token_names_dbg = ["Global(16)", "Local(32)", "Detail(64)"]
        for j, res in enumerate([16, 32, 64]):
            attn_w = attn_weights_list[j][0, 0, :].float().cpu().numpy()

            if model.voxel_uniform_pool_res > 0:
                R_scale = model.voxel_uniform_pool_res
            else:
                R_scale = max(2, R_base * res // 32)

            # ---- 诊断信息 ----
            logging.info(
                f"    {token_names_dbg[j]}: R_scale={R_scale}, K={R_scale**3}, "
                f"attn_w range=[{attn_w.min():.6e}, {attn_w.max():.6e}], "
                f"std={attn_w.std():.6e}, max-min={attn_w.max()-attn_w.min():.6e}"
            )
            print(                f"    {token_names_dbg[j]}: R_scale={R_scale}, K={R_scale**3}, "
                f"attn_w range=[{attn_w.min():.6e}, {attn_w.max():.6e}], "
                f"std={attn_w.std():.6e}, max-min={attn_w.max()-attn_w.min():.6e}")
            voxel_res_list.append(R_scale)
            attn_3d = attn_w.reshape(R_scale, R_scale, R_scale)
            attn_3d_list.append(attn_3d)

            point_attn = voxel_attn_to_point_attn(attn_3d, pts_np)
            point_attn_list.append(point_attn)

        # gt_mask & pred_mask
        gt_masks_raw = samples.get("masks", None)
        gt_mask_np = None
        if gt_masks_raw is not None:
            gt = gt_masks_raw[i]
            if gt.dim() == 3:
                gt = gt.squeeze(0)
            gt_mask_np = gt[0].float().cpu().numpy()

        pred_mask_np = None
        if i < len(pred_masks_scores) and pred_masks_scores[i] is not None:
            pm = pred_masks_scores[i]
            if pm.dim() == 3:
                pm = pm.squeeze(0)
            pred_mask_np = (pm[0] > 0.4).float().cpu().numpy()

        results.append({
            "points": pts_np,
            "point_attn": point_attn_list,
            "attn_3d": attn_3d_list,
            "voxel_res": voxel_res_list,
            "gt_mask": gt_mask_np,
            "pred_mask": pred_mask_np,
            "text": output_text[i] if i < len(output_text) else "",
            "question": questions[i],
            "shape_id": shape_id[i] if i < len(shape_id) else "",
            "label": samples.get("label", [""])[i] if "label" in samples else "",
        })

    model.voxel_token_multiscale.bfloat16()
    return results


# ─────────────────────────────────────────────────────────
# 4. Matplotlib 2D 投影渲染 (无坐标轴, 干净效果, headless 兼容)
# ─────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _project_points(points, elev=30, azim=135):
    """
    将 3D 点云投影到 2D 平面 (简易正交投影)。
    返回 (N, 2) 的 2D 坐标和深度排序索引 (远到近, 用于 painter's algorithm)。
    """
    elev_r = np.radians(elev)
    azim_r = np.radians(azim)

    # 旋转矩阵: 先绕 Z 转 azim, 再绕 X 转 elev
    ca, sa = np.cos(azim_r), np.sin(azim_r)
    ce, se = np.cos(elev_r), np.sin(elev_r)

    # 绕 Z 轴
    Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    # 绕 X 轴
    Rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]])

    R = Rx @ Rz
    rotated = points @ R.T  # (N, 3)

    xy = rotated[:, :2]
    depth = rotated[:, 2]

    # 远到近排序 (先画远的, 近的覆盖)
    order = np.argsort(depth)
    return xy, order


def _render_pointcloud(points, colors, save_path, elev=30, azim=135, figsize=6, point_size=8):
    """
    用 matplotlib scatter 渲染点云到图片。无坐标轴, 白色背景。
    colors: (N, 3) RGB 值 [0, 1]。
    """
    xy, order = _project_points(points, elev, azim)

    fig, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    ax.scatter(
        xy[order, 0], xy[order, 1],
        c=colors[order], s=point_size, edgecolors="none", rasterized=True,
    )
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    plt.tight_layout(pad=0)
    fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.02,
                facecolor="white", edgecolor="none")
    plt.close(fig)


def _norm_attn(a):
    """归一化 attention 到 [0, 1]。"""
    a_min, a_max = a.min(), a.max()
    if a_max - a_min > 1e-8:
        return (a - a_min) / (a_max - a_min)
    return np.zeros_like(a)


def visualize_sample(result, save_dir, sample_idx=0):
    """
    对单个样本生成可视化图片并保存。
    使用 matplotlib 2D 投影, 无坐标轴, 白色背景。

    生成的图片:
      - attn_global.png / attn_local.png / attn_detail.png
      - attn_combined.png
      - gt_mask.png / pred_mask.png
      - comparison.png (拼接对比大图)
    """
    os.makedirs(save_dir, exist_ok=True)

    pts = result["points"]
    attn_list = result["point_attn"]
    gt_mask = result["gt_mask"]
    pred_mask = result["pred_mask"]
    label = result["label"]
    shape_id = result["shape_id"]

    N = len(pts)

    token_names = ["global", "local", "detail"]
    highlight_colors = [
        np.array([0.9, 0.1, 0.1]),   # 红
        np.array([0.1, 0.8, 0.1]),   # 绿
        np.array([0.1, 0.3, 0.9]),   # 蓝
    ]
    base_color = np.array([0.82, 0.82, 0.82])

    attn_normed = [_norm_attn(a) for a in attn_list]

    elev, azim = 30, 135

    # ---- 每个 token 单独的热力图 ----
    single_paths = []
    for j in range(3):
        alpha = attn_normed[j][:, None]
        colors = base_color[None, :] * (1 - alpha) + highlight_colors[j][None, :] * alpha
        colors = np.clip(colors, 0, 1)

        path = os.path.join(save_dir, f"attn_{token_names[j]}.png")
        _render_pointcloud(pts, colors, path, elev=elev, azim=azim)
        single_paths.append(path)

    # ---- RGB 叠加 ----
    rgb = np.stack(attn_normed, axis=-1)  # (N, 3) = (R, G, B)
    row_max = rgb.max(axis=1, keepdims=True)
    rgb_enhanced = np.where(row_max > 0.05, rgb / (row_max + 1e-8), 0.0)
    rgb_enhanced = np.clip(rgb_enhanced, 0, 1)
    low_mask = row_max.squeeze() < 0.05
    rgb_enhanced[low_mask] = base_color

    path_combined = os.path.join(save_dir, "attn_combined.png")
    _render_pointcloud(pts, rgb_enhanced, path_combined, elev=elev, azim=azim)

    # ---- GT Mask ----
    gt_path = None
    if gt_mask is not None:
        gt_colors = np.full((N, 3), 0.82)
        gt_colors[gt_mask > 0.5] = [0.9, 0.15, 0.15]
        gt_path = os.path.join(save_dir, "gt_mask.png")
        _render_pointcloud(pts, gt_colors, gt_path, elev=elev, azim=azim)

    # ---- Pred Mask ----
    pred_path = None
    if pred_mask is not None:
        pred_colors = np.full((N, 3), 0.82)
        pred_colors[pred_mask > 0.5] = [0.15, 0.4, 0.9]
        pred_path = os.path.join(save_dir, "pred_mask.png")
        _render_pointcloud(pts, pred_colors, pred_path, elev=elev, azim=azim)

    # ---- 拼接对比大图 ----
    _make_comparison_image(
        single_paths, path_combined, gt_path, pred_path,
        save_dir, shape_id, label, result["text"],
    )

    logging.info(f"  Saved: {save_dir}/")


def _make_comparison_image(single_paths, combined_path, gt_path, pred_path,
                           save_dir, shape_id, label, text):
    """将各子图拼接成一张对比大图。"""
    from matplotlib.image import imread

    images = []
    titles = []

    for path, name in zip(single_paths, ["Global", "Local", "Detail"]):
        images.append(imread(path))
        titles.append(name)

    images.append(imread(combined_path))
    titles.append("Combined")

    if gt_path and os.path.exists(gt_path):
        images.append(imread(gt_path))
        titles.append("GT Mask")

    if pred_path and os.path.exists(pred_path):
        images.append(imread(pred_path))
        titles.append("Pred Mask")

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=4)
        ax.axis("off")

    fig.suptitle(
        f"[{shape_id}] {label}   |   {text[:100]}",
        fontsize=10, y=0.99,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "comparison.png"), dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)


# ─────────────────────────────────────────────────────────
# 5. 保存/加载 attention 数据 (npz)
# ─────────────────────────────────────────────────────────

def save_attention_npz(result, save_path):
    np.savez_compressed(
        save_path,
        points=result["points"],
        point_attn_0=result["point_attn"][0],
        point_attn_1=result["point_attn"][1],
        point_attn_2=result["point_attn"][2],
        attn_3d_0=result["attn_3d"][0],
        attn_3d_1=result["attn_3d"][1],
        attn_3d_2=result["attn_3d"][2],
        voxel_res=np.array(result["voxel_res"]),
        gt_mask=result["gt_mask"] if result["gt_mask"] is not None else np.array([]),
        pred_mask=result["pred_mask"] if result["pred_mask"] is not None else np.array([]),
        text=str(result["text"]),
        question=str(result["question"]),
        shape_id=str(result["shape_id"]),
        label=str(result["label"]),
    )


def load_attention_npz(npz_dir):
    npz_path = os.path.join(npz_dir, "attention_data.npz") if os.path.isdir(npz_dir) else npz_dir
    data = np.load(npz_path, allow_pickle=True)
    gt = data["gt_mask"]
    pred = data["pred_mask"]
    return {
        "points": data["points"],
        "point_attn": [data["point_attn_0"], data["point_attn_1"], data["point_attn_2"]],
        "attn_3d": [data["attn_3d_0"], data["attn_3d_1"], data["attn_3d_2"]],
        "voxel_res": data["voxel_res"].tolist(),
        "gt_mask": gt if len(gt) > 0 else None,
        "pred_mask": pred if len(pred) > 0 else None,
        "text": str(data["text"]),
        "question": str(data["question"]),
        "shape_id": str(data["shape_id"]),
        "label": str(data["label"]),
    }


# ─────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="可视化 SEG Token 在点云空间的 Attention")

    parser.add_argument("--cfg-path", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-type", type=str, default="auto")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--load-npz", type=str, default=None,
                        help="从已保存的 npz 目录/文件离线可视化")
    parser.add_argument("--output-dir", type=str, default="vis_output")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # ========= 离线模式 =========
    if args.load_npz:
        logging.info(f"离线模式: 从 {args.load_npz} 加载")
        if os.path.isfile(args.load_npz):
            result = load_attention_npz(args.load_npz)
            visualize_sample(result, args.output_dir, sample_idx=0)
        else:
            for fname in sorted(os.listdir(args.load_npz)):
                if fname.endswith(".npz"):
                    result = load_attention_npz(os.path.join(args.load_npz, fname))
                    idx = fname.replace("attention_data_", "").replace(".npz", "")
                    visualize_sample(result, os.path.join(args.output_dir, f"sample_{idx}"))
        logging.info("Done.")
        return

    # ========= 模型推理模式 =========
    assert args.cfg_path and args.checkpoint, "需要 --cfg-path 和 --checkpoint"

    from common.config import LLMConfig
    from common.registry import registry
    from torch.utils.data import DataLoader

    class ArgsWrapper:
        def __init__(self, cfg_path):
            self.cfg_path = cfg_path

    cfg = LLMConfig(ArgsWrapper(args.cfg_path))

    seed = cfg.run_cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    logging.info("Building model...")
    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config)

    sys.path.insert(0, os.path.dirname(__file__))
    from eval import load_checkpoint
    logging.info(f"Loading checkpoint: {args.checkpoint}")
    model = load_checkpoint(model, args.checkpoint, args.checkpoint_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).bfloat16()
    model.eval()

    logging.info("Building eval dataset...")
    eval_cfg = cfg.eval_datasets_cfg
    dataset = None
    for name in eval_cfg:
        dataset_config = eval_cfg[name]
        builder_func = registry.get_builder_func(dataset_config["type"])
        dataset = builder_func(name, dataset_config.path)
        logging.info(f"Using dataset: {name} ({len(dataset)} samples)")
        break

    assert dataset is not None, "配置中没有找到 eval 数据集"

    collate_fn = getattr(dataset, "collate", None)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )

    from eval import Bf16DeviceLoader
    dataloader = Bf16DeviceLoader(dataloader, device)

    os.makedirs(args.output_dir, exist_ok=True)
    sample_count = 0

    logging.info(f"开始推理, 最多可视化 {args.max_samples} 个样本...")
    for batch_idx, samples in enumerate(dataloader):
        if sample_count >= args.max_samples:
            break

        results = extract_attention_for_sample(model, samples)


        for r in results:
            if r is None:
                continue
            if sample_count >= args.max_samples:
                break

            sample_dir = os.path.join(args.output_dir, f"sample_{sample_count:03d}")
            os.makedirs(sample_dir, exist_ok=True)
            save_attention_npz(r, os.path.join(sample_dir, "attention_data.npz"))
            visualize_sample(r, sample_dir, sample_idx=sample_count)

            sample_count += 1
            logging.info(f"  [{sample_count}/{args.max_samples}] {r['shape_id']} / {r['label']}")

    logging.info(f"可视化完成! 共 {sample_count} 个样本, 保存在 {args.output_dir}/")


if __name__ == "__main__":
    main()
