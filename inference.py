import torch
import numpy as np
from models import load_model_and_preprocess
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


if __name__ == "__main__":
    print(torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    # load model
    model = load_model_and_preprocess(
        name="aff_phi3",
        model_type="phi",
        is_eval=True,
        device=device,
    )
    model.load_from_pretrained(
        "3D_ADLLM/model_pre/ckpts/Phi_Main/full_best.pth"
    )
    
    import pickle

    with open("3D_ADLLM/data/IRAS_data/val_data_new_prompt.pkl", "rb") as f:
        data = pickle.load(f)

    print(type(data))
    if isinstance(data, dict):
        print("Keys:", list(data.keys())[:10])
    elif isinstance(data, list):
        print("Length:", len(data))
        print("First item type:", type(data[0]))
        print("First item:", data[0])
    
    # load data
    # file = "/workspace/project/Research_3D_Aff/3D_ADLLM/demo/demo_data/Knife480.npy"
    # data = np.load(file)
    # if data.shape[1] >= 3:
    #     points = data[:, :3]
    data_input = data[0]
    points = data_input["full_shape_coordinate"]
    gt_mask = data_input["GT"]
    gt_mask = gt_mask.squeeze() 
    gt_mask = (gt_mask > 0.5).astype(bool)
    instruction = data_input['instruction']
    input_points = torch.tensor(pc_normalize(points)).float().to(device)
    input = [input_points]
    sample_affordance = {"question": instruction, "points": input}
    output_aff = model.generate(sample_affordance, num_beams=1, max_length=128)

    print(output_aff["text"])
    print(output_aff["masks"])
    # print(output_aff)
    masks_np = output_aff["masks"][0].cpu().numpy().reshape((1,2024))

    np.savetxt("mask.txt", masks_np, fmt="%.3f")
    
    mask = output_aff["masks"][0]      # shape: (N,) 或 (1, N)
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = mask.squeeze() 
    pts = points                        # 原始未归一化点云 (N, 3)
    mask = (mask > 0.5).astype(bool)    # 二值化
    
    same = mask & gt_mask       # ✅ 正确预测区域（真阳性）
    miss = gt_mask & ~mask      # ❌ GT中有但预测漏掉（漏检）
    false = ~gt_mask & mask     # ⚠️ 预测有但GT没有（误检）

    # ----------------------------
    # 5. 可视化（3子图）
    # ----------------------------
    save_dir = "/data1/user/zhangshaolong/3D_ADLLMresults_vis"
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(18, 6))

    # GT 可视化
    ax1 = fig.add_subplot(131, projection='3d')
    colors_gt = np.where(gt_mask, 'red', 'lightgray')
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors_gt, s=5)
    ax1.set_title("GT Mask (Red = Affordance)")

    # 预测可视化
    ax2 = fig.add_subplot(132, projection='3d')
    colors_pred = np.where(mask, 'blue', 'lightgray')
    ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors_pred, s=5)
    ax2.set_title("Predicted Mask (Blue = Affordance)")

    # 差异对比
    ax3 = fig.add_subplot(133, projection='3d')
    colors_diff = np.full(len(pts), 'lightgray')
    colors_diff[same] = 'green'   # 正确预测
    colors_diff[miss] = 'red'     # 漏检
    colors_diff[false] = 'blue'   # 误检
    ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors_diff, s=5)
    ax3.set_title("Comparison (Green=Correct, Red=Miss, Blue=False)")

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    plt.suptitle(f"Affordance Comparison: {data_input['instruction']}")
    save_path = os.path.join(save_dir, "affordance_compare.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 对比图已保存到: {save_path}")
