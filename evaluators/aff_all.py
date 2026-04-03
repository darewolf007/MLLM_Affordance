"""
=====================================================================
3D-ADLLM 评估器 v2 —— 严格对齐 OpenAD 评估方式 + argmax Acc^c
=====================================================================

相比 aff_all.py 的改动:
  1. OpenADAccumulator 新增 shape_scores 收集器，
     存储每个 (shape_id, affordance) 的 sigmoid 连续分数
  2. compute() 新增 argmax Acc^c:
     按 shape_id 分组 → 每个点跨所有 affordance 取 argmax → 多分类准确率
     这与 OpenAD 原始代码的 Acc = total_correct / total_seen 完全对齐
  3. 评估器传入 shape_id 和 masks_scores

使用方法:
  config 中设置 eval_type: affordance_openad_align_v2
  数据集需返回 "label", "shape_id" 字段
=====================================================================
"""

import os
import logging
import torch
import json
import numpy as np
import pickle as pkl
from typing import Any, Dict, List, Tuple
from collections import defaultdict

from common.logger import MetricLogger, SmoothedValue
from common.registry import registry
from common.utils import get_rank


# =====================================================================
#  基础函数
# =====================================================================

def compute_iou(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """单样本二值 mask 的 IoU"""
    pred, gt = pred.squeeze().bool(), gt.squeeze().bool()
    inter = (pred & gt).sum().float()
    union = (pred | gt).sum().float()
    return (inter / (union + 1e-8)).item() if union > 0 else 1.0


def compute_point_metrics(pred: torch.Tensor, gt: torch.Tensor) -> Tuple[float, float, float]:
    """单样本的 Accuracy, Precision, Recall"""
    pred, gt = pred.squeeze().bool(), gt.squeeze().bool()
    tp = (pred & gt).sum().item()
    fp = (pred & ~gt).sum().item()
    fn = (~pred & gt).sum().item()
    tn = (~pred & ~gt).sum().item()
    total = tp + fp + fn + tn
    return (
        (tp + tn) / total if total > 0 else 0.0,
        tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        tp / (tp + fn) if (tp + fn) > 0 else 0.0,
    )


# =====================================================================
#  OpenAD-style 全局累积器 (v2: 支持 argmax Acc^c)
# =====================================================================

class OpenADAccumulator:
    """
    对齐 OpenAD 的全局累积方式，新增 argmax Acc^c 计算。

    argmax Acc^c 的计算流程:
      1. 评估时收集每个 (shape_id, affordance) 的 sigmoid 连续分数
      2. compute() 时按 shape_id 分组
      3. 对每个 shape 的每个点，跨所有 affordance 取 argmax → 预测类别
      4. 跟 GT 多分类标签对比 → Acc = correct / total
    """

    def __init__(self):
        # class-level 全局累积 (OpenAD-style)
        self.class_tp = defaultdict(int)
        self.class_fp = defaultdict(int)
        self.class_fn = defaultdict(int)
        self.class_tn = defaultdict(int)
        self.class_gt_count = defaultdict(int)

        # instance-level 逐样本收集
        self.instance_ious = []
        self.instance_accs = []
        self.instance_precs = []
        self.instance_recs = []
        self.instance_ap50s = []
        self.instance_acc05s = []

        # 全局计数 (用于算 overall accuracy)
        self.total_correct = 0
        self.total_points = 0

        # ── 新增: 用于 argmax Acc^c ──
        # {shape_id: {affordance_label: scores_tensor}}
        self.shape_scores = defaultdict(dict)
        # {shape_id: {affordance_label: gt_tensor}}
        self.shape_gt = defaultdict(dict)

        # 保存详细结果
        self.detailed_results = []

    def add(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        label: str,
        scores: torch.Tensor = None,
        shape_id: str = None,
        question: str = "",
        pred_text: str = "",
        gt_text: str = "",
    ):
        """
        添加一个样本的评估结果

        Args:
            pred: [N] 二值预测 mask
            gt:   [N] 二值 GT mask
            label: affordance 类别名 (如 "grasp", "sit")
            scores: [N] sigmoid 连续分数 (用于 argmax Acc^c)
            shape_id: 物体标识符 (用于按物体分组)
        """
        pred = pred.squeeze().bool()
        gt = gt.squeeze().bool()

        tp = (pred & gt).sum().item()
        fp = (pred & ~gt).sum().item()
        fn = (~pred & gt).sum().item()
        tn = (~pred & ~gt).sum().item()
        total = tp + fp + fn + tn

        # ── 全局累积 (OpenAD-style) ──
        self.class_tp[label] += tp
        self.class_fp[label] += fp
        self.class_fn[label] += fn
        self.class_tn[label] += tn
        self.class_gt_count[label] += (tp + fn)

        self.total_correct += (tp + tn)
        self.total_points += total

        # ── 收集 argmax 所需数据 ──
        if shape_id is not None and scores is not None:
            scores_cpu = scores.squeeze().float().cpu()
            gt_cpu = gt.float().cpu()
            # 如果同一 shape_id 有多个同 label 的样本，保留最后一个
            # (正常情况下每个 shape 每个 affordance 只出现一次)
            self.shape_scores[shape_id][label] = scores_cpu
            self.shape_gt[shape_id][label] = gt_cpu

        # ── 逐样本指标 (instance-level) ──
        iou = compute_iou(pred.float(), gt.float())
        acc = (tp + tn) / total if total > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        acc05 = 1.0 if iou > 0.5 else 0.0
        ap50 = acc05

        self.instance_ious.append(iou)
        self.instance_accs.append(acc)
        self.instance_precs.append(prec)
        self.instance_recs.append(rec)
        self.instance_ap50s.append(ap50)
        self.instance_acc05s.append(acc05)

        self.detailed_results.append({
            "label": label,
            "shape_id": shape_id,
            "iou": iou,
            "acc05": acc05,
            "point_acc": acc,
            "point_prec": prec,
            "point_rec": rec,
            "ap50": ap50,
            "question": question,
            "pred_text": pred_text,
            "gt_text": gt_text,
        })

    def _compute_argmax_acc(self, exclude_none: bool = True) -> Dict:
        """
        按 shape_id 分组，跨 affordance 做 argmax，计算多分类 Acc^c。

        对齐 OpenAD 原始代码:
            afford_pred = np.argmax(afford_pred, axis=2)  # 每点预测一个类
            Acc = total_correct / total_seen

        Returns:
            dict with argmax_Acc_c and related stats
        """
        # ── 诊断打印: 检查数据是否满足 argmax 计算条件 ──
        logging.info("\n  [Argmax Acc^c 诊断]")
        logging.info(f"    收集到 shape_id 数量: {len(self.shape_scores)}")
        if self.shape_scores:
            # 统计每个 shape 有多少种 affordance
            aff_counts = [len(v) for v in self.shape_scores.values()]
            logging.info(f"    每个 shape 的 affordance 数量: min={min(aff_counts)}, max={max(aff_counts)}, "
                         f"mean={np.mean(aff_counts):.1f}, median={np.median(aff_counts):.0f}")
            logging.info(f"    affordance >= 2 的 shape 数量: {sum(1 for c in aff_counts if c >= 2)} "
                         f"(可参与 argmax 计算)")
            logging.info(f"    affordance == 1 的 shape 数量: {sum(1 for c in aff_counts if c == 1)} "
                         f"(将被跳过)")

            # 显示前 5 个 shape 的详情
            logging.info(f"    前 5 个 shape 示例:")
            for i, (sid, aff_dict) in enumerate(self.shape_scores.items()):
                if i >= 5:
                    break
                aff_names = list(aff_dict.keys())
                logging.info(f"      {sid}: {len(aff_names)} affordances → {aff_names}")

            # 统计所有出现过的 affordance 类别
            all_aff = set()
            for aff_dict in self.shape_scores.values():
                all_aff.update(aff_dict.keys())
            logging.info(f"    总 affordance 类别数: {len(all_aff)}")
            logging.info(f"    类别列表: {sorted(all_aff)}")
        else:
            logging.info("    没有收集到 shape_scores 数据!")
            logging.info("    可能原因: 1) 数据集没有 shape_id 字段  2) 模型没有输出 masks_scores")

        if not self.shape_scores:
            return {"argmax_Acc_c": -1, "argmax_shapes": 0, "argmax_classes": 0}

        # 收集所有出现过的 affordance 类别
        all_labels = set()
        for aff_dict in self.shape_scores.values():
            all_labels.update(aff_dict.keys())

        if exclude_none:
            all_labels -= {"none", "background", "bg", "None", "Background"}

        all_labels = sorted(all_labels)
        label2idx = {name: i for i, name in enumerate(all_labels)}
        num_classes = len(all_labels)

        if num_classes == 0:
            return {"argmax_Acc_c": -1, "argmax_shapes": 0, "argmax_classes": 0}

        total_correct = 0
        total_seen = 0
        # 也可以按类累积，用于计算 per-class argmax accuracy
        class_correct = defaultdict(int)
        class_seen = defaultdict(int)

        shapes_used = 0

        for shape_id in self.shape_scores:
            aff_scores = self.shape_scores[shape_id]
            aff_gt = self.shape_gt[shape_id]

            # 检查该 shape 有多少个 affordance 的预测
            available_labels = [l for l in aff_scores if l in label2idx]
            if len(available_labels) < 2:
                # 只有一个 affordance 的预测，无法做有意义的 argmax
                continue

            # 获取点数 N (取第一个 affordance 的分数长度)
            first_label = available_labels[0]
            N = aff_scores[first_label].shape[0]

            # 构建 [C_available, N] 的分数矩阵和 GT 矩阵
            # 只用该 shape 实际有预测的 affordance
            scores_matrix = torch.zeros(num_classes, N)
            gt_matrix = torch.zeros(num_classes, N)

            for lbl in available_labels:
                idx = label2idx[lbl]
                s = aff_scores[lbl]
                g = aff_gt[lbl]
                # 处理长度不匹配的情况
                n = min(s.shape[0], N)
                scores_matrix[idx, :n] = s[:n]
                gt_matrix[idx, :n] = g[:n]

            # argmax: 每个点预测属于哪个 affordance
            pred_class = scores_matrix.argmax(dim=0)  # [N]
            gt_class = gt_matrix.argmax(dim=0)         # [N]

            # 对于 GT 全为 0 的点(不属于任何 affordance)，标记为 "none" 类
            # OpenAD 包含 none 类，但论文说我们排除 none
            gt_has_any = gt_matrix.sum(dim=0) > 0      # [N] 是否有任何 affordance 标注

            if exclude_none:
                # 只在有 GT 标注的点上计算准确率
                valid_mask = gt_has_any
            else:
                valid_mask = torch.ones(N, dtype=torch.bool)

            if valid_mask.sum() == 0:
                continue

            correct = (pred_class[valid_mask] == gt_class[valid_mask]).sum().item()
            seen = valid_mask.sum().item()

            total_correct += correct
            total_seen += seen
            shapes_used += 1

            # 按 GT 类别累积
            for pt_idx in range(N):
                if not valid_mask[pt_idx]:
                    continue
                gt_label = all_labels[gt_class[pt_idx].item()]
                class_seen[gt_label] += 1
                if pred_class[pt_idx] == gt_class[pt_idx]:
                    class_correct[gt_label] += 1

        argmax_acc = total_correct / (total_seen + 1e-8) * 100 if total_seen > 0 else 0.0

        # 按类计算 argmax mAcc
        class_argmax_accs = []
        for lbl in all_labels:
            if class_seen[lbl] > 0:
                class_argmax_accs.append(class_correct[lbl] / class_seen[lbl])
        argmax_macc = np.mean(class_argmax_accs) * 100 if class_argmax_accs else 0.0

        return {
            "argmax_Acc_c": argmax_acc,
            "argmax_mAcc_c": argmax_macc,
            "argmax_shapes": shapes_used,
            "argmax_classes": num_classes,
            "argmax_total_correct": total_correct,
            "argmax_total_seen": total_seen,
        }

    def compute(self, exclude_none: bool = True) -> Dict:
        """
        计算所有指标

        Returns:
            dict with keys: instance, class_openad, per_class, argmax
        """
        # ============================================
        # Instance-level (Table 2) — 逐样本平均
        # ============================================
        instance = {
            "mIoU_i":  np.mean(self.instance_ious) * 100 if self.instance_ious else 0,
            "mAcc_i":  np.mean(self.instance_accs) * 100 if self.instance_accs else 0,
            "mPrec_i": np.mean(self.instance_precs) * 100 if self.instance_precs else 0,
            "mRec_i":  np.mean(self.instance_recs) * 100 if self.instance_recs else 0,
            "mAP50_i": np.mean(self.instance_ap50s) * 100 if self.instance_ap50s else 0,
        }

        # ============================================
        # Class-level (Table 1) — OpenAD 全局累积方式
        # ============================================
        class_ious = []
        class_accs = []
        per_class = {}

        for cls_name in sorted(self.class_tp.keys()):
            if exclude_none and cls_name.lower() in ["none", "background", "bg"]:
                continue

            tp = self.class_tp[cls_name]
            fp = self.class_fp[cls_name]
            fn = self.class_fn[cls_name]
            tn = self.class_tn[cls_name]
            gt_count = self.class_gt_count[cls_name]

            union = tp + fp + fn
            cls_iou = tp / (union + 1e-6) if union > 0 else 0.0
            cls_acc = tp / (gt_count + 1e-6) if gt_count > 0 else 0.0
            cls_prec = tp / (tp + fp + 1e-6) if (tp + fp) > 0 else 0.0

            class_ious.append(cls_iou)
            class_accs.append(cls_acc)

            per_class[cls_name] = {
                "IoU": cls_iou * 100,
                "Acc(Recall)": cls_acc * 100,
                "Precision": cls_prec * 100,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "GT_points": gt_count,
                "n_samples": sum(1 for r in self.detailed_results if r["label"] == cls_name),
            }

        overall_acc = self.total_correct / (self.total_points + 1e-8) * 100

        total_tp_all = sum(
            self.class_tp[c] for c in self.class_tp
            if not (exclude_none and c.lower() in ["none", "background", "bg"])
        )
        total_gt_all = sum(
            self.class_gt_count[c] for c in self.class_gt_count
            if not (exclude_none and c.lower() in ["none", "background", "bg"])
        )
        acc_c_approx = total_tp_all / (total_gt_all + 1e-8) * 100

        class_openad = {
            "mIoU_c": np.mean(class_ious) * 100 if class_ious else 0,
            "Acc_c_binary": overall_acc,
            "Acc_c_openad_approx": acc_c_approx,
            "mAcc_c": np.mean(class_accs) * 100 if class_accs else 0,
            "num_classes": len(class_ious),
        }

        # ============================================
        # Argmax Acc^c — 对齐 OpenAD 原始 Acc 计算
        # ============================================
        argmax_metrics = self._compute_argmax_acc(exclude_none=exclude_none)
        class_openad["Acc_c"] = argmax_metrics["argmax_Acc_c"]
        class_openad["argmax_mAcc_c"] = argmax_metrics["argmax_mAcc_c"]

        return {
            "instance": instance,
            "class_openad": class_openad,
            "per_class": per_class,
            "argmax": argmax_metrics,
        }


# =====================================================================
#  在线评估器 v2
# =====================================================================

@registry.register_evaluator("affordance_openad_align")
class AffordanceOpenADAlignEval:
    """
    严格对齐 OpenAD 的评估器 v2

    相比 v1 的改动:
      - 传入 shape_id 和 masks_scores 到 accumulator
      - 计算 argmax Acc^c (对齐 OpenAD 原始 Acc = total_correct / total_seen)

    config 中使用:
      eval_type: affordance_openad_align_v2

    数据集需返回 "label", "shape_id" 字段
    """

    def __init__(self, name) -> None:
        self.name = name

    def __call__(self, model, dataloader, dir, print_freq=100) -> Dict:
        logging.info(f"Start evaluating on {self.name}")

        accumulator = OpenADAccumulator()

        metric_logger = MetricLogger(delimiter="  ")
        for key in ["iou", "acc5", "pointAcc", "pointPrec", "pointRec", "mAP50"]:
            metric_logger.add_meter(
                key, SmoothedValue(fmt=f"global_{key}: {{global_avg:.4f}}")
            )

        for samples in metric_logger.log_every(dataloader, print_freq, self.name):
            samples.update({"category": "ref"})
            output = model.generate(samples, num_beams=1, max_length=30)

            pred_masks = output["masks"]
            masks_scores = output.get("masks_scores", [None] * len(pred_masks))
            answer_texts = output["text"]

            batch_ious = []

            for idx in range(len(pred_masks)):
                pred = pred_masks[idx]
                gt = samples["masks"][idx]

                n_pred, n_gt = pred.shape[0], gt.shape[0]
                if n_gt == 0:
                    continue
                if n_pred > n_gt:
                    pred = pred[:n_gt]

                labels = samples.get("label", [None] * len(pred_masks))
                label = labels[idx] if isinstance(labels, list) and idx < len(labels) else "unknown"

                shape_ids = samples.get("shape_id", [None] * len(pred_masks))
                shape_id = shape_ids[idx] if isinstance(shape_ids, list) and idx < len(shape_ids) else None

                score = masks_scores[idx] if idx < len(masks_scores) else None

                for m_idx in range(min(n_pred, n_gt)):
                    s = score[m_idx] if score is not None and m_idx < score.shape[0] else None
                    accumulator.add(
                        pred=pred[m_idx],
                        gt=gt[m_idx],
                        label=label,
                        scores=s,
                        shape_id=shape_id,
                        question=samples["question"][idx],
                        pred_text=answer_texts[idx],
                        gt_text=samples["answer"][idx],
                    )
                    batch_ious.append(accumulator.instance_ious[-1])

            if batch_ious:
                metric_logger.update(
                    iou=np.mean(batch_ious) * 100,
                    acc5=np.mean(accumulator.instance_acc05s[-len(batch_ious):]) * 100,
                    pointAcc=np.mean(accumulator.instance_accs[-len(batch_ious):]) * 100,
                    pointPrec=np.mean(accumulator.instance_precs[-len(batch_ious):]) * 100,
                    pointRec=np.mean(accumulator.instance_recs[-len(batch_ious):]) * 100,
                    mAP50=np.mean(accumulator.instance_ap50s[-len(batch_ious):]) * 100,
                )

        # ── 计算最终指标 ──
        metrics = accumulator.compute(exclude_none=True)

        # ── 日志 ──
        _log_final_metrics(self.name, metrics)

        # ── 保存 ──
        _save_final_results(dir, self.name, accumulator, metrics)

        metric_logger.synchronize_between_processes()

        # 返回给 Trainer
        result = {}
        result.update(metrics["instance"])
        result.update(metrics["class_openad"])
        result["agg_metrics"] = metrics["instance"]["mIoU_i"]
        return result


# =====================================================================
#  日志输出
# =====================================================================

def _log_final_metrics(name: str, metrics: Dict):
    inst = metrics["instance"]
    cls = metrics["class_openad"]
    per_cls = metrics["per_class"]
    argmax = metrics["argmax"]

    logging.info("=" * 75)
    logging.info(f"  {name} — Final Results (OpenAD-aligned v2)")
    logging.info("=" * 75)

    logging.info("\n  [Instance-level] (Table 2) — 逐样本平均")
    logging.info(f"    mIoU^i   = {inst['mIoU_i']:.2f}%")
    logging.info(f"    mAcc^i   = {inst['mAcc_i']:.2f}%")
    logging.info(f"    mPrec^i  = {inst['mPrec_i']:.2f}%")
    logging.info(f"    mRec^i   = {inst['mRec_i']:.2f}%")
    logging.info(f"    mAP50^i  = {inst['mAP50_i']:.2f}%")

    if cls:
        logging.info(f"\n  [Class-level] (Table 1) — OpenAD 全局累积 ({cls['num_classes']} classes)")
        logging.info(f"    mIoU^c       = {cls['mIoU_c']:.2f}%   ← mean(TP_c / union_c)")
        logging.info(f"    Acc^c        = {cls['Acc_c']:.2f}%   ← argmax 多分类准确率 (对齐 OpenAD)")
        logging.info(f"    Acc^c_binary = {cls['Acc_c_binary']:.2f}%   ← 二值: (TP+TN) / total")
        logging.info(f"    Acc^c_approx = {cls['Acc_c_openad_approx']:.2f}%   ← TP / GT_pos")
        logging.info(f"    mAcc^c       = {cls['mAcc_c']:.2f}%   ← mean(TP_c / GT_c)")

    if argmax["argmax_shapes"] > 0:
        logging.info(f"\n  [Argmax Acc^c Details]")
        logging.info(f"    shapes used  = {argmax['argmax_shapes']}")
        logging.info(f"    classes      = {argmax['argmax_classes']}")
        logging.info(f"    correct/seen = {argmax['argmax_total_correct']}/{argmax['argmax_total_seen']}")
        logging.info(f"    Acc^c        = {argmax['argmax_Acc_c']:.2f}%")
        logging.info(f"    mAcc^c(argmax) = {argmax['argmax_mAcc_c']:.2f}%")
    else:
        logging.info("\n  [Argmax Acc^c] skipped — no shape_id or scores available")

    if per_cls:
        logging.info(f"\n  [Per-class Details]")
        logging.info(
            f"    {'Class':25s} | {'IoU':>7s} | {'Acc(Rec)':>8s} | {'Prec':>7s} | "
            f"{'TP':>8s} | {'FP':>8s} | {'FN':>8s} | {'GT_pts':>8s} | {'N':>5s}"
        )
        logging.info(f"    {'-' * 100}")

        sorted_cls = sorted(per_cls.items(), key=lambda x: x[1]["IoU"], reverse=True)
        for cls_name, d in sorted_cls:
            logging.info(
                f"    {cls_name:25s} | {d['IoU']:6.2f}% | {d['Acc(Recall)']:7.2f}% | "
                f"{d['Precision']:6.2f}% | {d['TP']:8d} | {d['FP']:8d} | "
                f"{d['FN']:8d} | {d['GT_points']:8d} | {d['n_samples']:5d}"
            )

    logging.info("=" * 75)


# =====================================================================
#  保存结果
# =====================================================================

def _save_final_results(dir: str, name: str, accumulator: OpenADAccumulator, metrics: Dict):
    result_dir = os.path.join(dir, name)
    os.makedirs(result_dir, exist_ok=True)

    # 保存详细结果
    with open(os.path.join(result_dir, f"{get_rank()}_detailed.json"), "w") as f:
        json.dump(accumulator.detailed_results, f, indent=2)

    # 保存汇总
    summary = {
        "instance_level": metrics["instance"],
        "class_level_openad": metrics["class_openad"],
        "argmax_acc": metrics["argmax"],
        "per_class": {
            k: {kk: vv for kk, vv in v.items()}
            for k, v in metrics["per_class"].items()
        },
    }
    with open(os.path.join(result_dir, f"{get_rank()}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(f"Results saved to {result_dir}")


# =====================================================================
#  离线评估 (从 aff_eval 的 pkl 文件)
# =====================================================================

def offline_evaluate(pkl_path: str):
    """
    从 aff_eval 保存的 pkl 文件离线计算所有指标

    注意: 要计算 argmax Acc^c，pkl 中的 pred_mask 需要是 sigmoid 连续分数，
    而不是二值化后的 0/1。如果是二值化的，argmax Acc^c 可能不准确。
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with open(pkl_path, "rb") as f:
        data = pkl.load(f)

    logging.info(f"Loaded {len(data)} samples from {pkl_path}")

    accumulator = OpenADAccumulator()

    for sample in data:
        label = sample.get("label", "unknown")
        shape_id = sample.get("shape_id", None)
        pred_mask = sample["pred_mask"]
        gt_mask = sample["GT_masks"]

        if not isinstance(pred_mask, torch.Tensor):
            pred_mask = torch.tensor(pred_mask).float()
        if not isinstance(gt_mask, torch.Tensor):
            gt_mask = torch.tensor(gt_mask).float()

        n_pred, n_gt = pred_mask.shape[0], gt_mask.shape[0]
        if n_gt == 0:
            continue
        if n_pred > n_gt:
            pred_mask = pred_mask[:n_gt]

        for m_idx in range(min(n_pred, n_gt)):
            # pred_mask 可能是连续分数或二值，都可以用
            # 二值预测用于 TP/FP/FN 计算
            pred_binary = (pred_mask[m_idx] > 0.5) if pred_mask[m_idx].max() <= 1.0 else pred_mask[m_idx]
            accumulator.add(
                pred=pred_binary,
                gt=gt_mask[m_idx],
                label=label,
                scores=pred_mask[m_idx],  # 连续分数用于 argmax
                shape_id=shape_id,
                question=sample.get("question", ""),
                pred_text=sample.get("answer", ""),
                gt_text="",
            )

    metrics = accumulator.compute(exclude_none=True)
    _log_final_metrics("Offline Evaluation", metrics)
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type=str, required=True)
    args = parser.parse_args()
    offline_evaluate(args.pkl_path)
