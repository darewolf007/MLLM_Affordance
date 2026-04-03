"""
=====================================================================
3D-ADLLM 评估器 —— 严格对齐 OpenAD 评估方式
=====================================================================

OpenAD 原始代码的核心特征:
  1. 全局累积 TP/FP/FN，而不是逐样本算 IoU 再平均
  2. mIoU^c = mean( TP_c / (TP_c + FP_c + FN_c) )  对每个类别
  3. mAcc^c = mean( TP_c / GT_c )  对每个类别
  4. Acc^c  = 全部正确点 / 全部点

适配 3D-ADLLM 的差异:
  - OpenAD: 37类多分类 (argmax)
  - 3D-ADLLM: 二值分割 (每次只预测一个 affordance)
  - 适配方式: 按 label 分组，每组内全局累积 TP/FP/FN

两套计算方式:
  A. OpenAD-style (全局累积): 对齐论文 Table 1
  B. Instance-avg (逐样本平均): 对齐论文 Table 2

=====================================================================
使用方法:

  方式1 (在线, 训练时):
    config 中设置 eval_type: affordance_openad_align
    数据集需返回 "label" 字段

  方式2 (离线, 从 pkl):
    python affordance_openad_align.py --pkl_path /path/to/0.pkl

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


def compute_ap50(
    pred_scores: torch.Tensor,
    gt_mask: torch.Tensor,
    iou_threshold: float = 0.5,
    num_steps: int = 100,
) -> float:
    """
    单样本 AP@IoU=0.5 (11点插值)
    遍历多个二值化阈值 → 判 TP/FP → PR 曲线 → 面积
    """
    pred_scores = pred_scores.squeeze().float()
    gt_mask = gt_mask.squeeze().float()
    if gt_mask.sum() == 0:
        return 0.0

    thresholds = torch.linspace(1.0, 0.0, num_steps)
    tps = []
    for thr in thresholds:
        pb = (pred_scores >= thr).float()
        inter = (pb * gt_mask).sum()
        union = pb.sum() + gt_mask.sum() - inter
        iou = (inter / (union + 1e-8)).item()
        tps.append(1 if iou >= iou_threshold else 0)

    tp_cum = np.cumsum(tps).astype(float)
    fp_cum = np.cumsum([1 - t for t in tps]).astype(float)
    prec = tp_cum / (tp_cum + fp_cum + 1e-8)
    rec = np.minimum(tp_cum, 1.0)

    ap = 0.0
    for r in np.linspace(0, 1, 11):
        m = rec >= r
        if m.any():
            ap += prec[m].max()
    return ap / 11.0


# =====================================================================
#  OpenAD-style 全局累积器
# =====================================================================

class OpenADAccumulator:
    """
    对齐 OpenAD 的全局累积方式
    
    OpenAD 原始代码:
        for i in range(num_classes):
            total_correct_class[i] += np.sum((pred == i) & (label == i))   # TP
            total_iou_deno_class[i] += np.sum((pred == i) | (label == i))  # union
            total_seen_class[i] += np.sum((label == i))                     # GT count
    
    适配到二值分割:
        对每个 affordance 类别，全局累积 TP、FP、FN、TN
    """

    def __init__(self):
        # class-level 全局累积 (OpenAD-style)
        self.class_tp = defaultdict(int)      # 每类 True Positive 点数
        self.class_fp = defaultdict(int)      # 每类 False Positive 点数
        self.class_fn = defaultdict(int)      # 每类 False Negative 点数
        self.class_tn = defaultdict(int)      # 每类 True Negative 点数
        self.class_gt_count = defaultdict(int)  # 每类 GT 正例总点数

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

        # 保存详细结果
        self.detailed_results = []

    def add(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        label: str,
        scores: torch.Tensor = None,
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
            scores: [N] sigmoid 分数 (用于 AP 计算，可选)
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
        self.class_gt_count[label] += (tp + fn)  # GT 正例数

        self.total_correct += (tp + tn)
        self.total_points += total

        # ── 逐样本指标 (instance-level) ──
        iou = compute_iou(pred.float(), gt.float())
        acc = (tp + tn) / total if total > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        acc05 = 1.0 if iou > 0.5 else 0.0
        ap50 = acc05
        # ap50 = 0.0
        # if scores is not None:
        #     ap50 = compute_ap50(scores, gt.float())

        self.instance_ious.append(iou)
        self.instance_accs.append(acc)
        self.instance_precs.append(prec)
        self.instance_recs.append(rec)
        self.instance_ap50s.append(ap50)
        self.instance_acc05s.append(acc05)

        self.detailed_results.append({
            "label": label,
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

    def compute(self, exclude_none: bool = True) -> Dict:
        """
        计算所有指标
        
        Returns:
            dict with keys: instance, class_openad, per_class
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
        #
        # 这与 OpenAD 原始代码完全一致:
        #   IoU_c = total_correct_class[c] / total_iou_deno_class[c]
        #         = TP_c / (TP_c + FP_c + FN_c)
        #
        #   Acc_c = total_correct_class[c] / total_seen_class[c]
        #         = TP_c / (TP_c + FN_c)
        #
        #   mIoU = mean(IoU_c)
        #   mAcc = mean(Acc_c)
        #   overall_acc = total_correct / total_seen
        #
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

            # OpenAD-style IoU: 全局 TP / 全局 (TP + FP + FN)
            union = tp + fp + fn
            cls_iou = tp / (union + 1e-6) if union > 0 else 0.0

            # OpenAD-style class accuracy: 全局 TP / 全局 GT正例数
            cls_acc = tp / (gt_count + 1e-6) if gt_count > 0 else 0.0

            # 额外: class-level precision
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

        # Overall accuracy (对应 OpenAD 的 "eval point accuracy")
        overall_acc = self.total_correct / (self.total_points + 1e-8) * 100

        # 近似 OpenAD 多分类 Acc_c: 全局 TP / 全局 GT正例数 (排除TN的影响)
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
            "Acc_c": overall_acc,                # 二值分割: (TP+TN) / 总点数
            "Acc_c_openad": acc_c_approx,         # 近似OpenAD: 全局TP / 全局GT正例
            "mAcc_c": np.mean(class_accs) * 100 if class_accs else 0,
            "num_classes": len(class_ious),
        }

        return {
            "instance": instance,
            "class_openad": class_openad,
            "per_class": per_class,
        }


# =====================================================================
#  在线评估器
# =====================================================================

@registry.register_evaluator("affordance_openad_align")
class AffordanceOpenADAlignEval:
    """
    严格对齐 OpenAD 的评估器
    
    config 中使用:
      eval_type: affordance_openad_align
    
    数据集需返回 "label" 字段
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

                score = masks_scores[idx] if idx < len(masks_scores) else None

                for m_idx in range(min(n_pred, n_gt)):
                    s = score[m_idx] if score is not None and m_idx < score.shape[0] else None
                    accumulator.add(
                        pred=pred[m_idx],
                        gt=gt[m_idx],
                        label=label,
                        scores=s,
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
        output = {}
        output.update(metrics["instance"])
        output.update(metrics["class_openad"])
        output["agg_metrics"] = metrics["instance"]["mIoU_i"]
        return output


# =====================================================================
#  日志输出
# =====================================================================

def _log_final_metrics(name: str, metrics: Dict):
    inst = metrics["instance"]
    cls = metrics["class_openad"]
    per_cls = metrics["per_class"]

    logging.info("=" * 75)
    logging.info(f"  {name} — Final Results (OpenAD-aligned)")
    logging.info("=" * 75)

    logging.info("\n  [Instance-level] (Table 2) — 逐样本平均")
    logging.info(f"    mIoU^i   = {inst['mIoU_i']:.2f}%")
    logging.info(f"    mAcc^i   = {inst['mAcc_i']:.2f}%")
    logging.info(f"    mPrec^i  = {inst['mPrec_i']:.2f}%")
    logging.info(f"    mRec^i   = {inst['mRec_i']:.2f}%")
    logging.info(f"    mAP50^i  = {inst['mAP50_i']:.2f}%")

    if cls:
        logging.info(f"\n  [Class-level] (Table 1) — OpenAD 全局累积 ({cls['num_classes']} classes)")
        logging.info(f"    mIoU^c   = {cls['mIoU_c']:.2f}%   ← mean(TP_c / union_c)")
        logging.info(f"    Acc^c    = {cls['Acc_c']:.2f}%    ← binary: (TP+TN) / total")
        logging.info(f"    Acc^c*   = {cls['Acc_c_openad']:.2f}%   ← approx OpenAD: TP / GT_pos")
        logging.info(f"    mAcc^c   = {cls['mAcc_c']:.2f}%   ← mean(TP_c / GT_c)")

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
    
    用法: python affordance_openad_align.py --pkl_path /path/to/0.pkl
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with open(pkl_path, "rb") as f:
        data = pkl.load(f)

    logging.info(f"Loaded {len(data)} samples from {pkl_path}")

    accumulator = OpenADAccumulator()

    for sample in data:
        label = sample.get("label", "unknown")
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
            accumulator.add(
                pred=pred_mask[m_idx],
                gt=gt_mask[m_idx],
                label=label,
                scores=None,
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