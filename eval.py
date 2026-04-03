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

# =====================================================================
#  参数解析
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Evaluation")
    parser.add_argument(
        "--cfg-path",
        required=True,
        help="path to configuration file (same format as training config)",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="path to model checkpoint (.pth file or directory)",
    )
    parser.add_argument(
        "--checkpoint-type",
        choices=["state_dict", "accelerate", "deepspeed", "auto"],
        default="auto",
        help="checkpoint format (default: auto-detect)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="directory to save evaluation results (default: outputs/eval_<timestamp>)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="override eval batch size from config",
    )
    parser.add_argument(
        "--eval-type",
        default=None,
        help="override eval_type for all eval datasets (e.g., affordance_openad_align)",
    )
    parser.add_argument(
        "--use-accelerate",
        action="store_true",
        help="use accelerate for multi-GPU evaluation",
    )
    return parser.parse_args()


# =====================================================================
#  复用 train.py 的构建函数
# =====================================================================

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def build_model(cfg):
    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    return model_cls.from_config(model_config)


def build_eval_dataset(cfg, eval_type_override=None):
    datasets = []
    task_evals = []
    if cfg is not None:
        for name in cfg:
            dataset_config = cfg[name]
            builder_func = registry.get_builder_func(dataset_config["type"])
            dataset = builder_func(name, dataset_config.path)

            # 允许命令行覆盖 eval_type
            eval_type = eval_type_override or dataset_config["eval_type"]
            eval_func = registry.get_evaluator_func(eval_type)
            task_eval = eval_func(name)

            logging.info(f"Loaded eval dataset: {name} ({len(dataset)} samples, eval_type={eval_type})")
            datasets.append(dataset)
            task_evals.append(task_eval)
    return datasets, task_evals


# =====================================================================
#  Checkpoint 加载
# =====================================================================

def load_checkpoint(model, checkpoint_path, checkpoint_type="auto"):
    """
    加载 checkpoint 到模型
    
    支持三种格式:
      1. state_dict: 直接的 model.state_dict() 保存
      2. accelerate: accelerate.save_state() 保存的目录
      3. auto: 自动检测
    """
    if checkpoint_type == "auto":
        if os.path.isdir(checkpoint_path):
            checkpoint_type = "accelerate"
        elif checkpoint_path.endswith(".pth") or checkpoint_path.endswith(".pt"):
            checkpoint_type = "state_dict"
        elif checkpoint_path.endswith(".bin"):
            checkpoint_type = "state_dict"
        else:
            checkpoint_type = "state_dict"
        logging.info(f"Auto-detected checkpoint type: {checkpoint_type}")

    if checkpoint_type == "state_dict":
        logging.info(f"Loading state_dict from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        # 处理不同的保存格式
        if isinstance(ckpt, dict):
            if "model" in ckpt:
                state_dict = ckpt["model"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            elif "module" in ckpt:
                state_dict = ckpt["module"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        # 移除可能的 "module." 前缀 (DDP)
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                cleaned[k[7:]] = v
            else:
                cleaned[k] = v

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            logging.warning(f"Missing keys ({len(missing)}): {missing[:10]}...")
        if unexpected:
            logging.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}...")
        logging.info("State dict loaded successfully")

    elif checkpoint_type == "accelerate":
        logging.info(f"Loading accelerate checkpoint from: {checkpoint_path}")
        # accelerate 保存的格式通常在子目录中
        # 尝试找到 pytorch_model.bin 或 model.safetensors
        candidates = [
            os.path.join(checkpoint_path, "pytorch_model.bin"),
            os.path.join(checkpoint_path, "model.safetensors"),
            os.path.join(checkpoint_path, "pytorch_model-00001-of-00002.bin"),
        ]
        
        found = None
        for c in candidates:
            if os.path.exists(c):
                found = c
                break
        
        if found:
            load_checkpoint(model, found, "state_dict")
        else:
            # 尝试用 accelerate 的 load_state 方式
            try:
                from accelerate import load_checkpoint_in_model
                load_checkpoint_in_model(model, checkpoint_path)
                logging.info("Loaded via accelerate.load_checkpoint_in_model")
            except Exception as e:
                logging.error(f"Failed to load accelerate checkpoint: {e}")
                raise

    elif checkpoint_type == "deepspeed":
        logging.info(f"Loading DeepSpeed checkpoint from: {checkpoint_path}")
        # DeepSpeed 的 checkpoint 通常需要特殊处理
        ckpt_file = os.path.join(checkpoint_path, "mp_rank_00_model_states.pt")
        if os.path.exists(ckpt_file):
            load_checkpoint(model, ckpt_file, "state_dict")
        else:
            raise FileNotFoundError(f"DeepSpeed checkpoint not found at {ckpt_file}")

    return model


def run_evaluation(model, eval_datasets, task_evals, output_dir, cfg, batch_size=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).bfloat16()
    model.eval()

    bs = batch_size or cfg.run_cfg.get("batch_size_eval", 1)
    num_workers = cfg.run_cfg.get("num_workers", 4)

    all_results = {}

    for dataset, task_eval in zip(eval_datasets, task_evals):
        logging.info(f"\n{'='*60}")
        logging.info(f"Evaluating: {task_eval.name}")
        logging.info(f"{'='*60}")

        collate_fn = getattr(dataset, "collate", None)
        dataloader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        # 包一层 wrapper，在每个 batch 送入模型前转 bf16 + to(device)
        dataloader = Bf16DeviceLoader(dataloader, device)

        with torch.no_grad():
            results = task_eval(model, dataloader, output_dir)

        all_results[task_eval.name] = results

        logging.info(f"\n  Results for {task_eval.name}:")
        for k, v in results.items():
            if isinstance(v, (int, float)):
                logging.info(f"    {k}: {v:.4f}")

    summary_path = os.path.join(output_dir, "eval_summary.json")
    serializable = {}
    for name, res in all_results.items():
        serializable[name] = {
            k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
            for k, v in res.items()
        }
    with open(summary_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logging.info(f"\nSummary saved to {summary_path}")

    return all_results


class Bf16DeviceLoader:
    """包装 DataLoader，在迭代时自动把 batch 转成 bf16 并搬到 GPU"""
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        for batch in self.dataloader:
            yield self._move(batch)

    def __len__(self):
        return len(self.dataloader)

    def _move(self, data):
        if isinstance(data, torch.Tensor):
            if data.is_floating_point():
                return data.to(self.device, dtype=torch.bfloat16)
            else:
                return data.to(self.device)
        elif isinstance(data, dict):
            return {k: self._move(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._move(v) for v in data]
        elif isinstance(data, tuple):
            return tuple(self._move(v) for v in data)
        else:
            return data


def run_evaluation_with_accelerate(model, eval_datasets, task_evals, output_dir, cfg, batch_size=None):
    """使用 accelerate 进行多卡评估"""
    from accelerate import Accelerator

    accelerator = Accelerator()

    bs = batch_size or cfg.run_cfg.get("batch_size_eval", 1)
    num_workers = cfg.run_cfg.get("num_workers", 4)

    model = accelerator.prepare(model)
    model.eval()

    all_results = {}

    for dataset, task_eval in zip(eval_datasets, task_evals):
        logging.info(f"\nEvaluating: {task_eval.name}")

        dataloader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=getattr(dataset, "collater", None),
        )
        dataloader = accelerator.prepare(dataloader)

        with torch.no_grad():
            results = task_eval(model, dataloader, output_dir)

        all_results[task_eval.name] = results

    if accelerator.is_main_process:
        summary_path = os.path.join(output_dir, "eval_summary.json")
        serializable = {}
        for name, res in all_results.items():
            serializable[name] = {
                k: float(v) if isinstance(v, (int, float)) else str(v)
                for k, v in res.items()
            }
        with open(summary_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logging.info(f"Summary saved to {summary_path}")

    return all_results


def diagnose_embedding_tying(model):
    """检查 embed_tokens 和 lm_head 的权重是否仍然tied"""
    
    print("\n" + "="*70)
    print("  DIAGNOSTIC: Embedding/LM-Head Weight Tying Check")
    print("="*70)
    
    llm = model.llm_model
    
    # 1. 获取 input embedding 和 output embedding
    input_embed = llm.get_input_embeddings()
    output_embed = llm.get_output_embeddings()
    
    print(f"\n  Input embedding type:  {type(input_embed)}")
    print(f"  Output embedding type: {type(output_embed)}")
    
    # 2. 检查是否被PEFT包装
    from peft.tuners.tuners_utils import BaseTunerLayer
    
    has_wrapper_input = hasattr(input_embed, 'original_module')
    has_wrapper_output = hasattr(output_embed, 'original_module')
    print(f"\n  Input embed has PEFT wrapper:  {has_wrapper_input}")
    print(f"  Output embed has PEFT wrapper: {has_wrapper_output}")
    
    # 3. 获取实际权重
    def get_actual_weight(module):
        if hasattr(module, 'modules_to_save'):
            # PEFT ModulesToSaveWrapper
            active_adapter = module.active_adapter
            if isinstance(active_adapter, list):
                active_adapter = active_adapter[0]
            active_module = module.modules_to_save[active_adapter]
            return active_module.weight
        elif hasattr(module, 'weight'):
            return module.weight
        return None
    
    input_w = get_actual_weight(input_embed)
    output_w = get_actual_weight(output_embed)
    
    if input_w is not None and output_w is not None:
        same_object = input_w is output_w
        same_data = input_w.data_ptr() == output_w.data_ptr()
        
        print(f"\n  Same Python object: {same_object}")
        print(f"  Same data pointer:  {same_data}")
        print(f"  Input weight shape:  {input_w.shape}")
        print(f"  Output weight shape: {output_w.shape}")
        print(f"  Input requires_grad:  {input_w.requires_grad}")
        print(f"  Output requires_grad: {output_w.requires_grad}")
        
        if not same_data:
            print("\n  ⚠️⚠️⚠️ WARNING: TIED WEIGHTS ARE BROKEN! ⚠️⚠️⚠️")
            print("  embed_tokens and lm_head have SEPARATE weight tensors.")
            print("  This means:")
            print("    - CE loss gradient only updates lm_head copy")
            print("    - Input embedding uses a DIFFERENT copy")
            print("    - The two copies may diverge during training")
            print("    - This could explain the 17-point mIoU gap!")
            
            # 检查权重是否相同
            with __import__('torch').no_grad():
                diff = (input_w - output_w).abs().mean().item()
                print(f"\n  Mean absolute difference: {diff:.6f}")
                if diff < 1e-6:
                    print("  (Weights are currently identical - just initialized)")
                else:
                    print("  (Weights have already diverged!)")
        else:
            print("\n  ✅ Tied weights are preserved.")
    else:
        print("\n  ❌ Could not extract weights for comparison")
    
    # 4. 打印所有trainable参数中包含embed/lm_head的
    print("\n  Trainable params with 'embed' or 'lm_head':")
    for name, param in model.named_parameters():
        if param.requires_grad and ('embed' in name or 'lm_head' in name):
            print(f"    {name}: shape={list(param.shape)}, ptr={param.data_ptr()}")
    
    print("\n" + "="*70 + "\n")


# =====================================================================
#  Main
# =====================================================================

def main():
    args = parse_args()

    # 1. 加载配置
    class ArgsWrapper:
        def __init__(self, cfg_path):
            self.cfg_path = cfg_path
    
    cfg = LLMConfig(ArgsWrapper(args.cfg_path))
    cfg.device = "cuda"

    # 2. 设置输出目录
    output_dir = args.output_dir or path.join("outputs", f"eval_{now()}")
    os.makedirs(output_dir, exist_ok=True)

    # 3. 设置日志
    log_file = os.path.join(output_dir, "eval_log.txt")
    setup_logger(log_file)

    logging.info(f"Config: {args.cfg_path}")
    logging.info(f"Checkpoint: {args.checkpoint}")
    logging.info(f"Output: {output_dir}")

    # 4. 设置随机种子
    setup_seeds(cfg)

    # 5. 构建模型
    logging.info("Building model...")
    model = build_model(cfg)

    # 6. 加载 checkpoint
    logging.info("Loading checkpoint...")
    model = load_checkpoint(model, args.checkpoint, args.checkpoint_type)
    diagnose_embedding_tying(model)
    # 7. 构建评估数据集
    logging.info("Building eval datasets...")
    eval_datasets, task_evals = build_eval_dataset(
        cfg.eval_datasets_cfg,
        eval_type_override=args.eval_type,
    )

    if not eval_datasets:
        logging.error("No eval datasets found in config! Check eval_datasets_cfg.")
        sys.exit(1)

    # 8. 运行评估
    logging.info(f"Starting evaluation ({len(eval_datasets)} dataset(s))...")

    if args.use_accelerate:
        results = run_evaluation_with_accelerate(
            model, eval_datasets, task_evals, output_dir, cfg,
            batch_size=args.batch_size,
        )
    else:
        results = run_evaluation(
            model, eval_datasets, task_evals, output_dir, cfg,
            batch_size=args.batch_size,
        )

    logging.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()