import torch
import re
import numpy as np
from typing import Dict, List
import logging
import os

# 引入你的模型定义 (确保路径能被 python 找到)
from models.point_phi_model import AffordancePhiMGRPON  # 假设你的模型类在这个路径

class AffordanceRewardManager:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AffordanceRewardManager, cls).__new__(cls)
        return cls._instance

    def init_model(self, model_path, device="cuda"):
        """
        核心优化：只加载 PointEncoder 和 Decoder，不加载 LLM，节省显存。
        """
        if self._model is not None:
            return

        print(f"[RewardManager] Initializing Reward Model components from {model_path}...")
        
        # 1. 初始化空模型架构 (配置需要与训练时一致)
        # 这里的 config 需要能够复现你的模型结构
        # 技巧：如果你的模型支持 `load_from_pretrained` 且能控制只加载部分模块最好
        # 这里假设我们实例化完整类，但稍后通过 strict=False 加载权重，或者手动清理 LLM 权重
        
        # 为了极简，我们假设你有一个专门只构建 Encoder+Decoder 的辅助函数或配置
        # 这里演示加载完整架构但利用 delete 释放 LLM 显存的“土办法”
        
        try:
            # 实例化模型 (此时是在 CPU)
            # 注意：你需要传入正确的 args，这里简化处理
            model = AffordancePhiMGRPON(
                point_model_config_path="3D_ADLLM/configs/models/PointTransformer_2048point.yaml",
                # ... 其他必要的 config ...
                llm_model="microsoft/Phi-3-mini-4k-instruct", # 只需要 config，不需要加载真实权重
                freeze_llm=True,
                train_aff_decoder=False # 推理模式
            )
            
            # 2. 加载训练好的权重 (Checkpoint)
            # 通常 GRPO 训练初期用的是预训练好的 Encoder+Decoder
            state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            
            # 3. 显存优化：彻底删除 LLM 部分，只保留 Encoder 和 Decoder
            # 假设 model.llm_model 是 LLM 主干
            del model.llm_model
            # torch.cuda.empty_cache()
            
            # 4. 将保留的组件移动到 GPU
            model.point_encoder.to(device).eval()
            model.seg_point_encoder.to(device).eval() # 如果有独立的 backbone
            model.aff_model.to(device).eval()         # Decoder
            model.aff_proj.to(device).eval()          # Projector
            model.fuse_single.to(device).eval()       # Token Fuser
            
            # 保存必要的引用
            self._model = model
            self.device = device
            print("[RewardManager] Model loaded and LLM part stripped for efficiency.")
            
        except Exception as e:
            logging.error(f"Failed to load reward model: {e}")
            raise e

    @torch.no_grad()
    def compute_iou(self, points, token_embeddings, gt_mask):
        """
        执行点云解码和 IoU 计算
        """
        model = self._model
        
        # 1. Point Encoding
        # points: [1, 2048, 3]
        points_tensor = torch.tensor(points, dtype=torch.float16, device=self.device).unsqueeze(0)
        
        # 如果 Decoder 需要的是 PointBackbone 的特征
        # 注意：这里需要与你训练时的 forward 逻辑保持一致
        points_permuted = points_tensor.transpose(1, 2) # [1, 3, N]
        point_feat = model.seg_point_encoder(points_permuted) # [1, N, C]
        
        # 2. Decoder Forward
        # token_embeddings: [1, C] (这是 fuse 之后的 embedding)
        pred_mask_logits = model.aff_model(
            pointcloud_embeddings=point_feat.unsqueeze(0), # 可能需要调整维度
            pointcloud_emorigin=point_feat.unsqueeze(0),
            sparse_prompt_embeddings=token_embeddings.unsqueeze(1),
            multimask_output=False
        )
        
        # 3. Calculate IoU
        pred_mask = (torch.sigmoid(pred_mask_logits) > 0.5).float()
        gt_mask_tensor = torch.tensor(gt_mask, device=self.device).float()
        
        intersection = (pred_mask * gt_mask_tensor).sum()
        union = pred_mask.sum() + gt_mask_tensor.sum() - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        return iou.item()

    def get_token_embeddings_from_text(self, solution_str):
        """
        这是一个难点：GRPO 的输出是文本，我们怎么拿回 Token 的 Embedding？
        
        方案 A (低效): 重新跑一遍 LLM (Encoder) 拿 hidden states。
        方案 B (高效): 训练一个小的 Lookup Table 或者直接复用 model.llm_model.get_input_embeddings
                      但我们在上面把 LLM 删了。
                      
        **关键修正**: 
        为了拿到 Token Embedding，你**必须**保留 LLM 的 Embedding Layer 或者重新跑一遍 forward。
        鉴于你删除了 LLM，我们需要一种方式复原 Token Embedding。
        
        如果你的 Token 是 `<SEG_1>` 这种特殊 token，它们的 Embedding 是训练出来的。
        在 init_model 时，我们需要把 `model.llm_model.get_input_embeddings()` 的权重存下来！
        """
        pass # 具体逻辑见下方完整整合代码

# ------------------------------------------------------------------------------
# 真正的入口函数
# ------------------------------------------------------------------------------

# 全局管理器实例
reward_manager = AffordanceRewardManager()

def compute_affordance_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Verl 调用的入口函数。
    """
    # 1. 延迟初始化 (Lazy Init)
    # 确保只在 Worker 首次调用时加载模型
    if reward_manager._model is None:
        # 假设权重在 ref model 路径
        model_path = os.path.expandvars("$HOME/3D_ADLLM/model_pre/Phi_4_mini_instruct")
        reward_manager.init_model(model_path)

    # 2. 解析数据
    # 注意：在上一步数据处理中，我们需要确保 points 和 mask 被传进来了
    # Verl 通常把 data_source 或 ground_truth 传进来。
    # 我们假设在 make_map_fn 里把 points 和 masks 放进了 ground_truth 字典
    # 或者通过 extra_info 传递 (取决于 Verl 版本)
    
    # 这里做防御性编程
    points = None
    gt_mask = None
    
    # 尝试从各个可能的地方获取
    if isinstance(ground_truth, dict):
        points = ground_truth.get('points')
        gt_mask = ground_truth.get('masks')
    elif extra_info:
        points = extra_info.get('points')
        gt_mask = extra_info.get('masks')
        
    if points is None or gt_mask is None:
        return 0.0 # 数据缺失，返回 0

    # 3. 计算格式奖励 (Format Reward)
    format_score = 0.0
    pattern = r"\[SEG_\d+\]"
    tokens = re.findall(pattern, solution_str)
    if len(tokens) == 5: # 假设 N=5
        format_score += 0.5
    
    # 4. 计算几何奖励 (Geometric Reward)
    # 这里的难点是：如何从 solution_str (文本) 变回 Decoder 需要的 Embedding？
    # 因为 GRPO 输出的是纯文本，没有 Tensor。
    # 我们必须重新 Look up Embedding。
    
    # 获取 Token ID
    # 假设我们有 tokenizer
    # 这里为了演示，假设 manager 内部处理了 embedding lookup
    
    # ... (省略具体的 lookup 代码，见下文) ...
    
    # 模拟计算 IoU
    iou_score = reward_manager.compute_iou(points, token_embeddings, gt_mask)
    
    return format_score + iou_score * 2.0