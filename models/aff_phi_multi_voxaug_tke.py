"""
方案4E: Multi-Scale Voxel Ablation Study

Based on 方案4D (tkd), with 4 ablation flags to diagnose overfitting:
  1. voxel_uniform_pool_res > 0: uniform small pool for all scales (vs proportional)
  2. voxel_use_decoder: use decoder features (vs encoder)
  3. voxel_share_attn: share cross_attn across branches (reduce params)
  4. voxel_dropout > 0: dropout on cross-attention output

Each ablation can be enabled independently via config.
When all flags are off, behavior is identical to tkd.
When enable_voxel_token=False, the model is identical to the old version.
"""

import logging
from typing import List
import os
import yaml
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from common.registry import registry
from common.utils import get_rank
from models.base_model import BaseModel, disabled_train
from .AFD import build_AFD
from models.AFD.aff_utils.loss import dice_loss, sigmoid_ce_loss
from models.pointbert.point_encoder import PointTransformer
from models.openad.utils import build_model_checkpointfromddp
from gorilla.config import Config
from models.utils import Modify_cfg_from_yaml_file
from models.AFD.modeling.transformer import TokenAggregation, FeatureInjector

SEG_TOKEN1 = "<SEG_Global>"
SEG_TOKEN2 = "<SEG_Local>"
SEG_TOKEN3 = "<SEG_Detail>"


# ---------------------------------------------------------------------------
# VoxelTokenMultiScaleAblation: extends tkd's VoxelTokenMultiScale with
#   share_attn and dropout options for ablation study.
# ---------------------------------------------------------------------------
class VoxelTokenMultiScaleAblation(nn.Module):
    VOXEL_DIMS = [512, 128, 32]

    def __init__(self, hidden_dim, num_heads=8, share_attn=False, dropout=0.0):
        super().__init__()
        self.share_attn = share_attn

        # voxel_proj is always per-branch (different input dims)
        self.voxel_projs = nn.ModuleList([
            nn.Linear(vdim, hidden_dim) for vdim in self.VOXEL_DIMS
        ])

        if share_attn:
            # Shared cross_attn and gate across all 3 branches
            self.shared_cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
            )
            self.shared_gate = nn.Linear(hidden_dim, hidden_dim)
            nn.init.zeros_(self.shared_gate.weight)
            nn.init.constant_(self.shared_gate.bias, -2.0)
        else:
            # Per-branch cross_attn and gate (same as tkd)
            self.cross_attns = nn.ModuleList()
            self.gates = nn.ModuleList()
            for _ in self.VOXEL_DIMS:
                self.cross_attns.append(nn.MultiheadAttention(
                    embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
                ))
                gate = nn.Linear(hidden_dim, hidden_dim)
                nn.init.zeros_(gate.weight)
                nn.init.constant_(gate.bias, -2.0)
                self.gates.append(gate)

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, pred_tokens, multiscale_voxels):
        """
        Args:
            pred_tokens:       (B, 3, hidden_dim)
            multiscale_voxels: list of 3 tensors:
                [0]: (B, K, 512) — spatial tokens from feat_16
                [1]: (B, K, 128) — spatial tokens from feat_32
                [2]: (B, K,  32) — spatial tokens from feat_64
        Returns:
            enhanced: (B, 3, hidden_dim)
        """
        enhanced_list = []
        for j in range(3):
            token_j = pred_tokens[:, j:j+1, :]          # (B, 1, hidden_dim)
            voxel_feat = self.voxel_projs[j](multiscale_voxels[j])  # (B, K, hidden_dim)

            if self.share_attn:
                context, _ = self.shared_cross_attn(
                    query=token_j, key=voxel_feat, value=voxel_feat
                )
                gate = torch.sigmoid(self.shared_gate(context))
            else:
                context, _ = self.cross_attns[j](
                    query=token_j, key=voxel_feat, value=voxel_feat
                )
                gate = torch.sigmoid(self.gates[j](context))

            context = self.drop(context)
            enhanced_list.append(token_j + gate * context)
        return torch.cat(enhanced_list, dim=1)  # (B, 3, hidden_dim)


@registry.register_model("aff_phi_multi_voxaug_tke")
class AffordancePhiMVoxAugTKE(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "phi_default": "/data1/user/zhangshaolong/3D_ADLLM/configs/models/point_default.yaml",
    }

    def __init__(
        self,
        mix_precision="bf16",
        prompt_encoder_dim=32,
        # Loss
        ce_loss_weight=1.0,
        dice_loss_weight=1.0,
        bce_loss_weight=1.0,
        uselabelratio=False,
        # Point_encoder
        point_model_config_path=None,
        freeze_point=True,
        # Seg_Pointencoder(PointBackbone)
        free_seg_point_encoder=True,
        seg_point_encoder_config_path=None,
        seg_point_encoder_path=None,
        # AFD
        aff_path=None,
        train_aff_decoder=False,
        upscale_points=2048,
        # Lora
        lora_r=8,
        lora_alpha=16,
        target_modules=["qkv_proj"],
        # LLM
        llm_model=None,
        freeze_llm=True,
        prompt="",
        max_txt_len=256,
        max_output_txt_len=128,
        freeze_linear=True,
        label_ratio_path="3D_ADLLM/result_ratio.json",
        lora_llm_finetune=False,
        point_enhanced=False,
        learned_noise=False,
        noise_std=0.01,
        llm_model_type=None,
        # ===== Voxel Token MultiScale =====
        enable_voxel_token=False,
        vqvae_checkpoint=None,
        vqvae_num_embeddings=8192,
        voxel_spatial_res=4,
        enhance_feature_injector=False,
        # ===== [E] Ablation flags =====
        voxel_uniform_pool_res=0,   # >0: use this fixed R for all scales
        voxel_use_decoder=False,    # True: decoder features instead of encoder
        voxel_share_attn=False,     # True: share cross_attn across branches
        voxel_dropout=0.0,          # >0: dropout on cross-attn output
        **kwargs,
    ):
        super().__init__()
        self.mix_precision = mix_precision
        self.point_enhanced = point_enhanced
        self.learned_noise = learned_noise
        self.noise_std = noise_std
        self.llm_model_type = llm_model_type
        self.enable_voxel_token = enable_voxel_token
        self.voxel_spatial_res = voxel_spatial_res
        self.enhance_feature_injector = enhance_feature_injector
        # [E] ablation flags
        self.voxel_uniform_pool_res = voxel_uniform_pool_res
        self.voxel_use_decoder = voxel_use_decoder

        # Point_Encoder
        self.point_model_config = Modify_cfg_from_yaml_file(point_model_config_path)
        self.point_encoder = PointTransformer(self.point_model_config.model)
        self.point_encoder.load_checkpoint(self.point_model_config.model_path)

        # PointBackbone
        openadcfg = Config.fromfile(seg_point_encoder_config_path)
        self.seg_point_encoder = build_model_checkpointfromddp(
            openadcfg, seg_point_encoder_path, is_eval=False
        )
        self.upscale_points = upscale_points

        # Loss Weight
        self.ce_loss_weight = ce_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.bce_loss_weight = bce_loss_weight
        self.uselabelratio = uselabelratio

        if self.uselabelratio:
            with open(label_ratio_path, "r", encoding="utf-8") as ratio_file:
                self.label_ratio = json.load(ratio_file)

        if freeze_point:
            for name, param in self.point_encoder.named_parameters():
                param.requires_grad = False
            self.point_encoder = self.point_encoder.eval()
            self.point_encoder.train = disabled_train
            print("Freeze point encoder")

        if free_seg_point_encoder:
            for name, param in self.seg_point_encoder.named_parameters():
                param.requires_grad = False
            self.seg_point_encoder = self.seg_point_encoder.eval()
            self.seg_point_encoder.train = disabled_train
            print("Freeze Seg_point encoder (PointBackbone)")

        # LLM
        print("Start Load LLM")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            llm_model, trust_remote_code=True, use_fast=False, truncation_side="left"
        )
        self.llm_tokenizer.padding_side = "left"
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model, use_cache=False, trust_remote_code=True,
            attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map="cuda",
        )
        print("Load LLM Model Successfully")
        if freeze_llm:
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False
            print("Freeze LLM")

        if lora_llm_finetune:
            print("Using Lora LLM Finetune")
            lora_config = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, target_modules=target_modules,
                lora_dropout=0.1, bias="none", task_type=TaskType.CAUSAL_LM,
            )
            self.llm_model = get_peft_model(self.llm_model, lora_config)
            self.llm_model.print_trainable_parameters()

        # SEG tokens
        self.llm_model.base_model.model.model.embed_tokens.weight.requires_grad = True
        self.llm_tokenizer.add_tokens([SEG_TOKEN1])
        self.llm_tokenizer.add_tokens([SEG_TOKEN2])
        self.llm_tokenizer.add_tokens([SEG_TOKEN3])
        self.seg_token_id_1 = self.llm_tokenizer.convert_tokens_to_ids(SEG_TOKEN1)
        self.seg_token_id_2 = self.llm_tokenizer.convert_tokens_to_ids(SEG_TOKEN2)
        self.seg_token_id_3 = self.llm_tokenizer.convert_tokens_to_ids(SEG_TOKEN3)
        self.seg_token_ids = [self.seg_token_id_1, self.seg_token_id_2, self.seg_token_id_3]
        self.num_newtoken = len(self.seg_token_ids)
        self.hidden_dim = self.llm_model.config.hidden_size

        self.token_aggregation = TokenAggregation(embedding_dim=self.hidden_dim, num_heads=8)
        self.FeatureInjector = FeatureInjector(
            token_dim=self.hidden_dim, point_dim=prompt_encoder_dim, internal_dim=256
        )
        self.fuse_single = nn.Sequential(
            nn.Linear(self.hidden_dim, 1024), nn.ReLU(), nn.Linear(1024, 32)
        )

        # Legacy modules (kept for checkpoint compatibility)
        self.point_token_cross_attn = nn.MultiheadAttention(
            embed_dim=prompt_encoder_dim, num_heads=8, batch_first=True
        )
        self.token_weight = nn.Parameter(torch.ones(3))
        self.token_attn = nn.Linear(self.hidden_dim, 1)
        self.fuse = nn.Sequential(
            nn.Linear(self.hidden_dim * self.num_newtoken, 1024), nn.ReLU(), nn.Linear(1024, 32)
        )
        self.token_dim_reducer = nn.Linear(3072, 32)

        self.llm_proj = nn.Linear(
            self.point_model_config.model.trans_dim, self.llm_model.config.hidden_size
        )
        if freeze_linear:
            for name, param in self.llm_proj.named_parameters():
                param.requires_grad = False
            self.llm_proj = self.llm_proj.eval()
            self.llm_proj.train = disabled_train
            print("freeze point encoder to LLM liner")

        # AFD
        self.prompt_encoder_dim = prompt_encoder_dim
        self.aff_model, self.aff_proj = self.initialize_affordance_modules(
            aff_path, in_dim=self.llm_model.config.hidden_size,
            out_dim=self.prompt_encoder_dim, train_aff_decoder=train_aff_decoder,
        )

        # ===== VQVAE + VoxelTokenMultiScaleAblation =====
        if self.enable_voxel_token:
            from models.vqvae_utils import VQVAE3D, pointcloud_to_voxel
            self._pointcloud_to_voxel = pointcloud_to_voxel

            self.vqvae = VQVAE3D(num_embeddings=vqvae_num_embeddings)
            if vqvae_checkpoint is not None:
                state_dict = torch.load(vqvae_checkpoint, map_location="cpu", weights_only=True)
                self.vqvae.load_state_dict(state_dict)
                print(f"[E] Loaded VQVAE checkpoint from {vqvae_checkpoint}")
            self.vqvae.eval()
            for p in self.vqvae.parameters():
                p.requires_grad = False
            self.vqvae.train = disabled_train

            # [E] Use ablation-aware module
            self.voxel_token_multiscale = VoxelTokenMultiScaleAblation(
                hidden_dim=self.hidden_dim, num_heads=8,
                share_attn=voxel_share_attn, dropout=voxel_dropout,
            )

            # Logging
            R = voxel_spatial_res
            if voxel_uniform_pool_res > 0:
                uR = voxel_uniform_pool_res
                pool_desc = f"uniform R={uR} → {uR**3}tokens/scale"
            else:
                R_16 = max(2, R * 16 // 32)
                R_32 = R
                R_64 = R * 64 // 32
                pool_desc = (f"proportional: feat_16→{R_16}³={R_16**3}, "
                             f"feat_32→{R_32}³={R_32**3}, feat_64→{R_64}³={R_64**3}")
            feat_src = "decoder" if voxel_use_decoder else "encoder"
            print(f"[E] VoxelTokenMultiScaleAblation enabled: "
                  f"feat_src={feat_src}, pool={pool_desc}, "
                  f"share_attn={voxel_share_attn}, dropout={voxel_dropout}, "
                  f"enhance_fi={enhance_feature_injector}")

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt
        self.counting_training_parameters()

    def initialize_affordance_modules(self, aff_path, in_dim, out_dim, train_aff_decoder=True):
        aff_model = build_AFD(aff_path)
        if train_aff_decoder:
            aff_model.train()
            for param in aff_model.parameters():
                param.requires_grad = True
        text_fc = [
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0),
        ]
        aff_proj = nn.Sequential(*text_fc)
        aff_proj.train()
        for param in aff_proj.parameters():
            param.requires_grad = True
        return aff_model, aff_proj

    def counting_training_parameters(self):
        total = 0.0
        trainable_names = []
        all = 0.0
        for name, param in self.named_parameters():
            if param.requires_grad:
                total += param.nelement()
                trainable_names.append(name)
            all += param.nelement()
        print(trainable_names)
        print("  + Number of trainable params: %.2fM" % (total / 1e6))
        print("Number of all params: %.2fM" % (all / 1e6))
        return total

    # ===== [E] Changed: supports decoder/encoder + uniform/proportional pooling =====
    def _get_voxel_multiscale_tokens(self, points_batch):
        """
        Extract multi-scale features from VQVAE encoder or decoder.
        Supports uniform or proportional pooling based on ablation config.

        Args:
            points_batch: (B, N, 3) in [-0.5, 0.5]
        Returns:
            list of 3 tensors:
                [0]: (B, K_16, 512) — spatial tokens from feat_16
                [1]: (B, K_32, 128) — spatial tokens from feat_32
                [2]: (B, K_64,  32) — spatial tokens from feat_64
        """
        with torch.no_grad():
            voxel = self._pointcloud_to_voxel(points_batch, resolution=64)

            # [E] ablation: encoder vs decoder features
            if self.voxel_use_decoder:
                token_ids = self.vqvae.Encode(voxel)
                ms_feats = self.vqvae.decode_with_multiscale(token_ids)
            else:
                ms_feats = self.vqvae.encode_with_multiscale(voxel)
            # ms_feats: {64: (B,32,64,64,64), 32: (B,128,32,32,32), 16: (B,512,16,16,16)}

        R_base = self.voxel_spatial_res
        result = []
        for res in [16, 32, 64]:
            feat = ms_feats[res]  # (B, C, D, D, D)

            # [E] ablation: uniform vs proportional pooling
            if self.voxel_uniform_pool_res > 0:
                R_scale = self.voxel_uniform_pool_res
            else:
                R_scale = max(2, R_base * res // 32)

            pooled = F.adaptive_avg_pool3d(feat, R_scale)  # (B, C, R_s, R_s, R_s)
            spatial = pooled.flatten(2).transpose(1, 2)    # (B, R_s³, C)
            result.append(spatial)

        return result

    def encode_point(self, points):
        with self.maybe_autocast(self.mix_precision):
            points2bs = torch.stack(points)
            points_feat, points_pos = self.point_encoder(points2bs)
            inputs_llm = self.llm_proj(points_feat)
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(points2bs.device)
        return inputs_llm, atts_llm, points_pos

    def predict_mask(self, output_ids, last_hidden_states, points, shape_id, original_list=(1, 2048)):
        original_list = (1, self.upscale_points)
        if points is None:
            return []
        points2bs = torch.stack(points)
        points2bs_t = points2bs.transpose(1, 2)
        point_embedding = self.seg_point_encoder(points2bs_t)

        # get multiscale voxel tokens
        multiscale_voxels = None
        if self.enable_voxel_token:
            multiscale_voxels = self._get_voxel_multiscale_tokens(points2bs)

        bs, seq_len, hid = last_hidden_states.shape
        zero_vec = torch.zeros(1, self.hidden_dim, device=last_hidden_states.device)
        pred_masks = []

        def get_token_embedding(hidden_state_row, mask, zero_vec):
            if mask.sum() == 0:
                return zero_vec
            return hidden_state_row[mask]

        def last_occurrence_mask(ids, token_id):
            mask = (ids == token_id)
            idx = mask.nonzero(as_tuple=True)[0]
            out = torch.zeros_like(mask, dtype=torch.bool)
            if len(idx) > 0:
                out[idx[-1]] = True
            return out

        for i in range(bs):
            cur_ids = output_ids[i][:seq_len]
            seg_masks = [last_occurrence_mask(cur_ids, tid) for tid in self.seg_token_ids]
            if all(mask.sum() == 0 for mask in seg_masks):
                pred_masks.append(
                    torch.zeros((1, *original_list), dtype=torch.float32,
                                device=last_hidden_states.device)
                )
                continue
            pred_tokens = [
                get_token_embedding(last_hidden_states[i], seg_masks[j], zero_vec)
                for j in range(3)
            ]
            pred_tokens = torch.stack(pred_tokens, dim=0).squeeze(1)  # [3, C]
            pred_tokens = pred_tokens.unsqueeze(0)  # [1, 3, C]
            if self.training and self.learned_noise:
                noise = torch.randn_like(pred_tokens) * self.noise_std
                pred_tokens = pred_tokens + noise

            # enhance pred_tokens with multiscale voxel cross-attention
            if self.enable_voxel_token and multiscale_voxels is not None:
                ms_voxels_i = [v[i:i+1] for v in multiscale_voxels]  # list of (1, K, C_j)
                enhanced_pred_tokens = self.voxel_token_multiscale(pred_tokens, ms_voxels_i)
            else:
                enhanced_pred_tokens = pred_tokens

            # Aggregation on enhanced tokens
            aggregated_token = self.token_aggregation(enhanced_pred_tokens)
            aggregated_token = aggregated_token.squeeze(1)  # [1, C]
            pred_tokens_embedding = self.fuse_single(aggregated_token)

            point_embedding_ = point_embedding[i].unsqueeze(0)

            # FeatureInjector: use enhanced or original tokens based on config
            fi_tokens = enhanced_pred_tokens if self.enhance_feature_injector else pred_tokens
            point_embedding_enhanced = self.FeatureInjector(
                point_emb=point_embedding_, raw_tokens=fi_tokens
            )

            pred_mask = self.aff_model(
                pointcloud_embeddings=point_embedding_enhanced,
                pointcloud_emorigin=point_embedding_,
                sparse_prompt_embeddings=pred_tokens_embedding.unsqueeze(1),
                multimask_output=False,
            )
            pred_masks.append(pred_mask)
        return pred_masks

    # --- Text formatting ---
    def prepare_input_phi(self, question):
        return f"<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\n{question}<|end|>\n<|assistant|>\n"

    def prepare_response_phi(self, answer):
        return f"{answer}<|end|>\n"

    def prepare_input_qwen(self, question):
        return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

    def prepare_response_qwen(self, answer):
        return f"{answer}<|im_end|>\n"

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens["input_ids"].append(torch.cat([
                input_ids[i][:this_input_ones], output_ids[i][0:], input_ids[i][this_input_ones:],
            ]))
            llm_tokens["attention_mask"].append(torch.cat([
                input_atts[i][:this_input_ones], output_atts[i][0:], input_atts[i][this_input_ones:],
            ]))
        llm_tokens["input_ids"] = torch.stack(llm_tokens["input_ids"])
        llm_tokens["attention_mask"] = torch.stack(llm_tokens["attention_mask"])
        return llm_tokens, input_part_targets_len

    # --- Forward ---
    def forward(self, samples):
        questions = samples["question"]
        answers = samples["answer"]
        bs = len(questions)
        new_questions = []
        for i in range(bs):
            if self.llm_model_type == "phi3.5" or self.llm_model_type == "phi4":
                res = self.prepare_input_phi(questions[i])
            else:
                res = self.prepare_input_qwen(questions[i])
            new_questions.append(res)
        questions = new_questions
        new_answers = []
        for i in range(bs):
            if self.llm_model_type == "phi3.5" or self.llm_model_type == "phi4":
                res = self.prepare_response_phi(answers[i])
            else:
                res = self.prepare_response_qwen(answers[i])
            new_answers.append(res)
        answers = new_answers

        self.llm_tokenizer.truncation_side = "left"
        text_input_tokens = self.llm_tokenizer(
            questions, return_tensors="pt", padding="longest",
            truncation=True, max_length=self.max_txt_len, add_special_tokens=False,
        ).to(self.device)

        self.llm_tokenizer.truncation_side = "right"
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in answers],
            return_tensors="pt", padding="longest",
            truncation=True, max_length=self.max_output_txt_len, add_special_tokens=False,
        ).to(self.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids, text_input_tokens.attention_mask,
            text_output_tokens.input_ids, text_output_tokens.attention_mask,
        )
        targets = llm_tokens["input_ids"].masked_fill(
            llm_tokens["input_ids"] == self.llm_tokenizer.pad_token_id, -100
        )
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens["input_ids"])
        attention_mask = llm_tokens["attention_mask"]

        with self.maybe_autocast(self.mix_precision):
            if "points" in samples:
                points = samples["points"]
                inputs_llm, atts_llm, point_pos_em = self.encode_point(points)
                empty_targets = torch.ones(atts_llm.size(), dtype=torch.long).to(inputs_llm.device).fill_(-100)
                targets = torch.cat([empty_targets, targets], dim=1)
                inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm, attention_mask], dim=1)

            outputs = self.llm_model(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                return_dict=True, labels=targets, output_hidden_states=True,
            )
            points = samples.get("points", None)
            gt_masks = samples.get("masks", None)
            shape_id = samples.get("shape_id", None)
            labels = samples.get("label", None)
            if gt_masks is None:
                return {"loss": outputs.loss, "ce_loss": outputs.loss,
                        "mask_bce_loss": 0, "mask_dice_loss": 0, "mask_loss": 0}
            image_token_length = inputs_llm.shape[1]
            last_hidden_states = outputs.hidden_states[-1][:, image_token_length:, :]
            output_ids = llm_tokens["input_ids"][:, :]
            pred_masks = self.predict_mask(output_ids, last_hidden_states, points, shape_id)

        # Loss (standard)
        ce_loss = outputs.loss * self.ce_loss_weight
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]
            label = labels[batch_idx]
            if self.uselabelratio is True and label in self.label_ratio:
                labelr = self.label_ratio[label]
            else:
                labelr = 1.0
            assert gt_mask.shape[0] == pred_mask.shape[0]
            mask_bce_loss += labelr * (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
            )
            mask_dice_loss += labelr * (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]
        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss
        total_loss = ce_loss + mask_loss

        return {
            "loss": total_loss, "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss, "mask_dice_loss": mask_dice_loss, "mask_loss": mask_loss,
        }

    # --- Generate ---
    @torch.no_grad()
    def generate(self, samples, use_nucleus_sampling=False, num_beams=5, max_length=256,
                 min_length=2, top_p=0.9, repetition_penalty=1.0, length_penalty=1, temperature=1):
        questions = samples["question"]
        if isinstance(questions, str):
            questions = [questions]
        bs = len(questions)
        new_questions = []
        for i in range(bs):
            if self.llm_model_type == "phi3.5" or self.llm_model_type == "phi4":
                res = self.prepare_input_phi(question=questions[i])
            else:
                res = self.prepare_input_qwen(questions[i])
            new_questions.append(res)
        questions = new_questions
        llm_tokens = self.llm_tokenizer(
            questions, padding="longest", return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        with self.maybe_autocast(self.mix_precision):
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            attention_mask = llm_tokens.attention_mask
            if "points" in samples:
                points = samples["points"]
                shape_id = samples.get("shape_id", None)
                inputs_llm, atts_llm, points_pos = self.encode_point(points)
                inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm, attention_mask], dim=1)
            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                do_sample=use_nucleus_sampling, top_p=top_p, temperature=temperature,
                max_new_tokens=max_length, min_new_tokens=min_length,
                repetition_penalty=repetition_penalty, length_penalty=length_penalty,
                return_dict_in_generate=True, output_hidden_states=True,
                output_attentions=True, use_cache=False,
            )
            pred_masks = []
            if "points" in samples:
                output_ids = outputs.sequences[:, 1:-1]
                last_hidden_states = [
                    token_states[-1] for token_states in outputs.hidden_states[1:-1]
                ]
                if len(last_hidden_states) == 0:
                    return {"text": "", "masks": []}
                last_hidden_states = torch.concat(last_hidden_states, dim=1)
                pred_masks = self.predict_mask(output_ids, last_hidden_states, points, shape_id)
        try:
            output_text = self.llm_tokenizer.batch_decode(outputs["sequences"], skip_special_tokens=True)
        except Exception as e:
            print(outputs)
            raise e
        output_text = [text.strip("<|end|>") for text in output_text]
        masks_score = [score.sigmoid() for score in pred_masks]
        pro_pred_masks = [(m > 0.4).to(torch.float32) for m in masks_score]
        return {
            "text": output_text, "masks_scores": masks_score, "masks": pro_pred_masks,
            "output_ids": output_ids, "attentions": outputs.attentions, "seg_id": self.seg_token_id_1,
        }

    def load_from_pretrained(self, url_or_filename):
        if os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu", weights_only=False)
        else:
            raise RuntimeError("checkpoint url or path is invalid")
        state_dict = checkpoint["model"]
        msg = self.load_state_dict(state_dict, strict=False)
        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("Load Checkpoint From %s" % url_or_filename)
        return msg

    @classmethod
    def from_config(cls, cfg):
        mix_precision = cfg.get("mix_precision", "bf16")
        prompt_encoder_dim = cfg.get("prompt_encoder_dim", 32)
        ce_loss_weight = cfg.get("ce_loss_weight", 1.0)
        bce_loss_weight = cfg.get("bce_loss_weight", 1.0)
        dice_loss_weight = cfg.get("dice_loss_weight", 1.0)
        uselabelratio = cfg.get("uselabelratio", False)
        point_model_config_path = cfg.get("point_model_config_path",
            "3D_ADLLM/configs/models/PointTransformer_2048point.yaml")
        freeze_point = cfg.get("freeze_point", True)
        free_seg_point_encoder = cfg.get("free_seg_point_encoder", False)
        seg_point_encoder_config_path = cfg.get("seg_point_encoder_config_path", None)
        seg_point_encoder_path = cfg.get("seg_point_encoder_path", None)
        aff_path = cfg.get("aff_path", None)
        train_aff_decoder = cfg.get("train_aff_decoder", False)
        upscale_points = 2048
        lora_r = cfg.get("lora_r", 16)
        lora_alpha = cfg.get("lora_alpha", 32)
        target_modules = cfg.get("target_modules", ["qkv_proj"])
        llm_model = cfg.get("llm_model", None)
        freeze_llm = cfg.get("freeze_llm", True)
        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 128)
        freeze_linear = cfg.get("freeze_linear", False)
        label_ratio_path = cfg.get("label_ratio_path", "")
        lora_llm_finetune = cfg.get("lora_llm_finetune", False)
        point_enhanced = cfg.get("point_enhanced", False)
        learned_noise = cfg.get("learned_noise", False)
        noise_std = cfg.get("noise_std", 0.01)
        llm_model_type = cfg.get("llm_model_type", None)
        # Voxel token multiscale
        enable_voxel_token = cfg.get("enable_voxel_token", False)
        vqvae_checkpoint = cfg.get("vqvae_checkpoint", None)
        vqvae_num_embeddings = cfg.get("vqvae_num_embeddings", 8192)
        voxel_spatial_res = cfg.get("voxel_spatial_res", 4)
        enhance_feature_injector = cfg.get("enhance_feature_injector", False)
        # [E] Ablation flags
        voxel_uniform_pool_res = cfg.get("voxel_uniform_pool_res", 0)
        voxel_use_decoder = cfg.get("voxel_use_decoder", False)
        voxel_share_attn = cfg.get("voxel_share_attn", False)
        voxel_dropout = cfg.get("voxel_dropout", 0.0)

        model = cls(
            mix_precision=mix_precision, prompt_encoder_dim=prompt_encoder_dim,
            ce_loss_weight=ce_loss_weight, dice_loss_weight=dice_loss_weight,
            bce_loss_weight=bce_loss_weight, uselabelratio=uselabelratio,
            point_model_config_path=point_model_config_path, freeze_point=freeze_point,
            free_seg_point_encoder=free_seg_point_encoder,
            seg_point_encoder_config_path=seg_point_encoder_config_path,
            seg_point_encoder_path=seg_point_encoder_path,
            aff_path=aff_path, train_aff_decoder=train_aff_decoder, upscale_points=upscale_points,
            lora_r=lora_r, lora_alpha=lora_alpha, target_modules=target_modules,
            llm_model=llm_model, freeze_llm=freeze_llm, prompt=prompt,
            max_txt_len=max_txt_len, max_output_txt_len=max_output_txt_len,
            freeze_linear=freeze_linear, label_ratio_path=label_ratio_path,
            lora_llm_finetune=lora_llm_finetune, point_enhanced=point_enhanced,
            learned_noise=learned_noise, noise_std=noise_std, llm_model_type=llm_model_type,
            enable_voxel_token=enable_voxel_token, vqvae_checkpoint=vqvae_checkpoint,
            vqvae_num_embeddings=vqvae_num_embeddings, voxel_spatial_res=voxel_spatial_res,
            enhance_feature_injector=enhance_feature_injector,
            voxel_uniform_pool_res=voxel_uniform_pool_res, voxel_use_decoder=voxel_use_decoder,
            voxel_share_attn=voxel_share_attn, voxel_dropout=voxel_dropout,
        )
        model.load_checkpoint_from_config(cfg)
        return model
