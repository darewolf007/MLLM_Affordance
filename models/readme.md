aff_phi3_n1.py  ：在输入中拼接20个可训练的embedding，此类embedding不参与输出tokenloss的计算
aff_phi3_n2.py  ：

CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=true  torchrun --master_port 11555 --nproc_per_node=1 3D_ADLLM/train.py --cfg-path 3D_ADLLM/configs/phi_train/phi3n_train.yaml

在输入中拼接20个可训练的embedding
评估结果
dataname,acc5_global_avg,iou_global_avg,pointAcc_global_avg,pointPrecision_global_avg,pointRecall_global_avg
3D_Affordance_val_new_prompt0,23.797710134782864,30.696758717981005,71.12188208194179,41.69335733985743,59.48859648827794
3D_Affordance_openval_new_prompt0,18.244274914719675,25.32279537970783,66.52684283721639,33.575472441487314,51.98624866614881


CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=true  torchrun --master_port 11556 --nproc_per_node=1 3D_ADLLM/train.py --cfg-path 3D_ADLLM/configs/phi_train/phi3n2_train.yaml
将aff token拼接到输入中重新训练
评估结果
dataname,acc5_global_avg,iou_global_avg,pointAcc_global_avg,pointPrecision_global_avg,pointRecall_global_avg
3D_Affordance_val_new_prompt0,9.599236772260593,24.89483872306256,24.989992098043892,24.94310731006291,99.8844974362131
3D_Affordance_openval_new_prompt0,9.522900894397997,24.837571028079694,24.852928569298665,24.844707747103094,99.96332389271664


tmux attach -t phi4n1   0
正在训练
拼接了64个可训练的embedding
训练失败，召回率百分百，无法输出<AFF>token

训练phi4，  train_aff_decoder: True
训练结束，正在评估，效果变差
dataname,acc5_global_avg,iou_global_avg,pointAcc_global_avg,pointPrecision_global_avg,pointRecall_global_avg
3D_Affordance_val_new_prompt0,34.31721813260144,36.27783399683828,77.61235127922232,48.10935967973978,57.90963947831303
3D_Affordance_openval_new_prompt0,20.460135905797245,25.568474485673978,71.18296590065734,36.530132382077014,46.6799696438326

更改数据集标注，预测三个token（全局，局部，细节），将三个token对应的隐藏层取出拼接映射成一维输入到decoder
正在训练....
训练结束
评估中


tmux attach -t qwen3    3
正在训练 无改动 训练完成
评估中 效果很差
dataname,acc5_global_avg,iou_global_avg,pointAcc_global_avg,pointPrecision_global_avg,pointRecall_global_avg
3D_Affordance_val_new_prompt0,17.390267514090503,20.95682920367663,65.59288898497137,30.56282248160412,31.79209577678443
3D_Affordance_openval_new_prompt0,17.313931639867885,20.774693840105115,64.96156881783754,30.574096127968424,31.509766279393574

CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=true  torchrun --master_port 11556 --nproc_per_node=1 3D_ADLLM/train.py --cfg-path 3D_ADLLM/configs/phi_train/phi4_train.yaml
推理过程中添加一定的指导，训练效果很差，建议只在推理的时候加上可能有帮助


TODO：给点云加上位置编码信息


默认学习率 3e-4
bacth_size_per_gpu 8 accum_grad_iters 4

aff_phi3
复现
phi3_train
20251113175：添加全新token 学习率1e-4 bacth_size_per_gpu 4
20251117155：替换token  学习率1e-4 bacth_size_per_gpu 4 accum_grad_iters 4
20251207135：替换token 训了10个epoch

aff_phi3n
在phi3前拼接可训练的参数
20251207140：拼接20个可训练参数
20251209114：上一个评估结果

phi3n2没必要看（在输入拼接AFF，完全错误）

phi4
20251205184：best model
20251207123：上一个的评估结果

phi4_multi
20251212184：多个token拼接到一起经过mlp映射
20251213163：评估结果


调参小知识
batch_size 和 accum_grad_iters
通常batch_size越大效果越好，原因loss计算是取得整个batch的均值，batch_size越小，梯度估计的方差会较大，导致模型更新的方向波动很大


tmux attach -t phi4   1
目前是采用 attention pooling 对预测的多个token进行融合处理
正在训练
20251216211


tmux attach -t phi4n1   5
三个token进行加权相加，权重参数可学习
正在训练
CUDA_VISIBLE_DEVICES=5 TOKENIZERS_PARALLELISM=true  torchrun --master_port 11559 --nproc_per_node=1 3D_ADLLM/train.py --cfg-path 3D_ADLLM/configs/phi_train/phi4_multi_train.yaml

tmux attach -t phi4n1   4   20251216224
 三个token 进行sum pooling
正在训练
CUDA_VISIBLE_DEVICES=4 TOKENIZERS_PARALLELISM=true  torchrun --master_port 11559 --nproc_per_node=1 3D_ADLLM/train.py --cfg-path 3D_ADLLM/configs/phi_train/phi4_multi_train.yaml



/data1/user/zhangshaolong/3D_ADLLM_ZSL/data/single_token_data/train_data_new_prompt.pkl
/data1/user/zhangshaolong/3D_ADLLM_ZSL/data/single_token_data/openval_data_new_prompt.pkl
/data1/user/zhangshaolong/3D_ADLLM_ZSL/data/single_token_data/val_data_new_prompt.pkl

CUDA_VISIBLE_DEVICES=4 TOKENIZERS_PARALLELISM=true  torchrun --master_port 11566 --nproc_per_node=1 3D_ADLLM_ZSL/train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_ZSL/configs/train/multi_v1/phi4_multi_train.yaml


CUDA_VISIBLE_DEVICES=7 TOKENIZERS_PARALLELISM=true  torchrun --master_port 24440 --nproc_per_node=1 3D_ADLLM/train.py --cfg-path 3D_ADLLM/configs/phi_train/phi4_multi_train.yaml

CUDA_VISIBLE_DEVICES=7 TOKENIZERS_PARALLELISM=true  torchrun --master_port 24445 --nproc_per_node=1 3D_ADLLM_ZSL/train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_ZSL/configs/train/multi_v1/phi4_multi_train.yaml


aff_path =
'/data1/user/zhangshaolong/3D_ADLLM/model_pre/ckpts/PointBackbone/best_model.t7'
arch =
'aff_phi_multi'
bce_loss_weight =
1.0
ce_loss_weight =
1.0
dice_loss_weight =
1.0
free_seg_point_encoder =
False
freeze_linear =
False
freeze_llm =
True
freeze_point =
True
label_ratio_path =
'/data1/user/zhangshaolong/3D_ADLLM/result_ratio.json'
llm_model =
'3D_ADLLM/model_pre/Phi_4_mini_instruct'
load_finetuned =
False
load_pretrained =
False
lora_alpha =
48
lora_llm_finetune =
True
lora_r =
32
max_output_txt_len =
256
max_txt_len =
128
model_type =
'phi_default'
point_model_config_path =
'/data1/user/zhangshaolong/3D_ADLLM/configs/models/PointTransformer_2048point.yaml'
prompt =
''
prompt_encoder_dim =
32
seg_point_encoder_config_path =
'/data1/user/zhangshaolong/3D_ADLLM/models/openad/config/PT_modify.py'
seg_point_encoder_path =
'/data1/user/zhangshaolong/3D_ADLLM/model_pre/ckpts/PointBackbone/best_model.t7'
train_aff_decoder =
False
train_aff_encoder =
True
uselabelratio =
True