CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=true  torchrun --master_port 21467 --nproc_per_node=1 eval.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi_train/new/phi4_multi_eval3.yaml --checkpoint /hard_data1/user_dataset/zhangshaolong_dataset/3D_ADLLM_SHW/outputs_new/phi4_phase1_exp1_3/20260223082/checkpoint_best.pth 
CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=true  torchrun --master_port 24467 --nproc_per_node=1 eval.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi_train/new/phi4_multi_eval1.yaml --checkpoint /hard_data1/user_dataset/zhangshaolong_dataset/3D_ADLLM_SHW/outputs_new/phi4_phase1_exp1_1/20260223082/checkpoint_best.pth 

CUDA_VISIBLE_DEVICES=5 TOKENIZERS_PARALLELISM=true  torchrun --master_port 25467 --nproc_per_node=1 eval.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug_tke_1_3_4_eval.yaml --checkpoint /hard_data1/user_dataset/zhangshaolong_dataset/3D_ADLLM_SHW/best_full.pth

CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=true  torchrun --master_port 25167 --nproc_per_node=1 eval.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug_tke_1_3_4_partial_eval.yaml --checkpoint /hard_data1/user_dataset/zhangshaolong_dataset/3D_ADLLM_SHW/best_partial.pth

CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=true  torchrun --master_port 25467 --nproc_per_node=1 eval.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug_tke_1_3_4_eval_affpose.yaml --checkpoint /hard_data1/user_dataset/zhangshaolong_dataset/3D_ADLLM_SHW/best_full.pth

CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=true  torchrun --master_port 25467 --nproc_per_node=1 eval.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug_tke_1_3_4_eval.yaml --checkpoint /hard_data1/user_dataset/zhangshaolong_dataset/3D_ADLLM_SHW/best_full.pth
CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=true  torchrun --master_port 25167 --nproc_per_node=1 eval.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug_tke_1_3_4_partial_eval.yaml --checkpoint /hard_data1/user_dataset/zhangshaolong_dataset/3D_ADLLM_SHW/best_partial.pth


CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=true  torchrun --master_port 21467 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi_train/new/phi4_multi_train1.yaml

CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=true  torchrun --master_port 27467 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi_train/new/phi4_multi_train0.yaml

CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=true  torchrun --master_port 29467 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi_train/new/phi4_multi_train2.yaml

CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=true  torchrun --master_port 29467 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi_train/new/phi4_multi_train.yaml


CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=true  torchrun --master_port 20467 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi_train/new/phi4_multi_train1.yaml

CUDA_VISIBLE_DEVICES=7 TOKENIZERS_PARALLELISM=true  torchrun --master_port 20447 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi_train/new/phi4_multi_train5.yaml

CUDA_VISIBLE_DEVICES=6 TOKENIZERS_PARALLELISM=true  torchrun --master_port 20449 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi_train/new/phi4_multi_train6.yaml


CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=true  torchrun --master_port 21349 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/scheme1_train.yaml
CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=true  torchrun --master_port 22349 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/scheme2_train.yaml
CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=true  torchrun --master_port 23349 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/scheme4_train.yaml
CUDA_VISIBLE_DEVICES=6 TOKENIZERS_PARALLELISM=true  torchrun --master_port 29349 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/scheme4_2_train.yaml

CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=true  torchrun --master_port 29849 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/scheme4_A_train.yaml
CUDA_VISIBLE_DEVICES=6 TOKENIZERS_PARALLELISM=true  torchrun --master_port 29389 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/scheme4_B_train.yaml
CUDA_VISIBLE_DEVICES=6 TOKENIZERS_PARALLELISM=true  torchrun --master_port 29389 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/scheme4_A_qwen_train.yaml

CUDA_VISIBLE_DEVICES=5 TOKENIZERS_PARALLELISM=true  torchrun --master_port 21389 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_voxel_train.yaml

CUDA_VISIBLE_DEVICES=5 TOKENIZERS_PARALLELISM=true  torchrun --master_port 21389 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi_train/new/phi4_multi_train.yaml


CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=true  torchrun --master_port 22349 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug.yaml
CUDA_VISIBLE_DEVICES=5 TOKENIZERS_PARALLELISM=true  torchrun --master_port 29389 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/qwen_shapeomni_v2_train.yaml
CUDA_VISIBLE_DEVICES=6 TOKENIZERS_PARALLELISM=true  torchrun --master_port 2349 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug.yaml

CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=true  torchrun --master_port 22349 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_refined.yaml
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=true  torchrun --master_port 29389 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug_tkb.yaml
CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=true  torchrun --master_port 2147 --nproc_per_node=1 find_best_threshold.py --threshold-search --threshold-min 0.3 --threshold-max 0.7 --threshold-step 0.5 --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_eval_voxaug.yaml --checkpoint "/hard_data1/user_dataset/zhangshaolong_dataset/3D_ADLLM_SHW/outputs_new/phi4_multi_voxaug/20260313104/checkpoint_best.pth"
CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=true  torchrun --master_port 22349 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug_tka.yaml


CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=true  torchrun --master_port 22309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/ablation/tke_abl_t3r1_scale16.yaml
CUDA_VISIBLE_DEVICES=6 TOKENIZERS_PARALLELISM=true  torchrun --master_port 23409 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug_tkd.yaml


CUDA_VISIBLE_DEVICES=4 TOKENIZERS_PARALLELISM=true  torchrun --master_port 19389 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug_tkb.yaml

CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=true  torchrun --master_port 22309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/qwen_multi_train_voxaug_tkd.yaml


CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=true  torchrun --master_port 22309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug_tke_2.yaml
CUDA_VISIBLE_DEVICES=6 TOKENIZERS_PARALLELISM=true  torchrun --master_port 32309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug_tke_1.yaml
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=true  torchrun --master_port 38309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug_tke_1_2.yaml
CUDA_VISIBLE_DEVICES=4 TOKENIZERS_PARALLELISM=true  torchrun --master_port 19389 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug_tke_3.yaml


CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=true  torchrun --master_port 28309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug_tke_1_3.yaml
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=true  torchrun --master_port 10309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug_tke_1_3_4.yaml
CUDA_VISIBLE_DEVICES=6 TOKENIZERS_PARALLELISM=true  torchrun --master_port 32309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug_tke_1.yaml


CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=true  torchrun --master_port 28309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug_tkb_partial.yaml
CUDA_VISIBLE_DEVICES=6 TOKENIZERS_PARALLELISM=true  torchrun --master_port 32309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug_tke_1_3_4_new.yaml

CUDA_VISIBLE_DEVICES=5 TOKENIZERS_PARALLELISM=true  torchrun --master_port 39309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug_tke_1_3_4_partial.yaml
CUDA_VISIBLE_DEVICES=7 TOKENIZERS_PARALLELISM=true  torchrun --master_port 32301 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug_tke_1_3_4_pre.yaml

CUDA_VISIBLE_DEVICES=7 TOKENIZERS_PARALLELISM=true  torchrun --master_port 32301 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug_tkf_layer.yaml
CUDA_VISIBLE_DEVICES=6 TOKENIZERS_PARALLELISM=true  torchrun --master_port 32309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug_tkf_bottleneck256.yaml
CUDA_VISIBLE_DEVICES=7 TOKENIZERS_PARALLELISM=true  torchrun --master_port 18309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug_tke_1_3_4.yaml


CUDA_VISIBLE_DEVICES=6 TOKENIZERS_PARALLELISM=true  torchrun --master_port 32309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug_tkb_partial_continue.yaml
3D_Affordance_val_new_prompt  [1150/1187]  eta: 0:01:43  acc5: global_acc: 49.294092  iou: global_iou: 46.636257  pointAcc: global_pointAcc: 78.989565  pointPrecision: global_pointPrecision: 54.042325  pointRecall: global_pointRecall: 76.362909  time: 2.5062  data: 0.0012  max mem: 61440

CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=true  torchrun --master_port 28309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug_tke_1_3_4_v2.yaml

CUDA_VISIBLE_DEVICES=5 TOKENIZERS_PARALLELISM=true  torchrun --master_port 39309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug_tke_1_3_4.yaml
CUDA_VISIBLE_DEVICES=7 TOKENIZERS_PARALLELISM=true  torchrun --master_port 18309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug_tke_1_3_4_partial.yaml


CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=true  torchrun --master_port 28309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/ablation/tke_abl_t1r2_no_voxfusion.yaml
CUDA_VISIBLE_DEVICES=6 TOKENIZERS_PARALLELISM=true  torchrun --master_port 32309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/ablation/tke_abl_t1r3_no_fi.yaml
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=true  torchrun --master_port 10309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/ablation/tke_abl_t1r4_concat.yaml
CUDA_VISIBLE_DEVICES=7 TOKENIZERS_PARALLELISM=true  torchrun --master_port 12309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/ablation/tke_abl_t1r5_no_gate.yaml

CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=true  torchrun --master_port 10309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/ablation/tke_abl_t1r5_no_gate.yaml
CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=true  torchrun --master_port 28309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/ablation/tke_abl_t2r1_enh_orig.yaml
CUDA_VISIBLE_DEVICES=6 TOKENIZERS_PARALLELISM=true  torchrun --master_port 32309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/ablation/tke_abl_t2r2_orig_enh.yaml
CUDA_VISIBLE_DEVICES=7 TOKENIZERS_PARALLELISM=true  torchrun --master_port 12309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/ablation/tke_abl_t2r3_enh_enh.yaml


CUDA_VISIBLE_DEVICES=7 TOKENIZERS_PARALLELISM=true  torchrun --master_port 12309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/ablation/tke_abl_t2r4_orig_orig.yaml
CUDA_VISIBLE_DEVICES=6 TOKENIZERS_PARALLELISM=true  torchrun --master_port 32309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/ablation/tke_abl_t3r1_scale16.yaml
CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=true  torchrun --master_port 28309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/ablation/tke_abl_t3r2_scale32.yaml
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=true  torchrun --master_port 10309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/ablation/tke_abl_t3r3_scale64.yaml

CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=true  torchrun --master_port 28309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/ablation/tke_abl_t3r4_same32.yaml
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=true  torchrun --master_port 10309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/ablation/tke_abl_t4r2_voxel_input.yaml

CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=true  torchrun --master_port 10309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug_tke_1_3_4.yaml
CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=true  torchrun --master_port 28309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/ablation/tke_abl_t2r4_orig_orig.yaml


CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=true  torchrun --master_port 28309 --nproc_per_node=1 train.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug_tke_1_3_4.yaml

CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=true  torchrun --master_port 25467 --nproc_per_node=1 eval.py --cfg-path /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug_tke_1_3_4_eval.yaml --checkpoint /hard_data1/user_dataset/zhangshaolong_dataset/3D_ADLLM_SHW/best_full.pth


python visualize_attention.py \
      --cfg-path  /data1/user/zhangshaolong/3D_ADLLM_SHW/configs/phi4_multi_train_voxaug_tke_1_3_4.yaml \
      --checkpoint /hard_data1/user_dataset/zhangshaolong_dataset/3D_ADLLM_SHW/outputs_new/phi4_multi_voxaug_tke_a1_3_4_shareattn16/20260402074/checkpoint_16.pth \
      --output-dir vis_output/ \
      --max-samples 20