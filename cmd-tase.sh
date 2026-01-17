
## ------ Sim Transfer Cube Scripted ------ ##

## 训练 policy (diffusion transformer)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python train.py --config-name train_diffusion_transformer_image_workspace task=sim_transfer_cube_scripted

## 训练 keypose (Transformer)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python train.py --config-name train_keypose_transformer_workspace task=keypose_sim_transfer_cube_scripted

## 训练 keypose (Consistency)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python train.py --config-name train_keypose_cm_workspace task=keypose_sim_transfer_cube_scripted

## 训练 kt_policy (consistency)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python train.py --config-name train_trajectory_consistency_unet_workspace task=trajectory_sim_transfer_cube_scripted

## 测试 keypose predictor
python eval_sim_kp_predictor.py \
        --keypose_ckpt \
        /home/xuhang/TASE/data/outputs/2025.07.11/23.46.03_train_keypose_cm_keypose_sim_transfer_cube_scripted/checkpoints/epoch=1000-keypose_error_val=0.00063.ckpt \
        --dataset_dir \
        /ssd1/xuhang/dataset_sim/ \
        --task \
        sim_transfer_cube_scripted \
        --epi_num \
        50 \
        --step_num \
        2
    ## 模型库
    ## TFM版-基础：/home/xuhang/TASE/data/outputs/2025.07.08/17.29.48_train_keypose_transformer_keypose_sim_transfer_cube_scripted/checkpoints/epoch=0850-val_loss=0.0012.ckpt
    ## CM版-综合提升+改名：/home/xuhang/TASE/data/outputs/2025.07.11/23.46.03_train_keypose_cm_keypose_sim_transfer_cube_scripted/checkpoints/epoch=1000-keypose_error_val=0.00063.ckpt

## 测试 trajectory policy
python eval_sim_kp_policy.py \ 
        --keypose_ckpt \
        /home/xuhang/TASE/data/outputs/2025.07.11/23.46.03_train_keypose_cm_keypose_sim_transfer_cube_scripted/checkpoints/epoch=1000-keypose_error_val=0.00063.ckpt \
        --policy_ckpt \
        /home/xuhang/TASE/data/outputs/2025.07.11/23.46.54_train_trajectory_consistency_unet_trajectory_sim_transfer_cube_scripted/checkpoints/latest.ckpt \
        --task_cfg \
        /home/xuhang/TASE/diffusion_policy/config/task/sim_transfer_cube_scripted.yaml \ 
        --epsilon \
        0.07

## ------ Sim Insertion Scripted ------ ##

## 训练 keypose (Consistency)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python train.py --config-name train_keypose_cm_workspace task=keypose_sim_insertion_scripted

## 训练 keypose (Transformer)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python train.py --config-name train_keypose_transformer_workspace task=keypose_sim_insertion_scripted

## 训练 trajectory (consistency)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python train.py --config-name train_trajectory_consistency_unet_workspace task=trajectory_sim_insertion_scripted

## 测试 keypose predictor
        # /home/xuhang/TASE/data/outputs/2025.07.14/16.29.55_train_keypose_cm_keypose_sim_insertion_scripted/checkpoints/epoch=1000-keypose_error_val=0.00083.ckpt \
python eval_sim_kp_predictor.py \
        --keypose_ckpt \
        /home/xuhang/TASE/data/outputs/2025.07.22/13.37.24_train_keypose_cm_keypose_sim_insertion_scripted/checkpoints/latest.ckpt \
        --dataset_dir \
        /ssd1/xuhang/dataset_sim/ \
        --task \
        sim_insertion_scripted \
        --epi_num \
        50 \
        --step_num \
        1 \
        --model_type \
        cm


## ------ Aloha2 Screwdriver Fix ------ ##

## 训练 keypose (Consistency)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python train.py --config-name train_keypose_cm_workspace task=keypose_aloha2_screwdriver_fix

## 训练 keypose (Transformer)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python train.py --config-name train_keypose_transformer_workspace task=keypose_aloha2_screwdriver_fix

## 训练 trajectory (consistency)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python train.py --config-name train_trajectory_consistency_unet_workspace_real task=trajectory_aloha2_screwdriver_fix

## 测试 keypose predictor
python eval_real_kp_predictor.py \
        --keypose_ckpt \
        /home/xuhang/TASE/data/outputs/2025.07.23/23.14.18_train_keypose_cm_keypose_aloha2_screwdriver_fix/checkpoints/latest.ckpt \
        --dataset_dir \
        /ssd1/xuhang/datasets/ \
        --task \
        aloha2_screwdriver_fix \
        --epi_num \
        50 \
        --step_num \
        1 \
        --model_type \
        cm

## ------- ALOHA2 Pants ------ ##

## 训练 keypose (Consistency)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python train.py --config-name train_keypose_cm_workspace task=keypose_aloha2_pants

## 训练 keypose (Transformer)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python train.py --config-name train_keypose_transformer_workspace task=keypose_aloha2_pants

## 训练 trajectory (consistency)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python train.py --config-name train_trajectory_consistency_unet_workspace_real task=trajectory_aloha2_pants

## 测试 keypose predictor
python eval_real_kp_predictor.py \
        --keypose_ckpt \
        /home/xuhang/TASE/data/outputs/2025.07.23/23.27.59_train_keypose_cm_keypose_aloha2_pants/checkpoints/latest.ckpt \
        --dataset_dir \
        /ssd1/xuhang/datasets/ \
        --task \
        aloha2_pants \
        --epi_num \
        50 \
        --step_num \
        1 \
        --model_type \
        cm


## ------- ALOHA Cup Temporal ------ ##

## 训练 keypose (Consistency)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python train.py --config-name train_keypose_cm_workspace task=keypose_aloha2_cup_temporal

## 训练 keypose (Transformer)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python train.py --config-name train_keypose_transformer_workspace task=keypose_aloha2_cup_temporal

## 训练 trajectory (consistency)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python train.py --config-name train_trajectory_consistency_unet_workspace_real task=trajectory_aloha2_cup_temporal

## 测试 keypose predictor
python eval_real_kp_predictor.py \
        --keypose_ckpt \
        /home/xuhang/TASE/data/outputs/2025.07.26/00.00.20_train_keypose_cm_keypose_aloha2_cup_temporal/checkpoints/latest.ckpt \
        --dataset_dir \
        /ssd1/xuhang/datasets/ \
        --task \
        aloha2_cup_temporal \
        --epi_num \
        50 \
        --step_num \
        1 \
        --model_type \
        cm 

## ------- ALOHA Cup Spatial ------ ##
## 训练 keypose (Consistency)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python train.py --config-name train_keypose_cm_workspace task=keypose_aloha2_cup_spatial

## 训练 keypose (Transformer)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python train.py --config-name train_keypose_transformer_workspace task=keypose_aloha2_cup_spatial

## 训练 trajectory (consistency)
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python train.py --config-name train_trajectory_consistency_unet_workspace_real task=trajectory_aloha2_cup_spatial