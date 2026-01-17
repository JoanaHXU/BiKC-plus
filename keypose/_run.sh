#!/bin/bash

# 激活conda环境（如果需要）
source ~/miniforge3/etc/profile.d/conda.sh
conda deactivate
conda deactivate
conda activate /home/xuhang/miniforge3/envs/robodiff2

TASK="sim_transfer_cube_scripted"
NUM_EPISODES=50
ROOT="/ssd1/xuhang/dataset_sim/"
OUT="/home/xuhang/TASE/kp_dataset/"


## 1. Run `./extract_keypose.py` to extract keyposes from the dataset.
## IMPORTANT: 
## -- [1] Ensure task-specific keypose extraction logic is defined in ../task/{task}.py
## -- [2] Ajust the `DIST_MIN` parameter in ../task/{task}.py to control the keypose filter.
echo " #### Step 1: Extracting keyposes from the hdf5..."
torchrun extract_keypose.py \
    --task $TASK \
    --num_episodes $NUM_EPISODES \
    --root $ROOT \
    --to_mp4 False \


## 2. Run `./understand_video.py` to generate contact-change JSON files.
## IMPORTANT: Ensure task-specific imports are defined in understand_video.py
echo " #### Step 2: Understanding videos and generating contact-change JSON files..."
python understand_video.py \
    --dataset_folder "${ROOT}${TASK}/video/" \
    --num_video $NUM_EPISODES \
    --out_dir $OUT \
    --target_fps 10 \
    --use_image_list True
    

## 3.  Run `./get_bi_mode.py` to get the bi-mode for the dataset and visualize the contact changes.
## IMPORTANT: Ensure task-specific visualization are imported in get_bi_mode.py
echo " #### Step 3: Getting bimanual mode and visualizing contact changes..."
python get_bimanual_mode.py \
    --task_name $TASK \
    --file_dir $OUT \
    --num_episodes $NUM_EPISODES 


## 4. Run `./gen_dataset.py` to generate the training dataset from keypose and mode datasets.
echo " #### Step 4: Generating training dataset from keypose and mode datasets..."
python gen_dataset.py \
    --keypose_ds_dir "${OUT}${TASK}/dataset_keypose/" \
    --mode_ds_dir  "${OUT}${TASK}/dataset_mode/" \
    --demo_ds_dir "${ROOT}${TASK}/" \
    --output_dir "${ROOT}${TASK}/keypose/" \
    --num_episodes $NUM_EPISODES