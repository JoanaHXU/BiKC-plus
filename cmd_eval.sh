for i in {1} #{1..1}
do
    ### Screwdriver Fixing
    python eval_real_kp_policy.py \
        --keypose_ckpt /ssd1/xuhang/TASE/2025.07.23/23.14.18_train_keypose_cm_keypose_aloha2_screwdriver_fix/checkpoints/epoch=1000-keypose_error_val=0.00041.ckpt \
        --policy_ckpt /ssd1/xuhang/TASE/2025.07.23/23.22.01_train_trajectory_consistency_unet_trajectory_aloha2_screwdriver_fix/checkpoints/epoch=0900-train_action_mse=0.00001.ckpt \
        --max_timesteps 900 \
        --epsilon 0.045

    ### Cup Temporal
    python eval_real_kp_cup_temporal.py \
        --keypose_ckpt /ssd1/xuhang/TASE/2025.07.26/00.00.20_train_keypose_cm_keypose_aloha2_cup_temporal/checkpoints/epoch=1000-keypose_error_val=0.00054.ckpt \
        --policy_ckpt /ssd1/xuhang/TASE/2025.07.26/00.01.26_train_trajectory_consistency_unet_trajectory_aloha2_cup_temporal/checkpoints/epoch=1000-train_action_mse=0.00001.ckpt \
        --max_timesteps 500 \
        --epsilon 0.2

done