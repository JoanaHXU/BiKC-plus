## Pipeline for Pre-preprocessing Data

##### Preparation
- Collect demonstration trajectories as format of `.hdf5`
- Resize observation image by `/diffusion_policy/scripts/aloha_dataset_resize.py`


##### Step-1: Extract keypose from demonstration(.hdf5)

If process new tasks, please prepare the following items:

1. Ensure task-specific keypose extraction logic is defined in `./task/{task}.py`. 

2. Update `extract_keypose.py` with new task configuration, including 
   - `import`
   - `_find_keypose_idx()`
   - `_remove_too_close_keyposes()`.

Run the following commends for extracting the keypose by heuristic rules
```
echo " #### Step 1: Extracting keyposes from the hdf5..."
torchrun extract_keypose.py \
    --task $TASK \
    --num_episodes $NUM_EPISODES \
    --root $ROOT \
    --to_mp4 False \
```

#### Step-2: Understand Contact-relationship change from video (.mp4)

For each task, please perform the following operations:
1. Ensure update the `import keypose.task.{TASK_NAME}` based on specific task.
2. Check the prompt in `keypose.task.{TASK_NAME}.py`
3. Make sure the videos have been saved in `./data/{task}/videos/` directory. Notice that the video is composed by several views. Please select the view that is most suitable for understanding the contact-relationship change.
