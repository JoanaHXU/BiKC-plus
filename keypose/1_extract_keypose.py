import os
import sys
import click
import pathlib
import cv2
import h5py
import matplotlib.pyplot as plt
import modern_robotics as mr
from tqdm import tqdm
from typing import List
from einops import rearrange
import numpy as np

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)

from keypose.utils.hdf5_to_video import hdf5_to_video
from keypose.utils.visualize_keypose import (
    _plot_ee_and_gripper,
    _plot_keypose_imgs
)

from diffusion_policy.env.aloha.constants import (
    DT, vx300s, LEFT_BASE_POSE, RIGHT_BASE_POSE,
    GRIPPER_EPSILON, EE_VEL_EPSILONE, EE_DIST_BOUND
    )

### import functions from other tasks
from keypose.task.sim_transfer_cube_scripted import (
    _find_keypose_idx_transfer, 
    DIST_MIN_transfer, WINDOW_SIZE_transfer
    )
from keypose.task.sim_insertion_scripted import (
    _find_keypose_idx_insertion, 
    DIST_MIN_insertion, WINDOW_SIZE_insertion
    )
from keypose.task.aloha2_screwdriver import (
    _find_keypose_idx_screwdriver,
    DIST_MIN_screwdriver, WINDOW_SIZE_screwdriver
    )
from keypose.task.aloha2_pants import (
    _find_keypose_idx_hang,
    DIST_MIN_pants, WINDOW_SIZE_pants
    )

from keypose.task.aloha2_cup_temporal import (
    _find_keypose_idx_cup_temporal,
    DIST_MIN_cup_temporal, WINDOW_SIZE_cup_temporal
    )

from keypose.task.aloha2_cup_spatial import (
    _find_keypose_idx_cup_spatial,
    DIST_MIN_cup_spatial, WINDOW_SIZE_cup_spatial
    )


''' Update according to the task. '''
def _find_keypose_idx(task, **kwargs):
    if task == "sim_transfer_cube_scripted":
        return _find_keypose_idx_transfer(window_size=WINDOW_SIZE_transfer, **kwargs)
    elif task == "sim_insertion_scripted":
        return _find_keypose_idx_insertion(window_size=WINDOW_SIZE_insertion, **kwargs)
    elif task == "aloha2_screwdriver_fix" or task == "aloha2_screwdriver":
        return _find_keypose_idx_screwdriver(window_size=WINDOW_SIZE_screwdriver, **kwargs)
    elif task == "aloha2_pants":
        return _find_keypose_idx_hang(**kwargs)
    elif task == "aloha2_cup_temporal":
        return _find_keypose_idx_cup_temporal(**kwargs)
    elif task == "aloha2_cup_spatial":
        return _find_keypose_idx_cup_spatial(**kwargs)
    else:
        raise ValueError(f"Unknown task: {task}. Please specify the task to find keypose indices.")


def _remove_too_close_keyposes(keypose_indices, task, forward=False):
    ''' Remove keyposes that are too close to each other.
    This is to avoid redundant keyposes that are too close in time.
    Args:
        keypose_indices: list of keypose indices
        task: str, the task name, used to specify the minimum distance
        forward: bool, if True, keep the former keypose if the distance is larger than min_dist,
            if False, keep the later keypose if the distance is larger than min_dist
    Returns:
        refined_keypose_indices: list of refined keypose indices
    '''

    ## specify the minimum distance between two keyposes
    if task == "sim_transfer_cube_scripted":
        min_dist = DIST_MIN_transfer
    elif task == "sim_insertion_scripted":
        min_dist = DIST_MIN_insertion
    elif task == "aloha2_screwdriver_fix" or task == "aloha2_screwdriver":
        min_dist = DIST_MIN_screwdriver
    elif task == "aloha2_pants":
        min_dist = DIST_MIN_pants
    elif task == "aloha2_cup_temporal":
        min_dist = DIST_MIN_cup_temporal
    elif task == "aloha2_cup_spatial":
        min_dist = DIST_MIN_cup_spatial
    else:
        raise ValueError(f"Unknown task: {task}. Please specify the task to remove too close keyposes.")

    if forward:
        # from the first keypose, keep the former one if
        # the distance is larger than min_dist
        refined_keypose_indices = [keypose_indices[0]]
        for idx in keypose_indices[1:]:
            if idx - refined_keypose_indices[-1] > min_dist:
                refined_keypose_indices.append(idx)
        if refined_keypose_indices[-1] != keypose_indices[-1]:
            refined_keypose_indices.append(keypose_indices[-1])
    else:
        # from the last keypose, keep the later one if 
        # the distance is larger than min_dist
        refined_keypose_indices = [keypose_indices[-1]]
        for idx in reversed(keypose_indices[:-1]):
            if refined_keypose_indices[-1] - idx > min_dist:
                refined_keypose_indices.append(idx)
        if refined_keypose_indices[-1] != 0:
            if refined_keypose_indices[-1] > min_dist:
                refined_keypose_indices.append(0)
            else:
                refined_keypose_indices[-1] = 0
        refined_keypose_indices = refined_keypose_indices[::-1]
    
    return np.array(refined_keypose_indices, dtype=np.int32)


def _save_videos(video, dt, video_path=None):
    ''' Save a video from a list of images or a dict of camera images. '''
    if isinstance(video, list):
        cam_names = list(video[0].keys())
        h, w, _ = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        fps = int(1/dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]] # swap B and R channel
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()
        print(f'Saved video to: {video_path}')
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        all_cam_videos = []
        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])
        all_cam_videos = np.concatenate(all_cam_videos, axis=2) # width dimension

        n_frames, h, w, _ = all_cam_videos.shape
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # swap B and R channel
            out.write(image)
        out.release()
        print(f'Saved video to: {video_path}')




def _load_trajectory(dataset_dir, i, output_dir=None):
    '''load h5df trajectory and return dict of sequences of interest
    params:
        dataset_dir: str
        i: int, episode index
    return:
        dict of sequences of interest
    '''
    dataset_path = os.path.join(dataset_dir, f"episode_{i}.hdf5")
    with h5py.File(dataset_path, "r") as demo:
        
        ### load images and save videos
        this_image = dict()
        for cam_name in demo[f'/observations/images/'].keys(): # type: ignore
            this_image[cam_name] = demo[f'/observations/images/{cam_name}'][:].astype(np.uint8) # type: ignore
        
        out_video_dir = os.path.join(output_dir, "videos_demo") # type: ignore
        if not os.path.exists(out_video_dir):
            os.makedirs(out_video_dir)
        # _save_videos(this_image, DT, video_path=f'{out_video_dir}/episode_{i}_video.mp4')

        ### 从 hdf5 中加载数据：关节角度和夹爪开合
        this_qpos_full = demo["observations/qpos"][()].astype(np.float32) # type: ignore
        if this_qpos_full.shape[1] != 14: # type: ignore
            raise ValueError(f"Expected qpos shape (T, 14), but got {this_qpos_full.shape}.") # type: ignore

        this_qpos_left = demo["observations/qpos"][:, :6].astype(np.float32) # type: ignore
        this_qpos_right = demo["observations/qpos"][:, 6+1:6+7].astype(np.float32) # type: ignore
        this_gripper_left = demo["observations/qpos"][:, 6].astype(np.float32) # type: ignore
        this_gripper_right = demo["observations/qpos"][:, 13].astype(np.float32) # type: ignore

        this_qpos_act_left = demo["action"][:, :6].astype(np.float32) # type: ignore
        this_qpos_act_right = demo["action"][:, 6+1:6+7].astype(np.float32) # type: ignore
        this_gripper_act_left = demo["action"][:, 6].astype(np.float32) # type: ignore
        this_gripper_act_right = demo["action"][:, 13].astype(np.float32) # type: ignore

        T = this_qpos_left.shape[0] # type: ignore
        
        ## 末端执行器位置
        this_ee_pos_left = np.zeros((T, 3))
        this_ee_pos_right = np.zeros((T, 3))
        for j in range(T):
            ## 正向运动学计算 by modern_robotics.FKinSpace
            left_pose_mat = mr.FKinSpace(vx300s.M, vx300s.Slist, this_qpos_left[j])
            right_pose_mat = mr.FKinSpace(vx300s.M, vx300s.Slist, this_qpos_right[j])
            ## 坐标系变换: 从其局部坐标系变换到世界坐标系
            this_ee_pos_left[j] = np.dot(LEFT_BASE_POSE, left_pose_mat)[:3, 3]
            this_ee_pos_right[j] = np.dot(RIGHT_BASE_POSE, right_pose_mat)[:3, 3]
        
        ## 末端执行器的速度
        this_ee_dpos_left = np.diff(this_ee_pos_left, axis=0) / DT
        this_ee_dpos_right = np.diff(this_ee_pos_right, axis=0) / DT

        this_ee_vel_norm_left = np.linalg.norm(this_ee_dpos_left, axis=-1)
        this_ee_vel_norm_right = np.linalg.norm(this_ee_dpos_right, axis=-1)

        ## 两个末端执行器的距离 & 相对运动的速度（正值=相互远离，负值=相互接近，零值=距离保持不变。）
        this_ee_dist = np.linalg.norm(this_ee_pos_left - this_ee_pos_right, axis=-1)
        this_ee_ddist = np.diff(this_ee_dist) / DT

        this_trajectory = dict(
            qpos_full=this_qpos_full,
            qpos_left=this_qpos_left,
            qpos_right=this_qpos_right,
            gripper_left=this_gripper_left,
            gripper_right=this_gripper_right,
            qpos_act_left=this_qpos_act_left,
            qpos_act_right=this_qpos_act_right,
            gripper_act_left=this_gripper_act_left,
            gripper_act_right=this_gripper_act_right,
            ee_pos_left=this_ee_pos_left,
            ee_pos_right=this_ee_pos_right,
            ee_dpos_left=this_ee_dpos_left,
            ee_dpos_right=this_ee_dpos_right,
            ee_vel_norm_left=this_ee_vel_norm_left,
            ee_vel_norm_right=this_ee_vel_norm_right,
            ee_dist=this_ee_dist,
            ee_ddist=this_ee_ddist,
            image=this_image,
        )

        return this_trajectory


def _export_keypose_dataset(
        trajectory,
        keypose_indices,
        i,
        output_dir,
):
    '''
    ** Export keypose dataset to hdf5 file. **
    Args:
        trajectory: dict, the trajectory data
        keypose_indices: dict, the keypose indices for left and right arms
        i: int, the episode index
        output_dir: str, the output directory to save the keypose dataset
\
    HDF5 Structure:
        keypose_{i}.hdf5
        ├── left
        │   ├── timestep: array of keypose timesteps for left arm
        │   ├── qpos: array of keypose qpos for left arm
        │   └── image: dict of keypose images for left arm, each key is a camera name
        │       └── cam_name: array of keypose images for left arm
        └── right
            ├── timestep: array of keypose timesteps for right arm
            ├── qpos: array of keypose qpos for right arm
            └── image: dict of keypose images for right arm, each key is a camera name
                └── cam_name: array of keypose images for right arm
    '''
    output_path = os.path.join(output_dir, f"keypose_{i}.hdf5")

    keypose_dataset = dict()
    keypose_dataset["left"] = dict()
    keypose_dataset["right"] = dict()

    ## keypose timesteps
    keypose_dataset["left"]["timestep"] = keypose_indices["left"]
    keypose_dataset["right"]["timestep"] = keypose_indices["right"]

    ## keypose qpos [n, 14]
    # qpos_left = np.concatenate([trajectory["qpos_act_left"], trajectory["gripper_act_left"].reshape(-1, 1)], axis=1)
    # qpos_right = np.concatenate([trajectory["qpos_act_right"], trajectory["gripper_act_right"].reshape(-1, 1)], axis=1)

    # keypose_dataset["left"]["qpos"] = qpos_left[keypose_indices["left"]]
    # keypose_dataset["right"]["qpos"] = qpos_right[keypose_indices["right"]]

    keypose_dataset["left"]["qpos"] = trajectory["qpos_full"][keypose_indices["left"]]
    keypose_dataset["right"]["qpos"] = trajectory["qpos_full"][keypose_indices["right"]]

    # ## keypose images
    # keypose_dataset["left"]["image"] = dict()
    # keypose_dataset["right"]["image"] = dict()
    # for cam_name in trajectory["image"].keys():
    #     keypose_dataset["left"]["image"][cam_name] = trajectory["image"][cam_name][keypose_indices["left"]]
    #     keypose_dataset["right"]["image"][cam_name] = trajectory["image"][cam_name][keypose_indices["right"]]

    ## save keypose_dataset to hdf5
    with h5py.File(output_path, "w") as f:
        for side in ["left", "right"]:
            grp = f.create_group(side)
            grp.create_dataset("timestep", data=keypose_dataset[side]["timestep"])
            grp.create_dataset("qpos", data=keypose_dataset[side]["qpos"])
            # img_grp = grp.create_group("image")
            # for cam_name in keypose_dataset[side]["image"].keys():
            #     img_grp.create_dataset(cam_name, data=keypose_dataset[side]["image"][cam_name])

    # print(f"Keypose dataset saved to {output_path}")


@click.command()
@click.option('--task', '-t',  required=True)
@click.option('--num_episodes', '-n', default=1, type=int)
@click.option('--root', '-r', type=str, default="/ssd1/xuhang/dataset_sim/")
@click.option('--to_mp4', '-h', is_flag=True, default=False)
@click.option('--forward', '-f', is_flag=True, default=False)
def main(task, num_episodes, root, to_mp4, forward):
    ## input path
    if task.split("_")[0] == "sim":
        dataset_dir = os.path.join(root, task)
    elif task.split("_")[0] == "aloha2":
        dataset_dir = os.path.join(root, task, "original")
    else:
        raise ValueError(f"Check task title: {task}. Please begin with 'sim' or 'aloha'.")

    ## output path
    output_dir = os.path.join(ROOT_DIR, "kp_dataset", f"{task}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    print(f"\nExtracting keyposes of task {task}... using hueristic rules.")
    with tqdm(total=num_episodes, desc="Process", mininterval=1.0) as pbar:
        for i in range(num_episodes):
            this_trajectory = _load_trajectory(dataset_dir, i, output_dir)

            ### find keypose indices

            keypose_left, problem_left = _find_keypose_idx(
                task, 
                trajectory=this_trajectory, 
                side="left") # type: ignore
            keypose_right, problem_right = _find_keypose_idx(
                task, 
                trajectory=this_trajectory, 
                side="right") # type: ignore
            
            if problem_left:
                print(f'left problem in episode {i}')
            if problem_right:
                print(f'right problem in episode {i}')

            ## remove too close keyposes
            keypose_left = _remove_too_close_keyposes(keypose_left, task=task, forward=forward)
            keypose_right = _remove_too_close_keyposes(keypose_right, task=task, forward=forward)

            ## save keyposes as a dict
            keyposes = dict(
                left=keypose_left,
                right=keypose_right
            )

            ### export keypose dataset
            out_keypose_dir = os.path.join(output_dir, "dataset_keypose")
            if not os.path.exists(out_keypose_dir):
                os.makedirs(out_keypose_dir)
                
            _export_keypose_dataset(
                trajectory=this_trajectory,
                keypose_indices=keyposes,
                i=i,
                output_dir=out_keypose_dir,
            )
            
            ### plot ee and gripper curves based on curves and calculated keyposes
            out_img_dir = os.path.join(output_dir, "image_keypose")
            if not os.path.exists(out_img_dir):
                os.makedirs(out_img_dir)
            _plot_ee_and_gripper(this_trajectory, keyposes, out_img_dir, i)

            _plot_keypose_imgs(this_trajectory, keyposes, out_img_dir, i)

            pbar.update()
    
    ## generate demo video
    if to_mp4 == True:
        print("\nGenerating demo video...\n")
        hdf5_to_video(dataset_dir, num_episodes)
    else:
        print("\nSkipping demo video generation as hdf5_to_mp4 is set to False.\n")

    print("Done extracting keyposes.")
    print(f"Keypose dataset saved to {output_dir}/dataset_keypose")
    print(f"Keypose images saved to {output_dir}/image_keypose \n")

if __name__ == "__main__":
    main()