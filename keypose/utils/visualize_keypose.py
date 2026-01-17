import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
import os
import sys
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)

from diffusion_policy.env.aloha.constants import (
    DT, vx300s, LEFT_BASE_POSE, RIGHT_BASE_POSE,
    GRIPPER_EPSILON, EE_VEL_EPSILONE, EE_DIST_BOUND
    )

def _plot_ee_and_gripper(
    trajectory: dict,
    keyposes: dict,
    output_dir: str,
    i: int,
):
    ### load data
    this_qpos_left = trajectory["qpos_left"]
    this_gripper_left = trajectory["gripper_left"]
    this_gripper_right = trajectory["gripper_right"]
    this_gripper_act_left = trajectory["gripper_act_left"]
    this_gripper_act_right = trajectory["gripper_act_right"]
    this_image = trajectory["image"]
    this_ee_pos_left = trajectory["ee_pos_left"]
    this_ee_pos_right = trajectory["ee_pos_right"]
    this_ee_dpos_left = trajectory["ee_dpos_left"]
    this_ee_dpos_right = trajectory["ee_dpos_right"]
    this_ee_vel_norm_left = trajectory["ee_vel_norm_left"]
    this_ee_vel_norm_right = trajectory["ee_vel_norm_right"]
    this_ee_dist = trajectory["ee_dist"]

    keypose_left = keyposes["left"]
    keypose_right = keyposes["right"]
    # merge = keyposes["merge"]

    ### gripper, ee vel, and ee dist
    num_t, num_dim = this_qpos_left.shape[0], 6
    h, w = 2, num_dim*3
    num_figs = num_dim

    ### plot EE curves
    idx_ylabel_map = {
        0: r"$x$ [m]",
        1: r"$y$ [m]",
        2: r"$z$ [m]",
        3: "gripper",
        4: r"$v_{\rm ee}$ [m/s]",
        5: r"$d_{\rm ee}$ [m]",
    }

    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))
    t = np.arange(num_t) * DT
    for idx_dim in range(num_dim):
        ax = axs[idx_dim]

        if idx_dim < 3:
            ### x, y, z
            ax.plot(t, this_ee_pos_left[:, idx_dim], "r", label="left")
            ax.plot(t, this_ee_pos_right[:, idx_dim], "b", label="right")
            for idx in keypose_left:
                ax.axvline(x=t[idx], color='r', linewidth=0.5)
            for idx in keypose_right:
                ax.axvline(x=t[idx], color='b', linewidth=0.5)
            ax.legend()
        elif idx_dim == 3:
            ### gripper
            ax.plot(t, this_gripper_left, "r", label="left")
            ax.plot(t, this_gripper_right, "b", label="right")
            ax.plot(t, this_gripper_act_left, "r:")
            ax.plot(t, this_gripper_act_right, "b:")
            ax.plot(t, np.ones_like(t) * GRIPPER_EPSILON, 'k--')
            ax.plot(t, -np.ones_like(t) * GRIPPER_EPSILON, 'k--')
            ax.scatter(
                t[keypose_left],
                this_gripper_left[keypose_left],
                marker='x', color='r'
            )
            ax.scatter(
                t[keypose_right],
                this_gripper_right[keypose_right],
                marker='x', color='b'
            )
            for idx in keypose_left:
                ax.axvline(x=t[idx], color='r', linewidth=0.5)
            for idx in keypose_right:
                ax.axvline(x=t[idx], color='b', linewidth=0.5)
            ax.legend()
        elif idx_dim == 4:
            ### ee vel
            ax.plot(t[:-1], this_ee_vel_norm_left, "r", label="left")
            ax.plot(t[:-1], this_ee_vel_norm_right, "b", label="right")
            ax.plot(t, np.ones_like(t) * 0.05, 'k--')
            for idx in keypose_left:
                ax.axvline(x=t[idx], color='r', linewidth=0.5)
            for idx in keypose_right:
                ax.axvline(x=t[idx], color='b', linewidth=0.5)
            # set y limit
            ax.legend()
            ax.set_ylim([0, 0.1])
        elif idx_dim == 5:
            ### ee dist
            ax.plot(t, this_ee_dist, "r")
            ax.plot(t, np.ones_like(t) * EE_DIST_BOUND, 'k--')
            for idx in keypose_left:
                ax.axvline(x=t[idx], color='r', linewidth=0.5)
            for idx in keypose_right:
                ax.axvline(x=t[idx], color='b', linewidth=0.5)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(idx_ylabel_map[idx_dim])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/episode_{i}_ee.png', dpi=200)
    plt.close()


def _plot_keypose_imgs(
        trajectory: dict,
        keyposes: dict,
        output_dir: str,
        i: int,
):
    
    image = trajectory["image"]

    ## get keypose images based on keypose_idx
    keypose_image = dict()
    for cam_name in image.keys():
        keypose_image[cam_name] = dict()
        keypose_image[cam_name]['left'] = dict()
        keypose_image[cam_name]['right'] = dict()
        keypose_image[cam_name]['left']['image'] = list()
        keypose_image[cam_name]['right']['image'] = list()
        keypose_image[cam_name]['left']['timesteps'] = list()
        keypose_image[cam_name]['right']['timesteps'] = list()

        for idx in keyposes["left"]:
            keypose_image[cam_name]['left']['image'].append(image[cam_name][idx])
            keypose_image[cam_name]['left']['timesteps'].append(idx)
        for idx in keyposes["right"]:
            keypose_image[cam_name]['right']['image'].append(image[cam_name][idx])
            keypose_image[cam_name]['right']['timesteps'].append(idx)

    ## output the keypose images in two columns
    for cam_name, imgs in keypose_image.items():
        if len(imgs) == 0:
            print(f"Warning: no keypose images for {cam_name} in episode {i}.")
            continue
        imgs_left = rearrange(imgs['left']['image'], 'k h w c -> h (k w) c')
        imgs_right = rearrange(imgs['right']['image'], 'k h w c -> h (k w) c')
        timesteps_left = imgs['left']['timesteps']
        timesteps_right = imgs['right']['timesteps']
        if len(timesteps_left) == 0 or len(timesteps_right) == 0:
            print(f"Warning: no keypose images for {cam_name} in episode {i}.")
            continue

        ## two subplots, left arm on the left, right arm on the right
        plt.figure(figsize=(10, 10))

        plt.subplot(2, 1, 1)
        plt.imshow(imgs_left)
        plt.axis('off')
        plt.title(f'Keypose Images for {cam_name} - Left Arm\nTimesteps: {timesteps_left}')
       
        plt.subplot(2, 1, 2)
        plt.imshow(imgs_right)
        plt.axis('off')
        plt.title(f'Keypose Images for {cam_name} - Right Arm\nTimesteps: {timesteps_right}')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/episode_{i}_{cam_name}_keyposes.png', dpi=200)
        plt.close()

