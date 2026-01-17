import os
import numpy as np
import cv2
import h5py
import click
import argparse

import matplotlib.pyplot as plt

import IPython
e = IPython.embed

DT = 0.02  # seconds, 50Hz

JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
STATE_NAMES = JOINT_NAMES + ["gripper"]

def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys(): # type: ignore
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()] # type: ignore

    return image_dict

def save_videos(video, dt, video_path=None):
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


def visualize_joints(qpos_list, command_list, plot_path=None, ylim=None, label_overwrite=None):
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = 'State', 'Command'

    qpos = np.array(qpos_list) # ts, dim
    command = np.array(command_list)
    num_ts, num_dim = qpos.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # plot joint state
    all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1)
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    # plot arm command
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(command[:, dim_idx], label=label2)
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved qpos plot to: {plot_path}')
    plt.close()

def visualize_single(efforts_list, label, plot_path=None, ylim=None, label_overwrite=None):
    efforts = np.array(efforts_list) # ts, dim
    num_ts, num_dim = efforts.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # plot joint state
    all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(efforts[:, dim_idx], label=label)
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved effort plot to: {plot_path}')
    plt.close()


def visualize_timestamp(t_list, dataset_path):
    plot_path = dataset_path.replace('.pkl', '_timestamp.png')
    h, w = 4, 10
    fig, axs = plt.subplots(2, 1, figsize=(w, h*2))
    # process t_list
    t_float = []
    for secs, nsecs in t_list:
        t_float.append(secs + nsecs * 10E-10)
    t_float = np.array(t_float)

    ax = axs[0]
    ax.plot(np.arange(len(t_float)), t_float)
    ax.set_title(f'Camera frame timestamps')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    ax = axs[1]
    ax.plot(np.arange(len(t_float)-1), t_float[:-1] - t_float[1:])
    ax.set_title(f'dt')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved timestamp plot to: {plot_path}')
    plt.close()


def hdf5_to_video(dataset_root, num_episodes):

    ## dataset_path for original hdf5 files
    orignal_dir = os.path.join(dataset_root, 'original')
    if os.path.exists(orignal_dir):
        print(f'Found original dataset at {orignal_dir}, using it.')
        dataset_dir = orignal_dir
    else:
        print(f'No original dataset found at {orignal_dir}, using the dataset directory directly.')
        dataset_dir = dataset_root

    ## save video directory
    video_dir = os.path.join(dataset_root, 'video')
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    ## Process each episode in dataset_dir
    for i in range(num_episodes):
        dataset_name = f'episode_{i}'
        dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
        if not os.path.isfile(dataset_path):
            print(f'Dataset does not exist at \n{dataset_path}\n')
            continue

        # Load data
        image_dict = load_hdf5(dataset_dir, dataset_name)

        # Save video
        video_path = os.path.join(video_dir, dataset_name + '_video.mp4')
        save_videos(image_dict, DT, video_path=video_path)

@click.command()
@click.option('--dataset_root', type=str, default=None, help='Root directory of the dataset containing original hdf5 files')
@click.option('--num_episodes', type=int, default=2, help='Number of episodes to process')
def main(dataset_root, num_episodes):

    ## dataset_path for original hdf5 files
    orignal_dir = os.path.join(dataset_root, 'original')
    if os.path.exists(orignal_dir):
        print(f'Found original dataset at {orignal_dir}, using it.')
        dataset_dir = orignal_dir
    else:
        print(f'No original dataset found at {orignal_dir}, using the dataset directory directly.')
        dataset_dir = dataset_root

    ## save video directory
    video_dir = os.path.join(dataset_root, 'video')
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    ## Process each episode in dataset_dir
    for i in range(num_episodes):
        dataset_name = f'episode_{i}'
        dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
        if not os.path.isfile(dataset_path):
            print(f'Dataset does not exist at \n{dataset_path}\n')
            continue

        # Load data
        image_dict = load_hdf5(dataset_dir, dataset_name)

        # Save video
        video_path = os.path.join(video_dir, dataset_name + '_video.mp4')
        save_videos(image_dict, DT, video_path=video_path)


if __name__ == '__main__':
    main()
    # Example usage:
    # dataset_root = '/ssd1/xuhang/dataset_sim/sim_transfer_cube_scripted/'
    # num_episodes = 2  # Adjust as needed
    # hdf5_to_video(dataset_root, num_episodes) 
