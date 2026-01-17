# https://help.aliyun.com/zh/model-studio/vision?spm=5176.28630291.0.0.24bf7eb5h3MdqT&disableWebsiteRedirect=true#6b5c3f098fjfc

'''
python understand_video.py \
--dataset_folder /ssd1/xuhang/dataset_sim/sim_transfer_cube_scripted/video/ \
--num_video 1 \
--out_dir /home/xuhang/TASE/dataset/ \
--target_fps 2 \
'''

import os
import sys
import time
import cv2
import pathlib
import json
import click
import requests
from tqdm import tqdm
from dashscope import MultiModalConversation

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)



def save_video_frames(video_path, output_folder, target_fps=None):
    """
    Save frames from a video file as images with a specific FPS.
    
    Args:
        video_path (str): Path to the input video file
        output_folder (str): Folder to save the extracted frames
        target_fps (float): Desired frames per second (if None, saves all frames)
    """
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    else:
        # delete all files in the folder
        for file in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS) ## input_video 的原始帧率，即每秒帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps
    
    # print(f"Original FPS: {original_fps}")
    # print(f"Total frames: {total_frames}")
    # print(f"Duration: {duration:.2f} seconds")
    
    # If target_fps is None, save all frames
    if target_fps is None: ## output 的目标帧率
        target_fps = original_fps
    
    # Calculate frame interval based on target FPS
    frame_interval = max(1, int(round(original_fps / target_fps)))
    # print(f"Saving 1 frame every {frame_interval} frames")
    
    frame_count = 0
    saved_count = 0
    
    image_path_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save frame if it's the right interval
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            image_path_list.append(frame_filename)
            # print(f"Saved frame {saved_count} from original frame {frame_count}")
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    # print(f"Finished pre-processing.\nSaved {saved_count} frames from original {frame_count} frames.")

    return image_path_list



@click.command()
@click.option('--task_name', type=str, default='aloha2_screwdriver', help='Task name for video understanding')
@click.option('--dataset_root', type=str, help='Path to the input video file or a list of images')
@click.option('--num_video', type=int, default=1, help='Number of videos to process (default: 1)')
@click.option('--out_dir', type=str, help='Directory to save the output JSON file')
@click.option('--target_fps', type=float, default=2.0, help='Target frames per second for output video frames')
def main(task_name, dataset_root, num_video, out_dir, target_fps):
    '''
    Code logic:
    1. Convert the video into a list of images, or directly use the video path.
    2. Define the task name, object list, robot list, surface list, and video duration.
    '''

    dataset_folder = os.path.join(dataset_root, f'{task_name}', 'video')  # Ensure it ends with a slash
    assert os.path.exists(dataset_folder), f"Input path {dataset_folder} does not exist."

    if os.path.exists(out_dir) == False:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Processing {num_video} videos in {dataset_folder}...")

    ## understand each video in the dataset_folder
    for idx in tqdm(range(num_video), desc="Processing videos"):
        local_path = os.path.join(dataset_folder, f'episode_{idx}_video.mp4')

        ## pre-process video 
        ''' convert video to frame_images based on target_fps '''
        frame_images_folder = os.path.join(out_dir, f"{task_name}", 'image_frame_all', f'episode_{idx}')
        input_path = save_video_frames(
            local_path, 
            frame_images_folder, 
            target_fps=target_fps
            ) 
        print(f"Saved frame images to {frame_images_folder}\n")

    print("Done!")


if __name__ == "__main__":
    main()