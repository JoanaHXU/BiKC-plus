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


def import_task(task_name):
    if task_name == "sim_transfer_cube_scripted":
        from keypose.task.sim_transfer_cube_scripted import PROMPT
    elif task_name == "sim_insertion_scripted":
        from keypose.task.sim_insertion_scripted import PROMPT
    elif task_name == "aloha2_pants":
        from keypose.task.aloha2_pants import PROMPT
    elif task_name == "aloha2_screwdriver_fix":
        from keypose.task.aloha2_screwdriver import PROMPT
    elif task_name == "aloha2_cup_temporal":
        from keypose.task.aloha2_cup_temporal import PROMPT
    elif task_name == "aloha2_cup_spatial":
        from keypose.task.aloha2_cup_spatial import PROMPT
    else:
        raise ValueError(f"Unknown task name: {task_name}")

    return PROMPT


def video_understanding(video_path, prompt, fps):
    if isinstance(video_path, str):
        # print("loading a video:", video_path)
        # 处理视频时使用fps参数，表示每隔1/fps 秒抽取一帧
        content = [{'video': video_path, "fps": fps}, {'text': prompt}]
    else:
        # print("loading a list of images")
        # 处理图像列表时不使用fps参数，让API分析所有图像，to 避免二次降维
        content = [{'video': video_path, "fps": 1}, {'text': prompt}]

    messages = [
        {
            'role': 'system', 
            'content': [
                {'text': 'You are a helpful assistant.'}
            ]
        },
        {
            'role':'user',
            'content': content
        }
    ]

    # 添加重试机制
    max_retries = 3
    retry_delay = 5  # 秒
    
    for attempt in range(max_retries):
        try:
            response = MultiModalConversation.call(
                # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
                # api_key=os.getenv('DASHSCOPE_API_KEY'),
                api_key="sk-3415ed083d5349f990717cb39e4a0411",
                model='qwen2.5-vl-72b-instruct',
                messages=messages)
            
            # 如果调用成功，尝试解析响应
            json_output = response["output"]["choices"][0]["message"].content[0]["text"] # type: ignore
            return json_output
            
        except requests.exceptions.SSLError as e:
            print(f"SSL错误 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
            else:
                print("所有重试都失败了, 返回None")
                return None
                
        except Exception as e:
            print(f"其他错误 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print("所有重试都失败了, 返回None")
                return None
    
    return None




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




def save_text_to_json(data, output_file, indent=4):
    """
    Save the data as a JSON file.
    
    Args:
    - data (dict): The data structure to be saved.
    - output_file (str): The path of the output JSON file.
    - indent (int): The number of spaces for JSON indentation, with a default value of 4.
    """
    try:
        # 写入 JSON 文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        
        # print(f"\nSuccessfully saved the data to {output_file}")
    except Exception as e:
        print(f"\nError saving file: {e}")




def frameidx_to_timestep(frame_idx, fps, video_path):
    """
    Convert frame index to time step in seconds.
    
    Args:
        frame_idx (int): The index of the frame.
        fps (float): Frames per second of the video.
    
    Returns:
        float: Time step in seconds.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    original_fps = cap.get(cv2.CAP_PROP_FPS) ## input_video 的原始帧率，即每秒帧数
    
    frame_interval = original_fps / fps
    time_step = frame_idx * frame_interval

    return time_step




@click.command()
@click.option('--task_name', type=str, default='aloha2_screwdriver', help='Task name for video understanding')
@click.option('--dataset_root', type=str, help='Path to the input video file or a list of images')
@click.option('--num_video', type=int, default=1, help='Number of videos to process (default: 1)')
@click.option('--out_dir', type=str, help='Directory to save the output JSON file')
@click.option('--target_fps', type=float, default=10, help='Target frames per second for video processing (default: 2)')
@click.option('--use_image_list', type=bool, default=True, help='Use a list of images instead of a video file')
def main(task_name, dataset_root, num_video, out_dir, target_fps, use_image_list):
    '''
    Code logic:
    1. Convert the video into a list of images, or directly use the video path.
    2. Define the task name, object list, robot list, surface list, and video duration.
    3. Define the prompt, including task description, object, robot, surface information, and video duration.
    4. Call the video_understanding function for video understanding.
    5. Save the results as a JSON file.
    '''


    dataset_folder = os.path.join(dataset_root, f'{task_name}', 'video')  # Ensure it ends with a slash
    assert os.path.exists(dataset_folder), f"Input path {dataset_folder} does not exist."

    if os.path.exists(out_dir) == False:
        os.makedirs(out_dir, exist_ok=True)

    ## output path
    json_dir = os.path.join(out_dir, f"{task_name}", 'json_sg_vlm')
    if not os.path.exists(json_dir):
        os.makedirs(json_dir, exist_ok=True)

    print(f"Processing {num_video} videos in {dataset_folder}...")

    ## understand each video in the dataset_folder
    for idx in tqdm(range(num_video), desc="Processing videos"):
        local_path = os.path.join(dataset_folder, f'episode_{idx}_video.mp4')

        ## pre-process video 
        if use_image_list:
            ''' convert video to frame_images based on target_fps '''
            frame_images_folder = os.path.join(out_dir, f"{task_name}", 'image_frame')
            input_path = save_video_frames(
                local_path, 
                frame_images_folder, 
                target_fps=target_fps
                ) 
            # print(f"Saved frame images to {frame_images_folder}\n")

        else:
            input_path = local_path

        ## *** process the video or image list ***
        PROMPT = import_task(task_name)

        result = video_understanding(
            input_path, 
            PROMPT, 
            fps=target_fps
        )

        # Save the result to a JSON file
        if result is not None:
            json_str = "\n".join(result.split("```")[1].split("\n")[1:])
            json_output = json.loads(json_str)
        else:
            json_output = {}

        ## convert frame_idx to timestep, saving to json_output
        if 'ModeChangeDetection' in json_output:
            for change in json_output['ModeChangeDetection']:
                frame_idx = change['frame_idx']
                change['timestep_orig'] = frameidx_to_timestep(
                    frame_idx, 
                    target_fps,
                    video_path=local_path
                )

        json_path = os.path.join(json_dir, f"episode_{idx}.json")
        save_text_to_json(json_output, json_path, indent=4)

    print("Done!")
    print(f"Results saved to {json_dir}")


if __name__ == "__main__":
    main()