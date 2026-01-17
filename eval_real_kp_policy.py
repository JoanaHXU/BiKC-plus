'''
python eval_sim_kp_runner.py \
    --keypose_ckpt ... \
    --policy_dir ... \
    # --task_cfg ... \ 
    --epsilon 0.25 \

'''

import click
import os
import pathlib
import os
import torch
import hydra
import dill
import copy
import os
import json
import time
import random
import numpy as np
from einops import rearrange

# os.environ['MUJOCO_GL'] = 'egl'  # 如果使用 MuJoCo

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.kt_policy import KeyposeTrajectoryPolicy

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecorder

from aloha.aloha_scripts.robot_utils import move_grippers, move_arms, torque_on
from aloha.aloha_scripts.real_env import make_real_env

from interbotix_xs_modules.arm import InterbotixManipulatorXS


camera_names = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
rgb_keys = ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
lowdim_keys = ['qpos', 'keypose']


@click.command()
@click.option('-k', '--keypose_ckpt', required=True)
@click.option('-p', '--policy_ckpt', required=True)
# @click.option('-t', '--task_cfg', required=True)
@click.option('-mt', '--max_timesteps', default=500, help='Max duration for each epoch in seconds.')
@click.option('-e', '--epsilon', type=float, default=0.25, help="Distance threshold for keypose matching")
@click.option('-n', '--num_inference_steps', default=16, type=int, help="DDIM inference iterations.")
@click.option('-s', '--scale', type=float, default=4, help="Downsample scale for the real environment")
@click.option('-d', '--device', default='cuda:0', help="Device to run the evaluation on")
def main(keypose_ckpt, policy_ckpt, max_timesteps, num_inference_steps, epsilon, scale, device):

    seed = 36
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    policy_ckpt_root = "/".join(policy_ckpt.split('/')[:-2])
    output_dir = os.path.join(policy_ckpt_root, f"eval_epi={epsilon:.2f}")
    if not os.path.exists(output_dir):
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(device)

    """ --- 加载 keypose 模型 --- """
    k_payload = torch.load(open(keypose_ckpt, 'rb'), pickle_module=dill)
    k_cfg = k_payload['cfg']

    ## set the sampler to twostep
    # k_cfg['policy']['sample']['sampler'] = 'onestep'
    k_cfg['policy']['sample']['sampler'] = 'twostep'


    cls = hydra.utils.get_class(k_cfg._target_)
    k_workspace = cls(k_cfg, output_dir=output_dir)
    k_workspace: BaseWorkspace
    ### in case that model & opt are not defined in __init__ (e.g. ddp)
    if "model" not in k_workspace.__dict__.keys():
        k_workspace.model = hydra.utils.instantiate(k_cfg.policy)
    if "optimizer" not in k_workspace.__dict__.keys():
        k_workspace.optimizer = k_workspace.model.get_optimizer(**k_cfg.optimizer)
    k_workspace.load_payload(k_payload, exclude_keys=None, include_keys=None)

    kp_predictor = k_workspace.model.to(device)
    kp_predictor.eval()
    print("\nkeypose policy loaded")

    """ --- 加载 policy 模型 --- """
    p_payload = torch.load(open(policy_ckpt, 'rb'), pickle_module=dill)
    p_cfg = p_payload['cfg']

    # p_cfg['policy']['sample']['sampler'] = 'onestep'
    p_cfg['policy']['sample']['sampler'] = 'twostep'

    cls = hydra.utils.get_class(p_cfg._target_)
    p_workspace = cls(p_cfg, output_dir=output_dir)
    p_workspace: BaseWorkspace
    ### in case that model, ema_model & opt are not defined in __init__ (e.g. ddp)
    if "model" not in p_workspace.__dict__.keys():
        p_workspace.model = hydra.utils.instantiate(p_cfg.policy)
    if "ema_model" not in p_workspace.__dict__.keys() and p_cfg.training.use_ema:
        p_workspace.ema_model = copy.deepcopy(p_workspace.model)
    if "optimizer" not in p_workspace.__dict__.keys():
        p_workspace.optimizer = p_workspace.model.get_optimizer(**p_cfg.optimizer)
    p_workspace.load_payload(p_payload, exclude_keys=None, include_keys=None)

    policy = p_workspace.model
    if p_cfg.training.use_ema:
        policy = p_workspace.ema_model
    
    policy.to(device)
    policy.eval()
    print("\ntrajectory policy loaded")

    ## set inference steps for consistency model ?? ##
    # policy.num_inference_steps = num_inference_steps
    

    """ --- 创建联合模型 --- """
    policy_cond_kt = KeyposeTrajectoryPolicy(kp_predictor, policy, epsilon=epsilon)
    print("\nKeypose Trajectory Policy created")

    """ --- 设置 REAL 环境 ---"""
    shape_meta = p_cfg.task.shape_meta
    state_dim = shape_meta.obs.qpos.shape[0] ## qpos shape
    camera_names = [key for key in shape_meta.obs.keys() if key.startswith("cam")]
    c, h, w = shape_meta.obs.cam_high.shape ## [c, h, w]

    n_obs_steps = p_cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("max_timesteps:", max_timesteps)

    ## load aloha env
    print("Creating real environment...")
    env = make_real_env(init_node=True, downsample_scale=scale)

    ## rollout
    max_timesteps = int(max_timesteps) # T
    num_rollouts = 1

    """ --- 评估 --- """
    for rollout_idx in range(num_rollouts):
        rollout_idx += 1

        print(f"Rollout {rollout_idx} - Collecting observations...")

        ## reset env
        ts = env.reset()
        t_idx = pad_before = n_obs_steps - 1

        qpos_history = np.zeros(
            (max_timesteps + pad_before, state_dim), dtype=np.float32
        ) # [T+To-1, state_dim]
        images_history = dict()
        for cam_name in camera_names:
            images_history[cam_name] = np.zeros(
                (max_timesteps + pad_before, c, h, w), dtype=np.float32
            ) # [T+To-1, c, h, w]

        qpos_history, images_history = collect_obs(ts, t_idx, n_obs_steps, qpos_history, images_history, camera_names, shape_meta)
        ep_t0 = time.perf_counter()
        step_time_list = []
        with torch.inference_mode():
            ## loop max_timesteps
            while True:
                ''' construct observations_seq = {"images", "qpos"} '''
                # horizon indices: t-To+1, ..., t-1, t
                t_start = t_idx + 1 - n_obs_steps
                t_end = t_idx + 1

                obs_dict_np = dict()
                obs_dict_np["qpos"] = qpos_history[t_start:t_end] ## [n_obs_steps, state_dim]")
                for cam_name in camera_names:
                    obs_dict_np[cam_name] = images_history[cam_name][t_start:t_end] ## [n_obs_steps, c, h, w]

                print(f"obs_ts = [{t_start}:{t_end})")

                ''' get action sequence '''
                t0 = time.perf_counter()
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                
                result = policy_cond_kt.predict_action(obs_dict) ## !!
                
                t1 = time.perf_counter()
                step_time_list.append(t1 - t0)
                # print(f"Execution Policy: {t1 - t0:.4f} [s]")

                action_seq = result['action'][0].detach().to('cpu').numpy() # (Ta,Da)

                ''' implement action sequence '''
                for action in action_seq:
                    ts = env.step(action) # ts_(t+1)
                    t_idx += 1
                    
                    if t_idx == max_timesteps + pad_before:
                        break

                    qpos_history, images_history = collect_obs(ts, t_idx, n_obs_steps, qpos_history, images_history, camera_names, shape_meta)
                
                if t_idx >= max_timesteps + pad_before:
                    break

    ep_t1 = time.perf_counter()
    print(f"Average step time: {np.mean(step_time_list[1:]):.4f} +/- {np.std(step_time_list[1:]):.4f} s")
    print(f"Total time: {ep_t1 - ep_t0:.4f} s")


    """ --- 保存结果 --- """
    n_cam = len(camera_names)
    rgb_seq = np.full((max_timesteps, n_cam, 3, h, w), np.nan, dtype=np.uint8)
    for i, cam_name in enumerate(camera_names):
        rgb_seq[:, i] = (images_history[cam_name][pad_before:] * 255).astype(np.uint8)
    rgb_seq = rearrange(rgb_seq, 't n c h w -> t h (n w) c')

    video_recorder = VideoRecorder.create_h264(
        fps=50,
        codec="h264",
        input_pix_fmt="rgb24",
        crf=22,
        thread_type="FRAME",
        thread_count=1,
    )
    video_recorder.stop()
    video_recorder.start(os.path.join(output_dir, f"rollout_{rollout_idx}.mp4"))
    for rgb in rgb_seq:
        video_recorder.write_frame(rgb)
    video_recorder.stop()

    json_log = dict()
    json_log["ckpt_keypose"] = keypose_ckpt
    json_log["ckpt_trajectory"] = policy_ckpt
    json_log["epsilon"] = epsilon
    json_log["num_inference_steps"] = num_inference_steps
    output_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(output_path, 'w'), indent=2, sort_keys=True)

    ## sleep arms
    reset_arms(env.puppet_bot_left, env.puppet_bot_right)



def collect_obs(ts, idx, n_obs_steps, qpos_history, images_history, camera_names, shape_meta):
    obs = ts.observation
    ## get qpos input
    qpos = np.array(obs['qpos']) ## [state_dim,]
    if idx == n_obs_steps - 1:
        qpos_history[idx+1-n_obs_steps:idx+1] = qpos # broadcast here
    else:
        qpos_history[idx] = qpos

    ## get image input
    curr_image_dict = get_image(ts, camera_names, shape_meta) # it returns a dict
    if idx == n_obs_steps:
        for cam_name in camera_names:
            images_history[cam_name][idx+1-n_obs_steps:idx+1] = curr_image_dict[cam_name]
    else:
        for cam_name in camera_names:
            images_history[cam_name][idx] = curr_image_dict[cam_name]

    return qpos_history, images_history



def get_image(ts, camera_names, shape_meta):
    """
    @return
        curr_images_dict: {
            cam_name: image (c,h,w)
        }
    """
    curr_images_dict = dict()
    for cam_name in camera_names:
        # h,w,c --> c,h,w
        curr_image = np.moveaxis(ts.observation['images'][cam_name], -1, 0) / 255.0
        assert curr_image.shape == tuple((shape_meta.obs[cam_name]).shape), \
            f"{curr_image.shape} vs. {tuple((shape_meta.obs[cam_name]).shape)}"
        curr_images_dict[cam_name] = curr_image # [0, 1]^(c,h,w)

    return curr_images_dict


def reset_arms(puppet_bot_left=None, puppet_bot_right=None):
    all_bots = [puppet_bot_left, puppet_bot_right]
    for bot in all_bots:
        torque_on(bot)

    ### move grippers
    PUPPET_GRIPPER_JOINT_OPEN = 1.4910
    move_grippers([puppet_bot_left, puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open

    puppet_sleep_position = (0, -1.7, 1.55, 0.12, 0.65, 0)
    move_arms(all_bots, [puppet_sleep_position] * 2, move_time=2)



if __name__ == "__main__":
    main()