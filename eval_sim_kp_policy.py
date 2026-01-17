'''
python eval_sim_kp_runner.py \
    --keypose_ckpt /home/xuhang/TASE/data/outputs/2025.07.08/17.40.10_train_keypose_cm_keypose_sim_transfer_cube_scripted/checkpoints/epoch=0950-keypose_error_val=0.00106.ckpt \
    --policy_dir /home/xuhang/TASE/data/outputs/2025.07.04/15.02.34_train_diffusion_transformer_image_sim_transfer_cube_scripted/checkpoints/latest.ckpt \
    --task_cfg /home/xuhang/TASE/diffusion_policy/config/task/sim_transfer_cube_scripted.yaml 
'''

import click
import os
import pathlib
import os
import torch
import hydra
import dill
import copy
from omegaconf import OmegaConf
import os
import wandb
import json

# os.environ['MUJOCO_GL'] = 'egl'  # 如果使用 MuJoCo

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.kt_policy import KeyposeTrajectoryPolicy

camera_names = ["top"]
rgb_keys = ['images']
lowdim_keys = ['qpos', 'keypose']


@click.command()
@click.option('-k', '--keypose_ckpt', required=True)
@click.option('-p', '--policy_ckpt', required=True)
@click.option('-t', '--task_cfg', required=True)
@click.option('-e', '--epsilon', type=float, default=0.25, help="Distance threshold for keypose matching")
@click.option('-d', '--device', default='cuda:0', help="Device to run the evaluation on")
def main(keypose_ckpt, policy_ckpt, task_cfg, epsilon, device):

    policy_ckpt_root = "/".join(policy_ckpt.split('/')[:-2])
    output_dir = os.path.join(policy_ckpt_root, f"eval_epi={epsilon:.2f}")
    if not os.path.exists(output_dir):
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(device)

    """ --- 加载 keypose 模型 --- """
    k_payload = torch.load(open(keypose_ckpt, 'rb'), pickle_module=dill)
    k_cfg = k_payload['cfg']

    ## set the sampler to twostep
    k_cfg['policy']['sample']['sampler'] = 'onestep'

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
    print("keypose policy loaded")

    """ --- 加载 policy 模型 --- """
    p_payload = torch.load(open(policy_ckpt, 'rb'), pickle_module=dill)
    p_cfg = p_payload['cfg']

    p_cfg['policy']['sample']['sampler'] = 'onestep'

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
    print("trajectory policy loaded")

    """ --- 创建联合模型 --- """
    policy_cond_kt = KeyposeTrajectoryPolicy(kp_predictor, policy, epsilon=epsilon)

    """ --- 设置仿真环境 ---"""
    with open(task_cfg, 'r') as f:
        cfg = OmegaConf.load(f)
    # some modification on runner cfg
    cfg.env_runner.n_obs_steps = p_cfg.n_obs_steps
    cfg.env_runner.n_action_steps = p_cfg.n_action_steps
    cfg.env_runner.past_action = False
    cfg.env_runner.n_test = 20
    cfg.env_runner.n_test_vis = 5

    env_runner = hydra.utils.instantiate(
        cfg.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy_cond_kt) ## 重点

    # print(runner_log.keys())
    print(f"\nAverage Rewards: {runner_log['test/mean_score']:.4f}")
    print(f"Success Rate: \
          \n\t 1st stage: {runner_log['test/1st']} \
          \n\t 2nd stage: {runner_log['test/2nd']} \
          \n\t 3rd stage: {runner_log['test/3rd']}")

    # dump log to json
    json_log = dict()
    json_log["checkpoint_keypose"] = keypose_ckpt
    json_log["checkpoint_trajectory"] = policy_ckpt
    json_log["eposilon"] = epsilon
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)


if __name__ == "__main__":
    main()