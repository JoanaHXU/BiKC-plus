'''
python eval_sim_kp_dataset.py \
    -k data/outputs/2024.03.02/21.10.08_train_keypose_transformer_keypose_sim_transfer_cube_scripted/checkpoints/latest.ckpt \
    -d /ssd1/xuhang/dataset_sim/ \
    -t sim_transfer_cube_scripted \ 
    -i 0
'''

import click
import os
import pathlib
import os
import h5py
import numpy as np
import torch
from einops import rearrange
import hydra
import dill
import matplotlib.pyplot as plt
import time


from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace


camera_names = ["top"]
rgb_keys = ['images']
lowdim_keys = ['qpos', 'keypose']

def load_episode_to_data(dataset_path):

    with h5py.File(dataset_path, "r") as root: 

        all_cam_images = []
        for cam_name in camera_names:
            all_cam_images.append(root[f"/obs/images/{cam_name}"][()])
        all_cam_images = np.stack(all_cam_images, axis=0)  # [n, T, H, W, C]

        qpos = root["/obs/qpos"][()]  # [T, 14]
        keypose = root["/obs/keypose"][()]  # [T, 14]
        tgt_keypose = root["/tgt/keypose"][()]  # [T, 14]
        tgt_mode = root["/tgt/mode"][()]  # [T, 1]

    episode = {
        "images": all_cam_images.squeeze(),  # XXX: assume only one camera, [T, H, W, C]
        "qpos": qpos,  # [T, 14]
        "keypose": keypose,  # [T, 14]
        "tgt_keypose": tgt_keypose,  # [T, 14]
        "tgt_mode": tgt_mode,  # [T, 1]
    }

    return episode

def get_data(data, idx):

    obs_dict = dict()
    for key in rgb_keys:
        # move channel last to channel first
        # H,W,C
        # convert uint8 image to float32
        obs_dict[key] = (
            rearrange(data[key][idx], "... h w c -> ... c h w").astype(np.float32) / 255.0
        )
        obs_dict[key] = np.expand_dims(obs_dict[key], axis=0)  # [1, C, H, W]
        # C,H,W
    for key in lowdim_keys:
        obs_dict[key] = data[key][idx].astype(np.float32)
        obs_dict[key] = np.expand_dims(obs_dict[key], axis=0)  # [1, D]

    torch_data = {
        "obs": dict_apply(obs_dict, torch.from_numpy),
        "tgt_keypose": torch.from_numpy(data["tgt_keypose"][idx].astype(np.float32)).unsqueeze(0),  # [1, 14]
        "tgt_mode": torch.from_numpy(data["tgt_mode"][idx].astype(np.float32)).unsqueeze(0),  # [1, 1]
    }
    return torch_data


def visualize_keypose_performance(pred_keypose_all, tgt_keypose_all, output_dir, epi_idx):
    """
    1. 散点图对比 -- **关注点对点的预测准确性**
    ### X轴: 目标值 (Target) - 真实的关节角度; Y轴: 预测值 (Predicted) - 模型预测的关节角度
    ### 每个点: 代表某个时刻的 (真实值, 预测值) 对
    ### 预测精度: 点越接近 y=x 线，预测越准确
    """
    plt.figure(figsize=(20, 15))
    for i in range(14):
        plt.subplot(4, 4, i+1)
        plt.scatter(tgt_keypose_all[:, i], pred_keypose_all[:, i], alpha=0.6, s=20)
        plt.plot([tgt_keypose_all[:, i].min(), tgt_keypose_all[:, i].max()], 
                 [tgt_keypose_all[:, i].min(), tgt_keypose_all[:, i].max()], 
                 'r--', label='Perfect Prediction')
        plt.xlabel(f'Target Joint {i}')
        plt.ylabel(f'Predicted Joint {i}')
        plt.title(f'Joint {i} Prediction vs Target')
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'kp_scatter_{epi_idx}.png'), dpi=150)
    plt.close()


    '''
    2. 分布直方图对比 -- **关注整体分布的统计特性**
    ### X轴: 关节角度值; Y轴: 概率密
    ### 目标分布: 真实的关节角度分布；预测分布: 模型预测的关节角度分布
    ### 分析目标和预测的分布差异
        (1). 分布形状一致, 但中心位置偏移: 模型有系统性偏差(bias),可能需要调整模型的偏置项
        (2). 中心位置相同, 但一个更宽/更窄：模型预测方差与真实方差不匹配
        (3). 分布范围不同: 模型预测过于保守(pred值范围比tgt值范围小)或激进(pred值范围比tgt值范围大)
        (4). 目标分布有多个峰，但预测分布只有一个峰：模型无法捕捉数据的复杂模式
    '''
    plt.figure(figsize=(20, 15))
    for i in range(14):
        plt.subplot(4, 4, i+1)
        plt.hist(tgt_keypose_all[:, i], bins=20, alpha=0.7, label='Target', color='blue', density=True)
        plt.hist(pred_keypose_all[:, i], bins=20, alpha=0.7, label='Predicted', color='red', density=True)
        plt.xlabel(f'Joint {i} Value')
        plt.ylabel('Density')
        plt.title(f'Joint {i} Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'kp_histogram_{epi_idx}.png'), dpi=150)
    plt.close()

    """3. 时间序列对比 - 选择几个关键关节
    ### 选择几个代表性关节，观察它们在时间序列上的变化
    ### X轴: 时间步; Y轴: 关节角度值
    ### 关注点: 预测值与目标值的时间序列变化趋势是否一致
    """
    key_joints = list(range(14)) # 选择4个代表性关节
    plt.figure(figsize=(16, 10))
    for i, joint_idx in enumerate(key_joints):
        plt.subplot(4, 4, i+1)
        plt.plot(tgt_keypose_all[:, joint_idx], label='Target', linewidth=2)
        plt.plot(pred_keypose_all[:, joint_idx], label='Predicted', linewidth=2, alpha=0.8)
        plt.xlabel('Time Step')
        plt.ylabel(f'Joint {joint_idx} Angle')
        plt.title(f'Joint {joint_idx} Time Series Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'kp_timeseries_{epi_idx}.png'), dpi=150)
    plt.close()

    """
    4. 相关性分析热力图
    ### 计算每个关节的预测值与目标值之间的相关系数
    ### 绘制热力图展示每个关节的相关性
    ### 关注点: 哪些关节的预测与目标高度相关，哪些关节的预测较差
    """
    plt.figure(figsize=(12, 10))
    correlations = []
    for i in range(14):
        corr = np.corrcoef(tgt_keypose_all[:, i], pred_keypose_all[:, i])[0, 1]
        correlations.append(corr)
    
    # 创建相关性矩阵可视化
    corr_matrix = np.eye(14)
    for i in range(14):
        corr_matrix[i, i] = correlations[i]
    
    plt.imshow(corr_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation Coefficient')
    plt.title('Correlation between Target and Predicted Keypose\n(Diagonal shows per-joint correlation)')
    plt.xlabel('Joint Index')
    plt.ylabel('Joint Index')

    # 添加数值标注
    for i in range(14):
        plt.text(i, i, f'{correlations[i]:.3f}', ha='center', va='center', 
                color='white' if abs(correlations[i]) > 0.5 else 'black', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'kp_correlation_heatmap_{epi_idx}.png'), dpi=150)
    plt.close()

    # 相关系数阈值判断
    for i, corr in enumerate(correlations):
        if corr > 0.9:
            # print(f"    Joint {i}: 优秀预测 (r={corr:.3f})")
            pass
        elif corr > 0.7:
            print(f"    Joint {i}: 尚可预测 (r={corr:.3f})")
        elif corr > 0.5:
            print(f"    Joint {i}: 一般预测 (r={corr:.3f})")
        else:
            print(f"    Joint {i}: 较差预测 (r={corr:.3f}) - 需要改进\n")

    """ 5. 统计指标总结 """
    # 计算各种统计指标
    mae = np.mean(np.abs(pred_keypose_all - tgt_keypose_all), axis=0)
    mse = np.mean((pred_keypose_all - tgt_keypose_all)**2, axis=0)
    std_error = np.std(pred_keypose_all - tgt_keypose_all, axis=0)

    """ 6. 获取keypose跳变的时间点 """
    jump_threshold = 0.1  # 定义跳变阈值
    key_joints = [2, 9]  # 选择关节2和9进行跳变检测
    keypose_ts = dict()
    for idx in key_joints:
        joint_data = pred_keypose_all[:, idx]
        jump_indices = [0]

        # 检测跳变
        for i in range(1, joint_data.shape[0]):
            if np.abs(joint_data[i] - joint_data[i-1]) > jump_threshold:
                jump_indices.append(i)
        jump_indices.append(joint_data.shape[0] - 1)  # 添加最后一个时间步

        if idx == key_joints[0]:
            # print("     Left Wrist reaches keypose at time steps:", jump_indices)
            keypose_ts['left_wrist'] = jump_indices
        elif idx == key_joints[-1]:
            # print("     Right Wrist reaches keypose at time steps:", jump_indices)
            keypose_ts['right_wrist'] = jump_indices


    ## 7. 保存数值结果
    results_summary = {
        'episode_index': epi_idx,
        # 'std_error_per_joint': std_error.tolist(),
        'overall_mae': float(np.mean(mae)),  # 转换为 Python float
        'overall_mse': float(np.mean(mse)),  # 转换为 Python float
        'overall_correlation': float(np.mean(correlations)),  # 转换为 Python float
        'keypose_left_timesteps': keypose_ts.get('left_wrist', []),
        'keypose_right_timesteps': keypose_ts.get('right_wrist', []),
    }

    import json
    with open(os.path.join(output_dir, f'evaluation_results_epi_{epi_idx}.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # print(f"\n  Evaluation completed for episode {epi_idx}")
    print(f"    Overall MAE: {results_summary['overall_mae']:.4f}")
    print(f"    Overall MSE: {results_summary['overall_mse']:.6f}")
    print(f"    Overall Correlation: {results_summary['overall_correlation']:.4f}")
    # print(f"    Results saved to: {output_dir}")


def visualize_mode_performance(pred_mode_list, tgt_mode_list, output_dir, epi_idx):
    """
    可视化模式预测的性能
    1. 模式预测的散点图对比
    2. 模式预测的直方图对比
    """
    pred_mode_all = np.concatenate(pred_mode_list, axis=0)
    tgt_mode_all = np.concatenate(tgt_mode_list, axis=0)

    plt.figure(figsize=(10, 5))
    plt.scatter(tgt_mode_all, pred_mode_all, alpha=0.6, s=20)
    plt.plot([tgt_mode_all.min(), tgt_mode_all.max()], 
             [tgt_mode_all.min(), tgt_mode_all.max()], 
             'r--', label='Perfect Prediction')
    plt.xlabel('Target Mode')
    plt.ylabel('Predicted Mode')
    plt.title('Mode Prediction vs Target')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'mode_scatter_{epi_idx}.png'), dpi=150)
    plt.close()

    # 直方图对比
    plt.figure(figsize=(10, 5))
    plt.hist(tgt_mode_all, bins=20, alpha=0.7, label='Target', color='blue', density=True)
    plt.hist(pred_mode_all, bins=20, alpha=0.7, label='Predicted', color='red', density=True)
    plt.xlabel('Mode Value')
    plt.ylabel('Density')
    plt.title('Mode Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'mode_histogram_{epi_idx}.png'), dpi=150)
    plt.close()

    # 时间序列对比
    plt.figure(figsize=(10, 5))
    plt.plot(tgt_mode_all, label='Target Mode', linewidth=2)
    plt.plot(pred_mode_all, label='Predicted Mode', linewidth=2, alpha=0.8)
    plt.xlabel('Time Step')
    plt.ylabel('Mode Value')
    plt.title('Mode Time Series Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'mode_timeseries_{epi_idx}.png'), dpi=150)
    plt.close()

    print(f"    Mode Error: {np.mean(np.abs(pred_mode_all - tgt_mode_all)):.4f}")


def evaluate_episode(kp_predictor, episode_path, output_dir, epi_idx, device):
    """
    评估单个 episode 的数据
    """
    
    ## load episode to data
    episode_data = load_episode_to_data(episode_path)
    horizon = episode_data["qpos"].shape[0]

    pred_keypose_list = []
    pred_mode_list = []
    tgt_keypose_list = []
    tgt_mode_list = []
    time_list = []
    for idx in range(horizon):
        data = get_data(episode_data, idx)
        data = dict_apply(data, lambda x: x.to(device))

        with torch.no_grad():
            t_start = time.time()
            result = kp_predictor.predict_keypose_and_mode(data["obs"])
            t_end = time.time()
        time_list.append(t_end - t_start)

        ## prediction
        pred_keypose = result["keypose"]
        pred_mode = result["mode"]

        ## target
        tgt_keypose = data["tgt_keypose"]
        tgt_mode = data["tgt_mode"]

        ## collect results
        pred_keypose_list.append(pred_keypose.cpu().numpy())
        pred_mode_list.append(pred_mode.cpu().numpy())
        tgt_keypose_list.append(tgt_keypose.cpu().numpy())
        tgt_mode_list.append(tgt_mode.cpu().numpy())

    pred_keypose_all = np.concatenate(pred_keypose_list, axis=0)
    tgt_keypose_all = np.concatenate(tgt_keypose_list, axis=0)

    visualize_keypose_performance(pred_keypose_all, tgt_keypose_all, output_dir, epi_idx)
    visualize_mode_performance(pred_mode_list, tgt_mode_list, output_dir, epi_idx)
    print(f"    Time per step: {np.mean(time_list):.4f} seconds.")

@click.command()
@click.option('-k', '--keypose_ckpt', required=True)
@click.option('-d', '--dataset_dir', required=True)
@click.option('-t', '--task', required=True)
@click.option('-i', '--epi_num', type=int, default=5, help="Episode index to evaluate")
@click.option('-s', '--step_num', type=int, default=1, help="Number of steps for inference")
@click.option('-mt', '--model_type', type=str, default='cm', help="Model type:'cm' or 'tfm'")
@click.option('-dv', '--device', default='cuda:0', help="Device to run the evaluation on")
def main(keypose_ckpt, dataset_dir, task, epi_num, step_num, model_type, device):

    keypose_ckpt_root = "/".join(keypose_ckpt.split('/')[:-2])
    output_dir = os.path.join(keypose_ckpt_root, f"eval_{step_num}step")
    if not os.path.exists(output_dir):
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(device)

    """ --- 加载数据集 --- """
    dataset_path = os.path.join(dataset_dir, task, 'keypose')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

    """ --- 加载模型 ---"""
    k_payload = torch.load(open(keypose_ckpt, 'rb'), pickle_module=dill)
    k_cfg = k_payload['cfg']

    ## 设定 one-step 或 two-step 采样
    if model_type=='cm' and step_num == 1:
        k_cfg['policy']['sample']['sampler'] = 'onestep'
    elif model_type=='cm' and step_num == 2:
        k_cfg['policy']['sample']['sampler'] = 'twostep'
    elif model_type=='tfm':
        print("Using transformer model, step_num is ignored.")
    else:
        raise ValueError(f"Invalid model type {model_type} or step_num {step_num}.")

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

    """ --- 模型预测 --- """
    for epi_idx in range(epi_num):
        print(f"\n--- Evaluating episode {epi_idx} ---")
        ## episode path
        episode_path = os.path.join(dataset_path, f'kp_episode_{epi_idx}.hdf5')
        if not os.path.exists(episode_path):
            raise FileNotFoundError(f"Dataset path {episode_path} does not exist.")
        ## output path
        # output_path = os.path.join(output_dir, f'epi_{epi_idx}')\
        output_path = output_dir
        if not os.path.exists(output_path):
            pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
        ## 评估
        evaluate_episode(kp_predictor, episode_path, output_path, epi_idx, device)

    print("Evaluation completed for all episodes.")


if __name__ == "__main__":
    main()