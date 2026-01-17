from typing import Dict
from copy import deepcopy
import numpy as np
import torch
import modern_robotics as mr

from diffusion_policy.policy.keypose_base_policy import KeyposeBasePolicy
from diffusion_policy.policy.trajectory_base_policy import TrajectoryBasePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env.aloha.constants import (
    DT, vx300s, LEFT_BASE_POSE, RIGHT_BASE_POSE
)


class KeyposeTrajectoryPolicy:
    ### init 
    # load two sub policies
    # maintain target_keypose
    def __init__(self,
        keypose_model: KeyposeBasePolicy,
        trajectory_model: TrajectoryBasePolicy,
        epsilon: float = 0.35,
    ):
        self.keypose_model = keypose_model
        self.trajectory_model = trajectory_model
        self.epsilon = epsilon # distance threshold for keypose matching

        self.pre_keypose = None
        self.tgt_keypose = None
        self.tgt_mode = None
        self.min = epsilon

    def predict_action(self,
        obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        @params
            obs_dict: dict
                str: B,To,*
        @return
            result: dict
                action_pred: B,H,Da
                action: B,Ta,Da
                tgt_keypose: B,Da
        """

        k_nn = self.keypose_model
        t_nn = self.trajectory_model

        # current obs_dict: because obs_dict contains multi-step
        # o_(t-T_o+1), ..., o_(t-1), o_t,
        current_obs_dict = dict_apply(obs_dict, lambda x: x[:, -1].clone())
        current_pose = current_obs_dict["qpos"]
        device = current_pose.device

        ####### option 1: update keyposes after reaching target, better for real-world
        if self.tgt_keypose is None: # init step, all target_keypose should be updated
            ## pre_kp is the initial keypose
            self.pre_keypose = current_obs_dict["qpos"].clone() # B,Da
            current_obs_dict["keypose"] = self.pre_keypose
            ## predict target keypose and mode
            kp_result = k_nn.predict_keypose_and_mode(current_obs_dict) # B,Da
            self.tgt_keypose = kp_result["keypose"]
            self.tgt_mode = kp_result["mode"]

        ## check if current_pose is close to target_keypose
        flag = self._reach_target(current_pose, device)

        ## if reach target keypose, update pre_keypose and tgt_keypose
        if flag.any():
            # update pre_keypose
            self.pre_keypose[flag] = self.tgt_keypose[flag].clone()
            # update target keypose and mode
            current_obs_dict["keypose"] = self.pre_keypose
            kp_result = k_nn.predict_keypose_and_mode(current_obs_dict) # B,Da
            self.tgt_keypose[flag] = kp_result["keypose"][flag].clone()
            self.tgt_mode[flag] = kp_result["mode"][flag]
        ####### option 1 end

        # predict sub trajectory
        result = t_nn.predict_trajectory(obs_dict, self.tgt_keypose)
        result["tgt_keypose"] = self.tgt_keypose

        return result
    
    def _reach_target(self, current_pose: torch.Tensor, device: torch.device):
        # check if some current_poses are close to target_keyposes
        dist_left = dist_to_target(current_pose[:, :7], self.tgt_keypose[:, :7])
        dist_right = dist_to_target(current_pose[:, 7:], self.tgt_keypose[:, 7:])
        self.min = min(self.min, dist_left.min().item(), dist_right.min().item())
        # print(f"left_dist={dist_left} \t right_dist={dist_right}")

        print(f"LEFT={dist_left[0].item()} \t RIGHT={dist_right[0].item()}")

        reach_left_kp = (dist_left < self.epsilon).to(device) # (B,)
        reach_right_kp = (dist_right < self.epsilon).to(device) # (B,)

        ## element-wise 逐元素操作
        reach_any_kp = reach_left_kp | reach_right_kp  # Element-wise OR -- 任一手臂到达目标
        reach_both_kp = reach_left_kp & reach_right_kp  # Element-wise AND -- 双手臂都到达目标

        mode_0_mask = (self.tgt_mode == 0).squeeze(1) # (B,) - 非协调模式
        mode_1_mask = (self.tgt_mode == 1).squeeze(1) # (B,) - 协调模式

        ## 1. non-coordination mode: any arm reach target, the pre_kp & tgt_kp will be updated 
        flag_any_reach_kp = mode_0_mask & reach_any_kp
        ## 2. coordination mode: both arms reach target, the pre_kp & tgt_kp will be updated
        flag_both_reach_kp = mode_1_mask & reach_both_kp
        ## combine the flags
        flag = flag_any_reach_kp | flag_both_reach_kp

        if flag.any():
            print(f"    Reaching KP (both={flag_both_reach_kp.item()}): LEFT={dist_left[flag].item()} \t RIGHT={dist_right[flag].item()}")
            print("\n")

        return flag
    
    # reset state for stateful policies
    def reset(self):
        self.keypose_model.reset()
        self.trajectory_model.reset()
        self.target_keypose = None
        self.last_keypose = None

    @property
    def device(self):
        return self.keypose_model.device
    
    @property
    def dtype(self):
        return self.keypose_model.dtype


def dist_to_target(
    x: torch.Tensor,
    target: torch.Tensor,
    dim: int = -1
):
    """
    计算了两个张量之间的欧几里得距离(L2范数)
    @params
        x: torch.Tensor
            B,D
        target: torch.Tensor
            B,D
        dim: int

    @return
        d: torch.Tensor
            B
    """
    x = x.to(target.device)
    d = torch.linalg.vector_norm(x - target, dim=dim).to("cpu") # (B,)
    return d


def qpos_to_eepose(
    qpos: np.ndarray,
):
    """
    @params
        qpos: np.ndarray
            B, 14
    @return
        left_eepose: np.ndarray
            B, 4, 4
        right_eepose: np.ndarray
            B, 4, 4
    """
    if isinstance(qpos, torch.Tensor):
        qpos = qpos.cpu().numpy()
    left, right = np.split(qpos, 2, axis=-1)
    expected_shape = (qpos.shape[0], 4, 4)
    
    left_arm, right_arm = left[:, :-1], right[:, :-1]  # B, 6
    left_gripper, right_gripper = left[:, -1:], right[:, -1:] # B, 1
    relative_left_eepose = np.array(
        [mr.FKinSpace(vx300s.M, vx300s.Slist, this_qpos) for this_qpos in left_arm]
    )
    relative_right_eepose = np.array(
        [mr.FKinSpace(vx300s.M, vx300s.Slist, this_qpos) for this_qpos in right_arm]
    )
    left_eepose = np.matmul(LEFT_BASE_POSE, relative_left_eepose)
    right_eepose = np.matmul(RIGHT_BASE_POSE, relative_right_eepose)

    assert left_eepose.shape == expected_shape
    
    return left_eepose, right_eepose