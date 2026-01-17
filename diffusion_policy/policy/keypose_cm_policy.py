from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

## new package
from diffusion_policy.model.keypose.karras_diffusion import KarrasDenoiser, karras_sample ## updated for KP
from diffusion_policy.model.consistency.sampler import create_named_schedule_sampler, LossAwareSampler
from diffusion_policy.model.consistency.scripts_util import create_ema_and_scales_fn
from diffusion_policy.model.consistency.nn import update_ema
from diffusion_policy.model.consistency.fp16_utils import master_params_to_model_params, make_master_params, get_param_groups_and_shapes
## updated for KP
from diffusion_policy.model.consistency.nn import mean_flat
from diffusion_policy.model.keypose.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.keypose.classification_mlp import Binary_Classification_MLP
from diffusion_policy.model.common.normalizer_var_adjusted import LinearNormalizer


import time

import functools
import ipdb
import copy

class KeyposeConsistencyPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            # model: ConditionalUnet1D,
            noise_scheduler,
            ema_scale,
            sample,
            obs_encoder: MultiImageObsEncoder,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=False,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            enable_bias_correction=False,
            weighted_by_normalizer=False,
            weighted_by_batch=False,
            # model: mlp
            hidden_depth=2,
            hidden_dim=1024,
            **kwargs):
        super().__init__()

        # parse shapes
        tgt_keypose_shape = shape_meta['tgt']['keypose']['shape']
        tgt_model_shape = shape_meta['tgt']['mode']['shape']
        assert len(tgt_keypose_shape) == 1
        assert len(tgt_model_shape) == 1
        tgt_keypose_dim = tgt_keypose_shape[0]
        tgt_mode_dim = tgt_model_shape[0]

        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        ### -- mode_head: MLP --

        mode_net = Binary_Classification_MLP(
            input_dim=obs_feature_dim,
            output_dim=tgt_mode_dim,
            hidden_dims=[hidden_dim] * hidden_depth,  # hidden_dims is a list of hidden layer sizes
            dropout=0.1  # dropout can be adjusted as needed
        )

        ### -- keypose_head: consistency model --
        input_dim = tgt_keypose_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = tgt_keypose_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        ## -- create model --
        keypose_net = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            enable_bias_correction=enable_bias_correction,
        )

        ## -- create ema --
        self.ema_scale_fn = create_ema_and_scales_fn(
            target_ema_mode=ema_scale.target_ema_mode,
            start_ema=ema_scale.start_ema,
            scale_mode=ema_scale.scale_mode,
            start_scales=ema_scale.start_scales,
            end_scales=ema_scale.end_scales,
            total_steps=ema_scale.total_training_steps,
            distill_steps_per_iter=ema_scale.distill_steps_per_iter,
        )

        ## -- model follow DP --
        self.mode_net = mode_net
        self.keypose_net = keypose_net
        self.param_groups_and_shapes = get_param_groups_and_shapes(
            self.keypose_net.named_parameters()
        )
        self.master_params = make_master_params(
            self.param_groups_and_shapes
        )

        ## -- target model --
        self.target_keypose_net = copy.deepcopy(self.keypose_net)
        self.target_keypose_net.requires_grad_(False)
        self.target_keypose_net.train()

        self.target_keypose_net_param_groups_and_shapes = get_param_groups_and_shapes(
            self.target_keypose_net.named_parameters()
        )
        self.target_keypose_net_master_params = make_master_params(
            self.target_keypose_net_param_groups_and_shapes
        )

        ## -- scheduler follow CM --
        self.diffusion = KarrasDenoiser(
            sigma_data=noise_scheduler.sigma_data,
            sigma_max=noise_scheduler.sigma_max,
            sigma_min=noise_scheduler.sigma_min,
            distillation=noise_scheduler.distillation,
            weight_schedule=noise_scheduler.weight_schedule,
            noise_schedule=noise_scheduler.noise_schedule,
            loss_norm=noise_scheduler.loss_norm,
        )
        self.schedule_sampler = create_named_schedule_sampler(noise_scheduler.schedule_sampler, self.diffusion)

        self.obs_encoder = obs_encoder
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.tgt_keypose_dim = tgt_keypose_dim
        self.tgt_mode_dim = tgt_mode_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        self.training_mode = ema_scale.training_mode ## new
        #
        self.sampler = sample.sampler
        self.generator = sample.generator
        self.ts = sample.ts
        self.clip_denoised = sample.clip_denoised
        self.sigma_min = noise_scheduler.sigma_min
        self.sigma_max = noise_scheduler.sigma_max

        self.s_churn = sample.s_churn
        self.s_tmin = sample.s_tmin
        self.s_tmax = float(sample.s_tmax)
        self.s_noise = sample.s_noise
        self.steps = sample.steps
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.weighted_by_normalizer = weighted_by_normalizer
        self.weighted_by_batch = weighted_by_batch


    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

        ## initialize weights from normalizer
        if self.weighted_by_normalizer:
            print(f"Initialize weights based on dataset std ...")
            self.weight_by_std = self.initialize_weights_from_normalizer(normalizer=self.normalizer)
            self.weight_by_std = self.weight_by_std.unsqueeze(0).unsqueeze(0)  # [1, 1, D_kp]
        else:
            self.weight_by_std = None


    def get_optimizer(
        self, 
        keypose_weight_decay: float, 
        mode_weight_decay: float,
        obs_encoder_weight_decay: float,
        learning_rate: float, 
        betas: Tuple[float, float]
    ) -> torch.optim.Optimizer:
        """
        Create optimizer for the three modules:
        1. keypose_net: ConditionalUnet1D
        2. mode_net: MLP
        3. obs_encoder: MultiImageObsEncoder
        """
        optim_groups = []
        optim_groups.append({
            "params": self.keypose_net.parameters(),
            "weight_decay": keypose_weight_decay
        })
        optim_groups.append({
            "params": self.mode_net.parameters(),
            "weight_decay": mode_weight_decay
        })
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer
    

    def predict_keypose_and_mode(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        if len(nobs['keypose'].shape) == 2:
            for key in nobs.keys():
                nobs[key] = nobs[key].unsqueeze(1)  # [B, T, D0] for obs_encoder processing
        value = next(iter(nobs.values()))
        
        B, To = value.shape[:2]
        T = self.horizon
        Do = self.obs_feature_dim
        D_kp = self.tgt_keypose_dim
        D_mode = self.tgt_mode_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, D_kp), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, D_kp+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,D_kp:] = nobs_features
            cond_mask[:,:To,D_kp:] = True


        '''--- predict keypose ---'''

        # run sampling
        nkeypose_pred = karras_sample(
            self.diffusion,
            self.keypose_net,
            (B, T, D_kp),
            cond_data, 
            cond_mask,
            steps=self.steps,
            clip_denoised=self.clip_denoised,
            local_cond=None,
            global_cond=global_cond,
            device=self.device,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            sampler=self.sampler,
            s_churn=self.s_churn,
            s_tmin=self.s_tmin,
            s_tmax=self.s_tmax,
            s_noise=self.s_noise,
            generator=None,
            ts=self.ts,
        ) ## 重点！！ CM 定制！
        
        # unnormalize keypose prediction
        keypose_pred = self.normalizer['tgt_keypose'].unnormalize(nkeypose_pred) # B, 1, D_kp
        keypose_pred = keypose_pred.squeeze(1)  # B, D_kp

        '''--- predict mode ---'''
        mode_pred = self.mode_net.predict(nobs_features)
        
        '''--- output result ---'''
        result = {
            'keypose': keypose_pred, # B, D_kp
            'mode': mode_pred, # B, D_mode  
        }
        return result
    

    # ========= training  ============

    def compute_loss(self, batch, global_step):
        # normalize input
        assert 'valid_mask' not in batch

        ## get obs and keypose from batch, normalize them, and reshape them
        nobs = self.normalizer.normalize(batch['obs'])
        if len(nobs['keypose'].shape) == 2:
            horizon = 1
            for key in nobs.keys():
                nobs[key] = nobs[key].unsqueeze(1)  # [B, T, D0] for obs_encoder processing
        else:
            horizon = nobs['keypose'].shape[1]
        
        keypose = self.normalizer['tgt_keypose'].normalize(batch['tgt_keypose']) 
        keypose = keypose.unsqueeze(1) # B, 1, Do
        mode = batch['tgt_mode']

        batch_size = keypose.shape[0]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        cond_data = keypose
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([keypose, nobs_features], dim=-1)
            keypose = cond_data.detach()

        '''---- compute keypose loss ----'''

        ## -- Importance-sample timesteps for a batch
        t, weights = self.schedule_sampler.sample(keypose.shape[0], self.device)

        ema, num_scales = self.ema_scale_fn(global_step) ## 注意：参数设置 is different between CD and CT

        ## -- 声明 compute loss 模式 -- ##
        ## -- -- 定义于 karra_diffusion.py / consistency_loss()
        if self.training_mode == "consistency_training":
            compute_losses = functools.partial(
                self.diffusion.consistency_losses,
                self.keypose_net,
                keypose,
                num_scales,
                target_model=self.target_keypose_net,
                local_cond = local_cond,
                global_cond = global_cond,
            )
        else:
            raise ValueError(f"Warning training mode {self.training_mode}")

        ## -- 计算 loss -- ##
        #losses_kp = compute_losses() ## 重点
        kp_term = compute_losses()
        diffs = kp_term["diffs"]
        weights_by_schedule = kp_term["weights"]

        if self.weighted_by_normalizer==True and self.weighted_by_batch==False:
            ## -- 使用 normalizer 中的 std 初始化权重 -- ##
            weight_by_std = self.weight_by_std.to(self.device)  # [1, 1, D_kp]
            kp_losses = diffs * weight_by_std  # [B, 1, D_kp]
            kp_losses = mean_flat(kp_losses) * weights_by_schedule # [B]

        elif self.weighted_by_normalizer==False and self.weighted_by_batch==True:
            ## -- 使用 batch 中的 std 初始化权重 -- ##
            weight_by_std = self.initialize_weights_from_batch(batch).to(self.device)  # [1, 1, D_kp]
            kp_losses = diffs * weight_by_std  # [B, 1, D_kp]
            kp_losses = mean_flat(kp_losses) * weights_by_schedule # [B]

        elif self.weighted_by_normalizer==False and self.weighted_by_batch==False:
            ## -- 不使用权重 -- ##
            kp_losses = mean_flat(diffs) * weights_by_schedule

        else:
            ## -- 错误的权重组合 -- ##
            raise ValueError("Invalid combination of weighted_by_normalizer and weighted_by_batch") 

        ## 当 sampler = LossSecondMomentResampler 才会启动
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, kp_losses.detach()
            )

        ## -- loss 加权取平均 -- ##
        loss_kp = (kp_losses * weights).mean()

        """ -- 计算 mode loss -- """
        loss_mode = self.mode_net.compute_loss(
            nobs_features, 
            mode
        )

        """ -- 整合 loss -- """
        loss = loss_kp + loss_mode

        return loss
    

    def eval_sampling(self, predictor, sampling_batch, device):

        with torch.no_grad():
            batch = dict_apply(sampling_batch, lambda x: x.to(device, non_blocking=True))
            obs_dict = batch['obs']
            gt_keypose = batch['tgt_keypose']
            gt_mode = batch['tgt_mode']

            result = predictor.predict_keypose_and_mode(obs_dict)
            pre_keypose = result['keypose']
            pre_mode = result['mode']
            
            error_keypose_l1 = torch.mean(torch.abs(gt_keypose - pre_keypose)).item()
            ratio_mode_correct = (gt_mode.eq(pre_mode)).sum().item() / gt_mode.shape[0]

            if self.weighted_by_normalizer == True:
                error_keypose_l1_weighted= torch.mean(
                    torch.abs(gt_keypose - pre_keypose) * self.weight_by_std.to(device)
                ).item()

            del batch
            del obs_dict
            del gt_keypose
            del gt_mode
            del result
            del pre_keypose
            del pre_mode

        if self.weighted_by_normalizer == True:
            return error_keypose_l1, ratio_mode_correct, error_keypose_l1_weighted
        else:
            return error_keypose_l1, ratio_mode_correct, None


    def update_target_ema(self, global_step):

        ## Note: 此处针对原代码（consistency model）进行改动：
        ## 为防止 master_params 和 target_model_master_params 没有随着模型更新
        ## 此处 显性地实时地同步一遍 master_params 和 target_model_master_params
        self.master_params = make_master_params(
            self.param_groups_and_shapes
        )
        self.target_keypose_net_master_params = make_master_params(
            self.target_keypose_net_param_groups_and_shapes
        )

        target_ema, scales = self.ema_scale_fn(global_step)
        with torch.no_grad():
            update_ema(
                self.target_keypose_net_master_params,
                self.master_params,
                rate = target_ema,
            )
            master_params_to_model_params(
                self.target_keypose_net_param_groups_and_shapes,
                self.target_keypose_net_master_params,
            )
        # print(self.target_keypose_net_master_params)

    
    def initialize_weights_from_normalizer(self, 
                                           normalizer=None,
                                           base_weight=1.0):
        assert normalizer is not None, "normalizer must be provided to initialize weights"

        try:
            ## 获取 normalizer 中的 params_dict
            input_stats = normalizer.get_input_stats()
            if 'tgt_keypose' in input_stats:
                keypose_stats = input_stats['tgt_keypose']
                std_values = keypose_stats['std'].to(self.device)
            else:
                ## dimension mismatch, use default
                std_values = input_stats['std'].to(self.device)
            
            ## 使用连续的权重函数
            weights = base_weight / (std_values + 1e-8)
            weights = weights / weights.mean()  # normalize weights

            return weights

        except AttributeError:
            print("Error: Normalizer does not have the required attributes.")
            return None


    def initialize_weights_from_batch(self, 
                                    batch=None,
                                    base_weight=1.0):

        tgt_keypose = batch['tgt_keypose']
        if tgt_keypose is None:
            raise ValueError("tgt_keypose is None, cannot initialize weights from batch")

        try:
            ## 获取 tgt_keypose 中的 std
            std_values = torch.std(tgt_keypose, dim=0).to(self.device)
            
            ## 使用连续的权重函数
            weights = base_weight / (std_values + 1e-8)
            weights = weights / weights.mean()  # normalize weights

            weights = weights.unsqueeze(0).unsqueeze(0)  # [1, 1, D_kp]

            return weights

        except AttributeError:
            print("Error: Normalizer does not have the required attributes.")
            return None
