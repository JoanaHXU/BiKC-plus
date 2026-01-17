from typing import Dict, Union, Tuple
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.keypose_base_policy import KeyposeBasePolicy
from diffusion_policy.model.vision.dict_image_obs_encoder import DictImageObsEncoder
from diffusion_policy.model.keypose.transformer_for_keypose import TransformerForKeypose
from diffusion_policy.model.keypose.classification_mlp import Binary_Classification_MLP
from diffusion_policy.common.pytorch_util import dict_apply

logger = logging.getLogger(__name__)

class KeyposeTransformerPolicy(KeyposeBasePolicy):
    def __init__(self,
        shape_meta: dict,
        obs_encoder: DictImageObsEncoder,
        # transformer architecture
        n_layer: int=4,
        n_head: int=4,
        d_embedding: int=256,
        p_drop: float=0.1,
        ## model: mlp
        hidden_depth=2,
        hidden_dim=1024,
    ) -> None:
        super().__init__()

        # parse shapes
        tgt_keypose_dim = shape_meta['tgt']['keypose']['shape'][0]
        tgt_mode_dim = shape_meta['tgt']['mode']['shape'][0]
        # get feature dim dict
        obs_feature_dim_dict = obs_encoder.output_shape()
        # obs_feature_dim = obs_feature_dim_dict[0]

        rgb_keys = obs_encoder.rgb_keys
        low_dim_keys = obs_encoder.low_dim_keys

        # ### -- mode_head: MLP --
        obs_feature_dim = 0
        for key in obs_feature_dim_dict.keys():
            dim = math.prod(obs_feature_dim_dict[key])
            obs_feature_dim += dim

        mode_net = Binary_Classification_MLP(
            input_dim=obs_feature_dim,
            output_dim=tgt_mode_dim,
            hidden_dims=[hidden_dim] * hidden_depth,  # hidden_dims is a list of hidden layer sizes
            dropout=0.1  # dropout can be adjusted as needed
        )

        # create transformer model
        keypose_net = TransformerForKeypose(
            rgb_keys=rgb_keys,
            low_dim_keys=low_dim_keys,
            output_dim=tgt_keypose_dim,
            n_layer=n_layer,
            n_head=n_head,
            d_embedding=d_embedding,
            p_drop=p_drop
        )

        self.obs_encoder = obs_encoder
        self.keypose_net = keypose_net
        self.mode_net = mode_net
        self.normalizer = LinearNormalizer()
        self.obs_feature_dim_dict = obs_feature_dim_dict

    # ========== inference ==========
    def predict_keypose_and_mode(
        self,
        obs_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        @param obs_dict
            keys = shape_meta["obs"].keys()
        @return tgt_keypose: (B, next_keypose_dim)
        """
        # normalize obs
        nobs = self.normalizer.normalize(obs_dict)
        # encode obs
        nobs_feature = self.obs_encoder(nobs)

        # predict normalized next_keypose
        ntgt_keypose = self.keypose_net(nobs_feature)
        # unnormalize next_keypose
        tgt_keypose = self.normalizer["tgt_keypose"].unnormalize(ntgt_keypose)

        ## predict mode
        features = []
        batch_size = nobs['qpos'].shape[0]
        for key in nobs_feature.keys():
            feature_item = nobs_feature[key].reshape(batch_size,-1)
            features.append(feature_item)
        nobs_feature_embeddings = torch.cat(features, dim=-1)

        tgt_mode = self.mode_net.predict(nobs_feature_embeddings)

        result = {
            "keypose": tgt_keypose,
            "mode": tgt_mode,
        }

        return result

    # ========== training ==========
    def get_optimizer(
        self, 
        transformer_weight_decay: float, 
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
        optim_groups = self.keypose_net.get_optim_groups(
            weight_decay=transformer_weight_decay)
        optim_groups.append({
            "params": self.mode_net.parameters(),
            "weight_decay": mode_weight_decay  # mode_net does not need weight decay
        })
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(
        self,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
    ) -> torch.Tensor:
        nobs = self.normalizer.normalize(batch["obs"])
        tgt_keypose = self.normalizer["tgt_keypose"].normalize(batch["tgt_keypose"]).detach()
        tgt_mode = batch["tgt_mode"].detach()

        # encode obs
        nobs_feature = self.obs_encoder(nobs)

        # keypose loss
        pred_keypose = self.keypose_net(nobs_feature)
        loss_keypose = F.mse_loss(pred_keypose, tgt_keypose)

        # mode loss
        features = []
        batch_size = tgt_mode.shape[0]
        for key in nobs_feature.keys():
            feature_item = nobs_feature[key].reshape(batch_size,-1)
            features.append(feature_item)
        nobs_feature_embeddings = torch.cat(features, dim=-1)
        loss_mode = self.mode_net.compute_loss(nobs_feature_embeddings, tgt_mode)

        loss = loss_keypose + loss_mode
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

            del batch
            del obs_dict
            del gt_keypose
            del gt_mode
            del result
            del pre_keypose
            del pre_mode

        return error_keypose_l1, ratio_mode_correct
