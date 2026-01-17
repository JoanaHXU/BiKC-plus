if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)


import os
import h5py
from typing import Dict, List
import torch
import numpy as np
import copy
from tqdm import tqdm
import os
import shutil
from filelock import FileLock
from threadpoolctl import threadpool_limits
from einops import rearrange

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.sampler import get_val_mask
# from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.common.normalizer_var_adjusted import LinearNormalizer ## use var-adjusted normalizer
from diffusion_policy.dataset.keypose_base_dataset import KeyposeBaseDataset
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.normalize_util import get_image_range_normalizer


register_codecs()


class KeyposeAlohaDataset(KeyposeBaseDataset):
    def __init__(
        self,
        dataset_dir: str,
        shape_meta: dict,
        num_episodes=50,
        camera_names=["top"],
        seed=42,
        val_ratio=0.0,
        task="sim_transfer_cube_scripted",
        use_cache=False,
        normalizer=None,
    ):
        super().__init__()
        '''
            structure of self.data:
            {
                "images": np.array([T, H, W, C]),
                "qpo": np.array([T, 14]),
                "keypose": np.array([T, 14]),
                "tgt_keypose": np.array([T, 14]),
                "tgt_mode": np.array([T, 1]),
            }
        '''
        dataset_dir = os.path.join(dataset_dir, task, "keypose")
        if use_cache:
            cache_hdf5_path = os.path.join(dataset_dir, "keypose_cache.hdf5")
            cache_lock_path = cache_hdf5_path + ".lock"
            print("Acquiring lock on cache.")
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_hdf5_path):
                    try:
                        print("Cache does not exsit. Creating!")
                        data = self.load_episodes_to_data(
                            num_episodes=num_episodes,
                            dataset_dir=dataset_dir,
                            camera_names=camera_names,
                        )
                        print("Saving cache to disk.")
                        with h5py.File(cache_hdf5_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                            root.attrs["sim"] = True
                            for key, value in data.items():
                                root.create_dataset(key, data=value)
                    except Exception as e:
                        shutil.rmtree(cache_hdf5_path)
                        raise e
                else:
                    print("Loading cached keypose data from disk.")
                    with h5py.File(cache_hdf5_path, 'r') as root:
                        data = dict()
                        for key in root.keys():
                            data[key] = root[key][()]
                    print("Loaded!")
        else:
            data = self.load_episodes_to_data(
                num_episodes=num_episodes,
                dataset_dir=dataset_dir,
                camera_names=camera_names,
            )

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

        val_mask = get_val_mask(
            n_episodes=len(next(iter(data.values()))),
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        
        self.data = data
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.train_mask = train_mask
        self.train_indices = np.where(train_mask)[0]
        self.camera_names = camera_names


    def load_episodes_to_data(
        self,
        num_episodes,
        dataset_dir,
        camera_names,
    ):
        data = _create_empty_data()
        for i in tqdm(range(num_episodes)):  # num_episodes
            dataset_path = os.path.join(dataset_dir, f"kp_episode_{i}.hdf5")
            assert os.path.exists(dataset_path), f"Dataset file {dataset_path} does not exist."
            
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
            
            assert set(episode.keys()) == set(data.keys())

            for key, value in episode.items():
                data[key].append(value)

        ## Concatenate data from all episodes
        for key, value in data.items():
            data[key] = np.concatenate(value, axis=0)

        return data
    

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.train_mask = ~self.train_mask
        val_set.train_indices = np.where(val_set.train_mask)[0]
        return val_set
    

    # def get_normalizer(self, mode="limits", **kwargs):
    #     data = {
    #         "qpos": self.data["qpos"],
    #         "keypose": self.data["keypose"],
    #         "tgt_keypose": self.data["tgt_keypose"],
    #     }
    #     normalizer = LinearNormalizer()
    #     normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
    #     normalizer["images"] = get_image_range_normalizer()
    #     return normalizer


    def get_normalizer(self, 
                       mode="limits", 
                       adjust_small_variance=False,
                       small_variance_threshold=0.05,
                       min_range_multiplier=4.0,
                       verbose=False,
                       **kwargs):
        data = {
            "qpos": self.data["qpos"],
            "keypose": self.data["keypose"],
            "tgt_keypose": self.data["tgt_keypose"],
        }

        normalizer = LinearNormalizer()
        normalizer.fit(
            data=data, 
            last_n_dims=1, 
            mode=mode, 
            ## for var-adjusted normalizer
            adjust_small_variance=adjust_small_variance,
            small_variance_threshold=small_variance_threshold,
            min_range_multiplier=min_range_multiplier,
            verbose=verbose,
            **kwargs)

        normalizer["images"] = get_image_range_normalizer()

        return normalizer
    

    def __len__(self) -> int:
        return len(self.train_indices)


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.data
        idx = self.train_indices[idx]

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # H,W,C
            # convert uint8 image to float32
            obs_dict[key] = (
                rearrange(data[key][idx], "... h w c -> ... c h w").astype(np.float32) / 255.0
            )
            # C,H,W
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][idx].astype(np.float32)

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "tgt_keypose": torch.from_numpy(data["tgt_keypose"][idx].astype(np.float32)),
            "tgt_mode": torch.from_numpy(data["tgt_mode"][idx].astype(np.float32)),
        }
        return torch_data


def _create_empty_data():
    return {
        "images": list(),
        "qpos": list(),
        "keypose": list(),
        "tgt_keypose": list(),
        "tgt_mode": list(),
    }


def main():

    dataset_dir = "/ssd1/xuhang/dataset_sim/"
    task = "sim_insertion_scripted"

    shape_meta = {
        "obs":{
            "images": {"shape": [3, 120, 160], "type": "rgb"},
            "qpos": {"shape": [14], "type": "low_dim"},
            "keypose": {"shape": [14], "type": "low_dim"},
        },
        "tgt": {
            "keypose": {"shape": [14], "type": "low_dim"},
            "mode": {"shape": [1], "type": "low_dim"},
        },
        "keypose_ts": {"shape": [1], "type": "low_dim"},
    }

    np.set_printoptions(formatter={'int':lambda x: f"{x:>4}"}) ## 设置NumPy数组中整数的打印格式

    dataset = KeyposeAlohaDataset(
        dataset_dir=dataset_dir,
        shape_meta=shape_meta,
        num_episodes=5,
        camera_names=["top"],
        seed=42,
        val_ratio=0.1,
        task=task,
        use_cache=False,
    )

    qpos = dataset.data["qpos"]
    keypose = dataset.data["keypose"]
    tgt_keypose = dataset.data["tgt_keypose"]
    tgt_mode = dataset.data["tgt_mode"]

    print("dataset length:", len(dataset))
    print("train indices:", dataset.train_indices)
    print("data['images'].shape:", dataset.data["images"].shape)
    print("data['qpos'].shape:", qpos.shape)
    print("data['keypose'].shape:", keypose.shape)
    print("data['tgt_keypose'].shape:", tgt_keypose.shape)
    print("data['tgt_mode'].shape:", tgt_mode.shape)

    # test normalizer
    normalizer = dataset.get_normalizer()
    norm_qpos = normalizer['qpos'].normalize(dataset.data['qpos'][:])
    norm_keypose = normalizer['keypose'].normalize(dataset.data['keypose'][:])
    norm_tgt_keypose = normalizer['tgt_keypose'].normalize(dataset.data['tgt_keypose'][:])

    original_qpos = normalizer['qpos'].unnormalize(norm_qpos)
    original_keypose = normalizer['keypose'].unnormalize(norm_keypose)
    original_tgt_keypose = normalizer['tgt_keypose'].unnormalize(norm_tgt_keypose)


    # test getitem
    sample = dataset[0]
    print("sample['obs']['images'].shape:", sample["obs"]["images"].shape)
    print("sample['obs']['qpos'].shape:", sample["obs"]["qpos"].shape)
    print("sample['obs']['keypose'].shape:", sample["obs"]["keypose"].shape)
    print("sample['tgt_keypose'].shape:", sample["tgt_keypose"].shape)
    print("sample['tgt_mode'].shape:", sample["tgt_mode"].shape)

if __name__ == "__main__":
    main()
