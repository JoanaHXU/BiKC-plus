if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import time
import os
import h5py
from typing import Dict
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
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.keypose_base_dataset import KeyposeBaseDataset
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.normalize_util import get_image_range_normalizer


register_codecs()


class KeyposeAlohaMultiImageDataset(KeyposeBaseDataset):
    def __init__(
        self,
        dataset_dir: str,
        shape_meta: dict,
        num_episodes=50,
        camera_names=["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"],
        seed=42,
        val_ratio=0.0,
        task="aloha_insert_10s_random_init",
        use_cache=False,
    ):
        super().__init__()
        '''
            structure of self.data:
            {
                "cam_x": np.array([T, H, W, C]),
                "cam_y": np.array([T, H, W, C]),
                ...
                "qpos": np.array([T, 14]), # a.k.a. current joint position
                "last_keypose": np.array([T, 14]),
                "next_keypose": np.array([T, 14]),
            }
        '''
        dataset_dir = os.path.join(dataset_dir, task, "keypose")

        if use_cache:
            cache_hdf5_path = os.path.join(dataset_dir, f"{task}_keypose.hdf5")
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
            with h5py.File(dataset_path, "r") as root:

                all_cam_images = dict()
                for cam in camera_names:
                    all_cam_images[cam] = root[f"/obs/images/{cam}"][()] # [T, H, W, C]
                
                
                qpos = root["/obs/qpos"][()]  # [T, 14]
                keypose = root["/obs/keypose"][()]  # [T, 14]
                tgt_keypose = root["/tgt/keypose"][()]  # [T, 14]
                tgt_mode = root["/tgt/mode"][()]  # [T, 1]

            episode = {
                "qpos": qpos,  # [T, 14]
                "keypose": keypose,  # [T, 14]
                "tgt_keypose": tgt_keypose,  # [T, 14]
                "tgt_mode": tgt_mode,  # [T, 1]
            }
            episode.update(all_cam_images)
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
    

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "qpos": self.data["qpos"],
            "keypose": self.data["keypose"],
            "tgt_keypose": self.data["tgt_keypose"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
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
            # ..., H,W,C --> ..., C,H,W
            # convert uint8 image [0,255] to float32 [0, 1]
            obs_dict[key] = (
                np.moveaxis(data[key][idx], -1, -3).astype(np.float32) / 255.0
            )
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
        "cam_high": list(),
        "cam_low": list(),
        "cam_left_wrist": list(),
        "cam_right_wrist": list(),
        "qpos": list(),
        "keypose": list(),
        "tgt_keypose": list(),
        "tgt_mode": list(),
    } 


def main():

    dataset_dir = "/ssd1/xuhang/datasets/"
    task = "aloha2_screwdriver_fix"

    shape_meta = {
        "obs": {
            "cam_high": {
                "shape": (3, 120, 160),
                "type": "rgb",
            },
            "cam_low": {
                "shape": (3, 120, 160),
                "type": "rgb",
            },
            "cam_left_wrist": {
                "shape": (3, 120, 160),
                "type": "rgb",
            },
            "cam_right_wrist": {
                "shape": (3, 120, 160),
                "type": "rgb",
            },
            "qpos": {"shape": (14,), "type": "low_dim"},
            "keypose": {"shape": [14], "type": "low_dim"},
        },
        "tgt": {
            "keypose": {"shape": [14], "type": "low_dim"},
            "mode": {"shape": [1], "type": "low_dim"},
        },
        "keypose_ts": {"shape": [1], "type": "low_dim"},
    }

    np.set_printoptions(formatter={'int':lambda x: f"{x:>4}"})
    dataset = KeyposeAlohaMultiImageDataset(
        dataset_dir=dataset_dir,
        shape_meta=shape_meta,
        num_episodes=1,
        seed=42,
        val_ratio=0.2,
        task=task,
        use_cache=False,
    )

    print("dataset length:", len(dataset))
    print("training indices number:", len(dataset.train_indices))
    print("data['qpos'].shape:", dataset.data["qpos"].shape)

    # test normalizer
    normalizer = dataset.get_normalizer()

    nqpos = normalizer['qpos'].normalize(dataset.data['qpos'][:])
    nqpos_unnorm = normalizer['qpos'].unnormalize(nqpos)

    print(f"original qpos[0]: {dataset.data['qpos'][0]}")
    print(f"norm qpos[0]: {nqpos[0]}")
    print(f"unnorm qpos[0]: {nqpos_unnorm[0]}")

    # test getitem
    sample = dataset[0]
    print("sample['obs']['cam_low'].shape:", sample["obs"]["cam_low"].shape)
    print("sample['obs']['cam_left_wrist'].shape:", sample["obs"]["cam_left_wrist"].shape)
    print("sample['obs']['qpos'].shape:", sample["obs"]["qpos"].shape)
    print("sample['obs']['keypose'].shape:", sample["obs"]["keypose"].shape)
    print("sample['tgt']['keypose'].shape:", sample['tgt_keypose'].shape)

    # test loader speed
    val_set = dataset.get_validation_dataset()
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=8,
        shuffle=True,
        pin_memory=False,
        persistent_workers=True,
    )

    num_epochs = 1
    num_steps = 10
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}:")
        train_time_per_batch = []
        start = time.time()
        for i, batch in enumerate(tqdm(train_loader)):
            time_get = time.time()
            train_time_per_batch.append(time_get - start)
            start = time_get
            if i + 1 == num_steps:
                break
        train = np.array(train_time_per_batch)
        print(f"Train mean: {train.mean():.3f}, std: {train.std():.3f}, max: {train.max():.3f}")
        print("train:", train[:10])


if __name__ == "__main__":
    main()
