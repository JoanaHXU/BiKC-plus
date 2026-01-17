
import h5py
import numpy as np
import os
import click
from tqdm import tqdm


def read_keypose_hdf5(file_path):
    """
    Reads an HDF5 file and returns the data as a dictionary.
    
    Args:
        file_path (str): Path to the HDF5 file.
        
    Returns:
        dict: Dictionary containing the data from the HDF5 file.
    """
    with h5py.File(file_path, 'r') as f:

        left_kp_qpos = f['left/qpos/'][()] # type: ignore
        right_kp_qpos = f['right/qpos/'][()] # type: ignore

        left_kp_ts = f['left/timestep/'][()] # type: ignore
        right_kp_ts = f['right/timestep/'][()] # type: ignore

        # left_image = dict()
        # for cam_name in f[f'/left/image/'].keys(): # type: ignore
        #     left_image[cam_name] = f[f'/left/image/{cam_name}'][()].astype(np.uint8) # type: ignore

        # right_image = dict()
        # for cam_name in f[f'/right/image/'].keys(): # type: ignore
        #     right_image[cam_name] = f[f'/right/image/{cam_name}'][()].astype(np.uint8) # type: ignore

    kp_data = {
        'left_qpos': left_kp_qpos,
        'right_qpos': right_kp_qpos,
        'left_ts': left_kp_ts,
        'right_ts': right_kp_ts,
        # 'left_image': left_image,
        # 'right_image': right_image
    }
        
    return kp_data



def read_mode_hdf5(file_path):
    """
    Reads an HDF5 file and returns the data as a dictionary.
    
    Args:
        file_path (str): Path to the HDF5 file.
        
    Returns:
        dict: Dictionary containing the data from the HDF5 file.
    """
    with h5py.File(file_path, 'r') as f:
        co_data = list()
        ts_data = list()

        for key in f['Mode'].keys(): # type: ignore
            co_data.append(f[f'Mode/{key}/co/'][()]) # type: ignore
            ts_data.append(f[f'Mode/{key}/ts/'][()]) # type: ignore

        mode_data = {
            'co': np.array(co_data),
            'ts': np.array(ts_data)
        }
        
    return mode_data



def get_mode_at_biTS(kp_ts_full, mode_data):
    """
    Identifies the mode at each keypose timestamp.
    
    Args:
        kp_ts_full (np.ndarray): Array of keypose timestamps.
        mode_data (dict): Dictionary containing coordination modes and their timestamps.
        
    Returns:
        np.ndarray: Array of modes corresponding to each keypose timestamp.
    """
    kp_mode = dict()

    kp_mode_full = np.zeros_like(kp_ts_full, dtype=int)
    idx = 0
    for idx_base, ts_base in enumerate(mode_data['ts']): # type: ignore
        while idx < len(kp_ts_full):
            if kp_ts_full[idx] <= ts_base:
                kp_mode_full[idx] = mode_data['co'][idx_base] # type: ignore
                idx += 1
            else:
                break

    kp_mode = {
        'ts': list(kp_ts_full),
        'co': list(kp_mode_full),
    }

    return kp_mode



def get_uniKP_at_biTS(keypose_data, kp_mode, kp_ts_bi):
    """
    Re-organizes individual keypose data to match the unified keypose timestamps.
    Args:
        keypose_data (dict): Dictionary containing keypose data with left and right timestamps and qpos.
        kp_mode (dict): Dictionary containing the unified keypose timestamps and coordination modes.
    Returns:
        tuple: Two dictionaries for left and right keyposes, each containing 'ts', 'co', and 'qpos'.
        -- 'ts' is the keypose timestamp, 
        -- 'co' is the coordination mode, 
        -- 'qpos' is the joint positions (None for unavailable keyposes)
    """

    left_kp = {
        'ts': list(kp_ts_bi),
        'co': list(kp_mode['co']),
        'qpos': [None]*len(kp_ts_bi),
    }
    right_kp = {
        'ts': list(kp_ts_bi),
        'co': list(kp_mode['co']),
        'qpos': [None]*len(kp_ts_bi),
    }

    for ts in kp_ts_bi:
        if ts in keypose_data['left_ts']:
            idx = np.where(keypose_data['left_ts'] == ts)[0][0]
            left_kp['qpos'][kp_ts_bi.tolist().index(ts)] = keypose_data['left_qpos'][idx]
        if ts in keypose_data['right_ts']:
            idx = np.where(keypose_data['right_ts'] == ts)[0][0]
            right_kp['qpos'][kp_ts_bi.tolist().index(ts)] = keypose_data['right_qpos'][idx]

    return left_kp, right_kp



def get_bimanual_keypose(left_kp, right_kp, kp_mode, kp_ts_bi):
    """
    Generates a unified bimanual keypose dataset based on coordination mode.
    Returns:
        dict: Dictionary containing bimanual keypose data with 'ts', 'co', and 'qpo'.
        -- 'ts' is the keypose timestamp,
        -- 'co' is the coordination mode,
        -- 'qpo' is the joint positions ([0:7] for left hand, [7:14] for right hand).
    """

    ## 1. Fill missing keyposes, based on the mode
    ## -- Coordination is False: No Merging
    ## -- Coordination is True: Merging
    for idx in range(len(kp_ts_bi)-2, -1, -1):
        ## no merging -- fill with the next available keypose
        if left_kp['qpos'][idx] is None and left_kp['co'][idx] == 0:
            left_kp['qpos'][idx] = left_kp['qpos'][idx+1] if idx+1 < len(kp_ts_bi) else None
        if right_kp['qpos'][idx] is None and right_kp['co'][idx] == 0:
            right_kp['qpos'][idx] = right_kp['qpos'][idx+1] if idx+1 < len(kp_ts_bi) else None

        ## merging -- fill with right/left kp at the same timestamp if available
        if left_kp['qpos'][idx] is None and left_kp['co'][idx] == 1:
            left_kp['qpos'][idx] = right_kp['qpos'][idx] if right_kp['qpos'][idx] is not None else None
        if right_kp['qpos'][idx] is None and right_kp['co'][idx] == 1:
            right_kp['qpos'][idx] = left_kp['qpos'][idx] if left_kp['qpos'][idx] is not None else None

    assert not any(x is None for x in left_kp['qpos']), "Left keypose qpos contains None values."
    assert not any(x is None for x in right_kp['qpos']), "Right keypose qpos contains None values."

    ## 2. Get bimanual keypose by combining left keypose [0:7] and right keyposes [7:14]
    bimanual_kp = {
        'ts': kp_ts_bi,
        'co': kp_mode['co'],
        'qpos': [None]*len(kp_ts_bi)
    }

    for idx in range(len(kp_ts_bi)):
        left_qpos = left_kp['qpos'][idx][:7]
        right_qpos = right_kp['qpos'][idx][7:]

        bimanual_kp['qpos'][idx] = np.concatenate((left_qpos, right_qpos))

    # print("Unified keypose data prepared.")

    return bimanual_kp



def generate_training_dataset(demo_input_path, bimanual_kp):
    """
    Generates a training dataset for keypose prediction based on the demo input and bimanual keypose data.
    
    Args:
        demo_input_path (str): Path to the demo input file containing image data.
        bimanual_kp (dict): Dictionary containing bimanual keypose data with 'ts', 'co', and 'qpo'.
        
    Returns:
        dict: Training dataset containing obs and tgt data.
    """
    
    ## check if the input files exist
    if not os.path.exists(demo_input_path):
        raise FileNotFoundError(f"Demo input file {demo_input_path} does not exist.")

   ## load image data from demo
    if os.path.exists(demo_input_path):
        with h5py.File(demo_input_path, 'r') as f:
            this_image = dict()
            for cam_name in f['/observations/images/'].keys(): # type: ignore
                this_image[cam_name] = f[f'/observations/images/{cam_name}'][:].astype(np.uint8) # type: ignore
                horizon = this_image[cam_name].shape[0] 
            this_qpo = f['/observations/qpos/'][:].astype(np.float32) # type: ignore
            this_action = f['/action/'][:].astype(np.float32) # type: ignore
        # print("Demo image data loaded successfully.")
    else:
        print(f"Demo input file {demo_input_path} does not exist.")

    ## prepare the training dataset
    training_dataset = {
        "obs": {
            "images": this_image, 
            "qpos": this_qpo, # (horizon, 14)
            "keypose": np.zeros((horizon, 14)), # (horizon, 14) 
        },
        "tgt": {
            "keypose": np.zeros((horizon, 14)), # (horizon, 14) 
            "mode": np.zeros((horizon, 1)), # (horizon, 1) 
            "keypose_ts": bimanual_kp['ts'], 
        },
        "action": this_action, # (horizon, 14)
        
    }

    for ts in range(horizon): 

        ## based on kp_ts_bi, find the number before and after the current idx
        prev_idx = np.where(ts >= bimanual_kp['ts'])[0]
        next_idx = np.where(ts < bimanual_kp['ts'])[0]

        if len(prev_idx) > 0:
            prev_idx = prev_idx[-1]
        else:
            prev_idx = 0
        if len(next_idx) > 0:
            next_idx = next_idx[0]
        else:
            next_idx = len(bimanual_kp['ts']) - 1

        training_dataset['obs']['keypose'][ts] = bimanual_kp['qpos'][prev_idx]
        training_dataset['tgt']['keypose'][ts] = bimanual_kp['qpos'][next_idx]
        training_dataset['tgt']['mode'][ts] = bimanual_kp['co'][next_idx]

    # print("Training dataset prepared.")

    return training_dataset



def save_kp_training_dataset(output_path, training_dataset):

    with h5py.File(output_path, 'w') as f:
        grp = f.create_group('obs')
        grp.create_dataset('keypose', data=training_dataset['obs']['keypose'])
        grp.create_dataset('qpos', data=training_dataset['obs']['qpos']) # type: ignore
        img_grp = grp.create_group("images")
        for cam_name in training_dataset['obs']['images'].keys(): # type: ignore
            img_grp.create_dataset(cam_name, data=training_dataset['obs']['images'][cam_name]) # type: ignore

        grp = f.create_group('tgt')
        grp.create_dataset('keypose', data=training_dataset['tgt']['keypose'])
        grp.create_dataset('mode', data=training_dataset['tgt']['mode'])
        grp.create_dataset('ts', data=training_dataset['tgt']['keypose_ts']) # type: ignore

        f.create_dataset('action', data=training_dataset['action']) # type: ignore

    # print(f"Keypose training data saved to {output_path}")



def get_biKP_from_uniKP_and_mode(keypose_ds_path, mode_ds_path):
    """
    Generates a bimanual keypose dataset from individual keypose and mode datasets.
    Args:
        keypose_ds_path (str): Path to the keypose dataset file.
        mode_ds_path (str): Path to the mode dataset file.
    Returns:
        dict: Dictionary containing bimanual keypose data with 'ts', 'co', and 'qpo'.
        -- 'ts' is the keypose timestamp,
        -- 'co' is the coordination mode,
        -- 'qpo' is the joint positions ([0:7] for left hand, [7:14] for right hand).
    """
    ## check if the input files exist
    if not os.path.exists(keypose_ds_path):
        raise FileNotFoundError(f"Keypose dataset file {keypose_ds_path} does not exist.")
    if not os.path.exists(mode_ds_path):
        raise FileNotFoundError(f"Mode dataset file {mode_ds_path} does not exist.")

    ## read the keypose dataset
    if os.path.exists(keypose_ds_path):
        keypose_data = read_keypose_hdf5(keypose_ds_path)
        # print("Data loaded successfully.")
    else:
        print(f"File {keypose_ds_path} does not exist.")

    ## read the mode dataset
    if os.path.exists(mode_ds_path):
        mode_data = read_mode_hdf5(mode_ds_path)
        # print("Data loaded successfully.")
    else:
        print(f"File {mode_ds_path} does not exist.")


    ## all keypose timestep: combine keypose_data['left_ts'] and keypose_data['right_ts'], sort them, and remove duplicates
    kp_ts_bi = np.concatenate((keypose_data['left_ts'], keypose_data['right_ts'])) # type: ignore
    kp_ts_bi = np.unique(kp_ts_bi)
    kp_ts_bi.sort()
    # print(f"Combined timestamps: {kp_ts_bi}")

    ## identify the mode along all the keypose timestamps
    kp_mode = get_mode_at_biTS(kp_ts_bi, mode_data) # type: ignore
    print(f"Keypose modes identified: {kp_mode}")

    ## Re-organize individual keypose data, remain NA for unavailable keyposes
    left_kp, right_kp = get_uniKP_at_biTS(keypose_data, kp_mode, kp_ts_bi) # type: ignore

    ## generate a unified bimanual keypose dataset, depending on the coordination mode
    bimanual_kp = get_bimanual_keypose(left_kp, right_kp, kp_mode, kp_ts_bi) # type: ignore

    return bimanual_kp


@click.command()
@click.option('--keypose_ds_dir', type=str, help='Path to the keypose dataset directory.')
@click.option('--mode_ds_dir', type=str, help='Path to the mode dataset directory.')
@click.option('--demo_ds_dir', type=str, help='Path to the demo input directory containing image data.')
@click.option('--output_dir', type=str, help='Directory to save the output training dataset.')
@click.option('--num_episodes', type=int, default=1, help='Number of episodes to process.')
def main(keypose_ds_dir, mode_ds_dir, demo_ds_dir, output_dir, num_episodes):
    """
    Generate keypose training dataset from keypose and mode datasets.
    """

    print("\nStarting to generate keypose training dataset...")

    for episode_idx in tqdm(range(num_episodes), desc="Processing episodes"):
        ## input paths
        keypose_ds_path = os.path.join(keypose_ds_dir, f'keypose_{episode_idx}.hdf5')
        mode_ds_path = os.path.join(mode_ds_dir, f'mode_{episode_idx}.hdf5')
        demo_input_path = os.path.join(demo_ds_dir, f'episode_{episode_idx}.hdf5')

        ## output path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, f'kp_episode_{episode_idx}.hdf5')

        ## generate bimanual_keypose dataset from keypose and mode datasets
        bimanual_kp = get_biKP_from_uniKP_and_mode(keypose_ds_path, mode_ds_path)
        print(f"epi={episode_idx}: Keypose timestamp: {bimanual_kp['ts']} \t Coordination mode: {bimanual_kp['co']}")

        ## generate training dataset for keypose prediction
        training_dataset = generate_training_dataset(demo_input_path, bimanual_kp)

        # Save the unified keypose data to a file
        save_kp_training_dataset(output_path, training_dataset)


    print("\n🎉 All steps completed successfully!\n")


def test_single_episode():
    """
    Test function to process a single episode with default paths.
    """
    keypose_ds_path = '/home/xuhang/TASE/kp_dataset/sim_transfer_cube_scripted/dataset_keypose/keypose_0.hdf5' 
    mode_ds_path = '/home/xuhang/TASE/kp_dataset/sim_transfer_cube_scripted/dataset_mode/mode_0.hdf5'
    demo_input_path = '/ssd1/xuhang/dataset_sim/sim_transfer_cube_scripted/episode_0.hdf5'
    output_dir = '/home/xuhang/TASE/kp_dataset/sim_transfer_cube_scripted/dataset_training/'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'kp_episode_0.hdf5')

    ## generate bimanual_keypose dataset from keypose and mode datasets
    bimanual_kp = get_biKP_from_uniKP_and_mode(keypose_ds_path, mode_ds_path)

    ## generate training dataset for keypose prediction
    training_dataset = generate_training_dataset(demo_input_path, bimanual_kp)

    # Save the unified keypose data to a file
    save_kp_training_dataset(output_path, training_dataset)
    
    print("Test completed successfully!")


if __name__ == "__main__":
    import sys
    
    # Check if running with command line arguments
    if len(sys.argv) > 1:
        # Run with click command line interface
        main()
    else:
        # Run test function with default parameters
        print("No command line arguments provided. Running test with default parameters...")
        test_single_episode()
