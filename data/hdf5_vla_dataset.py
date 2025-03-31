import os
import fnmatch
import json
import glob
import time

import h5py
import yaml
import cv2
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import torch

from configs.state_vec import STATE_VEC_IDX_MAPPING

BASE_DIR = "data/datasets/"
DATASET_NAME = "Tabletop-Close-Door-v1"
DATASET_INSTRUCTIONS = {
    "Tabletop-Close-Door-v1": "Close the door.",
    # Add instructions for other datasets here
    # "Your-New-Task-v1": "Perform your new task.",
}

class HDF5VLADataset:
    """
    This class is used to sample episodes from the embododiment dataset
    """
    def __init__(self) -> None: 
        # Base directory for all datasets
        DATASET_PATH = os.path.join(BASE_DIR, DATASET_NAME)

        if not os.path.exists(DATASET_PATH):
            raise ValueError(f"Dataset {DATASET_NAME} not found.")
        if DATASET_NAME not in DATASET_INSTRUCTIONS:
            raise ValueError(f"Instructions for dataset {DATASET_NAME} not found. Available instructions: {list(DATASET_INSTRUCTIONS.keys())}")
        
        self.file_paths = []
        for root, _, files in os.walk(DATASET_PATH):
            for filename in fnmatch.filter(files, '*.h5'):
                file_path = os.path.join(root, filename)
                self.file_paths.append(file_path)
                
        # Load the config
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISTORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']
    
        # Get each episode's len
        episode_lens = []
        # print(f"Number of episodes(in HDF5VLADataset): {len(self.file_paths)}")
        for file_path in self.file_paths:
            valid, res = self.parse_hdf5_file_state_only(file_path)
            _len = res['state'].shape[0] if valid else 0
            episode_lens.append(_len)
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)
    
    def __len__(self):
        return len(self.file_paths)
    
    def get_dataset_name(self):
        return DATASET_NAME
    
    def get_item(self, index: int=None, state_only=False):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

        Returns:
           sample (dict): a dictionary containing the training sample.
        """
        while True:
            if index is None:
                file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)
            else:
                file_path = self.file_paths[index]
            
            # start_time = time.time()
            valid, sample = self.parse_hdf5_file(file_path) \
                if not state_only else self.parse_hdf5_file_state_only(file_path)
            # end_time = time.time()
            # print(f"Time taken to parse hdf5 file: {end_time - start_time:.2f} seconds, valid: {valid}")
            
            if valid:
                return sample
            else:
                index = np.random.randint(0, len(self.file_paths))
    
    def parse_hdf5_file_state_only(self, file_path):
        """Parse a hdf5 file to generate a training sample containing
            the complete trajectory.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid
            dict: a dictionary containing the training sample
        """
        with h5py.File(file_path, 'r') as f:
            # Get the qpos data from the file
            qpos = f['traj_0']['obs']['agent']['qpos'][:]
            actions_data = f['traj_0']['actions'][:]
            num_steps = qpos.shape[0]
            
            # [Optional] We drop too-short episode
            if num_steps < 128:
                return False, None

            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no qpos that exceeds the threshold.")
            
            qpos = qpos[first_idx:, :8]
            # Pad the actions array with zeros to make dimensions match observations
            actions = np.vstack([actions_data[first_idx:], np.zeros((1, actions_data.shape[1]), dtype=np.float32)])

            # Parse the state
            # Fill the state into the unified vector
            def fill_in_state(values):
                # Target indices corresponding to your state space
                # For single-arm robot with 7 joints + 1 gripper
                UNI_STATE_INDICES = [
                    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)
                ] + [
                    STATE_VEC_IDX_MAPPING["right_gripper_open"]
                ]
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec
            
            # Fill the action into the unified vector (for end-effector deltas)
            def fill_in_action(values):
                # For EEF control: [∆x, ∆y, ∆z, ∆φ, ∆θ, ∆ψ, g]
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                
                # Map EEF position deltas [∆x, ∆y, ∆z] to right EEF position indices
                uni_vec[..., STATE_VEC_IDX_MAPPING["right_eef_pos_x"]] = values[..., 0]  # ∆x
                uni_vec[..., STATE_VEC_IDX_MAPPING["right_eef_pos_y"]] = values[..., 1]  # ∆y
                uni_vec[..., STATE_VEC_IDX_MAPPING["right_eef_pos_z"]] = values[..., 2]  # ∆z
                
                # Convert Euler angles to 6D rotation representation (vectorized)
                # Extract Euler angles for all samples
                euler_angles = values[:, 3:6]
                
                # Convert to rotation matrices (batch operation)
                rot_matrices = R.from_euler('xyz', euler_angles).as_matrix()
                
                # Map the 6D rotation representation to the unified vector
                batch_size = values.shape[0]
                for i in range(batch_size):
                    # First column of rotation matrix
                    uni_vec[i, STATE_VEC_IDX_MAPPING["right_eef_angle_0"]] = rot_matrices[i, 0, 0]
                    uni_vec[i, STATE_VEC_IDX_MAPPING["right_eef_angle_1"]] = rot_matrices[i, 1, 0]
                    uni_vec[i, STATE_VEC_IDX_MAPPING["right_eef_angle_2"]] = rot_matrices[i, 2, 0]
                    # Second column of rotation matrix
                    uni_vec[i, STATE_VEC_IDX_MAPPING["right_eef_angle_3"]] = rot_matrices[i, 0, 1]
                    uni_vec[i, STATE_VEC_IDX_MAPPING["right_eef_angle_4"]] = rot_matrices[i, 1, 1]
                    uni_vec[i, STATE_VEC_IDX_MAPPING["right_eef_angle_5"]] = rot_matrices[i, 2, 1]
                
                # Map gripper [g] to right gripper open
                uni_vec[..., STATE_VEC_IDX_MAPPING["right_gripper_open"]] = values[..., 6]  # g
                
                return uni_vec
            
            # Create unified state/actions vector
            state = fill_in_state(qpos)  # Take only the first 8 dimensions, omitting the 9th
            actions = fill_in_action(actions)
            
            # Return the sample
            return True, {
                "state": state,
                "action": actions
            }
    
    def parse_hdf5_file(self, file_path):
        """Parse a hdf5 file to generate a training sample at
            a random timestep.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "meta": {
                        "dataset_name": str,    # the name of your dataset.
                        "#steps": int,          # the number of steps in the episode,
                                                # also the total timesteps.
                        "instruction": str      # the language instruction for this episode.
                    },                           
                    "step_id": int,             # the index of the sampled step,
                                                # also the timestep t.
                    "state": ndarray,           # state[t], (1, STATE_DIM).
                    "state_std": ndarray,       # std(state[:]), (STATE_DIM,).
                    "state_mean": ndarray,      # mean(state[:]), (STATE_DIM,).
                    "state_norm": ndarray,      # norm(state[:]), (STATE_DIM,).
                    "actions": ndarray,         # action[t:t+CHUNK_SIZE], (CHUNK_SIZE, STATE_DIM).
                    "state_indicator": ndarray, # indicates the validness of each dim, (STATE_DIM,).
                    "cam_high": ndarray,        # external camera image, (IMG_HISORY_SIZE, H, W, 3)
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_high_mask": ndarray,   # indicates the validness of each timestep, (IMG_HISORY_SIZE,) boolean array.
                                                # For the first IMAGE_HISTORY_SIZE-1 timesteps, the mask should be False.
                    "cam_left_wrist": ndarray,  # left wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_left_wrist_mask": ndarray,
                    "cam_right_wrist": ndarray, # right wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                                                # If only one wrist, make it right wrist, plz.
                    "cam_right_wrist_mask": ndarray
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, 'r') as f:
            # Get the qpos and action data from the file
            # print(f'h5 keys: {f.keys()}')
            qpos = f['traj_0']['obs']['agent']['qpos'][:]
            actions_data = f['traj_0']['actions'][:]
            num_steps = qpos.shape[0]
            
            # [Optional] We drop too-short episode
            if num_steps < 128:
                return False, None
            
            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no qpos that exceeds the threshold.")
            
            qpos = qpos[first_idx:, :8]
            # Pad the actions array with zeros to make dimensions match observations
            actions = np.vstack([actions_data[first_idx:], np.zeros((1, actions_data.shape[1]), dtype=np.float32)])

            # We randomly sample a timestep
            step_id = np.random.randint(first_idx, num_steps - 1)  # -1 because we need at least one future action
            
            # Load the instruction or precomputed embedding
            instruction = DATASET_INSTRUCTIONS[DATASET_NAME]
            # Assemble the meta
            meta = {
                "dataset_name": DATASET_NAME,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": instruction
            }
            
            # Parse the state and action
            state = qpos[step_id:step_id+1, :8]  # Take only the first 8 dimensions, omitting the 9th
            state_std = np.std(qpos[:, :8], axis=0)
            state_mean = np.mean(qpos[:, :8], axis=0)
            state_norm = np.sqrt(np.mean(qpos[:, :8]**2, axis=0))
            
            # Get actions for the next CHUNK_SIZE steps
            actions_slice = actions[step_id:step_id+self.CHUNK_SIZE]
            if actions_slice.shape[0] < self.CHUNK_SIZE:
                # Pad the actions using the last action
                actions_slice = np.concatenate([
                    actions_slice,
                    np.tile(actions_slice[-1:], (self.CHUNK_SIZE-actions_slice.shape[0], 1))
                ], axis=0)
            
            # Fill the state into the unified vector (for joint positions)
            def fill_in_state(values):
                # Target indices corresponding to your state space
                # For single-arm robot with 7 joints + 1 gripper
                UNI_STATE_INDICES = [
                    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)
                ] + [
                    STATE_VEC_IDX_MAPPING["right_gripper_open"]
                ]
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec
            
            # Fill the action into the unified vector (for end-effector deltas)
            def fill_in_action(values):
                # For EEF control: [∆x, ∆y, ∆z, ∆φ, ∆θ, ∆ψ, g]
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                
                # Map EEF position deltas [∆x, ∆y, ∆z] to right EEF position indices
                uni_vec[..., STATE_VEC_IDX_MAPPING["right_eef_pos_x"]] = values[..., 0]  # ∆x
                uni_vec[..., STATE_VEC_IDX_MAPPING["right_eef_pos_y"]] = values[..., 1]  # ∆y
                uni_vec[..., STATE_VEC_IDX_MAPPING["right_eef_pos_z"]] = values[..., 2]  # ∆z
                
                # Convert Euler angles to 6D rotation representation (vectorized)
                # Extract Euler angles for all samples
                euler_angles = values[:, 3:6]
                
                # Convert to rotation matrices (batch operation)
                rot_matrices = R.from_euler('xyz', euler_angles).as_matrix()
                
                # Map the 6D rotation representation to the unified vector
                batch_size = values.shape[0]
                for i in range(batch_size):
                    # First column of rotation matrix
                    uni_vec[i, STATE_VEC_IDX_MAPPING["right_eef_angle_0"]] = rot_matrices[i, 0, 0]
                    uni_vec[i, STATE_VEC_IDX_MAPPING["right_eef_angle_1"]] = rot_matrices[i, 1, 0]
                    uni_vec[i, STATE_VEC_IDX_MAPPING["right_eef_angle_2"]] = rot_matrices[i, 2, 0]
                    # Second column of rotation matrix
                    uni_vec[i, STATE_VEC_IDX_MAPPING["right_eef_angle_3"]] = rot_matrices[i, 0, 1]
                    uni_vec[i, STATE_VEC_IDX_MAPPING["right_eef_angle_4"]] = rot_matrices[i, 1, 1]
                    uni_vec[i, STATE_VEC_IDX_MAPPING["right_eef_angle_5"]] = rot_matrices[i, 2, 1]
                
                # Map gripper [g] to right gripper open
                uni_vec[..., STATE_VEC_IDX_MAPPING["right_gripper_open"]] = values[..., 6]  # g
                
                return uni_vec
            
            # Create unified state and action vectors
            state = fill_in_state(state)
            state_indicator = fill_in_state(np.ones_like(state_std))
            state_std = fill_in_state(state_std)
            state_mean = fill_in_state(state_mean)
            state_norm = fill_in_state(state_norm)
            actions = fill_in_action(actions_slice)  # Use fill_in_action for actions
            
            def parse_img(key):
                imgs = []
                for i in range(max(0, step_id - self.IMG_HISTORY_SIZE + 1), step_id + 1):
                    img = f['traj_0']['obs']['sensor_data'][key]['rgb'][i]
                    imgs.append(img)
                imgs = np.array(imgs)
                if imgs.shape[0] < self.IMG_HISTORY_SIZE:
                    imgs = np.concatenate([
                        np.tile(imgs[:1], (self.IMG_HISTORY_SIZE - imgs.shape[0], 1, 1, 1)), 
                        imgs
                        ], axis=0)
                return imgs
            
            cam_front = parse_img('base_front_camera')
            # For step_id = first_idx - 1, the valid_len should be one
            valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISTORY_SIZE)
            cam_front_mask = np.array(
                [False] * (self.IMG_HISTORY_SIZE - valid_len) + [True] * valid_len
            )
            cam_right_wrist = parse_img('hand_camera')
            cam_right_wrist_mask = cam_front_mask.copy()
            cam_left_wrist = np.zeros((self.IMG_HISTORY_SIZE, 0, 0, 0))
            cam_left_wrist_mask = np.zeros(self.IMG_HISTORY_SIZE, dtype=bool)
            
            # Return the sample
            return True, {
                "meta": meta,
                "step_id": step_id,
                "state": state,
                "state_std": state_std,
                "state_mean": state_mean,
                "state_norm": state_norm,
                "actions": actions,
                "state_indicator": state_indicator,
                "cam_high": cam_front,
                "cam_high_mask": cam_front_mask,
                "cam_left_wrist": cam_left_wrist,
                "cam_left_wrist_mask": cam_left_wrist_mask,
                "cam_right_wrist": cam_right_wrist,
                "cam_right_wrist_mask": cam_right_wrist_mask
            } 


if __name__ == "__main__":
    print("Initializing dataset...")
    start_time = time.time()
    ds = HDF5VLADataset()
    init_time = time.time() - start_time
    print(f"Dataset initialization took {init_time:.2f} seconds")
    print(f"Total episodes: {len(ds)}")
    
    # Test data preparation time for both methods
    print("\nTesting data preparation time...")
    
    # Test state_only mode
    print("\n1. Testing state_only mode:")
    num_samples = len(ds)  # Test with all episodes
    total_time = 0
    for i in range(num_samples):
        print(f"Processing episode {i+1}/{num_samples} (state_only=True)...")
        start_time = time.time()
        sample = ds.get_item(i, state_only=True)
        elapsed = time.time() - start_time
        total_time += elapsed
        print(f"  - Time taken: {elapsed:.2f} seconds")
    print(f"Average time for state_only mode: {total_time/num_samples:.2f} seconds")
    
    # Test full parsing mode
    print("\n2. Testing full parsing mode:")
    total_time = 0
    for i in range(num_samples):
        print(f"Processing episode {i+1}/{num_samples} (state_only=False)...")
        start_time = time.time()
        sample = ds.get_item(i, state_only=False)
        elapsed = time.time() - start_time
        total_time += elapsed
        print(f"  - Time taken: {elapsed:.2f} seconds")
    print(f"Total time for full parsing mode: {total_time:.2f} seconds")
    print(f"Average time for full parsing mode: {total_time/num_samples:.2f} seconds")
