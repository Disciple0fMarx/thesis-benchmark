import os
import pandas as pd
import numpy as np


class TrajectoryLoader:
    def __init__(self, data_dir):
        """
        Standardized loader for ETH/UCY datasets.
        :param data_dir: Path to 'data/raw'
        """
        self.data_dir = data_dir
        # Standard definitions for the 5 scenes
        self.scenes = {
            'eth': ['hotel', 'univ'],
            'ucy': ['univ', 'zara1', 'zara2']
        }

    def load_scene(self, dataset_type, scene_name):
        """
        Loads a specific scene and its metadata.
        """
        scene_path = os.path.join(self.data_dir, dataset_type, scene_name)
        obsmat_path = os.path.join(scene_path, 'obsmat.txt')
        
        if not os.path.exists(obsmat_path):
            raise FileNotFoundError(f"Could not find {obsmat_path}")

        # 1. Load Trajectories
        # [Frame, ID, X, Z, Y, vX, vZ, vY] -> We take [0, 1, 2, 4]
        # Using float32 for deep learning compatibility later
        data = pd.read_csv(obsmat_path, sep=r'\s+', header=None, usecols=[0, 1, 2, 4])
        data.columns = ['frame', 'id', 'x', 'y']
        data = data.astype({'frame': int, 'id': int, 'x': np.float32, 'y': np.float32})

        # 2. Load Homography Matrix (if needed for visualization later)
        h_matrix = self._load_h_matrix(scene_path)

        # 3. Load Social Groups
        groups = self._load_groups(scene_path)

        return {
            'df': data,
            'H': h_matrix,
            'groups': groups,
            'fps': 15 if dataset_type == 'eth' else 25
        }

    def _load_h_matrix(self, scene_path):
        h_path = os.path.join(scene_path, 'H.txt')
        if os.path.exists(h_path):
            return np.loadtxt(h_path)
        return None

    def _load_groups(self, scene_path):
        """Returns a list of lists, where each sublist is a group of IDs"""
        group_path = os.path.join(scene_path, 'groups.txt')
        groups = []
        if os.path.exists(group_path):
            with open(group_path, 'r') as f:
                for line in f:
                    ids = [int(i) for i in line.split() if i.isdigit()]
                    if ids:
                        groups.append(ids)
        return groups

    def load_all_except(self, test_scene_name):
        """
        Helper for Leave-One-Out training.
        e.g., loader.load_all_except('hotel')
        """
        combined_df = []
        for d_type, scenes in self.scenes.items():
            for s_name in scenes:
                if s_name == test_scene_name:
                    continue
                
                scene_data = self.load_scene(d_type, s_name)
                # We add a prefix to IDs to avoid collisions between different scenes
                df = scene_data['df'].copy()
                df['id'] = df['id'] + (hash(s_name) % 100000) 
                combined_df.append(df)
        
        return pd.concat(combined_df, ignore_index=True)
