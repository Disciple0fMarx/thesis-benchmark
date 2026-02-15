import os
import pandas as pd
import numpy as np
import torch


class TrajectoryLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_scene(self, dataset_type, scene_name):
        """
        Loads a specific scene (e.g., 'eth', 'hotel')
        Returns a DataFrame with [frame, id, x, y]
        """
        file_path = os.path.join(self.data_dir, dataset_type, scene_name, 'obsmat.txt')
        
        # Load columns 0, 1, 2, 4 (Frame, ID, X, Y)
        data = pd.read_csv(file_path, sep=r'\s+', header=None, usecols=[0, 1, 2, 4])
        data.columns = ['frame', 'id', 'x', 'y']
        return data


# Example Usage
loader = TrajectoryLoader('data/raw')
eth_univ = loader.load_scene('eth', 'univ')
print(eth_univ.head())
