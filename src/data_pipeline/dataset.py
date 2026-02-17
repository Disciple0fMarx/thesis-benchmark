import torch
from torch.utils.data import Dataset
import numpy as np


class SocialDataset(Dataset):
    def __init__(self, scenes, loader, generator, config):
        """
        Args:
            scenes: List of tuples [('eth', 'univ'), ...]
            loader: TrajectoryLoader instance
            generator: SocialSequenceGenerator instance
            config: Data configuration
        """
        self.obs_list = []
        self.pred_list = []
        self.obs_rel_list = []
        self.pred_rel_list = []
        
        for dataset_type, scene_name in scenes:
            # 1. Load data via our new loader
            scene_data = loader.load_scene(dataset_type, scene_name)
            
            # 2. Generate sequences (now returns 4 components)
            # Note: We pass the scene's specific FPS-based step to the generator
            generator.frame_step = 6 if dataset_type == 'eth' else 10
            out = generator.generate(scene_data['df'])
            
            # Unpack: each is a list of [Num_Peds, Len, 2]
            obs, pred, obs_rel, pred_rel = out
            
            self.obs_list.extend(obs)
            self.pred_list.extend(pred)
            self.obs_rel_list.extend(obs_rel)
            self.pred_rel_list.extend(pred_rel)

    def __len__(self):
        return len(self.obs_list)

    def __getitem__(self, idx):
        """
        Returns one 'social window' containing all peds present in that time.
        """
        # Convert to tensors
        obs = torch.from_numpy(self.obs_list[idx]).float()
        pred = torch.from_numpy(self.pred_list[idx]).float()
        obs_rel = torch.from_numpy(self.obs_rel_list[idx]).float()
        pred_rel = torch.from_numpy(self.pred_rel_list[idx]).float()

        # --- COORDINATE NORMALIZATION ---
        # Shift the absolute coordinates so the last observation point is (0,0)
        # This helps models generalize across different scenes/origins.
        origin = obs[:, -1:, :] # Last seen position [Num_Peds, 1, 2]
        obs_norm = obs - origin
        
        return {
            'obs': obs_norm,      # Normalized absolute (for input)
            'pred': pred,         # Raw absolute (for metric calculation)
            'obs_rel': obs_rel,   # Velocity/Offsets (for input)
            'pred_rel': pred_rel, # Velocity/Offsets (for training loss)
            'origin': origin      # To reconstruct absolute predictions later
        }


def social_collate(batch):
    """
    Collects scenes with different numbers of pedestrians into lists.
    """
    return {
        'obs': [item['obs'] for item in batch],
        'pred': [item['pred'] for item in batch],
        'obs_rel': [item['obs_rel'] for item in batch],
        'pred_rel': [item['pred_rel'] for item in batch],
        'origin': [item['origin'] for item in batch]
    }
