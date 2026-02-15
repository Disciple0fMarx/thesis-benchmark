import torch
from torch.utils.data import Dataset
from src.utils.graph_tools import get_spatial_edges


class SocialDataset(Dataset):
    def __init__(self, scenes, loader, generator, config):
        """
        Args:
            scenes: List of tuples [('eth', 'univ'), ('ucy', 'zara1'), ...]
            loader: TrajectoryLoader instance
            generator: SocialSequenceGenerator instance
            config: Data configuration dictionary
        """
        self.all_sequences = []
        
        for dataset_type, scene_name in scenes:
            df = loader.load_scene(dataset_type, scene_name)
            scene_sequences = generator.generate(df)
            self.all_sequences.extend(scene_sequences)
            
        self.t_obs = config['data']['t_obs']

    def __len__(self):
        return len(self.all_sequences)

    def __getitem__(self, idx):
        # scene_data shape: [Num_Peds, 20, 2]
        scene_data = torch.from_numpy(self.all_sequences[idx]).float()
        
        # Split into Observation and Prediction
        obs = scene_data[:, :self.t_obs, :]
        pred = scene_data[:, self.t_obs:, :]
        
        # Generate Social Graphs for the observation period
        # Shape: [T_obs, Num_Peds, Num_Peds]
        adj = get_spatial_edges(obs)
        
        return obs, pred, adj


def social_collate(batch):
    """
    Since each scene has a different number of pedestrians, 
    we pack them into lists.
    """
    obs_list = [item[0] for item in batch]
    pred_list = [item[1] for item in batch]
    adj_list = [item[2] for item in batch]
    
    return obs_list, pred_list, adj_list
