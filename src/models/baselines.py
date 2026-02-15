import torch
import torch.nn as nn


class ConstantVelocityModel(nn.Module):
    """
    A non-trainable baseline that projects the last observed velocity forward.
    """
    def __init__(self, pred_len):
        super().__init__()
        self.pred_len = pred_len

    def forward(self, obs_list, adj_list=None):
        """
        Args:
            obs_list: List of tensors [Num_Peds, T_obs, 2]
            adj_list: Ignored by this model (it's not socially aware).
        Returns:
            pred_list: List of tensors [Num_Peds, T_pred, 2]
        """
        pred_list = []
        
        # Iterate through each scene in the batch
        for obs in obs_list:
            # obs shape: [N, T_obs, 2]
            
            # 1. Calculate last velocity vector
            # v_last = pos[t] - pos[t-1]
            last_pos = obs[:, -1, :]
            second_to_last_pos = obs[:, -2, :]
            velocity = last_pos - second_to_last_pos # [N, 2]
            
            # 2. Project forward
            scene_preds = []
            for t in range(1, self.pred_len + 1):
                # p[t+k] = p[t] + k * v
                future_pos = last_pos + velocity * t
                scene_preds.append(future_pos)
                
            # Stack into [N, T_pred, 2]
            pred_list.append(torch.stack(scene_preds, dim=1))
            
        return pred_list
