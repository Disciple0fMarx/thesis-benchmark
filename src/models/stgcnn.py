import torch
import torch.nn as nn


class SocialGCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SocialGCN, self).__init__()
        self.W = nn.Linear(in_channels, out_channels)

    def forward(self, v, a):
        """
        Args:
            v: Node features [Num_Peds, T, In_Channels]
            a: Adjacency Matrix [Num_Peds, Num_Peds]
        """
        # 1. Message Passing: A * V
        # This aggregates features from neighbors
        out = torch.matmul(a, v)
        
        # 2. Linear Transformation
        out = self.W(out)
        return out


class STGCNN(nn.Module):
    def __init__(self, obs_len, pred_len, input_dim=2):
        super(STGCNN, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        
        # Simple STGCNN Backbone
        self.gcn = SocialGCN(input_dim, 64)
        self.temporal_cnn = nn.Conv1d(obs_len, pred_len, kernel_size=1)
        self.fc = nn.Linear(64, 2) # Back to (x, y)

    def forward(self, obs_list, adj_list):
        """
        Processes a list of scene tensors.
        """
        pred_list = []
        for obs, adj in zip(obs_list, adj_list):
            # obs: [N, T_obs, 2], adj: [T_obs, N, N]
            
            # Use the graph at the last observed timestep
            # Simplified version for the first draft
            a = adj[-1] 
            
            # GCN Layer
            x = torch.relu(self.gcn(obs, a)) # [N, T_obs, 64]
            
            # Temporal Shift (Obs -> Pred)
            x = self.temporal_cnn(x) # [N, T_pred, 64]
            
            # Output coordinates
            out = self.fc(x) # [N, T_pred, 2]
            pred_list.append(out)
            
        return pred_list
