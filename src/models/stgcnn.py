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
        # v is [N, T, C]. We want to multiply A [N, N] by V [N, T, C]
        # We need to treat T as a batch dimension for matmul.
        # Permute v to [T, N, C] so matmul can broadcast across T
        v = v.permute(1, 0, 2) 
        
        # Now: [T, N, N] matmul [T, N, C] -> [T, N, C]
        # (a is automatically broadcasted across T)
        out = torch.matmul(a, v)
        
        # Permute back to [N, T, C]
        out = out.permute(1, 0, 2)
        
        # Apply the linear weight
        out = self.W(out)
        return out


class STGCNN(nn.Module):
    def __init__(self, obs_len, pred_len, input_dim=2):
        super(STGCNN, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        
        self.gcn = SocialGCN(input_dim, 64)
        
        # Change this: Input channels = 64, Output channels = 64
        # We will use a kernel or interpolation to change the time dimension later,
        # OR we keep it as is but change how we apply it.
        # Standard STGCNN uses a Conv layer that preserves channels but changes Time.
        self.temporal_cnn = nn.Conv1d(64, 64, kernel_size=3, padding=1) 
        
        # We need a way to resize Time from 8 to 12. 
        # A simple linear layer or interpolation works best for a first draft.
        self.time_extrapolator = nn.Linear(obs_len, pred_len)
        
        self.fc = nn.Linear(64, 2)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)

    def forward(self, obs_list, adj_list):
        """
        Processes a list of scene tensors.
        """
        pred_list = []
        for obs, adj in zip(obs_list, adj_list):
            # 1. Convert absolute to relative: obs[:, 1:] - obs[:, :-1]
            # This makes the first frame the "origin" for every pedestrian
            last_obs_pos = obs[:, -1:, :] 
            rel_obs = torch.zeros_like(obs)
            rel_obs[:, 1:, :] = obs[:, 1:, :] - obs[:, :-1, :]
            
            # 2. Process through GCN (using relative motion)
            a = adj[-1] 
            x = torch.relu(self.gcn(rel_obs, a)) 
            x = x.permute(0, 2, 1) 
            x = torch.relu(self.temporal_cnn(x)) 
            x = self.time_extrapolator(x) 
            x = x.permute(0, 2, 1)
            
            # 3. Model predicts relative displacements for the future
            rel_pred = self.fc(x) 
            
            # 4. Convert back to absolute for evaluation
            # Current_pos = Last_Obs + cumulative sum of predicted deltas
            abs_pred = last_obs_pos + torch.cumsum(rel_pred, dim=1)
            pred_list.append(abs_pred)
            
        return pred_list
