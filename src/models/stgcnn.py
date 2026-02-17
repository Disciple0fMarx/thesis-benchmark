import torch
import torch.nn as nn


class SocialGCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SocialGCN, self).__init__()
        self.W = nn.Linear(in_channels, out_channels)

        # NEW: Projection layer to match dimensions for the residual connection
        self.residual_project = nn.Linear(in_channels, out_channels)

    def forward(self, v, a):
        # 1. Add Self-loops (Identity matrix) so a ped always listens to themselves
        I = torch.eye(a.size(0)).to(a.device)
        a_hat = a + I 
        
        # 2. Normalize the Adjacency (Standard GCN trick)
        # This prevents the signal from "exploding" in crowded scenes
        degree = torch.sum(a_hat, dim=1)
        d_inv = torch.diag(torch.pow(degree, -0.5))
        a_norm = d_inv @ a_hat @ d_inv
        
        v_perm = v.permute(1, 0, 2) 
        out = torch.matmul(a_norm, v_perm)
        out = out.permute(1, 0, 2).contiguous()
        out = self.W(out)

        # 3. Residual branch (Project v from 2 channels to 64)
        res = self.residual_project(v)
        
        return out + res


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

    def forward(self, obs_norm, obs_rel, k=20):
        pred_list = []
        for obs, rel in zip(obs_norm, obs_rel):
            # 1. Use relative motion as features
            # Standard STGCNN uses a graph over all peds in the scene
            # Here we simplify: use the last frame's spatial distance for Adjacency
            dist_mat = torch.cdist(obs[:, -1, :], obs[:, -1, :])
            adj = (dist_mat < 2.0).float().to(obs.device) # 2-meter social threshold
            
            v = torch.relu(self.gcn(rel, adj))
            v = v.permute(0, 2, 1)
            v = torch.relu(self.temporal_cnn(v))
            v = self.time_extrapolator(v).permute(0, 2, 1)
            
            # 2. To get K samples, we can add a small stochastic noise to the bottleneck
            # or use a dropout layer during inference.
            base_rel_pred = self.fc(v) # [N, T_pred, 2]
            
            k_samples = []
            for _ in range(k):
                noise = torch.randn_like(base_rel_pred) * 0.05 # 5cm variance noise
                k_samples.append(base_rel_pred + noise)
            
            pred_list.append(torch.stack(k_samples))
            
        return pred_list
