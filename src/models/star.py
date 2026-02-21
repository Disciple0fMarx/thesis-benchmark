import torch
import torch.nn as nn


class STAR(nn.Module):
    def __init__(self, obs_len=8, pred_len=12, d_model=128, nhead=8, noise_dim=16):
        super(STAR, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.noise_dim = noise_dim

        # 1. Embedding & Positional Encoding
        self.input_fc = nn.Linear(2, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, obs_len, d_model))

        # 2. Transformer Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, batch_first=True
        )
        self.temporal_tf = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.spatial_tf = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 3. Output Head (Handles noise for Best-of-K)
        self.output_fc = nn.Linear(d_model + noise_dim, pred_len * 2)

    def forward(self, obs_norm, obs_rel, k=1, **kwargs):
        # --- A. Flatten the Batch of Lists ---
        # Capture the sizes so we can split them later
        scene_sizes = [scene.shape[0] for scene in obs_rel]
        
        # Flatten all pedestrians from all scenes into one dimension
        # Shape: [Total_N, 8, 2]
        obs_flat = torch.cat(obs_rel, dim=0)
        Total_N = obs_flat.shape[0]

        # --- B. Transformer Encoding ---
        # 1. Temporal: Each person's history
        x = self.input_fc(obs_flat) + self.pos_encoder
        x = self.temporal_tf(x) # [Total_N, 8, d_model]
        
        # 2. Spatial: Interaction between all people in the batch
        # (Standard for STAR: treating the batch as a global interaction space)
        x = x.permute(1, 0, 2) # [8, Total_N, d_model]
        x = self.spatial_tf(x)
        h = x[-1, :, :] # [Total_N, d_model]

        # --- C. Stochastic Generation (Best-of-K) ---
        # Generate K samples if Trainer asks for them
        if k > 1:
            all_samples = []
            for _ in range(k):
                z = torch.randn(Total_N, self.noise_dim).to(obs_flat.device)
                h_combined = torch.cat([h, z], dim=1)
                out = self.output_fc(h_combined)
                # Reshape to [1, Total_N, 12, 2]
                all_samples.append(out.view(1, Total_N, self.pred_len, 2))
            
            # Combined shape: [K, Total_N, 12, 2]
            preds_flat = torch.cat(all_samples, dim=0)
            
            # --- D. Split back into List for Trainer ---
            # Split along the 'N' dimension (dim 1)
            preds_list = torch.split(preds_flat, scene_sizes, dim=1)
        else:
            # Deterministic path (K=1)
            z = torch.zeros(Total_N, self.noise_dim).to(obs_flat.device)
            h_combined = torch.cat([h, z], dim=1)
            out = self.output_fc(h_combined)
            preds_flat = out.view(Total_N, self.pred_len, 2)
            
            # Split along the 'N' dimension (dim 0)
            preds_list = torch.split(preds_flat, scene_sizes, dim=0)

        return list(preds_list)
