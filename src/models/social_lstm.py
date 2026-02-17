import torch
import torch.nn as nn


class SocialLSTM(nn.Module):
    def __init__(self, obs_len, pred_len, hidden_dim=64, noise_dim=16):
        super(SocialLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        self.noise_dim = noise_dim
        
        self.encoder = nn.LSTM(2, hidden_dim, batch_first=True)
        self.social_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        
        # Decoder input: [hidden + social + noise]
        self.decoder = nn.LSTM(hidden_dim + hidden_dim + noise_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 2)

    def forward(self, obs_norm, obs_rel, k=1):
        """
        Args:
            obs_norm: List of [N, T_obs, 2] (Normalized absolute)
            obs_rel:  List of [N, T_obs, 2] (Relative displacements)
            k: Number of samples to generate
        """
        pred_list = []
        for obs in obs_rel: # We use rel for motion encoding
            num_peds = obs.shape[0]
            _, (h_n, _) = self.encoder(obs)
            h_n = h_n.squeeze(0) # [N, hidden_dim]
            
            # Simple Pooling (Global average for this draft)
            social_context = self.social_mlp(h_n.mean(dim=0, keepdim=True).repeat(num_peds, 1))
            
            # Base hidden state
            base_h = torch.cat([h_n, social_context], dim=-1) # [N, hidden*2]
            
            # Sample K times
            k_preds = []
            for _ in range(k):
                z = torch.randn(num_peds, self.noise_dim).to(obs.device)
                dec_input_h = torch.cat([base_h, z], dim=-1)
                
                # Recursive Decoding (more stable than sequence repeat)
                curr_h = dec_input_h.unsqueeze(1).repeat(1, self.pred_len, 1)
                out, _ = self.decoder(curr_h)
                k_preds.append(self.output_layer(out)) # [N, T_pred, 2]
            
            # Stack to [K, N, T_pred, 2]
            pred_list.append(torch.stack(k_preds))
            
        return pred_list
