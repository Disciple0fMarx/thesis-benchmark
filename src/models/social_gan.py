import torch
import torch.nn as nn


class PoolingModule(nn.Module):
    def __init__(self, hidden_dim, pooling_dim):
        super(PoolingModule, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, pooling_dim),
            nn.ReLU(),
            nn.Linear(pooling_dim, pooling_dim),
            nn.ReLU()
        )

    def forward(self, hidden_states):
        # hidden_states: [N, hidden_dim]
        # 1. Pass each ped's hidden state through MLP
        out = self.mlp(hidden_states) # [N, pooling_dim]
        
        # 2. Max-pool across the pedestrian dimension
        # This creates a single vector representing the "social vibe" of the crowd
        pooled = torch.max(out, dim=0, keepdim=True)[0] # [1, pooling_dim]
        return pooled


class SocialGANGenerator(nn.Module):
    def __init__(self, obs_len, pred_len, input_dim=2, hidden_dim=64, noise_dim=16):
        super(SocialGANGenerator, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim

        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Social Pooling
        self.pooler = PoolingModule(hidden_dim, hidden_dim)
        
        # Decoder input = hidden_state + pooled_context + noise
        decoder_input_dim = hidden_dim + hidden_dim + noise_dim  # 144
        self.decoder = nn.LSTM(decoder_input_dim, hidden_dim, batch_first=True)
        
        # NEW: Bridge the gap between the 144-dim context and the 64-dim LSTM state
        self.context_to_hidden = nn.Linear(decoder_input_dim, hidden_dim)
        self.context_to_cell = nn.Linear(decoder_input_dim, hidden_dim)

        self.hidden_to_rel = nn.Linear(hidden_dim, 2)

    def forward(self, obs_norm, obs_rel, k=20):
        pred_list = []
        
        for obs, rel in zip(obs_norm, obs_rel):
            num_peds = rel.size(0)
            
            # 1. Encode all pedestrians in the scene
            _, (h_n, c_n) = self.encoder(rel)
            h_n = h_n.squeeze(0) # [N, hidden_dim]
            
            # 2. Get the Social Context (Pooled vector)
            social_context = self.pooler(h_n) # [1, hidden_dim]
            social_context = social_context.repeat(num_peds, 1) # [N, hidden_dim]
            
            k_samples = []
            for _ in range(k):
                # 3. Inject Noise (z)
                z = torch.randn(num_peds, self.noise_dim).to(rel.device)

                # This 144-dim vector is our "Combined Context"
                combined_context = torch.cat([h_n, social_context, z], dim=1) # [N, 144]
                
                # NEW: Project it down to 64 to fit into h and c
                h_d = self.context_to_hidden(combined_context).unsqueeze(0) # [1, N, 64]
                c_d = self.context_to_cell(combined_context).unsqueeze(0)   # [1, N, 64]

                outputs = []
                for _ in range(self.pred_len):
                    # Use the 144-dim context as the input at each step
                    inp = combined_context.unsqueeze(1) # [N, 1, 144]
                    out, (h_d, c_d) = self.decoder(inp, (h_d, c_d))
                    
                    prediction = self.hidden_to_rel(out.squeeze(1))
                    outputs.append(prediction)
                
                k_samples.append(torch.stack(outputs, dim=1)) # [N, T_pred, 2]
            
            pred_list.append(torch.stack(k_samples)) # [K, N, T_pred, 2]
            
        return pred_list


class SocialGANDiscriminator(nn.Module):
    def __init__(self, obs_len, pred_len, input_dim=2, hidden_dim=64):
        super(SocialGANDiscriminator, self).__init__()
        
        # Total length of the trajectory the D sees: 8 + 12 = 20
        self.full_len = obs_len + pred_len
        
        # Encoder for the trajectory
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid() # Output probability [0, 1]
        )

    def forward(self, traj_rel):
        # traj_rel: [N, total_len, 2]
        _, (h_n, _) = self.encoder(traj_rel)
        
        # Use the final hidden state to classify the motion
        prob = self.classifier(h_n.squeeze(0)) # [N, 1]
        return prob
