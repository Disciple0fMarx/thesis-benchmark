import torch
import torch.nn as nn


class SocialLSTM(nn.Module):
    def __init__(self, obs_len, pred_len, hidden_dim=64):
        super(SocialLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        
        # Encoder: Processes X, Y coordinates
        self.encoder = nn.LSTM(2, hidden_dim, batch_first=True)
        
        # Social Layer: Aggregates neighbor hidden states
        # In this variant, we use a simple MLP to process aggregated social context
        self.social_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Decoder: Predicts future coordinates
        self.decoder = nn.LSTM(hidden_dim + hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 2)

    def forward(self, obs_list, adj_list):
        pred_list = []
        
        for obs, adj in zip(obs_list, adj_list):
            num_peds = obs.shape[0]
            
            # 1. Encode each pedestrian
            _, (h_n, c_n) = self.encoder(obs) # h_n: [1, N, hidden_dim]
            h_n = h_n.squeeze(0) # [N, hidden_dim]
            
            # 2. Social Pooling
            # Use the adjacency matrix (binary) to mask who interacts
            # adj[-1] shape: [N, N]
            social_mask = (adj[-1] > 0).float()
            
            # Compute average hidden state of neighbors for each ped
            # [N, N] @ [N, hidden_dim] -> [N, hidden_dim]
            neighbor_sum = torch.matmul(social_mask, h_n)
            neighbor_count = social_mask.sum(dim=1, keepdim=True).clamp(min=1)
            social_context = self.social_mlp(neighbor_sum / neighbor_count)
            
            # 3. Concatenate Social Context with individual Hidden State
            decoder_input_h = torch.cat([h_n, social_context], dim=-1) # [N, hidden_dim*2]
            
            # 4. Decode (Recursive prediction)
            # For simplicity, we'll repeat the context and project
            # A more advanced version would use a true recursive loop
            curr_h = decoder_input_h.unsqueeze(1).repeat(1, self.pred_len, 1)
            
            # This is a simplified "sequence-to-sequence" projection
            # To match your GNN's flow:
            out, _ = self.decoder(curr_h)
            preds = self.output_layer(out) # [N, T_pred, 2]
            
            pred_list.append(preds)
            
        return pred_list
