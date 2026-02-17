import torch
import torch.optim as optim
import torch.nn as nn
import os


class Trainer:
    def __init__(self, model, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['training']['lr'])
        self.best_ade = float('inf')
        
        # We define MSE with 'none' reduction to handle Best-of-K manually
        self.mse = nn.MSELoss(reduction='none')

    def variety_loss(self, preds, target):
        """
        Implements Variety Loss (Best-of-K loss).
        Args:
            preds:  [K, N, T, 2]
            target: [N, T, 2]
        """
        # Calculate MSE for every sample: [K, N, T, 2]
        # We sum the (x,y) and time dimensions to get error per pedestrian per sample
        error = torch.sum(self.mse(preds, target.expand_as(preds)), dim=(2, 3)) # [K, N]
        
        # For each pedestrian, find the index of the best sample (minimum error)
        best_error, _ = torch.min(error, dim=0) # [N]
        
        # Return the mean of the best errors across all pedestrians
        return torch.mean(best_error)

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        
        for batch in loader:
            # Unpack our rich batch structure
            obs_norm = [o.to(self.device) for o in batch['obs']]
            target_rel = [tr.to(self.device) for tr in batch['pred_rel']]
            obs_rel = [or_to.to(self.device) for or_to in batch['obs_rel']]
            
            self.optimizer.zero_grad()
            
            # Forward pass: request K samples even during training for Variety Loss
            # k=20 is standard, but some use k=1 during early warm-up
            k_samples = self.config['training'].get('k_samples', 20)
            
            # preds_rel_list is a list of [K, N, T, 2]
            preds_rel_list = self.model(obs_norm, obs_rel, k=k_samples)
            
            loss = 0
            for p_rel, t_rel in zip(preds_rel_list, target_rel):
                # We train on RELATIVE offsets
                loss += self.variety_loss(p_rel, t_rel)
            
            loss = loss / len(obs_norm)
            loss.backward()
            
            # Gradient clipping is highly recommended for LSTMs and GCNs
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training'].get('clip', 1.0))
            
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(loader)

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'state_dict': self.model.state_dict(), 'config': self.config}, path)
