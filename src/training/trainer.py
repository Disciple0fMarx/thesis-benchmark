import torch
import torch.optim as optim
import torch.nn as nn
from src.evaluation.metrics import calculate_ade
import os


class Trainer:
    def __init__(self, model, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        # Loss and Optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['training']['lr'])
        
        self.best_ade = float('inf')

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        
        for batch in loader:
            obs, target, adj = batch
            # Move to device
            obs = [o.to(self.device) for o in obs]
            target = [t.to(self.device) for t in target]
            adj = [a.to(self.device) for a in adj]
            
            self.optimizer.zero_grad()
            
            # Forward pass
            preds = self.model(obs, adj)
            
            # Compute loss
            loss = 0
            for p, t in zip(preds, target):
                loss += self.criterion(p, t)
            
            # Normalize by number of scenes in batch
            loss = loss / len(obs)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(loader)

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
