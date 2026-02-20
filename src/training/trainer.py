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


class GANTrainer:
    def __init__(self, generator, discriminator, config):
        self.gen = generator
        self.disc = discriminator
        self.config = config
        
        self.opt_g = torch.optim.Adam(self.gen.parameters(), lr=config['training']['lr'])
        self.opt_d = torch.optim.Adam(self.disc.parameters(), lr=config['training']['lr'])
        
        self.criterion = nn.BCELoss() # For GAN loss
        self.l2_loss = nn.MSELoss()  # For Variety (L2) loss

    def train_epoch(self, loader):
        self.gen.train()
        self.disc.train()
        
        for batch in loader:
            obs_rel = [r.to(self.device) for r in batch['obs_rel']]
            pred_rel = [p.to(self.device) for p in batch['pred_rel']]
            
            for obs_r, target_r in zip(obs_rel, pred_rel):
                # --- 1. Train Discriminator ---
                self.opt_d.zero_grad()
                
                # Real Trajectories
                real_traj = torch.cat([obs_r, target_r], dim=1)
                real_label = torch.ones(real_traj.size(0), 1).to(self.device)
                prob_real = self.disc(real_traj)
                loss_d_real = self.criterion(prob_real, real_label)
                
                # Fake Trajectories
                # We only take the 'best' of K samples to keep it stable
                fake_pred_list = self.gen([None], [obs_r], k=1) # Single sample for D update
                fake_traj = torch.cat([obs_r, fake_pred_list[0][0]], dim=1)
                fake_label = torch.zeros(fake_traj.size(0), 1).to(self.device)
                prob_fake = self.disc(fake_traj.detach()) # Detach so G isn't updated
                loss_d_fake = self.criterion(prob_fake, fake_label)
                
                loss_d = loss_d_real + loss_d_fake
                loss_d.backward()
                self.opt_d.step()

                # --- 2. Train Generator (Variety Loss + Adversarial) ---
                self.opt_g.zero_grad()
                
                # Variety Loss (Best of K)
                k_samples = self.gen([None], [obs_r], k=self.config['training']['k'])
                # (Logic to find best sample and calc L2 loss goes here, same as previous Trainer)
                loss_variety = self.calculate_variety_loss(k_samples[0], target_r)
                
                # Adversarial Loss (Trick D into thinking Fake is Real)
                prob_fake_for_g = self.disc(fake_traj)
                loss_adv = self.criterion(prob_fake_for_g, real_label)
                
                loss_g = loss_variety + 0.1 * loss_adv # Weighting the GAN influence
                loss_g.backward()
                self.opt_g.step()
