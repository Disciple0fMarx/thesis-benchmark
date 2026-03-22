import os
import torch
import torch.optim as optim
import torch.nn as nn
 
 
class Trainer:
    """
    General-purpose trainer for deterministic and stochastic trajectory models.
 
    Model output contract
    ---------------------
    The model's forward(obs, obs_rel, k) must return a FloatTensor of shape
    [K, N, pred_len, 2] in the NORMALISED absolute coordinate frame.
 
    Training target
    ---------------
    We train against 'pred_rel' (displacement targets) by default via the
    variety loss, but the loss is computed in whichever space the model
    outputs.  If your model outputs absolute normalised positions, override
    this trainer and compute the loss against 'obs' + cumulative displacements.
    For all baseline models (Social-LSTM, STGCNN) that output displacements,
    this trainer is correct as-is.
 
    Parameters
    ----------
    model  : nn.Module
    config : dict
        Expected keys under 'training': lr, k_samples, clip.
    """
 
    def __init__(self, model, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = model.to(self.device)
 
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['lr'],
        )
        self.best_ade = float('inf')
 
        # reduction='none' so we can implement Best-of-K manually.
        self.mse = nn.MSELoss(reduction='none')
 
    def variety_loss(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Variety Loss (Best-of-K regression loss).
 
        For each pedestrian, only the sample closest to the ground truth
        contributes to the gradient.  This encourages the model to produce
        at least one accurate prediction rather than K averaged ones.
 
        Parameters
        ----------
        preds  : FloatTensor [K, N, T, 2]
        target : FloatTensor    [N, T, 2]
 
        Returns
        -------
        Scalar FloatTensor
        """
        # Sum MSE over the (x, y) and time dimensions → [K, N]
        error = self.mse(
            preds, target.unsqueeze(0).expand_as(preds)
        ).sum(dim=(2, 3))
 
        # Per-pedestrian minimum error across K samples → [N]
        best_error, _ = error.min(dim=0)
 
        return best_error.mean()
 
    def train_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0
 
        k_samples = self.config['training'].get('k_samples', 20)
        clip      = self.config['training'].get('clip', 1.0)
 
        for batch in loader:
            obs_list      = [o.to(self.device)  for o in batch['obs']]
            obs_rel_list  = [r.to(self.device)  for r in batch['obs_rel']]
            target_list   = [t.to(self.device)  for t in batch['pred_rel']]
 
            self.optimizer.zero_grad()
 
            loss = torch.tensor(0.0, device=self.device)
 
            for obs, obs_rel, target_rel in zip(obs_list, obs_rel_list, target_list):
                # preds: [K, N, T, 2]
                preds = self.model(obs, obs_rel, k=k_samples)
                loss  = loss + self.variety_loss(preds, target_rel)
 
            # Average over scenes in the batch so the loss scale is
            # independent of batch size.
            loss = loss / len(obs_list)
            loss.backward()
 
            nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()
 
            total_loss += loss.item()
 
        return total_loss / len(loader)
 
    def save_model(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {'state_dict': self.model.state_dict(), 'config': self.config},
            path,
        )
 
 
class GANTrainer:
    """
    Trainer for GAN-based trajectory models (e.g. Social-GAN).
 
    Model output contract
    ---------------------
    The generator's forward(obs, obs_rel, k) returns [K, N, pred_len, 2] in
    the normalised absolute coordinate frame, matching the evaluator contract.
 
    The discriminator receives concatenated [obs | pred] trajectories and
    returns a per-pedestrian real/fake probability [N, 1].
 
    Parameters
    ----------
    generator     : nn.Module
    discriminator : nn.Module
    config        : dict
        Expected keys under 'training': lr, k_samples, clip, adv_weight.
    """
 
    def __init__(self, generator, discriminator, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
        self.gen  = generator.to(self.device)
        self.disc = discriminator.to(self.device)
 
        lr = config['training']['lr']
        self.opt_g = optim.Adam(self.gen.parameters(),  lr=lr)
        self.opt_d = optim.Adam(self.disc.parameters(), lr=lr)
 
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss(reduction='none')
 
    def variety_loss(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Best-of-K MSE loss. See Trainer.variety_loss for full docstring."""
        error = self.mse(
            preds, target.unsqueeze(0).expand_as(preds)
        ).sum(dim=(2, 3))           # [K, N]
        best_error, _ = error.min(dim=0)   # [N]
        return best_error.mean()
 
    def train_epoch(self, loader) -> dict:
        self.gen.train()
        self.disc.train()
 
        k_samples  = self.config['training'].get('k_samples', 20)
        clip       = self.config['training'].get('clip', 1.0)
        adv_weight = self.config['training'].get('adv_weight', 0.1)
 
        total_loss_g = 0.0
        total_loss_d = 0.0
        n_scenes     = 0
 
        for batch in loader:
            obs_list     = [o.to(self.device) for o in batch['obs']]
            obs_rel_list = [r.to(self.device) for r in batch['obs_rel']]
            pred_rel_list = [p.to(self.device) for p in batch['pred_rel']]
 
            for obs, obs_rel, target_rel in zip(
                obs_list, obs_rel_list, pred_rel_list
            ):
                n_peds = obs.shape[0]
 
                real_label = torch.ones(n_peds,  1, device=self.device)
                fake_label = torch.zeros(n_peds, 1, device=self.device)
 
                # ---- 1. Train Discriminator --------------------------------
                self.opt_d.zero_grad()
 
                # Real: concatenate observed and ground-truth future [N, T_obs+T_pred, 2]
                real_traj  = torch.cat([obs, target_rel], dim=1)
                loss_d_real = self.bce(self.disc(real_traj), real_label)
 
                # Fake: generate a single sample for the D update.
                # Detach so gradients do not flow into G here.
                with torch.no_grad():
                    fake_pred = self.gen(obs, obs_rel, k=1)[0]   # [N, T, 2]
                fake_traj   = torch.cat([obs, fake_pred], dim=1)
                loss_d_fake = self.bce(self.disc(fake_traj), fake_label)
 
                loss_d = loss_d_real + loss_d_fake
                loss_d.backward()
                nn.utils.clip_grad_norm_(self.disc.parameters(), clip)
                self.opt_d.step()
 
                # ---- 2. Train Generator ------------------------------------
                self.opt_g.zero_grad()
 
                # Variety loss: Best-of-K over fresh samples from the current G.
                preds_k = self.gen(obs, obs_rel, k=k_samples)   # [K, N, T, 2]
                loss_variety = self.variety_loss(preds_k, target_rel)
 
                # Adversarial loss: fool D with a fresh single sample.
                # Must NOT reuse the detached fake_traj from the D step —
                # that was generated before the D update and has no gradient
                # path through G.
                fresh_pred = self.gen(obs, obs_rel, k=1)[0]     # [N, T, 2]
                fresh_traj = torch.cat([obs, fresh_pred], dim=1)
                loss_adv   = self.bce(self.disc(fresh_traj), real_label)
 
                loss_g = loss_variety + adv_weight * loss_adv
                loss_g.backward()
                nn.utils.clip_grad_norm_(self.gen.parameters(), clip)
                self.opt_g.step()
 
                total_loss_g += loss_g.item()
                total_loss_d += loss_d.item()
                n_scenes     += 1
 
        return {
            'loss_g': total_loss_g / n_scenes,
            'loss_d': total_loss_d / n_scenes,
        }
 
    def save_model(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                'generator_state_dict':     self.gen.state_dict(),
                'discriminator_state_dict': self.disc.state_dict(),
                'config':                   self.config,
            },
            path,
        )
