"""
MoFlow teacher and IMLE student trainers.

Normalisation contract
----------------------
All normalisation is handled by the shared TrajectoryNormaliser in the
data pipeline — NOT inside these trainers.  The trainers:

  1. Receive a pre-fitted TrajectoryNormaliser at construction time.
  2. Call normaliser.fit(train_dataset) before training (MoFlowTrainer only).
  3. Call train_dataset.set_normaliser(normaliser) and
     test_dataset.set_normaliser(normaliser) before creating DataLoaders.
  4. Read 'pred_norm' directly from batch items — already in [-1, 1].
  5. Call normaliser.inverse_transform() only for denormalisation at eval
     time, before calling SocialDataset.reconstruct_abs().
  6. Save and load the normaliser alongside the model checkpoint so that
     fold-specific bounds are always paired with the correct model weights.

Leave-one-out usage (one fold)
------------------------------
  train_scenes  = loader.load_train_scenes(test_scene='hotel')
  test_scene    = loader.load_scene('hotel')

  train_dataset = SocialDataset(train_scenes, generator)
  test_dataset  = SocialDataset(test_scene,   generator)

  normaliser    = TrajectoryNormaliser(mode='minmax')
  normaliser.fit(train_dataset)           # training fold ONLY

  train_dataset.set_normaliser(normaliser)
  test_dataset.set_normaliser(normaliser) # same bounds on test set

  teacher_trainer = MoFlowTrainer(flow_matcher, normaliser, config)
  teacher_trainer.train(train_loader, val_loader)
  teacher_trainer.save_teacher_samples(train_loader_no_shuffle, path)
  
  imle_dataset = IMLEDataset(train_dataset, path)
  imle_trainer = IMLETrainer(imle, normaliser, config)
  imle_trainer.train(imle_loader, val_loader)
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.moflow import FlowMatcher, IMLE
from src.data_pipeline.dataset import SocialDataset
from src.data_pipeline.normaliser import TrajectoryNormaliser
from src.evaluation.metrics import calculate_best_of_k
from src.models.moflow_adapter import _build_context


# ---------------------------------------------------------------------------
# MoFlow Teacher Trainer
# ---------------------------------------------------------------------------

class MoFlowTrainer:
    """
    Trainer for the MoFlow teacher FlowMatcher model.

    Parameters
    ----------
    flow_matcher : FlowMatcher
    normaliser   : TrajectoryNormaliser
        Must be fitted on the training fold before constructing this trainer.
        Both train_dataset and test_dataset must have set_normaliser() called
        with this same instance before DataLoaders are created.
    config : dict
        Sub-dicts:
            'training'   : lr, epochs, clip, weight_decay
            'evaluation' : k_samples (default 20)
        Keys:
            'checkpoint_dir' : str
            'sample_dir'     : str  (for save_teacher_samples)

    Typical leave-one-out usage
    ---------------------------
        normaliser = TrajectoryNormaliser(mode='minmax')
        normaliser.fit(train_dataset)

        train_dataset.set_normaliser(normaliser)
        test_dataset.set_normaliser(normaliser)   # training bounds on test

        train_loader = DataLoader(train_dataset, batch_size=32,
                                  shuffle=True,  collate_fn=social_collate)
        val_loader   = DataLoader(test_dataset,  batch_size=32,
                                  shuffle=False, collate_fn=social_collate)

        trainer = MoFlowTrainer(flow_matcher, normaliser, config)
        trainer.train(train_loader, val_loader)

        # After training, generate teacher samples for IMLE:
        ordered_loader = DataLoader(train_dataset, batch_size=32,
                                    shuffle=False, collate_fn=social_collate)
        trainer.save_teacher_samples(ordered_loader, 'data/eth_ucy/imle/eth_samples.pkl')
    """

    def __init__(
        self,
        flow_matcher: FlowMatcher,
        normaliser:   TrajectoryNormaliser,
        config:       dict,
    ):
        if not normaliser.is_fitted:
            raise RuntimeError(
                "normaliser must be fitted before constructing MoFlowTrainer. "
                "Call normaliser.fit(train_dataset) first."
            )

        self.fm         = flow_matcher
        self.normaliser = normaliser
        self.config     = config
        self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fm.to(self.device)

        train_cfg      = config['training']
        self.optimizer = torch.optim.AdamW(
            self.fm.parameters(),
            lr=train_cfg['lr'],
            weight_decay=train_cfg.get('weight_decay', 1e-4),
        )
        self.epochs    = train_cfg['epochs']
        self.clip      = train_cfg.get('clip', 1.0)
        self.k_eval    = config.get('evaluation', {}).get('k_samples', 20)
        self.best_ade  = float('inf')
        self.lr        = train_cfg['lr']

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Full training loop with per-epoch validation and checkpointing."""
        ckpt_dir = self.config.get('checkpoint_dir', 'results/checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)

        total_steps  = self.epochs * len(train_loader)
        warmup_steps = max(1, int(total_steps * 0.05))
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=1e-6 / self.lr,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=total_steps - warmup_steps,
                    eta_min=1e-6,
                ),
            ],
            milestones=[warmup_steps],
        )

        for epoch in range(self.epochs):
            loss    = self._train_epoch(train_loader, scheduler)
            metrics = self.evaluate(val_loader)
            print(
                f"[Teacher] Epoch {epoch+1:>3}/{self.epochs}  "
                f"loss={loss:.4f}  "
                f"ADE={metrics['ADE']:.4f}  FDE={metrics['FDE']:.4f}"
            )
            if metrics['ADE'] < self.best_ade:
                self.best_ade = metrics['ADE']
                self.save_checkpoint(
                    os.path.join(ckpt_dir, 'moflow_teacher_best.pt')
                )

        self.save_checkpoint(
            os.path.join(ckpt_dir, 'moflow_teacher_last.pt')
        )

    def _train_epoch(self, loader: DataLoader, scheduler=None) -> float:
        self.fm.train()
        total = 0.0

        for batch in loader:
            obs_list       = [o.to(self.device) for o in batch['obs']]
            obs_rel_list   = [r.to(self.device) for r in batch['obs_rel']]
            pred_norm_list = [p.to(self.device) for p in batch['pred_norm']]

            self.optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=self.device)

            for obs, obs_rel, pred_norm in zip(
                obs_list, obs_rel_list, pred_norm_list
            ):
                past_traj  = _build_context(obs, obs_rel)
                loss, loss_reg, loss_cls = self.fm(past_traj, pred_norm)
                # print(f"  loss_reg={loss_reg.item():.4f}  loss_cls={loss_cls.item():.4f}")
                batch_loss = batch_loss + loss

            batch_loss = batch_loss / len(obs_list)
            batch_loss.backward()
            nn.utils.clip_grad_norm_(self.fm.parameters(), self.clip)
            self.optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total += batch_loss.item()

        return total / len(loader)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        k:      int | None = None,
        steps:  int        = 100,
    ) -> dict:
        """
        min-K ADE/FDE in world coordinates.

        Sample (normalised) → inverse_transform (origin-centred) → reconstruct_abs (world).
        """
        self.fm.eval()
        k = k or self.k_eval

        ade_sum = fde_sum = 0.0
        ped_count = 0

        for batch in loader:
            obs_list     = [o.to(self.device) for o in batch['obs']]
            obs_rel_list = [r.to(self.device) for r in batch['obs_rel']]
            target_list  = [t.to(self.device) for t in batch['pred']]
            origin_list  = [o.to(self.device) for o in batch['origin']]

            for obs, obs_rel, target, origin in zip(
                obs_list, obs_rel_list, target_list, origin_list
            ):
                past_traj     = _build_context(obs, obs_rel)
                preds_norm    = self.fm.sample(past_traj, K=k, steps=steps)
                preds_centred = self.normaliser.inverse_transform(preds_norm)
                preds_world   = SocialDataset.reconstruct_abs(preds_centred, origin)

                n          = obs.shape[0]
                ade_sum   += calculate_best_of_k(preds_world, target, 'ade').item() * n
                fde_sum   += calculate_best_of_k(preds_world, target, 'fde').item() * n
                ped_count += n

        if ped_count == 0:
            return {'ADE': float('inf'), 'FDE': float('inf')}
        return {'ADE': ade_sum / ped_count, 'FDE': fde_sum / ped_count}

    # ------------------------------------------------------------------
    # Teacher sample generation for IMLE
    # ------------------------------------------------------------------

    @torch.no_grad()
    def save_teacher_samples(
        self,
        loader:    DataLoader,
        save_path: str,
        k:         int | None = None,
        steps:     int        = 100,
    ) -> None:
        """
        Run teacher inference on every window and save predictions to pickle.

        CRITICAL: loader MUST be created with shuffle=False.  The saved list
        is positionally aligned with the dataset — sample at index i
        corresponds to dataset window i.  IMLETrainer reads it by position.

        Predictions are saved in the NORMALISED frame (not world coords)
        because the IMLE Chamfer loss operates in normalised space.

        Parameters
        ----------
        loader    : DataLoader with shuffle=False over the training set.
                    Must use the same SocialDataset that will be passed to
                    IMLETrainer, with the same batch_size.
        save_path : Full path for the output pickle file.
        k         : Number of teacher predictions per window.
        steps     : ODE steps (100 for full quality, fewer for debugging).
        """
        self.fm.eval()
        k = k or self.k_eval

        samples = []   # list[np.ndarray], each [K, N_i, T, 2]

        for batch in loader:
            obs_list     = [o.to(self.device) for o in batch['obs']]
            obs_rel_list = [r.to(self.device) for r in batch['obs_rel']]

            for obs, obs_rel in zip(obs_list, obs_rel_list):
                past_traj  = _build_context(obs, obs_rel)
                preds_norm = self.fm.sample(past_traj, K=k, steps=steps)
                samples.append(preds_norm.cpu().numpy().astype(np.float32))

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(samples, f)

        print(f"Saved {len(samples)} teacher samples → {save_path}")

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                'model':      self.fm.state_dict(),
                'opt':        self.optimizer.state_dict(),
                'normaliser': self.normaliser.state_dict(),
                'config':     self.config,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        data = torch.load(path, map_location=self.device, weights_only=True)
        self.fm.load_state_dict(data['model'])
        self.optimizer.load_state_dict(data['opt'])
        self.normaliser.load_state_dict(data['normaliser'])
        print(f"Loaded teacher checkpoint from {path}")


# ---------------------------------------------------------------------------
# IMLE Student Trainer
# ---------------------------------------------------------------------------

class IMLETrainer:
    """
    Trainer for the IMLE student model.

    Uses the shared SocialDataset and social_collate — no custom dataset.
    Teacher samples are loaded once at construction time and indexed
    positionally during training.

    Parameters
    ----------
    imle              : IMLE
    normaliser        : TrajectoryNormaliser
        Pass teacher_trainer.normaliser directly — must be the identical
        instance used during teacher training so bounds are guaranteed equal.
    config            : dict
    teacher_samples_path : str
        Path to the pickle file written by MoFlowTrainer.save_teacher_samples.
    num_to_gen        : int   — M, IMLE student samples per forward pass (20).

    DataLoader requirements
    -----------------------
    The DataLoader passed to train() MUST satisfy:
        shuffle=False
        drop_last=False
    so that batch positions correspond predictably to dataset indices.
    The batch_size must match the one used when save_teacher_samples() was
    called.  A mismatch will silently produce wrong teacher-sample alignment.

    Typical usage
    -------------
        imle_loader = DataLoader(
            train_dataset,          # same instance as teacher training
            batch_size=32,
            shuffle=False,          # REQUIRED
            drop_last=False,        # REQUIRED
            collate_fn=social_collate,
        )
        imle_trainer = IMLETrainer(
            imle, teacher_trainer.normaliser, config,
            teacher_samples_path='data/eth_ucy/imle/eth_samples.pkl',
        )
        imle_trainer.train(imle_loader, val_loader)
    """

    def __init__(
        self,
        imle:                 IMLE,
        normaliser:           TrajectoryNormaliser,
        config:               dict,
        teacher_samples_path: str,
        num_to_gen:           int = 20,
    ):
        if not normaliser.is_fitted:
            raise RuntimeError(
                "normaliser must be fitted.  Pass teacher_trainer.normaliser."
            )

        self.imle       = imle
        self.normaliser = normaliser
        self.config     = config
        self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.imle.to(self.device)
        self.M          = num_to_gen

        # Load teacher samples once — list[np.ndarray [K, N_i, T, 2]]
        with open(teacher_samples_path, 'rb') as f:
            self._teacher_samples = pickle.load(f)

        print(
            f"Loaded {len(self._teacher_samples)} teacher samples "
            f"from {teacher_samples_path}"
        )

        train_cfg      = config['training']
        self.optimizer = torch.optim.AdamW(
            self.imle.parameters(),
            lr=train_cfg['lr'],
            weight_decay=train_cfg.get('weight_decay', 1e-4),
        )
        self.epochs    = train_cfg['epochs']
        self.clip      = train_cfg.get('clip', 1.0)
        self.k_eval    = config.get('evaluation', {}).get('k_samples', 20)
        self.best_ade  = float('inf')

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, imle_loader: DataLoader, val_loader: DataLoader) -> None:
        """
        Full IMLE training loop.

        imle_loader must have shuffle=False and drop_last=False.
        """
        ckpt_dir = self.config.get('checkpoint_dir', 'results/checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)

        for epoch in range(self.epochs):
            loss    = self._train_epoch(imle_loader)
            metrics = self.evaluate(val_loader)
            print(
                f"[IMLE]    Epoch {epoch+1:>3}/{self.epochs}  "
                f"loss={loss:.4f}  "
                f"ADE={metrics['ADE']:.4f}  FDE={metrics['FDE']:.4f}"
            )
            if metrics['ADE'] < self.best_ade:
                self.best_ade = metrics['ADE']
                self.save_checkpoint(
                    os.path.join(ckpt_dir, 'moflow_imle_best.pt')
                )

        self.save_checkpoint(
            os.path.join(ckpt_dir, 'moflow_imle_last.pt')
        )

    def _train_epoch(self, loader: DataLoader) -> float:
        """
        One epoch of IMLE training.

        A running counter (global_idx) tracks the absolute position of each
        scene across all batches.  This is correct regardless of batch size
        and handles the last batch correctly even when it is smaller than
        the others (drop_last=False).

        Using b_idx * current_batch_size + p_idx is WRONG for the last
        batch because that batch may have fewer items, causing the computed
        index to wrap back.  The running counter avoids this entirely.
        """
        self.imle.train()
        total      = 0.0
        global_idx = 0   # increments once per scene regardless of batch size

        for batch in loader:
            obs_list       = [o.to(self.device) for o in batch['obs']]
            obs_rel_list   = [r.to(self.device) for r in batch['obs_rel']]
            pred_norm_list = [p.to(self.device) for p in batch['pred_norm']]

            self.optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=self.device)

            for obs, obs_rel, pred_norm in zip(
                obs_list, obs_rel_list, pred_norm_list
            ):
                # Running counter → teacher sample lookup
                teacher_np     = self._teacher_samples[global_idx]   # [K, N, T, 2]
                global_idx    += 1
                teacher_tensor = torch.from_numpy(
                    teacher_np.astype(np.float32)
                ).to(self.device)

                past_traj  = _build_context(obs, obs_rel)
                loss, _, _ = self.imle(
                    past_traj, pred_norm, teacher_tensor, M=self.M
                )
                batch_loss = batch_loss + loss

            # batch_loss = batch_loss / batch_size
            batch_loss = batch_loss / len(obs_list)

            batch_loss.backward()
            nn.utils.clip_grad_norm_(self.imle.parameters(), self.clip)
            self.optimizer.step()
            total += batch_loss.item()

        return total / len(loader)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, k: int | None = None) -> dict:
        """
        One-step min-K ADE/FDE in world coordinates.

        Uses the shared val_loader — no teacher samples needed at eval time.
        """
        self.imle.eval()
        k = k or self.k_eval

        ade_sum = fde_sum = 0.0
        ped_count = 0

        for batch in loader:
            obs_list     = [o.to(self.device) for o in batch['obs']]
            obs_rel_list = [r.to(self.device) for r in batch['obs_rel']]
            target_list  = [t.to(self.device) for t in batch['pred']]
            origin_list  = [o.to(self.device) for o in batch['origin']]

            for obs, obs_rel, target, origin in zip(
                obs_list, obs_rel_list, target_list, origin_list
            ):
                past_traj     = _build_context(obs, obs_rel)
                preds_norm    = self.imle(
                    past_traj, pred_norm=None, teacher_samples=None, M=1
                )
                preds_centred = self.normaliser.inverse_transform(preds_norm)
                preds_world   = SocialDataset.reconstruct_abs(preds_centred, origin)

                n          = obs.shape[0]
                ade_sum   += calculate_best_of_k(preds_world, target, 'ade').item() * n
                fde_sum   += calculate_best_of_k(preds_world, target, 'fde').item() * n
                ped_count += n

        return {'ADE': ade_sum / ped_count, 'FDE': fde_sum / ped_count}

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                'model':      self.imle.state_dict(),
                'opt':        self.optimizer.state_dict(),
                'normaliser': self.normaliser.state_dict(),
                'config':     self.config,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        data = torch.load(path, map_location=self.device, weights_only=True)
        self.imle.load_state_dict(data['model'])
        self.optimizer.load_state_dict(data['opt'])
        self.normaliser.load_state_dict(data['normaliser'])
        print(f"Loaded IMLE checkpoint from {path}")