"""
train_moflow_teacher.py
=======================
Train the MoFlow teacher (FlowMatcher) on one fold of the ETH-UCY
leave-one-out benchmark, then save teacher samples for IMLE distillation.

Config is loaded from configs/moflow.yml.  Any value can be overridden
at the command line with --override key=value (dot-separated path), e.g.:
    python train_moflow_teacher.py --subset eth --override training.lr=5e-5

Usage
-----
    # Single fold
    python train_moflow_teacher.py --subset eth

    # All five folds
    for subset in eth hotel univ zara1 zara2; do
        python train_moflow_teacher.py --subset $subset
    done

Outputs
-------
    results/moflow/teacher/<subset>/checkpoint_best.pt
    results/moflow/teacher/<subset>/checkpoint_last.pt
    data/processed/imle_samples/<subset>_teacher_samples.pkl
"""

import argparse
import os
import random
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data_pipeline.loader import TrajectoryLoader, ALL_SCENES
from src.data_pipeline.sequence_generator import SocialSequenceGenerator
from src.data_pipeline.dataset import SocialDataset, social_collate
from src.data_pipeline.normaliser import TrajectoryNormaliser
from src.models.moflow import ETHMotionTransformer, FlowMatcher
from src.training.moflow_trainer import MoFlowTrainer


# ---------------------------------------------------------------------------
# Config utilities
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """
    Apply dot-separated key=value overrides to a nested config dict.

    Example:
        apply_overrides(cfg, ['training.lr=5e-5', 'model.dropout=0.2'])
    """
    for override in overrides:
        if '=' not in override:
            raise ValueError(
                f"Override '{override}' must be in key=value format."
            )
        key_path, value_str = override.split('=', 1)
        keys   = key_path.strip().split('.')
        target = config
        for k in keys[:-1]:
            if k not in target:
                raise KeyError(
                    f"Config key '{k}' not found in path '{key_path}'."
                )
            target = target[k]
        # Attempt to parse as float, int, bool, then fall back to string
        leaf = keys[-1]
        for cast in (int, float):
            try:
                target[leaf] = cast(value_str)
                break
            except ValueError:
                continue
        else:
            if value_str.lower() == 'true':
                target[leaf] = True
            elif value_str.lower() == 'false':
                target[leaf] = False
            else:
                target[leaf] = value_str

    return config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict) -> FlowMatcher:
    mc  = cfg['model']
    fc  = cfg['flow']
    net = ETHMotionTransformer(
        d_model        = mc['d_model'],
        K              = mc['K'],
        pred_len       = cfg['data']['pred_len'],
        n_enc_heads    = mc['n_enc_heads'],
        n_enc_layers   = mc['n_enc_layers'],
        n_dec_heads    = mc['n_dec_heads'],
        n_dec_layers   = mc['n_dec_layers'],
        ffn_multiplier = mc['ffn_multiplier'],
        dropout        = mc['dropout'],
        drop_logi_k    = mc['drop_logi_k'],
        drop_logi_m    = mc['drop_logi_m'],
    )
    fm = FlowMatcher(
        model           = net,
        K               = mc['K'],
        pred_len        = cfg['data']['pred_len'],
        logit_norm_mean = fc['logit_norm_mean'],
        logit_norm_std  = fc['logit_norm_std'],
        tied_noise      = fc['tied_noise'],
        fm_in_scaling   = fc['fm_in_scaling'],
        loss_nn_mode    = fc['loss_nn_mode'],
        loss_weight_reg = fc['loss_weight_reg'],
        loss_weight_cls = fc['loss_weight_cls'],
    )
    n_params = sum(p.numel() for p in fm.parameters())
    print(f"  Model parameters: {n_params:,}")
    return fm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train MoFlow teacher on one ETH-UCY fold."
    )
    parser.add_argument(
        '--subset', type=str, required=True, choices=ALL_SCENES,
        help="Test scene (held-out fold).",
    )
    parser.add_argument(
        '--config', type=str, default='configs/moflow.yml',
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        '--data_dir', type=str, default='data/raw',
    )
    parser.add_argument(
        '--results_dir', type=str, default='results/moflow',
    )
    parser.add_argument(
        '--samples_dir', type=str, default='data/processed/imle_samples',
        help="Where to save teacher samples for IMLE.",
    )
    parser.add_argument(
        '--no_save_samples', action='store_true',
        help="Skip saving teacher samples (useful for quick runs).",
    )
    parser.add_argument(
        '--override', nargs='*', default=[],
        metavar='key=value',
        help="Override config values, e.g. --override training.lr=5e-5",
    )
    args = parser.parse_args()

    # Load and patch config
    cfg = load_config(args.config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    test_scene   = args.subset
    ckpt_dir     = os.path.join(args.results_dir, 'teacher', test_scene)
    samples_path = os.path.join(
        args.samples_dir, f'{test_scene}_teacher_samples.pkl'
    )

    print(f"\n{'='*60}")
    print(f"  MoFlow Teacher Training")
    print(f"  Config       : {args.config}")
    print(f"  Test scene   : {test_scene}")
    print(f"  Train scenes : {[s for s in ALL_SCENES if s != test_scene]}")
    print(f"  Checkpoint   : {ckpt_dir}")
    print(f"{'='*60}\n")

    set_seed(cfg['seed'])

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dc        = cfg['data']
    loader    = TrajectoryLoader(args.data_dir)
    generator = SocialSequenceGenerator(
        obs_len  = dc['obs_len'],
        pred_len = dc['pred_len'],
        min_peds = dc['min_peds'],
    )

    print("Loading scenes...")
    train_scenes = loader.load_train_scenes(test_scene)
    test_scene_d = loader.load_scene(test_scene)

    train_dataset = SocialDataset(train_scenes, generator)
    test_dataset  = SocialDataset(test_scene_d, generator)

    print(f"  Train: {train_dataset}")
    print(f"  Test : {test_dataset}")

    # ------------------------------------------------------------------
    # Normaliser — fit on training fold only
    # ------------------------------------------------------------------
    print("\nFitting normaliser on training fold...")
    normaliser = TrajectoryNormaliser(mode='minmax')
    normaliser.fit(train_dataset)
    print(f"  {normaliser}")

    train_dataset.set_normaliser(normaliser)
    test_dataset.set_normaliser(normaliser)

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    tc         = cfg['training']
    batch_size = tc['batch_size']

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        drop_last   = False,
        collate_fn  = social_collate,
        num_workers = 0,
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        drop_last   = False,
        collate_fn  = social_collate,
        num_workers = 0,
    )

    print(f"\n  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")

    # ------------------------------------------------------------------
    # Model and trainer
    # ------------------------------------------------------------------
    print("\nBuilding model...")
    fm = build_model(cfg)

    trainer_config = {
        'training':       tc,
        'evaluation':     cfg['evaluation'],
        'checkpoint_dir': ckpt_dir,
    }
    trainer = MoFlowTrainer(fm, normaliser, trainer_config)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    ec         = cfg['evaluation']
    eval_every = ec['eval_every']
    ode_steps  = ec['ode_steps']

    print(f"\nTraining for {tc['epochs']} epochs "
          f"(evaluating every {eval_every})...\n")

    for epoch in range(tc['epochs']):
        loss = trainer._train_epoch(train_loader)

        if (epoch + 1) % eval_every == 0 or epoch == tc['epochs'] - 1:
            metrics = trainer.evaluate(
                val_loader, k=ec['k_samples'], steps=ode_steps
            )
            ade    = metrics['ADE']
            fde    = metrics['FDE']
            marker = ' ← best' if ade < trainer.best_ade else ''
            print(
                f"Epoch {epoch+1:>3}/{tc['epochs']}  "
                f"loss={loss:.4f}  "
                f"ADE={ade:.4f}  FDE={fde:.4f}"
                f"{marker}"
            )
            if ade < trainer.best_ade:
                trainer.best_ade = ade
                trainer.save_checkpoint(
                    os.path.join(ckpt_dir, 'checkpoint_best.pt')
                )

    trainer.save_checkpoint(os.path.join(ckpt_dir, 'checkpoint_last.pt'))
    print(f"\nBest val ADE : {trainer.best_ade:.4f}")
    print(f"Checkpoints  : {ckpt_dir}")

    # ------------------------------------------------------------------
    # Save teacher samples for IMLE
    # ------------------------------------------------------------------
    if not args.no_save_samples:
        print(f"\nLoading best checkpoint for sample generation...")
        trainer.load_checkpoint(os.path.join(ckpt_dir, 'checkpoint_best.pt'))

        ordered_loader = DataLoader(
            train_dataset,
            batch_size  = cfg['sampling']['batch_size'],
            shuffle     = False,
            drop_last   = False,
            collate_fn  = social_collate,
            num_workers = 0,
        )

        print(f"Saving teacher samples → {samples_path}")
        trainer.save_teacher_samples(
            ordered_loader,
            samples_path,
            k     = ec['k_samples'],
            steps = cfg['sampling']['ode_steps'],
        )

    print(f"\n{'='*60}")
    print(f"  Done. Next: python train_moflow_imle.py --subset {test_scene}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
