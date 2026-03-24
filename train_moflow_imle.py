"""
train_moflow_imle.py
====================
Train the MoFlow IMLE student on one fold of the ETH-UCY benchmark.

Must be run AFTER train_moflow_teacher.py for the same --subset.

Config is loaded from configs/moflow.yml.  Any value can be overridden
at the command line with --override key=value, e.g.:
    python train_moflow_imle.py --subset eth --override imle_training.lr=5e-5

Usage
-----
    python train_moflow_imle.py --subset eth

    for subset in eth hotel univ zara1 zara2; do
        python train_moflow_imle.py --subset $subset
    done

Outputs
-------
    results/moflow/imle/<subset>/checkpoint_best.pt
    results/moflow/imle/<subset>/checkpoint_last.pt
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
from src.models.moflow import (
    ETHMotionTransformer, FlowMatcher,
    ETHIMLETransformer, IMLE,
)
from src.training.moflow_trainer import IMLETrainer


# ---------------------------------------------------------------------------
# Config utilities (shared with teacher script)
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    for override in overrides:
        if '=' not in override:
            raise ValueError(f"Override '{override}' must be in key=value format.")
        key_path, value_str = override.split('=', 1)
        keys   = key_path.strip().split('.')
        target = config
        for k in keys[:-1]:
            if k not in target:
                raise KeyError(f"Config key '{k}' not found in path '{key_path}'.")
            target = target[k]
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


def build_student(cfg: dict) -> IMLE:
    mc = cfg['model']
    ic = cfg['imle']
    net = ETHIMLETransformer(
        d_model        = mc['d_model'],
        K              = mc['K'],
        pred_len       = cfg['data']['pred_len'],
        n_enc_heads    = mc['n_enc_heads'],
        n_enc_layers   = mc['n_enc_layers'],
        n_dec_heads    = mc['n_dec_heads'],
        n_dec_layers   = mc['n_dec_layers'],
        ffn_multiplier = mc['ffn_multiplier'],
        dropout        = mc['dropout'],
    )
    imle = IMLE(
        model          = net,
        K              = mc['K'],
        pred_len       = cfg['data']['pred_len'],
        chamfer_weight = ic['chamfer_weight'],
        gt_weight      = ic['gt_weight'],
        loss_reduction = ic['loss_reduction'],
    )
    n_params = sum(p.numel() for p in imle.parameters())
    print(f"  Student parameters: {n_params:,}")
    return imle


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train MoFlow IMLE student on one ETH-UCY fold."
    )
    parser.add_argument(
        '--subset', type=str, required=True, choices=ALL_SCENES,
    )
    parser.add_argument(
        '--config', type=str, default='configs/moflow.yml',
    )
    parser.add_argument('--data_dir',    type=str, default='data/raw')
    parser.add_argument('--results_dir', type=str, default='results/moflow')
    parser.add_argument(
        '--samples_dir', type=str, default='data/processed/imle_samples',
    )
    parser.add_argument(
        '--load_pretrained', action='store_true', default=True,
        help="Initialise student encoder from teacher checkpoint (recommended).",
    )
    parser.add_argument(
        '--override', nargs='*', default=[], metavar='key=value',
    )
    args = parser.parse_args()

    cfg        = load_config(args.config)
    if args.override:
        cfg    = apply_overrides(cfg, args.override)

    test_scene   = args.subset
    teacher_ckpt = os.path.join(
        args.results_dir, 'teacher', test_scene, 'checkpoint_best.pt'
    )
    samples_path = os.path.join(
        args.samples_dir, f'{test_scene}_teacher_samples.pkl'
    )
    ckpt_dir = os.path.join(args.results_dir, 'imle', test_scene)

    if not os.path.exists(teacher_ckpt):
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {teacher_ckpt}\n"
            f"Run train_moflow_teacher.py --subset {test_scene} first."
        )
    if not os.path.exists(samples_path):
        raise FileNotFoundError(
            f"Teacher samples not found: {samples_path}\n"
            f"Run train_moflow_teacher.py --subset {test_scene} first."
        )

    print(f"\n{'='*60}")
    print(f"  MoFlow IMLE Student Training")
    print(f"  Config         : {args.config}")
    print(f"  Test scene     : {test_scene}")
    print(f"  Train scenes   : {[s for s in ALL_SCENES if s != test_scene]}")
    print(f"  Teacher ckpt   : {teacher_ckpt}")
    print(f"  Teacher samples: {samples_path}")
    print(f"  Output         : {ckpt_dir}")
    print(f"{'='*60}\n")

    set_seed(cfg['seed'])

    # ------------------------------------------------------------------
    # Data — identical to teacher training
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
    # Restore normaliser from teacher checkpoint
    # CRITICAL: must use the exact same bounds as teacher training.
    # ------------------------------------------------------------------
    print("\nRestoring normaliser from teacher checkpoint...")
    ckpt_data  = torch.load(teacher_ckpt, map_location='cpu', weights_only=True)
    normaliser = TrajectoryNormaliser(mode='minmax')
    normaliser.load_state_dict(ckpt_data['normaliser'])
    print(f"  {normaliser}")

    train_dataset.set_normaliser(normaliser)
    test_dataset.set_normaliser(normaliser)

    # ------------------------------------------------------------------
    # DataLoaders
    # IMLE loader: shuffle=False, drop_last=False — required for
    # positional alignment with teacher samples.
    # ------------------------------------------------------------------
    tc         = cfg['imle_training']
    batch_size = tc['batch_size']

    imle_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = False,
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

    print(f"\n  Train batches : {len(imle_loader)}")
    print(f"  Val   batches : {len(val_loader)}")

    # ------------------------------------------------------------------
    # Build student model
    # ------------------------------------------------------------------
    print("\nBuilding student model...")
    imle = build_student(cfg)

    # ------------------------------------------------------------------
    # Optionally transfer encoder weights from teacher
    # ------------------------------------------------------------------
    if args.load_pretrained:
        print("Transferring encoder weights from teacher checkpoint...")
        teacher_state = ckpt_data['model']
        student_state = imle.state_dict()
        transferred   = 0
        for key, val in teacher_state.items():
            if key.startswith('model.encoder.') and key in student_state:
                if student_state[key].shape == val.shape:
                    student_state[key] = val
                    transferred += 1
        imle.load_state_dict(student_state)
        print(f"  Transferred {transferred} encoder parameter tensors.")

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer_config = {
        'training':       tc,
        'evaluation':     cfg['evaluation'],
        'checkpoint_dir': ckpt_dir,
    }
    trainer = IMLETrainer(
        imle                 = imle,
        normaliser           = normaliser,
        config               = trainer_config,
        teacher_samples_path = samples_path,
        num_to_gen           = cfg['imle']['num_to_gen'],
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    ec         = cfg['evaluation']
    eval_every = ec['eval_every']

    print(f"\nTraining for {tc['epochs']} epochs "
          f"(evaluating every {eval_every})...\n")

    for epoch in range(tc['epochs']):
        loss = trainer._train_epoch(imle_loader)

        if (epoch + 1) % eval_every == 0 or epoch == tc['epochs'] - 1:
            metrics = trainer.evaluate(val_loader, k=ec['k_samples'])
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

    print(f"\n{'='*60}")
    print(f"  Done.")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
