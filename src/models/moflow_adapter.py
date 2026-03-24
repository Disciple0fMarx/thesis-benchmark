"""
MoFlowAdapter / IMLEAdapter: thin wrappers that make FlowMatcher and IMLE
satisfy the shared evaluator contract:

    model.forward(obs, obs_rel, k) → FloatTensor [K, N, pred_len, 2]

This allows MoFlow (both teacher and student) to be evaluated by the same
generic Evaluator used for all other models, and to be included in
run_all_benchmarks.py without any special-casing.

Responsibilities of the adapter
--------------------------------
1. Build the 6-feature context tensor from obs and obs_rel.
2. Call the underlying model's sampling method.
3. Denormalise predictions from [-1, 1] back to origin-centred absolute
   coordinates using the TrajectoryNormaliser.

The adapter does NOT reconstruct world coordinates — that is the
evaluator's responsibility via SocialDataset.reconstruct_abs(pred, origin).
This matches the contract for every other model.

Normaliser dependency
---------------------
The adapter requires a fitted TrajectoryNormaliser to denormalise
predictions.  This must be the SAME normaliser that was attached to the
dataset during training — guaranteed in practice by passing the trainer's
normaliser attribute to the adapter constructor.
"""

import torch
import torch.nn as nn

from src.models.moflow import FlowMatcher, IMLE


def _build_context(obs: torch.Tensor, obs_rel: torch.Tensor) -> torch.Tensor:
    """
    Construct the 6-feature context tensor expected by ETHContextEncoder.

    Parameters
    ----------
    obs     : [N, T_o, 2]  origin-centred absolute positions
    obs_rel : [N, T_o, 2]  displacements (= velocities at 2.5 Hz)

    Returns
    -------
    [N, T_o, 6]  — [abs_x, abs_y, rel_x, rel_y, vx, vy]

    'obs' appears twice: once as absolute coordinates and once as
    coordinates relative to the last observed frame.  Since 'obs' is
    already origin-centred (last frame = 0), both representations are
    numerically identical.  The encoder learns to use each channel
    independently.
    """
    return torch.cat([obs, obs, obs_rel], dim=-1)


class MoFlowAdapter(nn.Module):
    """
    Adapter for the MoFlow teacher model (FlowMatcher).

    Wraps FlowMatcher.sample() to satisfy:
        forward(obs, obs_rel, k) → [K, N, pred_len, 2]

    in the origin-centred absolute frame expected by the Evaluator.

    Parameters
    ----------
    flow_matcher : FlowMatcher
        Trained teacher model.
    normaliser   : TrajectoryNormaliser
        The fitted normaliser from MoFlowTrainer — must be identical to
        the one used during training.
    steps        : int
        ODE solver steps.  Default 100 (full quality).  Reduce for speed.
    solver       : str
        ODE solver: 'lin_poly' (default, matches authors) or 'euler'.
    """

    def __init__(
        self,
        flow_matcher: FlowMatcher,
        normaliser,
        steps:  int = 100,
        solver: str = 'lin_poly',
    ):
        super().__init__()
        self.fm         = flow_matcher
        self.normaliser = normaliser
        self.steps      = steps
        self.solver     = solver

    def forward(
        self,
        obs:     torch.Tensor,   # [N, T_o, 2]
        obs_rel: torch.Tensor,   # [N, T_o, 2]
        k:       int,
    ) -> torch.Tensor:
        """
        Returns [K, N, pred_len, 2] in the origin-centred absolute frame.

        The Evaluator will call SocialDataset.reconstruct_abs(preds, origin)
        to convert to world coordinates before computing ADE/FDE.
        """
        past_traj   = _build_context(obs, obs_rel)               # [N, T_o, 6]

        # Sample in normalised frame: [K, N, T, 2]
        preds_norm  = self.fm.sample(
            past_traj, K=k, steps=self.steps, solver=self.solver
        )

        # Denormalise to origin-centred absolute frame: [K, N, T, 2]
        return self.normaliser.inverse_transform(preds_norm)


class IMLEAdapter(nn.Module):
    """
    Adapter for the MoFlow student model (IMLE).

    Wraps IMLE one-step inference to satisfy:
        forward(obs, obs_rel, k) → [K, N, pred_len, 2]

    in the origin-centred absolute frame.

    Parameters
    ----------
    imle       : IMLE
        Trained student model.
    normaliser : TrajectoryNormaliser
        Must be identical to the one used during IMLE training.
    """

    def __init__(self, imle: IMLE, normaliser):
        super().__init__()
        self.imle       = imle
        self.normaliser = normaliser

    def forward(
        self,
        obs:     torch.Tensor,   # [N, T_o, 2]
        obs_rel: torch.Tensor,   # [N, T_o, 2]
        k:       int,
    ) -> torch.Tensor:
        """
        Returns [K, N, pred_len, 2] in the origin-centred absolute frame.
        K comes from the IMLE model's configured self.K — the k argument
        is accepted for interface compatibility but the student always
        generates its full K predictions.
        """
        past_traj   = _build_context(obs, obs_rel)               # [N, T_o, 6]

        # One-step inference: [K, N, T, 2] normalised
        preds_norm  = self.imle(
            past_traj,
            pred_norm=None,
            teacher_samples=None,
            M=1,
        )

        # Denormalise to origin-centred absolute frame
        return self.normaliser.inverse_transform(preds_norm)
