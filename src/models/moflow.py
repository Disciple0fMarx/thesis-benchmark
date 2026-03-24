"""
MoFlow: One-Step Flow Matching for Human Trajectory Forecasting
via Implicit Maximum Likelihood Estimation based Distillation

Adapted from Fu et al., CVPR 2025 (https://github.com/DSL-Lab/MoFlow)

Normalisation contract
----------------------
This file contains NO normalisation logic.  All models in this codebase
receive exactly the same data from SocialDataset:

  'obs'       [N, T_o, 2]  origin-centred absolute positions
  'obs_rel'   [N, T_o, 2]  frame-to-frame displacements
  'pred'      [N, T,   2]  raw world coordinates (metrics only)
  'pred_norm' [N, T,   2]  min-max normalised to [-1, 1] (when normaliser set)
  'origin'    [N, 1,   2]  world position of the normalisation origin

FlowMatcher.forward() receives 'pred_norm' directly from the batch.
FlowMatcher.sample() returns predictions in the same normalised frame.
MoFlowTrainer is responsible for denormalisation before metric computation.

Context tensor construction
---------------------------
ETHContextEncoder expects [N, T_o, 6].  The 6 features are assembled from
the shared pipeline tensors inside _build_context() in moflow_trainer.py:
    [obs, obs, obs_rel]  →  [abs_x, abs_y, rel_x, rel_y, vx, vy]
This is a model-specific feature engineering step, not a preprocessing
change — the underlying obs and obs_rel tensors are identical to what
every other model receives.

Tensor shape conventions
------------------------
N   : pedestrians in current scene (variable per window)
K   : joint predictions (default 20)
T   : future frames (default 12)
M   : IMLE student samples per noise draw
T_o : observed frames (default 8)
D   : model hidden dimension
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embedding.  Used for both agent index encoding
    and flow time encoding.  Identical to the authors' implementation.
    """
    def __init__(self, dim: int, theta: int = 10000):
        super().__init__()
        self.dim   = dim
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: 1-D tensor of positions or times
        device   = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)  # [..., D]


# ---------------------------------------------------------------------------
# Context Encoder
# ---------------------------------------------------------------------------

class SocialTransformer(nn.Module):
    """
    First-stage encoder: flattens each agent's T_o-frame trajectory into a
    single vector, then runs a transformer over the agent axis.

    Input  : [N, T_o, F]  where F=6
    Output : [N, D]
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.proj    = nn.Linear(in_dim, hidden_dim, bias=False)
        layer        = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=2,
            dim_feedforward=hidden_dim, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.out_proj    = nn.Linear(hidden_dim, out_dim)

    def forward(self, past_traj: torch.Tensor) -> torch.Tensor:
        N, T, F  = past_traj.shape
        x        = past_traj.reshape(N, T * F)          # [N, T_o*F]
        x        = self.proj(x).unsqueeze(0)            # [1, N, hidden]
        x        = x + self.transformer(x)              # residual
        return self.out_proj(x.squeeze(0))              # [N, D]


class ETHContextEncoder(nn.Module):
    """
    Full context encoder: SocialTransformer + sinusoidal agent PE +
    transformer encoder over agents.

    Input  : past_traj [N, T_o, 6]
    Output : H_enc     [N, D]
    """
    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        n_layers: int,
        dropout:  float,
    ):
        super().__init__()
        # T_o=8, F=6  →  flattened = 48
        self.social_enc = SocialTransformer(
            in_dim=48, hidden_dim=256, out_dim=d_model
        )
        self.pos_emb = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.agent_emb = nn.Embedding(256, d_model)
        self.pe_fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, past_traj: torch.Tensor) -> torch.Tensor:
        N      = past_traj.shape[0]
        device = past_traj.device

        feat   = self.social_enc(past_traj)                        # [N, D]
        idx    = torch.arange(N, device=device).float()
        sin_pe = self.pos_emb(idx)                                 # [N, D]
        agt_pe = self.agent_emb(torch.arange(N, device=device))   # [N, D]
        pe     = self.pe_fusion(torch.cat([sin_pe, agt_pe], -1))  # [N, D]

        feat   = (feat + pe).unsqueeze(0)                          # [1, N, D]
        return self.transformer(feat).squeeze(0)                   # [N, D]


# ---------------------------------------------------------------------------
# Motion Decoder with AdaLN
# ---------------------------------------------------------------------------

class AdaLN(nn.Module):
    """
    Adaptive LayerNorm: scale and shift predicted from a conditioning vector.
    When use_adaln=False (student model) degenerates to plain LayerNorm.
    """
    def __init__(self, d_model: int, cond_dim: int, use_adaln: bool):
        super().__init__()
        self.use_adaln = use_adaln
        self.norm      = nn.LayerNorm(d_model)
        if use_adaln:
            self.proj  = nn.Linear(cond_dim, 2 * d_model)

    def forward(
        self,
        x:    torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.norm(x)
        if self.use_adaln and cond is not None:
            # cond must already be shaped to broadcast against x.
            # DecoderLayer is responsible for passing the right shape:
            #   norm_k: x is [N, K, D], cond must be [N, 1, D]
            #   norm_a: x is [K, N, D], cond must be [1, N, D]
            #   norm_ff: x is [K, N, D], cond must be [1, N, D]
            scale, shift = self.proj(cond).chunk(2, dim=-1)
            x = x * (1 + scale) + shift
        return x


class DecoderLayer(nn.Module):
    """
    Transformer decoder layer with factorised K/A attention.

    Attention over K dimension (per agent, K predictions attend each other),
    then attention over N dimension (per prediction, agents attend each other),
    then FFN.  Each sub-layer preceded by AdaLN.

    Input/output: [K, N, D]
    """
    def __init__(
        self,
        d_model:   int,
        n_heads:   int,
        ffn_dim:   int,
        dropout:   float,
        cond_dim:  int,
        use_adaln: bool,
    ):
        super().__init__()
        self.attn_k  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.attn_a  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm_k  = AdaLN(d_model, cond_dim, use_adaln)
        self.norm_a  = AdaLN(d_model, cond_dim, use_adaln)
        self.norm_ff = AdaLN(d_model, cond_dim, use_adaln)
        self.drop    = nn.Dropout(dropout)

    def forward(
        self,
        x:    torch.Tensor,           # [K, N, D]
        cond: torch.Tensor | None,    # [N, D] time embedding
    ) -> torch.Tensor:
        K, N, D = x.shape

        # Shape cond for each attention axis:
        #   norm_k operates on xk: [N, K, D] — needs cond [N, 1, D]
        #   norm_a and norm_ff operate on x: [K, N, D] — needs cond [1, N, D]
        cond_k  = cond.unsqueeze(1) if cond is not None else None   # [N, 1, D]
        cond_kn = cond.unsqueeze(0) if cond is not None else None   # [1, N, D]

        # Attention over K: each agent's K predictions attend each other
        xk      = rearrange(x, 'k n d -> n k d')
        xk      = self.norm_k(xk, cond_k)
        xk, _   = self.attn_k(xk, xk, xk)
        x       = x + self.drop(rearrange(xk, 'n k d -> k n d'))

        # Attention over N: each prediction's agents attend each other
        xa      = self.norm_a(x, cond_kn)
        xa, _   = self.attn_a(xa, xa, xa)
        x       = x + self.drop(xa)

        # FFN
        xf      = self.norm_ff(x, cond_kn)
        x       = x + self.drop(self.ffn(xf))
        return x


class MotionDecoder(nn.Module):
    """Stack of DecoderLayer blocks.  Input/output: [K, N, D]."""
    def __init__(
        self,
        d_model:   int,
        n_heads:   int,
        n_layers:  int,
        ffn_dim:   int,
        dropout:   float,
        cond_dim:  int,
        use_adaln: bool,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, ffn_dim, dropout, cond_dim, use_adaln)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        x:    torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, cond)
        return x


# ---------------------------------------------------------------------------
# Teacher model
# ---------------------------------------------------------------------------

class ETHMotionTransformer(nn.Module):
    """
    MoFlow teacher: denoising network for the flow matching objective.

    Not called directly during training or evaluation — always accessed
    through the FlowMatcher wrapper below.

    forward(y, t, past_traj) → (pred [K, N, T*2], logits [K, N])

    Parameters (all have sensible defaults for ETH-UCY)
    ----------
    d_model       : hidden dimension D
    K             : joint predictions
    pred_len      : future frames T
    drop_logi_k/m : logistic masking parameters (k=20, m=0.5 from paper)
    """
    def __init__(
        self,
        d_model:        int   = 128,
        K:              int   = 20,
        pred_len:       int   = 12,
        n_enc_heads:    int   = 4,
        n_enc_layers:   int   = 2,
        n_dec_heads:    int   = 4,
        n_dec_layers:   int   = 4,
        ffn_multiplier: int   = 4,
        dropout:        float = 0.1,
        drop_logi_k:    float = 20.0,
        drop_logi_m:    float = 0.5,
    ):
        super().__init__()
        self.K           = K
        self.pred_len    = pred_len
        self.out_dim     = pred_len * 2
        self.D           = d_model
        self.drop_logi_k = drop_logi_k
        self.drop_logi_m = drop_logi_m

        self.encoder = ETHContextEncoder(d_model, n_enc_heads, n_enc_layers, dropout)

        # Flow time → embedding (scale ×1000 before sinusoidal as per authors)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.noisy_y_mlp = nn.Sequential(
            nn.Linear(self.out_dim, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model),      nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Factorised attention on noisy trajectory embeddings before fusion
        self.noisy_attn_k = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.noisy_attn_a = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )

        self.k_emb = nn.Embedding(K,   d_model)
        self.a_emb = nn.Embedding(256, d_model)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.LayerNorm(d_model), nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.post_pe_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model), nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.decoder = MotionDecoder(
            d_model=d_model, n_heads=n_dec_heads, n_layers=n_dec_layers,
            ffn_dim=d_model * ffn_multiplier, dropout=dropout,
            cond_dim=d_model, use_adaln=True,
        )

        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, self.out_dim),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        y:         torch.Tensor,   # [K, N, T*2]  noisy trajectories
        t:         torch.Tensor,   # [1]           flow time ∈ [0, 1]
        past_traj: torch.Tensor,   # [N, T_o, 6]  context
    ) -> tuple[torch.Tensor, torch.Tensor]:
        K, N, _ = y.shape
        device  = y.device

        enc_out  = self.encoder(past_traj)                         # [N, D]
        t_emb    = self.time_mlp(t * 1000.0).expand(N, -1)        # [N, D]

        y_emb    = self.noisy_y_mlp(y)                             # [K, N, D]
        k_pe     = self.k_emb(torch.arange(K, device=device))     # [K, D]
        a_pe     = self.a_emb(torch.arange(N, device=device))     # [N, D]
        k_pe     = k_pe.unsqueeze(1).expand(-1, N, -1)            # [K, N, D]
        a_pe     = a_pe.unsqueeze(0).expand(K, -1, -1)            # [K, N, D]

        y_emb    = y_emb + k_pe + a_pe

        # Factorised attention before fusion
        y_k      = rearrange(y_emb, 'k n d -> n k d')
        y_k      = self.noisy_attn_k(y_k)
        y_emb    = rearrange(y_k, 'n k d -> k n d')
        y_emb    = self.noisy_attn_a(y_emb)

        # Flow-time masking: zero noisy embedding with logistic probability.
        # Prevents the model from copying y_t near t=1 where y_t ≈ x_1.
        if self.training:
            t_val = t[0].item()
            p_m   = 1.0 / (1.0 + math.exp(
                -self.drop_logi_k * (t_val - self.drop_logi_m)
            ))
            y_emb = y_emb.masked_fill(
                torch.rand(K, N, 1, device=device) < p_m, 0.0
            )

        enc_exp  = enc_out.unsqueeze(0).expand(K, -1, -1)         # [K, N, D]
        t_exp    = t_emb.unsqueeze(0).expand(K, -1, -1)           # [K, N, D]

        fused    = self.fusion_mlp(
            torch.cat([enc_exp, y_emb, t_exp], dim=-1)
        )                                                           # [K, N, D]
        query    = self.post_pe_mlp(fused + k_pe + a_pe)          # [K, N, D]
        decoded  = self.decoder(query, cond=t_emb)                 # [K, N, D]

        pred     = self.reg_head(decoded)                          # [K, N, T*2]
        logits   = self.cls_head(decoded).squeeze(-1)              # [K, N]
        return pred, logits


# ---------------------------------------------------------------------------
# Student model
# ---------------------------------------------------------------------------

class ETHIMLETransformer(nn.Module):
    """
    MoFlow student: one-step generator trained via IMLE distillation.

    Identical architecture to the teacher except:
      - No time input or AdaLN (standard LayerNorm in decoder)
      - Input is a latent noise vector Z ∈ R^D rather than noisy trajectories
      - Generates M × K predictions per call to support IMLE nearest-neighbour

    forward(past_traj, M) → [M, K, N, T*2]
    """
    def __init__(
        self,
        d_model:        int   = 128,
        K:              int   = 20,
        pred_len:       int   = 12,
        n_enc_heads:    int   = 4,
        n_enc_layers:   int   = 2,
        n_dec_heads:    int   = 4,
        n_dec_layers:   int   = 4,
        ffn_multiplier: int   = 4,
        dropout:        float = 0.1,
    ):
        super().__init__()
        self.K       = K
        self.pred_len = pred_len
        self.out_dim  = pred_len * 2
        self.D        = d_model

        self.encoder = ETHContextEncoder(d_model, n_enc_heads, n_enc_layers, dropout)

        self.noise_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.k_emb = nn.Embedding(K,   d_model)
        self.a_emb = nn.Embedding(256, d_model)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model), nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.pe_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.decoder = MotionDecoder(
            d_model=d_model, n_heads=n_dec_heads, n_layers=n_dec_layers,
            ffn_dim=d_model * ffn_multiplier, dropout=dropout,
            cond_dim=d_model, use_adaln=False,
        )
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, self.out_dim),
        )

    def forward(
        self,
        past_traj: torch.Tensor,   # [N, T_o, 6]
        M:         int,
    ) -> torch.Tensor:
        """Returns [M, K, N, T*2]."""
        K      = self.K
        N      = past_traj.shape[0]
        device = past_traj.device

        enc_out   = self.encoder(past_traj)                        # [N, D]
        noise     = torch.randn(M, self.D, device=device)
        noise_emb = self.noise_mlp(noise)                          # [M, D]

        k_pe = self.k_emb(torch.arange(K, device=device))         # [K, D]
        a_pe = self.a_emb(torch.arange(N, device=device))         # [N, D]

        enc_exp   = enc_out[None, None].expand(M, K, -1, -1)      # [M, K, N, D]
        noise_exp = noise_emb[:, None, None].expand(-1, K, N, -1) # [M, K, N, D]
        k_exp     = k_pe[None, :, None].expand(M, -1, N, -1)      # [M, K, N, D]
        a_exp     = a_pe[None, None, :].expand(M, K, -1, -1)      # [M, K, N, D]

        fused  = self.fusion_mlp(
            torch.cat([enc_exp, noise_exp], dim=-1)
        )                                                           # [M, K, N, D]
        query  = self.pe_mlp(fused + k_exp + a_exp)               # [M, K, N, D]

        # Run decoder for each of the M noise samples independently,
        # preserving the K/N factorised attention semantics.
        outputs = []
        for m in range(M):
            dec_out = self.decoder(query[m], cond=None)            # [K, N, D]
            outputs.append(self.reg_head(dec_out))                 # [K, N, T*2]

        return torch.stack(outputs, dim=0)                         # [M, K, N, T*2]


# ---------------------------------------------------------------------------
# Flow Matching wrapper
# ---------------------------------------------------------------------------

def _pad_t(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape [1] time tensor to broadcast against x."""
    return t.reshape(-1, *([1] * (x.dim() - 1)))


class FlowMatcher(nn.Module):
    """
    Wraps ETHMotionTransformer with flow matching training logic and ODE
    sampling.

    Velocity wrapper (paper Eq. 4.1):
        D_θ(y_t, C, t) = y_t + (1 − t) · F_θ(y_t, C, t)

    D_θ predicts the data x_1 from the noisy state y_t.  The loss
    compares D_θ to the ground-truth x_1 (normalised future trajectories
    from 'pred_norm' in the batch).

    Training
    --------
        loss, loss_reg, loss_cls = flow_matcher(past_traj, pred_norm)

    pred_norm is 'pred_norm' directly from the batch — already in [-1, 1]
    via the shared TrajectoryNormaliser in the pipeline.

    Sampling (teacher inference)
    ----------------------------
        preds_norm = flow_matcher.sample(past_traj, K, steps)
        # [K, N, T, 2] in the normalised frame

    The caller (MoFlowTrainer) denormalises before metric computation.
    """

    def __init__(
        self,
        model:           ETHMotionTransformer,
        K:               int   = 20,
        pred_len:        int   = 12,
        logit_norm_mean: float = -0.5,
        logit_norm_std:  float = 1.5,
        tied_noise:      bool  = True,
        fm_in_scaling:   bool  = True,
        loss_nn_mode:    str   = 'scene',
        loss_weight_reg: float = 1.0,
        loss_weight_cls: float = 1.0,
    ):
        super().__init__()
        self.model           = model
        self.K               = K
        self.pred_len        = pred_len
        self.out_dim         = pred_len * 2
        self.logit_norm_mean = logit_norm_mean
        self.logit_norm_std  = logit_norm_std
        self.tied_noise      = tied_noise
        self.fm_in_scaling   = fm_in_scaling
        self.loss_nn_mode    = loss_nn_mode
        self.w_reg           = loss_weight_reg
        self.w_cls           = loss_weight_cls

    def _sample_t(self, device: torch.device) -> torch.Tensor:
        """Sample one flow time from logit-normal distribution → [1]."""
        kappa = (
            torch.randn(1, device=device) * self.logit_norm_std
            + self.logit_norm_mean
        )
        return torch.sigmoid(kappa)

    def _input_scaling(self, t: torch.Tensor) -> torch.Tensor:
        """1 / sqrt(t² + (1−t)²) — keeps noisy input variance near 1."""
        return 1.0 / (t.pow(2) + (1 - t).pow(2)).sqrt().clamp(min=1e-4)

    def _velocity_wrapper(
        self, y_t: torch.Tensor, t: torch.Tensor, f_out: torch.Tensor
    ) -> torch.Tensor:
        """D_θ = y_t + (1 − t) · F_θ  →  predicts x_1."""
        return y_t + (1 - t) * f_out

    def _vel_from_data(
        self, x1: torch.Tensor, xt: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """v = (x_1 − x_t) / (1 − t)."""
        return (x1 - xt) / (1 - t).clamp(min=1e-6)

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def forward(
        self,
        past_traj: torch.Tensor,   # [N, T_o, 6]  — from _build_context
        pred_norm: torch.Tensor,   # [N, T,   2]  — 'pred_norm' from batch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute flow matching loss for one scene.
        Returns (loss, loss_reg, loss_cls) — all scalar tensors.

        pred_norm comes directly from batch['pred_norm'] which is produced
        by the shared TrajectoryNormaliser attached to the dataset.  No
        normalisation happens here.
        """
        K      = self.K
        N, T   = pred_norm.shape[0], self.pred_len
        device = pred_norm.device

        # Ground-truth in flattened form, repeated K times
        x1 = pred_norm.reshape(N, T * 2).unsqueeze(0).expand(K, -1, -1)  # [K, N, T*2]

        t = self._sample_t(device)                                         # [1]

        # Tied noise: one noise vector shared across all K components
        if self.tied_noise:
            noise = torch.randn(1, N, T * 2, device=device).expand(K, -1, -1)
        else:
            noise = torch.randn(K, N, T * 2, device=device)

        # Linear interpolation: y_t = t·x_1 + (1−t)·noise
        y_t  = t * x1 + (1 - t) * noise                                  # [K, N, T*2]
        y_in = y_t * self._input_scaling(t) if self.fm_in_scaling else y_t

        f_out, logits = self.model(y_in, t, past_traj)                    # [K,N,T*2], [K,N]
        pred_x1       = self._velocity_wrapper(y_t, t, f_out)             # [K, N, T*2]

        # Per-component error: [K, N, T]  →  mean over T  →  [K, N]
        error = (
            (pred_x1 - x1).reshape(K, N, T, 2).norm(dim=-1).mean(dim=-1)
        )                                                                  # [K, N]

        if self.loss_nn_mode == 'scene':
            # Scene-level: mean over agents, then argmin over K
            err_scene = error.mean(dim=-1)                                 # [K]
            best_k    = err_scene.argmin()
            loss_reg  = err_scene[best_k]
            loss_cls  = F.cross_entropy(
                logits.mean(dim=-1).unsqueeze(0),
                best_k.unsqueeze(0),
            )

        elif self.loss_nn_mode == 'agent':
            best_k_a = error.argmin(dim=0)                                 # [N]
            loss_reg = error.gather(0, best_k_a.unsqueeze(0)).squeeze(0).mean()
            loss_cls = F.cross_entropy(
                rearrange(logits, 'k n -> n k'), best_k_a
            )

        else:
            raise ValueError(f"Unknown loss_nn_mode: '{self.loss_nn_mode}'")

        loss = self.w_reg * loss_reg # + self.w_cls * loss_cls
        return loss, loss_reg.detach(), loss_cls.detach()

    # ------------------------------------------------------------------
    # ODE sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        past_traj:          torch.Tensor,
        K:                  int | None = None,
        steps:              int        = 100,
        solver:             str        = 'lin_poly',
        lin_poly_p:         int        = 5,
        lin_poly_long_step: int        = 1000,
    ) -> torch.Tensor:
        """
        Run ODE to generate K joint predictions in the normalised frame.

        Returns [K, N, T, 2].  The caller must denormalise using the same
        TrajectoryNormaliser that was used during training before computing
        metrics.
        """
        K      = K or self.K
        N      = past_traj.shape[0]
        T      = self.pred_len
        device = past_traj.device

        if self.tied_noise:
            y_t = torch.randn(1, N, T * 2, device=device).expand(K, -1, -1).clone()
        else:
            y_t = torch.randn(K, N, T * 2, device=device)

        # Build time schedule
        if solver == 'euler':
            dt   = 1.0 / steps
            t_ls = [dt * i for i in range(steps)]
            dt_ls = [dt] * steps

        elif solver == 'lin_poly':
            n_lin  = steps // 2
            n_poly = steps - n_lin
            dt_lin = 1.0 / lin_poly_long_step
            t_lin  = [dt_lin * i for i in range(n_lin)]

            def poly_pts(a, b, n, p):
                return [a + (b - a) * (i ** p) / (n ** p) for i in range(n + 1)]

            t_poly_start = t_lin[-1] + dt_lin
            t_poly       = poly_pts(t_poly_start, 1.0, n_poly, lin_poly_p)
            dt_poly      = [t_poly[i + 1] - t_poly[i] for i in range(n_poly)]

            t_ls  = t_lin + t_poly[:-1]
            dt_ls = [dt_lin] * n_lin + dt_poly
        else:
            raise ValueError(f"Unknown solver: '{solver}'")

        for cur_t, cur_dt in zip(t_ls, dt_ls):
            t_tensor = torch.tensor([cur_t], device=device, dtype=torch.float32)
            y_in     = y_t * self._input_scaling(t_tensor) if self.fm_in_scaling else y_t
            f_out, _ = self.model(y_in, t_tensor, past_traj)
            pred_x1  = self._velocity_wrapper(y_t, t_tensor, f_out)
            velocity = self._vel_from_data(pred_x1, y_t, t_tensor)
            y_t      = y_t + velocity * cur_dt

        return y_t.reshape(K, N, T, 2)


# ---------------------------------------------------------------------------
# IMLE wrapper
# ---------------------------------------------------------------------------

class IMLE(nn.Module):
    """
    Wraps ETHIMLETransformer with the IMLE Chamfer-distance training objective.

    During training, generates M candidate K-prediction sets per scene.
    The nearest candidate (by Chamfer distance to the teacher sample) is
    selected; only that candidate contributes to the gradient.

    Training:
        loss, loss_chamfer, loss_gt = imle(past_traj, pred_norm, teacher_samples, M)

    Inference (one-step):
        preds_norm = imle(past_traj, pred_norm=None, teacher_samples=None, M=1)
        # [K, N, T, 2]  —  normalised frame
    """

    def __init__(
        self,
        model:          ETHIMLETransformer,
        K:              int   = 20,
        pred_len:       int   = 12,
        chamfer_weight: float = 1.0,
        gt_weight:      float = 0.0,
        loss_reduction: str   = 'mean',
    ):
        super().__init__()
        self.model          = model
        self.K              = K
        self.pred_len       = pred_len
        self.out_dim        = pred_len * 2
        self.w_chamfer      = chamfer_weight
        self.w_gt           = gt_weight
        self.loss_reduction = loss_reduction

    def _chamfer(
        self,
        gen:    torch.Tensor,   # [M, K, N, T, 2]
        target: torch.Tensor,   # [K, N, T, 2]
    ) -> torch.Tensor:
        """
        Per-M Chamfer distance between generated and teacher trajectory sets.
        Reduction over T first, then pairwise over K, then mean over N → [M].
        """
        if self.loss_reduction == 'mean':
            gen_r    = gen.mean(dim=-2)     # [M, K, N, 2]
            target_r = target.mean(dim=-2)  # [K, N, 2]
        else:
            gen_r    = gen.sum(dim=-2)
            target_r = target.sum(dim=-2)

        # Pairwise distances per agent: [M, N, K1, K2]
        gen_a    = gen_r.permute(0, 2, 1, 3)    # [M, N, K, 2]
        target_a = target_r.permute(1, 0, 2)    # [N, K, 2]

        diff      = gen_a.unsqueeze(3) - target_a[None, :, None, :]
        pair_dist = diff.norm(dim=-1)            # [M, N, K1, K2]

        c_fwd = pair_dist.min(dim=-1)[0].mean(dim=-1)   # [M, N]
        c_bwd = pair_dist.min(dim=-2)[0].mean(dim=-1)   # [M, N]
        return (c_fwd + c_bwd).mean(dim=-1)              # [M]

    def forward(
        self,
        past_traj:       torch.Tensor,
        pred_norm:       torch.Tensor | None,
        teacher_samples: torch.Tensor | None,
        M:               int = 1,
    ) -> torch.Tensor | tuple:
        """
        Training  : returns (loss, loss_chamfer, loss_gt)
                    when teacher_samples is provided.
        Inference : returns FloatTensor [K, N, T, 2]
                    when teacher_samples is None.

        pred_norm comes from batch['pred_norm'] — already normalised by the
        shared pipeline.  No normalisation happens inside this method.
        """
        K      = self.K
        N      = past_traj.shape[0]
        T      = self.pred_len
        device = past_traj.device

        gen_flat = self.model(past_traj, M=M)              # [M, K, N, T*2]
        gen      = gen_flat.reshape(M, K, N, T, 2)

        if teacher_samples is None:
            assert M == 1
            return gen.squeeze(0)                          # [K, N, T, 2]

        # --- Chamfer loss against teacher samples ---
        loss_chamfer = torch.tensor(0.0, device=device)
        if self.w_chamfer > 0:
            chamfer_m    = self._chamfer(gen, teacher_samples)   # [M]
            best_m       = chamfer_m.argmin()
            loss_chamfer = chamfer_m[best_m] * self.w_chamfer

        # --- Optional GT supervision ---
        loss_gt = torch.tensor(0.0, device=device)
        if self.w_gt > 0 and pred_norm is not None:
            gt_exp  = pred_norm[None, None]                # [1, 1, N, T, 2]
            gt_dist = (gen - gt_exp).norm(dim=-1)          # [M, K, N, T]
            if self.loss_reduction == 'mean':
                gt_dist = gt_dist.mean(dim=-1)
            else:
                gt_dist = gt_dist.sum(dim=-1)
            best_per_m = gt_dist.min(dim=1)[0].mean(dim=-1)  # [M]
            best_m_gt  = best_per_m.argmin()
            loss_gt    = best_per_m[best_m_gt] * self.w_gt

        loss = loss_chamfer + loss_gt
        return loss, loss_chamfer.detach(), loss_gt.detach()
