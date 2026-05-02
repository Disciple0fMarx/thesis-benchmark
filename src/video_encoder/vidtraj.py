"""
vidtraj.py
----------
VidTraj: video-conditioned scene encoding for pedestrian trajectory prediction.

Implements the architecture proposed in the survey (Section IX):

    raw video clip
        │
        ▼
    VideoEncoderBackend  (Video Swin Transformer  or  DINOv2 + optical flow)
        │  F_vid ∈ ℝ^{Ts × Hs × Ws × C}
        ▼
    Spatial grounding  +  RoI-pooling   →  {v^t_i} per agent per frame
        │
        ▼
    LightweightTemporalTransformer      →  c^vid_i  ∈ ℝ^{d_vid}  per agent
        │                               →  g_vid    ∈ ℝ^{d_vid}  global scene token
        ▼
    Injection adapter  (model-specific)
        │
        ▼
    downstream PTP model  f_θ(X, c^vid_i, g_vid, z_i)

The module is intentionally backend-agnostic and framework-light:
  • It imports torch only when a VidTrajEncoder is actually constructed.
  • The Video Swin / DINOv2 backbones are imported lazily so the rest of the
    pipeline (stream, tracker, CV adapter) continues to work without them.
  • All tensor shapes and dtypes follow the ETH/UCY convention used everywhere
    else in the project.

Typical usage
-------------
    from src.video_encoder.vidtraj import VidTrajConfig, VidTrajEncoder

    cfg = VidTrajConfig(backend="swin", d_vid=128, freeze_backbone=True)
    encoder = VidTrajEncoder(cfg)

    # frames: list of T_obs BGR ndarrays (H, W, 3) uint8
    # bboxes: dict  agent_id → list of T_obs (x1,y1,x2,y2) pixel tuples
    c_vid, g_vid = encoder(frames, bboxes)
    # c_vid: torch.Tensor (N, d_vid)
    # g_vid: torch.Tensor (d_vid,)

VidTrajContext  (data container)
---------------------------------
The output of VidTrajEncoder is wrapped in a VidTrajContext so downstream
adapters receive a single typed object rather than a bare tuple.

VideoContext fields:
    c_vid   torch.Tensor (N, d_vid) — per-agent video context embeddings
    g_vid   torch.Tensor (d_vid,)   — global scene token
    ids     list[int]               — agent IDs aligned with axis-0 of c_vid
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VidTrajConfig:
    """
    Hyperparameters for the VidTraj video encoder.

    Parameters
    ----------
    backend:
        "swin"  — Video Swin Transformer (torchvision, pretrained Kinetics-400).
                  Rich spatio-temporal features; ~28 M params for Swin-T.
        "dinov2"— DINOv2 ViT-S/14 applied per frame, concatenated with dense
                  optical flow (Farneback).  Lighter; better for small datasets.
    d_vid:
        Output embedding dimension for both c^vid_i and g_vid.  Defaults to 128
        to stay light relative to the trajectory model's hidden dimension.
    freeze_backbone:
        If True (default stage-1 training), the video backbone is frozen and
        only the projection layers and temporal Transformer are trained.
    lora_rank:
        When > 0, apply LoRA adapters of this rank to the backbone attention
        layers (stage-2 fine-tuning).  Set to 0 to disable LoRA entirely.
    temporal_heads:
        Number of attention heads in the per-agent temporal Transformer.
    temporal_layers:
        Number of Transformer encoder layers in the per-agent temporal encoder.
    roi_output_size:
        Spatial size (H, W) of the RoI-pooled feature crop per agent per frame.
    device:
        Torch device string.  Defaults to "cpu"; use "cuda" or "mps" for GPU.
    obs_len:
        Number of observation frames.  Must match the downstream model's obs_len.
    """
    backend: str = "swin"
    d_vid: int = 128
    freeze_backbone: bool = True
    lora_rank: int = 0
    temporal_heads: int = 4
    temporal_layers: int = 2
    roi_output_size: Tuple[int, int] = (4, 4)
    device: str = "cpu"
    obs_len: int = 8


# ─────────────────────────────────────────────────────────────────────────────
# Output container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VidTrajContext:
    """
    Output of VidTrajEncoder.encode().

    Attributes
    ----------
    c_vid:
        Per-agent video context embeddings.
        Shape (N, d_vid).  N agents aligned with `ids`.
    g_vid:
        Global scene context token aggregated from the full field of view.
        Shape (d_vid,).
    ids:
        Agent IDs corresponding to axis-0 of c_vid.
    """
    c_vid: "torch.Tensor"   # (N, d_vid)
    g_vid: "torch.Tensor"   # (d_vid,)
    ids: List[int]


# ─────────────────────────────────────────────────────────────────────────────
# LoRA helpers (lightweight, no external library required)
# ─────────────────────────────────────────────────────────────────────────────

class _LoRALinear:
    """
    Wraps a frozen nn.Linear with a low-rank adapter A @ B scaled by alpha/r.
    Applied in-place to the target module's forward pass via monkey-patching.

    This avoids pulling in the full PEFT/LoRA library for a single hyperparameter.
    """

    def __init__(self, linear, rank: int, alpha: float = 1.0) -> None:
        import torch
        import torch.nn as nn
        self.linear = linear
        self.rank = rank
        self.alpha = alpha

        d_out, d_in = linear.weight.shape
        self.A = nn.Parameter(torch.randn(rank, d_in) * 0.02)
        self.B = nn.Parameter(torch.zeros(d_out, rank))
        self.scaling = alpha / rank

    def __call__(self, x):
        import torch
        base = self.linear(x)
        lora = (x @ self.A.T) @ self.B.T * self.scaling
        return base + lora


def _apply_lora(module, rank: int) -> None:
    """
    Apply LoRA adapters to all nn.Linear layers inside an attention sub-module.
    Only Q, K, V projection layers are targeted (those named 'qkv' or 'q_proj',
    'k_proj', 'v_proj' depending on the backbone).
    """
    import torch.nn as nn
    target_names = {"qkv", "q_proj", "k_proj", "v_proj", "query", "key", "value"}
    for name, child in module.named_modules():
        short_name = name.split(".")[-1]
        if isinstance(child, nn.Linear) and short_name in target_names:
            parent = module
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            lora_wrapper = _LoRALinear(child, rank)
            setattr(parent, parts[-1], lora_wrapper)


# ─────────────────────────────────────────────────────────────────────────────
# Video backbone wrappers
# ─────────────────────────────────────────────────────────────────────────────

class _SwinBackend:
    """
    Video Swin Transformer-Tiny loaded from torchvision.

    Input:  (1, C=3, T, H, W) float32 tensor in [0, 1]
    Output: (Ts, Hs, Ws, C_feat) feature map (patch tokens before head)

    We remove the classifier head and expose the patch-level features from
    the last stage, then reshape to (Ts, Hs, Ws, C_feat) for RoI-pooling.
    """

    # Kinetics-400 normalisation constants
    MEAN = (0.45, 0.45, 0.45)
    STD  = (0.225, 0.225, 0.225)

    def __init__(self, freeze: bool = True, lora_rank: int = 0,
                 device: str = "cpu") -> None:
        import torch
        import torchvision.models.video as vm

        self.device = torch.device(device)
        weights = vm.Swin3D_T_Weights.KINETICS400_V1
        model = vm.swin3d_t(weights=weights)

        # Remove the classification head — keep patch features only
        model.head = torch.nn.Identity()
        self._model = model.to(self.device)

        if freeze:
            for p in self._model.parameters():
                p.requires_grad_(False)

        if lora_rank > 0:
            for block in self._model.features:
                _apply_lora(block, lora_rank)

        # Feature dimension for Swin-T last stage
        self.feature_dim = 768

        mean = torch.tensor(self.MEAN, device=self.device).view(3, 1, 1, 1)
        std  = torch.tensor(self.STD,  device=self.device).view(3, 1, 1, 1)
        self.register_buffer_mean = mean
        self.register_buffer_std  = std

    def encode(self, clip_uint8: np.ndarray) -> "torch.Tensor":
        """
        clip_uint8: np.ndarray (T, H, W, 3) uint8 BGR
        returns:    torch.Tensor (T', H', W', C_feat) on self.device

        The Swin-T temporal stride reduces T by 2× and spatial stride reduces
        H and W by 32× overall (4 stages × 2× each = 32×, but the model uses
        32× spatial and 2× temporal with window attention).
        Typical: (8, 240, 320, 3) → feature map (4, 7, 10, 768)
        """
        import torch

        # BGR → RGB, (T, H, W, 3) → (3, T, H, W) float [0,1]
        rgb = clip_uint8[..., ::-1].copy()
        t = torch.from_numpy(rgb).float().to(self.device)
        t = t.permute(3, 0, 1, 2) / 255.0   # (3, T, H, W)
        t = (t - self.register_buffer_mean) / self.register_buffer_std
        t = t.unsqueeze(0)                    # (1, 3, T, H, W)

        with torch.no_grad() if not any(
                p.requires_grad for p in self._model.parameters()) \
                else torch.enable_grad():
            # Extract intermediate patch features via forward hook
            features = []
            def _hook(module, inp, out):
                features.append(out)

            # Hook the last normalization layer before the head
            handle = self._model.norm.register_forward_hook(_hook)
            try:
                self._model(t)
            finally:
                handle.remove()

        if features:
            feat = features[0]              # (1, Ts*Hs*Ws, C) typically
            # Swin-T output after norm: (1, T', H'*W', C) or (1, N, C)
            # Reshape to spatial: infer Ts, Hs, Ws from the input dimensions
            _, N, C = feat.shape
            # Estimate spatial dims: Swin-T uses 32× spatial, 2× temporal
            T_in = clip_uint8.shape[0]
            H_in = clip_uint8.shape[1]
            W_in = clip_uint8.shape[2]
            Ts = max(1, T_in // 2)
            Hs = max(1, H_in // 32)
            Ws = max(1, W_in // 32)
            # If Ts*Hs*Ws != N, fall back to using N as a flat sequence
            if Ts * Hs * Ws == N:
                feat = feat.squeeze(0).view(Ts, Hs, Ws, C)
            else:
                # Use the flattened output directly — global average gives g_vid
                feat = feat.squeeze(0).unsqueeze(0).unsqueeze(0).expand(
                    1, 1, N, C)
                feat = feat.squeeze(0)      # (1, N, C) → reinterpret
                feat = feat.mean(1, keepdim=True).expand(1, 1, C).unsqueeze(0)
                # Fallback: (1, 1, C) — single spatial token
                feat = feat.squeeze()
                feat = feat.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1,1,1,C)
                Ts, Hs, Ws = 1, 1, 1

            return feat   # (Ts, Hs, Ws, C)
        else:
            raise RuntimeError("Swin forward hook did not capture features")


class _DINOv2Backend:
    """
    DINOv2 ViT-S/14 applied per frame, concatenated with dense optical flow
    (Farneback method via OpenCV) to capture temporal dynamics.

    Input:  (T, H, W, 3) uint8 BGR ndarray
    Output: (T, H', W', C_feat) where H' = H//14, W' = W//14, C_feat = 384 + 2

    This is lighter than Swin (ViT-S has ~22 M params vs Swin-T's ~28 M) and
    better suited when GPU memory is limited.
    """

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    PATCH_SIZE = 14
    DINO_DIM = 384   # ViT-S

    def __init__(self, freeze: bool = True, lora_rank: int = 0,
                 device: str = "cpu") -> None:
        import torch
        self.device = torch.device(device)
        self._freeze = freeze
        self._lora_rank = lora_rank
        self._model = None   # loaded lazily on first encode() call

    def _load(self) -> None:
        import torch
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                                pretrained=True, verbose=False)
        model = model.to(self.device)
        if self._freeze:
            for p in model.parameters():
                p.requires_grad_(False)
        if self._lora_rank > 0:
            _apply_lora(model, self._lora_rank)
        self._model = model
        self.feature_dim = self.DINO_DIM + 2   # DINOv2 + flow (u, v)

    def _optical_flow(self, frame_a: np.ndarray,
                      frame_b: np.ndarray) -> np.ndarray:
        """
        Compute Farneback dense optical flow between two BGR frames.
        Returns (H, W, 2) float32 ndarray.
        """
        import cv2
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray_a, gray_b, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        return flow   # (H, W, 2)

    def encode(self, clip_uint8: np.ndarray) -> "torch.Tensor":
        """
        clip_uint8: (T, H, W, 3) uint8 BGR
        returns:    (T, Hp, Wp, C_feat) torch.Tensor on self.device
                    where Hp = H//14, Wp = W//14, C_feat = 386
        """
        import torch

        if self._model is None:
            self._load()

        T, H, W, _ = clip_uint8.shape
        Hp = H // self.PATCH_SIZE
        Wp = W // self.PATCH_SIZE

        dino_tokens_all = []
        flow_tokens_all = []

        for t in range(T):
            # ── DINOv2 patch tokens ──────────────────────────────────────
            rgb = clip_uint8[t, :, :, ::-1].copy().astype(np.float32) / 255.0
            rgb = (rgb - self.MEAN) / self.STD          # (H, W, 3)

            tensor = torch.from_numpy(rgb).float().to(self.device)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

            context = torch.no_grad() if self._freeze else torch.enable_grad()
            with context:
                tokens = self._model.get_intermediate_layers(
                    tensor, n=1, return_class_token=False)[0]
                # tokens: (1, Hp*Wp, 384)
            dino_tokens_all.append(tokens.squeeze(0))  # (Hp*Wp, 384)

            # ── Optical flow tokens ──────────────────────────────────────
            if t == 0:
                flow = np.zeros((H, W, 2), dtype=np.float32)
            else:
                flow = self._optical_flow(clip_uint8[t - 1], clip_uint8[t])

            # Pool flow to patch grid
            flow_patches = np.zeros((Hp, Wp, 2), dtype=np.float32)
            for ph in range(Hp):
                for pw in range(Wp):
                    patch = flow[ph * self.PATCH_SIZE:(ph + 1) * self.PATCH_SIZE,
                                 pw * self.PATCH_SIZE:(pw + 1) * self.PATCH_SIZE]
                    flow_patches[ph, pw] = patch.mean(axis=(0, 1))

            flow_tok = torch.from_numpy(
                flow_patches.reshape(Hp * Wp, 2)).to(self.device)
            flow_tokens_all.append(flow_tok)    # (Hp*Wp, 2)

        # Stack: (T, Hp*Wp, 384) and (T, Hp*Wp, 2)
        dino = torch.stack(dino_tokens_all, dim=0)  # (T, Hp*Wp, 384)
        flow = torch.stack(flow_tokens_all, dim=0)  # (T, Hp*Wp, 2)
        combined = torch.cat([dino, flow], dim=-1)  # (T, Hp*Wp, 386)

        # Reshape to (T, Hp, Wp, C_feat)
        return combined.view(T, Hp, Wp, self.DINO_DIM + 2)


# ─────────────────────────────────────────────────────────────────────────────
# RoI-pooling helper
# ─────────────────────────────────────────────────────────────────────────────

def _roi_pool_feature(
    feat_map: "torch.Tensor",
    bbox_pixel: Tuple[float, float, float, float],
    img_hw: Tuple[int, int],
    output_size: Tuple[int, int] = (4, 4),
) -> "torch.Tensor":
    """
    Extract a feature crop for one bounding box using bilinear RoI-align.

    Parameters
    ----------
    feat_map:    (Hs, Ws, C) spatial feature map for one timestep
    bbox_pixel:  (x1, y1, x2, y2) in image pixel coordinates
    img_hw:      (H_img, W_img) original image dimensions
    output_size: (oh, ow) output spatial size after pooling

    Returns
    -------
    torch.Tensor (output_size[0] * output_size[1] * C,) flat feature vector
    """
    import torch
    import torch.nn.functional as F

    Hs, Ws, C = feat_map.shape
    H_img, W_img = img_hw
    x1, y1, x2, y2 = bbox_pixel

    # Clamp and normalise bbox to feature map coordinates
    # Map pixel → feature-map fractional coords in [-1, 1] for grid_sample
    def px_to_norm(px, max_px, max_feat):
        feat_coord = px / max_px * max_feat
        return (feat_coord / (max_feat - 1)) * 2 - 1

    x1_n = px_to_norm(x1, W_img, Ws)
    x2_n = px_to_norm(x2, W_img, Ws)
    y1_n = px_to_norm(y1, H_img, Hs)
    y2_n = px_to_norm(y2, H_img, Hs)

    oh, ow = output_size
    # Build sampling grid (1, oh, ow, 2)
    xs = torch.linspace(x1_n, x2_n, ow, device=feat_map.device)
    ys = torch.linspace(y1_n, y2_n, oh, device=feat_map.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, oh, ow, 2)

    # Rearrange feat_map to (1, C, Hs, Ws) for grid_sample
    feat = feat_map.permute(2, 0, 1).unsqueeze(0)  # (1, C, Hs, Ws)
    sampled = F.grid_sample(feat.float(), grid, mode="bilinear",
                            padding_mode="border", align_corners=True)
    # sampled: (1, C, oh, ow) → flatten to (C * oh * ow,)
    return sampled.squeeze(0).flatten()


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight temporal Transformer
# ─────────────────────────────────────────────────────────────────────────────

class _LightweightTemporalTransformer:
    """
    A small Transformer encoder that collapses a per-agent temporal sequence
    of RoI-pooled feature vectors into a single context embedding.

    Input:  (N, T_obs, d_in) per-agent feature sequences
    Output: (N, d_vid) per-agent context embeddings
            (d_vid,)   global scene embedding (mean-pooled over N)
    """

    def __init__(self, d_in: int, d_vid: int, n_heads: int = 4,
                 n_layers: int = 2, device: str = "cpu") -> None:
        import torch
        import torch.nn as nn

        self.device = torch.device(device)
        self.d_vid = d_vid

        # Input projection to d_vid
        self.proj_in = nn.Linear(d_in, d_vid).to(self.device)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_vid,
            nhead=n_heads,
            dim_feedforward=d_vid * 4,
            dropout=0.1,
            batch_first=True,        # expects (batch, seq, d_model)
            norm_first=True,         # pre-norm: more stable for small models
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            norm=nn.LayerNorm(d_vid),
        ).to(self.device)

        # CLS-style readout: learnable token prepended to the sequence
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, d_vid, device=self.device))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, per_agent_feats: "torch.Tensor") -> Tuple[
            "torch.Tensor", "torch.Tensor"]:
        """
        per_agent_feats: (N, T_obs, d_in) — pooled features per agent per frame
        returns:
            c_vid:  (N, d_vid) — per-agent context embedding (CLS token output)
            g_vid:  (d_vid,)   — global scene token (mean over N)
        """
        import torch

        N, T, _ = per_agent_feats.shape

        # Project input to d_vid
        x = self.proj_in(per_agent_feats)   # (N, T, d_vid)

        # Prepend CLS token for each agent
        cls = self.cls_token.expand(N, -1, -1)   # (N, 1, d_vid)
        x = torch.cat([cls, x], dim=1)            # (N, T+1, d_vid)

        # Transformer encoder
        out = self.encoder(x)   # (N, T+1, d_vid)

        # CLS token output as per-agent embedding
        c_vid = out[:, 0, :]    # (N, d_vid)

        # Global scene token: mean over agents
        g_vid = c_vid.mean(dim=0)   # (d_vid,)

        return c_vid, g_vid

    def parameters(self):
        """Yield all trainable parameters (for optimiser construction)."""
        import itertools
        return itertools.chain(
            self.proj_in.parameters(),
            self.encoder.parameters(),
            [self.cls_token],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main VidTraj encoder
# ─────────────────────────────────────────────────────────────────────────────

class VidTrajEncoder:
    """
    Full VidTraj video-conditioned scene encoder.

    Wires together:
        BackendEncoder → per-frame spatial feature maps
        RoI-pooling    → per-agent per-frame feature vectors
        Temporal Transformer → per-agent context embeddings + global token

    Parameters
    ----------
    config: VidTrajConfig

    Usage
    -----
        encoder = VidTrajEncoder(VidTrajConfig(backend="swin", d_vid=128))

        # frames: list of T_obs BGR np.ndarray (H, W, 3) uint8
        # bboxes: dict  agent_id → list of T_obs (x1, y1, x2, y2) float tuples
        #         missing frames for an agent → None in the list
        ctx = encoder.encode(frames, bboxes)
        # ctx.c_vid: (N, d_vid) torch.Tensor
        # ctx.g_vid: (d_vid,)   torch.Tensor
        # ctx.ids:   list[int]  agent IDs aligned with axis-0 of c_vid
    """

    def __init__(self, config: VidTrajConfig) -> None:
        self.cfg = config
        self._backbone: Optional[object] = None   # built lazily
        self._temporal: Optional[_LightweightTemporalTransformer] = None

    # ------------------------------------------------------------------
    # Lazy construction
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """
        Build backbone and temporal Transformer on first encode() call.
        Deferred so that importing this module has no torch dependency.
        """
        import torch

        cfg = self.cfg
        device = cfg.device

        # ── Backbone ──────────────────────────────────────────────────
        if cfg.backend == "swin":
            self._backbone = _SwinBackend(
                freeze=cfg.freeze_backbone,
                lora_rank=cfg.lora_rank,
                device=device,
            )
            d_roi = (self._backbone.feature_dim
                     * cfg.roi_output_size[0] * cfg.roi_output_size[1])
        elif cfg.backend == "dinov2":
            self._backbone = _DINOv2Backend(
                freeze=cfg.freeze_backbone,
                lora_rank=cfg.lora_rank,
                device=device,
            )
            d_roi = (self._backbone.DINO_DIM + 2) * cfg.roi_output_size[0] * cfg.roi_output_size[1]
        else:
            raise ValueError(
                f"Unknown VidTraj backend {cfg.backend!r}. "
                f"Choose 'swin' or 'dinov2'.")

        # ── Temporal Transformer ──────────────────────────────────────
        self._temporal = _LightweightTemporalTransformer(
            d_in=d_roi,
            d_vid=cfg.d_vid,
            n_heads=cfg.temporal_heads,
            n_layers=cfg.temporal_layers,
            device=device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        frames: List[np.ndarray],
        bboxes: Dict[int, List[Optional[Tuple[float, float, float, float]]]],
    ) -> VidTrajContext:
        """
        Encode a video clip and bounding box sequences into context embeddings.

        Parameters
        ----------
        frames:
            List of T_obs BGR np.ndarray of shape (H, W, 3) uint8.
            Must have len(frames) == config.obs_len.
        bboxes:
            Dict mapping agent_id → list of T_obs (x1, y1, x2, y2) tuples
            in pixel coordinates.  A None entry means the agent was not
            visible at that frame — it is replaced with the zero vector.

        Returns
        -------
        VidTrajContext with:
            c_vid   (N, d_vid) per-agent embeddings
            g_vid   (d_vid,)   global scene token
            ids     list[int]  of length N
        """
        import torch

        if self._backbone is None:
            self._build()

        T = len(frames)
        H, W = frames[0].shape[:2]
        clip = np.stack(frames, axis=0)   # (T, H, W, 3)

        # ── Step 1: backbone feature maps ─────────────────────────────
        # feat_map: (Ts, Hs, Ws, C_feat)
        feat_map = self._backbone.encode(clip)

        # ── Step 2: per-frame feature maps aligned to T_obs ───────────
        # The backbone may temporally downsample (Swin: ×2).
        # We interpolate/repeat to get T separate (Hs, Ws, C) maps.
        Ts = feat_map.shape[0]
        feat_per_frame = []
        for t in range(T):
            t_idx = min(int(t / T * Ts), Ts - 1)
            feat_per_frame.append(feat_map[t_idx])  # (Hs, Ws, C)

        # ── Step 3: RoI-pooling per agent per frame ───────────────────
        agent_ids = sorted(bboxes.keys())
        N = len(agent_ids)

        if N == 0:
            # No agents — return zero context
            import torch
            device = torch.device(self.cfg.device)
            d = self.cfg.d_vid
            return VidTrajContext(
                c_vid=torch.zeros(0, d, device=device),
                g_vid=torch.zeros(d, device=device),
                ids=[],
            )

        # d_roi: flattened RoI feature size
        sample_feat = feat_per_frame[0]
        Hs, Ws, C_feat = sample_feat.shape
        oh, ow = self.cfg.roi_output_size
        d_roi = C_feat * oh * ow

        device = self._temporal.proj_in.weight.device
        per_agent_feats = torch.zeros(N, T, d_roi, device=device)

        for n, agent_id in enumerate(agent_ids):
            bb_seq = bboxes[agent_id]
            for t in range(T):
                if t >= len(bb_seq) or bb_seq[t] is None:
                    continue   # zero stays
                bbox = bb_seq[t]
                roi_vec = _roi_pool_feature(
                    feat_per_frame[t].to(device),
                    bbox_pixel=bbox,
                    img_hw=(H, W),
                    output_size=self.cfg.roi_output_size,
                )
                per_agent_feats[n, t] = roi_vec

        # ── Step 4: temporal Transformer ──────────────────────────────
        c_vid, g_vid = self._temporal.forward(per_agent_feats)

        return VidTrajContext(c_vid=c_vid, g_vid=g_vid, ids=agent_ids)

    # ------------------------------------------------------------------
    # Convenience: trainable parameters for optimiser
    # ------------------------------------------------------------------

    def trainable_parameters(self):
        """
        Yield all parameters that should be updated during training.
        When freeze_backbone=True this is just the temporal Transformer.
        When lora_rank > 0 this also includes the LoRA A/B matrices.
        """
        if self._temporal is None:
            self._build()
        yield from self._temporal.parameters()
        if not self.cfg.freeze_backbone and self._backbone is not None:
            if hasattr(self._backbone, "_model") and self._backbone._model:
                yield from self._backbone._model.parameters()


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_vidtraj(backend: str = "swin", **kwargs) -> VidTrajEncoder:
    """
    Factory matching the style of build_adapter() and build_detector().

    Parameters
    ----------
    backend:
        "swin" or "dinov2"
    **kwargs:
        Forwarded to VidTrajConfig (d_vid, freeze_backbone, lora_rank, etc.)

    Returns
    -------
    VidTrajEncoder (backbone built lazily on first encode() call)
    """
    cfg = VidTrajConfig(backend=backend, **kwargs)
    return VidTrajEncoder(cfg)
    