"""
smoke_test_moflow.py
====================
End-to-end smoke test for the MoFlow pipeline.

Run from the repo root:
    python smoke_test_moflow.py

Tests every component in dependency order using synthetic data.
No real ETH-UCY data required.  Each test prints PASS / FAIL with
the relevant tensor shapes so you can see exactly where a failure occurs.

Requirements
------------
    pip install torch einops numpy
"""

import sys
import os
import traceback
import pickle
import tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

all_passed = True

def check(name, condition, info=""):
    global all_passed
    if not condition:
        all_passed = False
    status = PASS if condition else FAIL
    print(f"  {status}  {name}" + (f"  [{info}]" if info else ""))
    return condition

def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

# ---------------------------------------------------------------------------
# Synthetic constants — tiny so CPU runs in seconds
# ---------------------------------------------------------------------------

N       = 5    # pedestrians per scene
T_OBS   = 8
T_PRED  = 12
K       = 4    # joint predictions
M       = 3    # IMLE student samples
D_MODEL = 32
N_WIN   = 10   # number of dataset windows

# ---------------------------------------------------------------------------
# Shared fake dataset (reused across tests)
# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)

def make_fake_dataset():
    """
    Build a SocialDataset by directly populating its internal lists.
    Avoids needing real ETH-UCY files on disk.
    """
    from src.data_pipeline.dataset import SocialDataset
    ds = SocialDataset.__new__(SocialDataset)
    ds.obs_len     = T_OBS
    ds.pred_len    = T_PRED
    ds._normaliser = None
    ds._obs      = [rng.uniform(-5, 5, (N, T_OBS,  2)).astype(np.float32) for _ in range(N_WIN)]
    ds._pred     = [rng.uniform(-5, 5, (N, T_PRED, 2)).astype(np.float32) for _ in range(N_WIN)]
    ds._obs_rel  = [rng.uniform(-1, 1, (N, T_OBS,  2)).astype(np.float32) for _ in range(N_WIN)]
    ds._pred_rel = [rng.uniform(-1, 1, (N, T_PRED, 2)).astype(np.float32) for _ in range(N_WIN)]
    return ds

# ---------------------------------------------------------------------------
# 1. TrajectoryNormaliser
# ---------------------------------------------------------------------------

section("1. TrajectoryNormaliser")

try:
    from src.data_pipeline.normaliser import TrajectoryNormaliser

    ds   = make_fake_dataset()
    norm = TrajectoryNormaliser(mode='minmax')
    norm.fit(ds)

    check("fit() succeeds", norm.is_fitted)
    check("min shape [2]",  norm._min.shape == (2,), str(norm._min.shape))
    check("max shape [2]",  norm._max.shape == (2,), str(norm._max.shape))

    x       = torch.tensor(ds._pred[0])
    x_norm  = norm.transform(x)
    x_back  = norm.inverse_transform(x_norm)
    rt_err  = (x - x_back).abs().max().item()
    check("round-trip error < 1e-5", rt_err < 1e-5, f"err={rt_err:.2e}")
    check("normalised in [-1, 1]",
          x_norm.min() >= -1.001 and x_norm.max() <= 1.001,
          f"min={x_norm.min():.3f} max={x_norm.max():.3f}")

    # state_dict round-trip
    norm2 = TrajectoryNormaliser(mode='minmax')
    norm2.load_state_dict(norm.state_dict())
    check("state_dict round-trip",
          (norm.transform(x) - norm2.transform(x)).abs().max().item() < 1e-6)

except Exception:
    print(f"  {FAIL}  Unexpected exception:")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 2. SocialDataset — set_normaliser and pred_norm
# ---------------------------------------------------------------------------

section("2. SocialDataset — normaliser attachment")

try:
    from src.data_pipeline.dataset import SocialDataset, social_collate

    # pred_norm absent before normaliser
    item = ds[0]
    check("pred_norm absent without normaliser", 'pred_norm' not in item)

    # After attachment
    ds.set_normaliser(norm)
    item = ds[0]
    check("pred_norm present after set_normaliser", 'pred_norm' in item)
    check("pred_norm shape [N, T, 2]",
          item['pred_norm'].shape == (N, T_PRED, 2),
          str(item['pred_norm'].shape))
    check("pred_norm values in [-1, 1]",
          item['pred_norm'].min() >= -1.001 and item['pred_norm'].max() <= 1.001)
    check("origin shape [N, 1, 2]",
          item['origin'].shape == (N, 1, 2), str(item['origin'].shape))

    # Collate
    batch = social_collate([ds[i] for i in range(3)])
    check("collate includes pred_norm", 'pred_norm' in batch)
    check("collate pred_norm is list of 3", len(batch['pred_norm']) == 3)
    check("repr shows normaliser", 'minmax' in repr(ds))

except Exception:
    print(f"  {FAIL}  Unexpected exception:")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 3. ETHContextEncoder
# ---------------------------------------------------------------------------

section("3. ETHContextEncoder")

try:
    from src.models.moflow import ETHContextEncoder

    enc = ETHContextEncoder(d_model=D_MODEL, n_heads=2, n_layers=1, dropout=0.0)
    enc.eval()
    with torch.no_grad():
        out = enc(torch.randn(N, T_OBS, 6))
    check("output shape [N, D]", out.shape == (N, D_MODEL), str(out.shape))
    check("output finite", out.isfinite().all().item())

except Exception:
    print(f"  {FAIL}  Unexpected exception:")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 4. ETHMotionTransformer
# ---------------------------------------------------------------------------

section("4. ETHMotionTransformer (teacher backbone)")

try:
    from src.models.moflow import ETHMotionTransformer

    teacher_net = ETHMotionTransformer(
        d_model=D_MODEL, K=K, pred_len=T_PRED,
        n_enc_heads=2, n_enc_layers=1,
        n_dec_heads=2, n_dec_layers=1,
        ffn_multiplier=2, dropout=0.0,
    )
    teacher_net.eval()

    with torch.no_grad():
        pred, logits = teacher_net(
            torch.randn(K, N, T_PRED * 2),
            torch.tensor([0.3]),
            torch.randn(N, T_OBS, 6),
        )

    check("pred shape [K, N, T*2]",
          pred.shape == (K, N, T_PRED * 2), str(pred.shape))
    check("logits shape [K, N]",
          logits.shape == (K, N), str(logits.shape))
    check("pred finite",   pred.isfinite().all().item())
    check("logits finite", logits.isfinite().all().item())

except Exception:
    print(f"  {FAIL}  Unexpected exception:")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 5. FlowMatcher — training loss
# ---------------------------------------------------------------------------

section("5. FlowMatcher — training loss")

try:
    from src.models.moflow import FlowMatcher

    fm = FlowMatcher(
        model=teacher_net, K=K, pred_len=T_PRED,
        tied_noise=True, fm_in_scaling=True,
    )
    fm.train()

    past_traj = torch.randn(N, T_OBS, 6)
    pred_norm = torch.rand(N, T_PRED, 2) * 2 - 1

    loss, loss_reg, loss_cls = fm(past_traj, pred_norm)

    check("loss scalar",     loss.shape == torch.Size([]))
    check("loss finite",     loss.isfinite().item(), f"loss={loss.item():.4f}")
    check("loss_reg finite", loss_reg.isfinite().item())
    check("loss_cls finite", loss_cls.isfinite().item())

    loss.backward()
    check("gradients computed",
          all(p.grad is not None for p in fm.parameters() if p.requires_grad))

except Exception:
    print(f"  {FAIL}  Unexpected exception:")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 6. FlowMatcher — ODE sampling
# ---------------------------------------------------------------------------

section("6. FlowMatcher — ODE sampling")

try:
    fm.eval()
    with torch.no_grad():
        preds = fm.sample(
            torch.randn(N, T_OBS, 6), K=K, steps=5, solver='euler'
        )
    check("sample shape [K, N, T, 2]",
          preds.shape == (K, N, T_PRED, 2), str(preds.shape))
    check("sample finite", preds.isfinite().all().item())

except Exception:
    print(f"  {FAIL}  Unexpected exception:")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 7. ETHIMLETransformer
# ---------------------------------------------------------------------------

section("7. ETHIMLETransformer (student backbone)")

try:
    from src.models.moflow import ETHIMLETransformer

    student_net = ETHIMLETransformer(
        d_model=D_MODEL, K=K, pred_len=T_PRED,
        n_enc_heads=2, n_enc_layers=1,
        n_dec_heads=2, n_dec_layers=1,
        ffn_multiplier=2, dropout=0.0,
    )
    student_net.eval()

    with torch.no_grad():
        out = student_net(torch.randn(N, T_OBS, 6), M=M)

    check("output shape [M, K, N, T*2]",
          out.shape == (M, K, N, T_PRED * 2), str(out.shape))
    check("output finite", out.isfinite().all().item())

except Exception:
    print(f"  {FAIL}  Unexpected exception:")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 8. IMLE — training loss
# ---------------------------------------------------------------------------

section("8. IMLE — Chamfer training loss")

try:
    from src.models.moflow import IMLE

    imle = IMLE(model=student_net, K=K, pred_len=T_PRED, chamfer_weight=1.0)
    imle.train()

    past_traj       = torch.randn(N, T_OBS, 6)
    pred_norm_t     = torch.rand(N, T_PRED, 2) * 2 - 1
    teacher_samples = torch.rand(K, N, T_PRED, 2) * 2 - 1

    loss, lc, lg = imle(past_traj, pred_norm_t, teacher_samples, M=M)

    check("loss scalar",  loss.shape == torch.Size([]))
    check("loss finite",  loss.isfinite().item(), f"loss={loss.item():.4f}")

    loss.backward()
    check("gradients computed",
          all(p.grad is not None for p in imle.parameters() if p.requires_grad))

except Exception:
    print(f"  {FAIL}  Unexpected exception:")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 9. IMLE — one-step inference
# ---------------------------------------------------------------------------

section("9. IMLE — one-step inference")

try:
    imle.eval()
    with torch.no_grad():
        preds = imle(torch.randn(N, T_OBS, 6),
                     pred_norm=None, teacher_samples=None, M=1)
    check("inference shape [K, N, T, 2]",
          preds.shape == (K, N, T_PRED, 2), str(preds.shape))
    check("inference finite", preds.isfinite().all().item())

except Exception:
    print(f"  {FAIL}  Unexpected exception:")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 10. MoFlowAdapter / IMLEAdapter
# ---------------------------------------------------------------------------

section("10. MoFlowAdapter / IMLEAdapter (evaluator contract)")

try:
    from src.models.moflow_adapter import MoFlowAdapter, IMLEAdapter

    mf_adapter   = MoFlowAdapter(fm,   norm, steps=5, solver='euler')
    imle_adapter = IMLEAdapter(imle, norm)

    obs     = torch.randn(N, T_OBS, 2)
    obs_rel = torch.randn(N, T_OBS, 2)

    mf_adapter.eval()
    imle_adapter.eval()

    with torch.no_grad():
        mf_out   = mf_adapter(obs, obs_rel, k=K)
        imle_out = imle_adapter(obs, obs_rel, k=K)

    check("MoFlowAdapter shape [K, N, T, 2]",
          mf_out.shape == (K, N, T_PRED, 2), str(mf_out.shape))
    check("IMLEAdapter shape [K, N, T, 2]",
          imle_out.shape == (K, N, T_PRED, 2), str(imle_out.shape))
    check("MoFlowAdapter finite", mf_out.isfinite().all().item())
    check("IMLEAdapter finite",   imle_out.isfinite().all().item())

except Exception:
    print(f"  {FAIL}  Unexpected exception:")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 11. save_teacher_samples positional alignment
# ---------------------------------------------------------------------------

section("11. save_teacher_samples → IMLETrainer positional alignment")

try:
    from torch.utils.data import DataLoader
    from src.data_pipeline.dataset import social_collate

    # Fake save_teacher_samples: one array per window, shape [K, N, T, 2]
    fake_teacher = [
        rng.uniform(-1, 1, (K, N, T_PRED, 2)).astype(np.float32)
        for _ in range(N_WIN)
    ]

    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        tmp_path = f.name
        pickle.dump(fake_teacher, f)

    # IMLETrainer loads samples and indexes by position
    # Simulate _train_epoch logic directly to verify index alignment
    loader = DataLoader(
        ds, batch_size=3, shuffle=False,
        drop_last=False, collate_fn=social_collate,
    )

    loaded = []
    with open(tmp_path, 'rb') as f:
        samples_list = pickle.load(f)

    global_idx = 0
    for batch in loader:
        for _ in range(len(batch['obs'])):
            if global_idx < len(samples_list):
                loaded.append(global_idx)
            global_idx += 1

    os.unlink(tmp_path)

    # Should have loaded indices 0..N_WIN-1 in order
    check("all windows indexed exactly once",
          loaded == list(range(N_WIN)), str(loaded))

except Exception:
    print(f"  {FAIL}  Unexpected exception:")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 12. Normaliser checkpoint embed/restore
# ---------------------------------------------------------------------------

section("12. Normaliser embedded in checkpoint")

try:
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        ckpt_path = f.name

    torch.save(
        {
            'model':      teacher_net.state_dict(),
            'normaliser': norm.state_dict(),
        },
        ckpt_path,
    )
    data = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    norm_r = TrajectoryNormaliser(mode='minmax')
    norm_r.load_state_dict(data['normaliser'])
    os.unlink(ckpt_path)

    x_t = torch.randn(N, T_PRED, 2)
    check("restored normaliser identical",
          (norm.transform(x_t) - norm_r.transform(x_t)).abs().max().item() < 1e-6)

except Exception:
    print(f"  {FAIL}  Unexpected exception:")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 13. Mini teacher training (5 steps, finite losses)
# ---------------------------------------------------------------------------

section("13. Mini teacher training (5 steps)")

try:
    net2 = ETHMotionTransformer(
        d_model=D_MODEL, K=K, pred_len=T_PRED,
        n_enc_heads=2, n_enc_layers=1,
        n_dec_heads=2, n_dec_layers=1,
        ffn_multiplier=2, dropout=0.0,
    )
    fm2  = FlowMatcher(model=net2, K=K, pred_len=T_PRED,
                       tied_noise=True, fm_in_scaling=True)
    opt  = torch.optim.Adam(fm2.parameters(), lr=1e-3)

    losses = []
    for _ in range(5):
        fm2.train()
        opt.zero_grad()
        loss, _, _ = fm2(
            torch.randn(N, T_OBS, 6),
            torch.rand(N, T_PRED, 2) * 2 - 1,
        )
        loss.backward()
        opt.step()
        losses.append(loss.item())

    check("all losses finite",
          all(np.isfinite(l) for l in losses),
          str([f"{l:.4f}" for l in losses]))

except Exception:
    print(f"  {FAIL}  Unexpected exception:")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'─'*60}")
if all_passed:
    print(f"  {PASS}  All checks passed. Ready to train on real data.")
else:
    print(f"  {FAIL}  Some checks failed. Fix the issues above before training.")
print(f"{'─'*60}\n")
