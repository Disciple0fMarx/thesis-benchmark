from __future__ import annotations
import queue, threading
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from .tracker_thread import ObsWindow, world_to_pixel

@dataclass
class PredResult:
    pred_world: np.ndarray
    pred_pixel: np.ndarray
    ids: List[int]
    frame_idx: int
    pixel_positions: Dict[int, np.ndarray]
    group_labels: Optional[List[int]] = None
    vidtraj_used: bool = False

class ConstantVelocityAdapter:
    def __init__(self, pred_len: int = 12) -> None:
        self._pred_len = pred_len
    def predict(self, obs: np.ndarray, video_context=None) -> np.ndarray:
        vel = obs[:, -1, :] - obs[:, -2, :]
        steps = np.arange(1, self._pred_len + 1)[:, None]
        return obs[:, -1:, :] + vel[:, None, :] * steps[None, :, :]

class SocialLSTMAdapter:
    def __init__(self, checkpoint_path: str, device: str = "cpu", pred_len: int = 12) -> None:
        import torch
        from src.models.social_lstm import SocialLSTM
        self._device = torch.device(device); self._pred_len = pred_len
        ckpt = torch.load(checkpoint_path, map_location=self._device)
        self._model = SocialLSTM(**ckpt.get("model_kwargs", {}))
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval(); self._model.to(self._device)
    def predict(self, obs: np.ndarray, video_context=None) -> np.ndarray:
        import torch
        with torch.no_grad():
            t = torch.from_numpy(obs).float().to(self._device).permute(1, 0, 2)
            kw = {"pred_len": self._pred_len}
            if video_context is not None and video_context.c_vid is not None:
                kw["h0_context"] = _align_context(video_context, obs.shape[0], self._device).c_vid
            return self._model(t, **kw).permute(1, 0, 2).cpu().numpy()

class STGCNNAdapter:
    def __init__(self, checkpoint_path: str, device: str = "cpu", pred_len: int = 12, k_neighbours: int = 4) -> None:
        import torch
        from src.models.stgcnn import STGCNN
        self._device = torch.device(device); self._pred_len = pred_len; self._k = k_neighbours
        ckpt = torch.load(checkpoint_path, map_location=self._device)
        self._model = STGCNN(**ckpt.get("model_kwargs", {}))
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval(); self._model.to(self._device)
    def _build_adj(self, obs):
        last = obs[:, -1, :]; N = last.shape[0]
        if N == 1: return np.eye(1)
        dist = np.linalg.norm(last[:, None, :] - last[None, :, :], axis=-1)
        adj = np.zeros((N, N)); k = min(self._k, N - 1)
        for i in range(N):
            nb = np.argsort(dist[i])[1:k+1]; adj[i, nb] = adj[nb, i] = 1.0
        np.fill_diagonal(adj, 1.0); return adj
    def predict(self, obs: np.ndarray, video_context=None) -> np.ndarray:
        import torch
        adj = self._build_adj(obs)
        with torch.no_grad():
            t = torch.from_numpy(obs).float().to(self._device).unsqueeze(0)
            a = torch.from_numpy(adj).float().to(self._device).unsqueeze(0)
            kw = {"pred_len": self._pred_len}
            if video_context is not None and video_context.g_vid is not None:
                kw["scene_context"] = video_context.g_vid.to(self._device)
            return self._model(t, a, **kw).squeeze(0).cpu().numpy()

class AgentFormerAdapter:
    def __init__(self, checkpoint_path: str, device: str = "cpu", pred_len: int = 12, num_samples: int = 20) -> None:
        import torch
        from src.models.agentformer import AgentFormer
        self._device = torch.device(device); self._pred_len = pred_len; self._num_samples = num_samples
        ckpt = torch.load(checkpoint_path, map_location=self._device)
        self._model = AgentFormer(**ckpt.get("model_kwargs", {}))
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval(); self._model.to(self._device)
    def predict(self, obs: np.ndarray, video_context=None) -> np.ndarray:
        import torch
        with torch.no_grad():
            t = torch.from_numpy(obs).float().to(self._device)
            kv = None
            if video_context is not None and video_context.c_vid is not None:
                ctx = _align_context(video_context, obs.shape[0], self._device)
                c = ctx.c_vid.unsqueeze(1)
                g = ctx.g_vid.unsqueeze(0).unsqueeze(0).expand(c.shape[0], 1, -1)
                kv = torch.cat([c, g], dim=1)
            samples = []
            for _ in range(self._num_samples):
                kw = {"pred_len": self._pred_len}
                if kv is not None: kw["video_kv_tokens"] = kv
                samples.append(self._model(t, **kw).cpu().numpy())
            stack = np.stack(samples); mean = stack.mean(0)
            dists = np.linalg.norm((stack - mean[None]).reshape(self._num_samples, -1), axis=1)
            return samples[np.argmin(dists)]

class MoFlowAdapter:
    def __init__(self, checkpoint_path: str, device: str = "cpu", pred_len: int = 12, temperature: float = 1.0) -> None:
        import torch
        from src.models.moflow import MoFlow
        self._device = torch.device(device); self._pred_len = pred_len; self._temperature = temperature
        ckpt = torch.load(checkpoint_path, map_location=self._device)
        self._model = MoFlow(**ckpt.get("model_kwargs", {}))
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval(); self._model.to(self._device)
    def predict(self, obs: np.ndarray, video_context=None) -> np.ndarray:
        import torch
        with torch.no_grad():
            t = torch.from_numpy(obs).float().to(self._device)
            kw = {"pred_len": self._pred_len, "temperature": self._temperature}
            if video_context is not None and video_context.c_vid is not None:
                ctx = _align_context(video_context, obs.shape[0], self._device)
                c = ctx.c_vid.unsqueeze(1)
                g = ctx.g_vid.unsqueeze(0).unsqueeze(0).expand(c.shape[0], 1, -1)
                kw["video_kv_tokens"] = torch.cat([c, g], dim=1)
            return self._model.sample(t, **kw).cpu().numpy()

def _align_context(video_context, N_obs: int, device):
    import torch
    from .vidtraj import VidTrajContext
    c = video_context.c_vid.to(device); g = video_context.g_vid.to(device)
    N_enc, d_vid = c.shape
    if N_enc < N_obs:
        c = torch.cat([c, torch.zeros(N_obs - N_enc, d_vid, device=device)], dim=0)
    elif N_enc > N_obs:
        c = c[:N_obs]
    return VidTrajContext(c_vid=c, g_vid=g, ids=video_context.ids)

class GroupInferenceAdapter:
    def __init__(self, base_adapter, prox_threshold=2.5, vel_threshold=0.5, min_cooccurrence=3, alpha=0.7):
        self._base = base_adapter; self._prox = prox_threshold
        self._vel = vel_threshold; self._min_cooc = min_cooccurrence; self._alpha = alpha
    def _infer_groups(self, obs):
        N, T, _ = obs.shape
        if N == 1: return [0]
        vel = obs[:, 1:, :] - obs[:, :-1, :]
        adj = np.zeros((N, N), dtype=bool)
        for i in range(N):
            for j in range(i+1, N):
                pd = np.linalg.norm(obs[i]-obs[j], axis=-1)
                vd = np.linalg.norm(vel[i]-vel[j], axis=-1)
                T_s = min(len(pd), len(vd))
                if int(np.sum((pd[:T_s] < self._prox) & (vd[:T_s] < self._vel))) >= self._min_cooc:
                    adj[i,j] = adj[j,i] = True
        labels = [-1]*N; cl = 0
        for s in range(N):
            if labels[s] != -1: continue
            labels[s] = cl; frontier = [s]
            while frontier:
                node = frontier.pop()
                for nb in range(N):
                    if adj[node,nb] and labels[nb]==-1:
                        labels[nb]=cl; frontier.append(nb)
            cl += 1
        return labels
    def predict(self, obs: np.ndarray, video_context=None) -> np.ndarray:
        self.last_group_labels = self._infer_groups(obs)
        return self._base.predict(obs, video_context=video_context)

def build_adapter(model_type: str, **kwargs):
    if model_type.startswith("group+"):
        base_type = model_type[len("group+"):]
        gkw = {k: kwargs.pop(k) for k in ("prox_threshold","vel_threshold","min_cooccurrence","alpha") if k in kwargs}
        return GroupInferenceAdapter(build_adapter(base_type, **kwargs), **gkw)
    registry = {"cv": ConstantVelocityAdapter, "social_lstm": SocialLSTMAdapter,
                "stgcnn": STGCNNAdapter, "agentformer": AgentFormerAdapter, "moflow": MoFlowAdapter}
    if model_type not in registry:
        raise ValueError(f"Unknown model type {model_type!r}. Available: {list(registry)} or 'group+<base>'")
    import inspect
    cls = registry[model_type]
    return cls(**{k: v for k, v in kwargs.items() if k in inspect.signature(cls.__init__).parameters})

class InferenceThread:
    def __init__(self, obs_q, pred_q, adapter, H_inv, vidtraj_encoder=None):
        self._obs_q = obs_q; self._pred_q = pred_q; self._adapter = adapter
        self._H_inv = H_inv; self._vidtraj = vidtraj_encoder
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="InferenceThread")
    def __enter__(self): self._thread.start(); return self
    def __exit__(self, *_): self._stop_event.set(); self._thread.join(timeout=10.0)
    def is_alive(self): return self._thread.is_alive()
    def _run(self):
        while not self._stop_event.is_set():
            try:
                window = self._obs_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if window is None:
                self._pred_q.put(None); return
            video_context = None; vidtraj_used = False
            if self._vidtraj is not None and window.frames is not None and window.bboxes is not None:
                try:
                    video_context = self._vidtraj.encode(window.frames, window.bboxes)
                    vidtraj_used = True
                except Exception as exc:
                    print(f"[InferenceThread] VidTraj encode() raised: {exc}")
            try:
                pred_world = self._adapter.predict(window.obs, video_context=video_context)
            except Exception as exc:
                print(f"[InferenceThread] predict() raised: {exc}"); continue
            group_labels = getattr(self._adapter, "last_group_labels", None)
            N, pred_len, _ = pred_world.shape
            pred_pixel = np.zeros_like(pred_world)
            for i in range(N):
                for t in range(pred_len):
                    pred_pixel[i, t] = world_to_pixel(pred_world[i, t], self._H_inv)
            result = PredResult(pred_world=pred_world, pred_pixel=pred_pixel, ids=window.ids,
                                frame_idx=window.frame_idx, pixel_positions=window.pixel_positions,
                                group_labels=group_labels, vidtraj_used=vidtraj_used)
            if self._pred_q.full():
                try: self._pred_q.get_nowait()
                except queue.Empty: pass
            self._pred_q.put_nowait(result)
