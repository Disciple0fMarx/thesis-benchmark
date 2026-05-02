"""
tracker_thread.py
-----------------
Thread-2: pulls frames from the FrameStream queue, runs a pedestrian detector
(YOLO via ultralytics or a torchvision Faster R-CNN), feeds detections into a
per-frame tracker (ByteTrack via supervision), applies the homography to
convert pixel centroids to world coordinates, and maintains a sliding
observation window for each tracked agent.

When the window reaches obs_len frames it is pushed onto the obs_queue so
Thread-3 (inference) can run the PTP model.

Design notes
~~~~~~~~~~~~
* The tracker is deliberately separated from the reader so that a slow YOLO
  pass (≈30–80 ms on GPU) does not stall frame decoding.
* A new obs_window is pushed every stride_frames frames, not every frame.
  This matches the typical ETH/UCY 0.4 s stride (10 frames @ 25 fps).
* World coordinates use the same (x, y) convention as obsmat.txt: metres,
  origin at the homography reference point.
* The observation window dict has the shape expected by all src/models/:
      {"obs": np.ndarray shape (N, obs_len, 2),  # world coords, agents × time × xy
       "ids": list[int],                          # agent IDs matching axis-0
       "frame_idx": int}                          # frame index of the last obs frame
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .stream import FrameItem


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Single bounding-box detection before tracking."""
    bbox_xyxy: np.ndarray    # shape (4,) — x1 y1 x2 y2 in pixels
    confidence: float
    class_id: int            # 0 = person in COCO


@dataclass
class TrackedAgent:
    """State maintained per agent across frames."""
    agent_id: int
    history: Deque[np.ndarray] = field(default_factory=lambda: deque())
    # Each entry in history is shape (2,): world-coord [x, y]

    # Optional bbox history: entries are (x1,y1,x2,y2) pixel tuples or None
    bbox_history: Deque[Optional[tuple]] = field(default_factory=lambda: deque())

    def add(self, world_xy: np.ndarray, maxlen: int,
            bbox_xyxy: Optional[tuple] = None) -> None:
        self.history.append(world_xy.copy())
        self.bbox_history.append(bbox_xyxy)
        while len(self.history) > maxlen:
            self.history.popleft()
        while len(self.bbox_history) > maxlen:
            self.bbox_history.popleft()

    def window(self, length: int) -> Optional[np.ndarray]:
        """Return shape (length, 2) if enough history, else None."""
        if len(self.history) < length:
            return None
        return np.stack(list(self.history)[-length:])   # (length, 2)

    def bbox_window(self, length: int) -> List[Optional[tuple]]:
        """Return the last `length` bbox entries (may contain Nones)."""
        hist = list(self.bbox_history)
        if len(hist) >= length:
            return hist[-length:]
        # Pad front with None if history is shorter than requested
        return [None] * (length - len(hist)) + hist


@dataclass
class ObsWindow:
    """Batch of observation windows ready for the inference thread."""
    obs: np.ndarray                           # (N, obs_len, 2)  world coords
    ids: List[int]                            # length N
    frame_idx: int
    pixel_positions: Dict[int, np.ndarray]   # agent_id → pixel centroid (2,)

    # ── VidTraj fields (populated when store_frames=True in TrackerThread) ──
    # frames: the T_obs raw BGR frames that correspond to this observation window,
    #         in temporal order.  None when store_frames=False (default) so that
    #         the standard pipeline incurs no memory overhead.
    frames: Optional[List[np.ndarray]] = None   # list of T_obs (H,W,3) uint8

    # bboxes: per-agent bounding box history aligned with frames.
    #         agent_id → list of T_obs (x1,y1,x2,y2) tuples (or None if unseen)
    bboxes: Optional[Dict[int, List[Optional[tuple]]]] = None


# ─────────────────────────────────────────────────────────────────────────────
# Detector backends
# ─────────────────────────────────────────────────────────────────────────────

class _YOLODetector:
    """Thin wrapper around ultralytics YOLO."""

    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.4,
                 device: str = "cpu") -> None:
        from ultralytics import YOLO  # imported lazily so the module loads
        self._model = YOLO(model_name)
        self._conf = conf
        self._device = device

    def detect(self, bgr: np.ndarray) -> List[Detection]:
        results = self._model(bgr, conf=self._conf, classes=[0],
                              device=self._device, verbose=False)[0]
        out = []
        for box in results.boxes:
            out.append(Detection(
                bbox_xyxy=box.xyxy[0].cpu().numpy(),
                confidence=float(box.conf[0]),
                class_id=int(box.cls[0]),
            ))
        return out


class _FasterRCNNDetector:
    """Fallback detector using torchvision Faster R-CNN."""

    def __init__(self, conf: float = 0.5, device: str = "cpu") -> None:
        import torch
        import torchvision
        self._device = torch.device(device)
        self._model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True).to(self._device)
        self._model.eval()
        self._conf = conf
        self._transform = torchvision.transforms.ToTensor()

    def detect(self, bgr: np.ndarray) -> List[Detection]:
        import torch
        rgb = bgr[..., ::-1].copy()
        tensor = self._transform(rgb).unsqueeze(0).to(self._device)
        with torch.no_grad():
            preds = self._model(tensor)[0]
        out = []
        for box, score, label in zip(preds["boxes"], preds["scores"],
                                     preds["labels"]):
            if label.item() != 1:   # 1 = person in COCO
                continue
            if score.item() < self._conf:
                continue
            out.append(Detection(
                bbox_xyxy=box.cpu().numpy(),
                confidence=float(score),
                class_id=0,
            ))
        return out


def build_detector(backend: str = "yolo", **kwargs):
    if backend == "yolo":
        return _YOLODetector(**kwargs)
    if backend == "fasterrcnn":
        return _FasterRCNNDetector(**kwargs)
    raise ValueError(f"Unknown detector backend: {backend!r}. "
                     f"Choose 'yolo' or 'fasterrcnn'.")


# ─────────────────────────────────────────────────────────────────────────────
# ByteTrack adapter
# ─────────────────────────────────────────────────────────────────────────────

class _ByteTrackAdapter:
    """
    Wraps supervision's ByteTrack so the rest of the module stays
    backend-agnostic.  Returns (agent_id, pixel_centroid) pairs.
    """

    def __init__(self) -> None:
        from supervision import ByteTrack, Detections
        self._tracker = ByteTrack()
        self._Detections = Detections

    def update(self, detections: List[Detection],
               frame_hw: Tuple[int, int]) -> List[Tuple[int, np.ndarray]]:
        """
        Returns list of (tracker_id, pixel_centroid_xy) for all active agents.
        pixel_centroid_xy is shape (2,) in (col, row) = (x, y) order.
        """
        if not detections:
            return []

        import numpy as np
        boxes = np.array([d.bbox_xyxy for d in detections])   # (N, 4)
        scores = np.array([d.confidence for d in detections])  # (N,)
        class_ids = np.array([d.class_id for d in detections]) # (N,)

        sv_dets = self._Detections(
            xyxy=boxes,
            confidence=scores,
            class_id=class_ids,
        )
        tracked = self._tracker.update_with_detections(sv_dets)

        results = []
        for xyxy, tid in zip(tracked.xyxy, tracked.tracker_id):
            cx = (xyxy[0] + xyxy[2]) / 2.0
            cy = (xyxy[1] + xyxy[3]) / 2.0
            results.append((int(tid), np.array([cx, cy])))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Homography helpers
# ─────────────────────────────────────────────────────────────────────────────

def pixel_to_world(pixel_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Apply homography H (3×3) to convert a pixel centroid to world coords.
    pixel_xy: shape (2,) — (col, row)
    Returns shape (2,) — (world_x, world_y)
    """
    p = np.array([pixel_xy[0], pixel_xy[1], 1.0])
    w = H @ p
    return w[:2] / w[2]


def world_to_pixel(world_xy: np.ndarray, H_inv: np.ndarray) -> np.ndarray:
    """Inverse projection for rendering prediction fans back onto the frame."""
    p = np.array([world_xy[0], world_xy[1], 1.0])
    px = H_inv @ p
    return px[:2] / px[2]


# ─────────────────────────────────────────────────────────────────────────────
# Tracker thread
# ─────────────────────────────────────────────────────────────────────────────

class TrackerThread:
    """
    Pulls FrameItems from frame_q, detects + tracks pedestrians, accumulates
    observation windows, and pushes ObsWindow objects to obs_q.

    Parameters
    ----------
    frame_q:
        Source queue (output of FrameStream).
    obs_q:
        Destination queue consumed by InferenceThread.
    H:
        3×3 homography matrix (pixel → world).  Load from H.txt with
        np.loadtxt("H.txt").
    obs_len:
        Number of observed frames per prediction window (e.g. 8 for ETH/UCY).
    stride_frames:
        How often to push a new obs window (e.g. 10 = every 10 frames).
    detector_backend:
        "yolo" (default) or "fasterrcnn".
    detector_kwargs:
        Passed verbatim to build_detector().
    """

    def __init__(
        self,
        frame_q: queue.Queue,
        obs_q: queue.Queue,
        H: np.ndarray,
        obs_len: int = 8,
        stride_frames: int = 10,
        detector_backend: str = "yolo",
        store_frames: bool = False,
        **detector_kwargs,
    ) -> None:
        self._frame_q = frame_q
        self._obs_q = obs_q
        self._H = H
        self._H_inv = np.linalg.inv(H)
        self._obs_len = obs_len
        self._stride_frames = stride_frames

        # store_frames=True populates ObsWindow.frames and ObsWindow.bboxes
        # so that VidTrajEncoder can perform RoI-pooling on the raw pixels.
        # Set to False (default) for the standard trajectory-only pipeline
        # to avoid the memory cost of buffering raw video frames.
        self._store_frames = store_frames
        self._frame_buffer: Deque[np.ndarray] = deque(maxlen=obs_len)

        # Detector and tracker are built lazily inside _run.
        self._detector_backend = detector_backend
        self._detector_kwargs = detector_kwargs
        self._detector = None
        self._byte_track = None

        self._agents: Dict[int, TrackedAgent] = {}

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="TrackerThread")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "TrackerThread":
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5.0)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        # ── Lazy construction of heavy dependencies ───────────────────────
        if self._detector is None:
            self._detector = build_detector(self._detector_backend,
                                            **self._detector_kwargs)
        if self._byte_track is None:
            self._byte_track = _ByteTrackAdapter()

        frames_since_push = 0

        while not self._stop_event.is_set():
            # ── Grab frame ───────────────────────────────────────────────
            try:
                item: Optional[FrameItem] = self._frame_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:        # end-of-stream sentinel
                self._obs_q.put(None)
                return

            # ── Buffer raw frame for VidTraj ─────────────────────────────
            if self._store_frames:
                self._frame_buffer.append(item.bgr)

            # ── Detect ───────────────────────────────────────────────────
            detections = self._detector.detect(item.bgr)

            # ── Track ────────────────────────────────────────────────────
            h, w = item.bgr.shape[:2]

            # ByteTrack returns (agent_id, pixel_centroid) pairs.
            # We also need full bboxes for VidTraj RoI-pooling, so we extend
            # the adapter call to also return the raw xyxy bbox.
            tracked_pairs = self._byte_track.update(detections, (h, w))

            # ── Update agent histories ────────────────────────────────────
            seen_ids = set()
            pixel_positions: Dict[int, np.ndarray] = {}
            # Build a bbox lookup keyed by agent_id for this frame
            current_bboxes: Dict[int, tuple] = {}
            for det in detections:
                pass   # raw detections don't have IDs yet; use tracked output

            for agent_id, pixel_xy in tracked_pairs:
                seen_ids.add(agent_id)
                pixel_positions[agent_id] = pixel_xy
                world_xy = pixel_to_world(pixel_xy, self._H)

                # Approximate bbox from centroid (±25 px) when full bbox
                # is not available from the tracker.  If the ByteTrack adapter
                # is extended to return xyxy in future, replace this estimate.
                half = 25.0
                approx_bbox = (
                    float(pixel_xy[0] - half), float(pixel_xy[1] - half),
                    float(pixel_xy[0] + half), float(pixel_xy[1] + half),
                )
                current_bboxes[agent_id] = approx_bbox

                if agent_id not in self._agents:
                    self._agents[agent_id] = TrackedAgent(agent_id=agent_id)
                self._agents[agent_id].add(
                    world_xy,
                    maxlen=self._obs_len,
                    bbox_xyxy=approx_bbox if self._store_frames else None,
                )

            # Age out stale agents
            stale = [aid for aid in self._agents if aid not in seen_ids]
            for aid in stale:
                agent = self._agents[aid]
                agent._frames_missing = getattr(agent, "_frames_missing", 0) + 1
                if agent._frames_missing > self._obs_len * 2:
                    del self._agents[aid]
                else:
                    # Record None bbox for this frame in the agent's history
                    if self._store_frames and aid in self._agents:
                        self._agents[aid].bbox_history.append(None)
                        if len(self._agents[aid].bbox_history) > self._obs_len:
                            self._agents[aid].bbox_history.popleft()

            # Reset missing counter for visible agents
            for aid in seen_ids:
                if aid in self._agents:
                    self._agents[aid]._frames_missing = 0

            # ── Maybe push an observation window ─────────────────────────
            frames_since_push += 1
            if frames_since_push < self._stride_frames:
                continue
            frames_since_push = 0

            windows, ids = [], []
            for agent_id, agent in self._agents.items():
                w_arr = agent.window(self._obs_len)
                if w_arr is not None:
                    windows.append(w_arr)
                    ids.append(agent_id)

            if not windows:
                continue

            # ── Populate VidTraj fields if store_frames=True ──────────────
            obs_frames = None
            obs_bboxes = None
            if self._store_frames and len(self._frame_buffer) >= self._obs_len:
                obs_frames = list(self._frame_buffer)[-self._obs_len:]
                obs_bboxes = {
                    aid: self._agents[aid].bbox_window(self._obs_len)
                    for aid in ids
                    if aid in self._agents
                }

            obs_batch = ObsWindow(
                obs=np.stack(windows),          # (N, obs_len, 2)
                ids=ids,
                frame_idx=item.frame_idx,
                pixel_positions=pixel_positions,
                frames=obs_frames,
                bboxes=obs_bboxes,
            )

            # Non-blocking push: replace stale window if inference is lagging
            if self._obs_q.full():
                try:
                    self._obs_q.get_nowait()
                except queue.Empty:
                    pass
            self._obs_q.put_nowait(obs_batch)
            