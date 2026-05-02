"""
live.py
-------
Main entry point for live trajectory prediction from video.

Orchestrates the three-thread pipeline:

    FrameStream  →  TrackerThread  →  InferenceThread
                                            ↓
                            renderer loop (this thread) reads pred_q
                            and overlays predictions on each display frame

Usage
-----
From Python:

    from src.video_encoder.live import LivePredictor, LiveConfig

    cfg = LiveConfig(
        source="path/to/video.mp4",   # or 0 for webcam
        H=np.loadtxt("data/raw/eth/hotel/H.txt"),
        model_type="social_lstm",
        checkpoint_path="checkpoints/social_lstm_eth.pt",
        obs_len=8,
        pred_len=12,
        display=True,
        output_path="out.mp4",        # optional — None to skip writing
    )
    predictor = LivePredictor(cfg)
    predictor.run()

From the command line:

    python -m src.video_encoder.live \
        --source data/raw/eth/hotel/video.avi \
        --homography data/raw/eth/hotel/H.txt \
        --model social_lstm \
        --checkpoint checkpoints/social_lstm_eth.pt \
        --display

Overlay description
-------------------
For each tracked agent the renderer draws:

  ● Observation trail  — thin solid polyline in the agent's colour showing
                          the last obs_len pixel positions.
  ● Prediction fan     — dashed polyline in the same colour, lighter opacity,
                          showing the predicted future positions.
  ● Agent ID badge     — small filled circle with the integer tracker ID.

Colours cycle through a fixed palette (one colour per agent ID mod palette
size) so the same agent retains its colour across frames.
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .stream import FrameStream, FrameItem
from .tracker_thread import TrackerThread, ObsWindow, world_to_pixel
from .inference_thread import InferenceThread, PredResult, build_adapter


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LiveConfig:
    """
    All parameters for a live prediction session.

    source:
        Video file path or integer camera index.
    H:
        3×3 homography matrix (pixel → world).  Load with np.loadtxt("H.txt").
    model_type:
        One of "cv", "social_lstm", "stgcnn", "agentformer".
    checkpoint_path:
        Path to a .pt checkpoint.  Not required for model_type="cv".
    obs_len:
        Observation window length in frames.  Default 8 matches ETH/UCY.
    pred_len:
        Prediction horizon in frames.  Default 12 matches ETH/UCY.
    stride_frames:
        How often (in frames) a new prediction is triggered.  Default 10.
    detector_backend:
        "yolo" (default) or "fasterrcnn".
    device:
        Torch device string, e.g. "cpu", "cuda", "mps".
    display:
        If True, open an OpenCV window showing the annotated video.
    output_path:
        If set, write the annotated video to this path.
    target_fps:
        Cap the reader at this fps (useful for file replay).  None = uncapped.
    trail_len:
        How many past pixel positions to draw as the observation trail.
    """
    source: Union[str, Path, int]
    H: np.ndarray
    model_type: str = "cv"
    checkpoint_path: Optional[str] = None
    obs_len: int = 8
    pred_len: int = 12
    stride_frames: int = 10
    detector_backend: str = "yolo"
    detector_kwargs: dict = field(default_factory=dict)
    device: str = "cpu"
    display: bool = True
    output_path: Optional[Union[str, Path]] = None
    target_fps: Optional[float] = None
    trail_len: int = 20
    output_fps: Optional[float] = None
    """
    fps for the output video file.  When None (default), the codec metadata
    from the source file is used.  Override this when the AVI metadata is
    wrong — e.g. the ETH hotel video reports ~59 fps but plays at 25 fps:
        LiveConfig(..., output_fps=25.0)
    """


# ─────────────────────────────────────────────────────────────────────────────
# Overlay drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

# 12-colour palette — one per agent ID mod 12
_PALETTE = [
    (235,  87,  87),   # coral
    ( 87, 181, 235),   # sky blue
    ( 87, 235, 148),   # mint
    (235, 185,  87),   # amber
    (148,  87, 235),   # purple
    ( 87, 235, 218),   # teal
    (235, 120,  87),   # orange
    ( 87, 130, 235),   # periwinkle
    (200, 235,  87),   # lime
    (235,  87, 160),   # pink
    ( 87, 235,  87),   # green
    (160, 160, 235),   # lavender
]


def _colour(agent_id: int) -> Tuple[int, int, int]:
    return _PALETTE[agent_id % len(_PALETTE)]


def draw_overlay(
    frame: np.ndarray,
    result: Optional[PredResult],
    agent_trail: Dict[int, List[np.ndarray]],
    trail_len: int,
) -> np.ndarray:
    """
    Draws observation trails and prediction fans onto a copy of frame.

    Parameters
    ----------
    frame:
        BGR image to annotate.
    result:
        Latest PredResult from the inference thread (may be None before
        the first prediction arrives).
    agent_trail:
        Running dict mapping agent_id → list of recent pixel (x, y) positions.
        Updated in-place by this function.
    trail_len:
        Maximum number of past positions to draw.
    """
    out = frame.copy()

    # ── Update trails from current pixel positions ──────────────────────
    if result is not None:
        for agent_id, px in result.pixel_positions.items():
            if agent_id not in agent_trail:
                agent_trail[agent_id] = []
            agent_trail[agent_id].append(px.astype(int))
            if len(agent_trail[agent_id]) > trail_len:
                agent_trail[agent_id].pop(0)

    # ── Draw observation trails ──────────────────────────────────────────
    for agent_id, trail in agent_trail.items():
        if len(trail) < 2:
            continue
        colour = _colour(agent_id)
        pts = np.array(trail, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=False, color=colour,
                      thickness=2, lineType=cv2.LINE_AA)
        # Agent ID badge at the last known position
        last = trail[-1]
        cv2.circle(out, tuple(last), 6, colour, -1, cv2.LINE_AA)
        cv2.putText(out, str(agent_id),
                    (int(last[0]) + 8, int(last[1]) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

    # ── Draw prediction fans ─────────────────────────────────────────────
    if result is None:
        return out

    for i, agent_id in enumerate(result.ids):
        if i >= len(result.pred_pixel):
            continue
        colour = _colour(agent_id)
        light = tuple(min(255, int(c * 0.55 + 180)) for c in colour)

        pred_pts = result.pred_pixel[i].astype(int)   # (pred_len, 2)

        # Connect last observed position to start of prediction
        if agent_id in agent_trail and agent_trail[agent_id]:
            start = agent_trail[agent_id][-1]
            if pred_pts.shape[0] > 0:
                cv2.line(out, tuple(start), tuple(pred_pts[0]),
                         light, 1, cv2.LINE_AA)

        # Dashed prediction polyline
        for t in range(len(pred_pts) - 1):
            p1 = tuple(pred_pts[t])
            p2 = tuple(pred_pts[t + 1])
            # Simulate dashes by drawing short segments
            if t % 2 == 0:
                cv2.line(out, p1, p2, light, 1, cv2.LINE_AA)

        # Terminal dot at the predicted destination
        if len(pred_pts) > 0:
            cv2.circle(out, tuple(pred_pts[-1]), 4, light, -1, cv2.LINE_AA)

    # ── Draw group brackets ──────────────────────────────────────────────
    # When a GroupInferenceAdapter is active, draw a thin line between the
    # current pixel positions of agents that share a group label.  This gives
    # an immediate visual signal of inferred group structure inspired by
    # Ge et al.'s group annotations.
    if result.group_labels is not None:
        from collections import defaultdict
        group_members: dict = defaultdict(list)
        for agent_id, label in zip(result.ids, result.group_labels):
            if agent_id in agent_trail and agent_trail[agent_id]:
                group_members[label].append(agent_trail[agent_id][-1])

        for label, positions in group_members.items():
            if len(positions) < 2:
                continue
            # Draw a thin white bracket connecting all members of the group
            bracket_colour = (220, 220, 220)
            for k in range(len(positions) - 1):
                p1 = tuple(positions[k].astype(int))
                p2 = tuple(positions[k + 1].astype(int))
                cv2.line(out, p1, p2, bracket_colour, 1, cv2.LINE_AA)

    return out


def draw_hud(
    frame: np.ndarray,
    frame_idx: int,
    n_agents: int,
    model_type: str,
    inference_lag_frames: int,
) -> np.ndarray:
    """Draws a small heads-up display in the top-left corner."""
    out = frame.copy()
    lines = [
        f"frame {frame_idx:06d}",
        f"agents: {n_agents}",
        f"model: {model_type}",
        f"pred lag: {inference_lag_frames}f",
    ]
    x, y0 = 10, 20
    for i, line in enumerate(lines):
        y = y0 + i * 18
        cv2.putText(out, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (200, 200, 200), 1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# LivePredictor
# ─────────────────────────────────────────────────────────────────────────────

class LivePredictor:
    """
    Orchestrates the full three-thread pipeline and owns the render loop.

    Call .run() to start.  Press 'q' in the display window to quit early,
    or send KeyboardInterrupt.
    """

    def __init__(self, config: LiveConfig) -> None:
        self.cfg = config
        self._H_inv = np.linalg.inv(config.H)

        # Assemble the full set of candidate kwargs, then let build_adapter
        # filter them to what the chosen adapter actually accepts.
        adapter_kwargs: dict = {"pred_len": config.pred_len}
        if config.checkpoint_path:
            adapter_kwargs["checkpoint_path"] = config.checkpoint_path
        if config.device:
            adapter_kwargs["device"] = config.device
        self._adapter = build_adapter(config.model_type, **adapter_kwargs)

    def run(self) -> None:
        cfg = self.cfg

        # render_q is large enough to buffer the full video so the fanout
        # thread is never blocked by a slow renderer.  track_q stays small
        # (recency-only) because the tracker only needs the latest frames.
        render_q: queue.Queue = queue.Queue(maxsize=0)  # 0 = unbounded
        track_q:  queue.Queue = queue.Queue(maxsize=2)
        obs_q:    queue.Queue = queue.Queue(maxsize=4)
        pred_q:   queue.Queue = queue.Queue(maxsize=1)

        writer: Optional[cv2.VideoWriter] = None
        agent_trail: Dict[int, List[np.ndarray]] = {}
        latest_result: Optional[PredResult] = None
        frame_idx = 0

        def _fanout(stream: FrameStream) -> None:
            """
            Drain FrameStream and push each item to both render_q and track_q.

            render_q uses a blocking put (with a generous timeout) so that
            every frame reaches the writer — dropping render frames would
            produce a choppy or truncated output video.

            track_q drops the oldest frame when full so the tracker always
            works on nearly-current data and never falls behind the decoder.
            """
            while True:
                item = stream.get(timeout=0.5)
                # render queue: blocking — preserve every frame for the writer
                try:
                    render_q.put(item, timeout=2.0)
                except queue.Full:
                    pass  # extremely slow renderer; drop rather than deadlock
                # track queue: non-blocking evict-oldest for recency guarantee
                if track_q.full():
                    try:
                        track_q.get_nowait()
                    except queue.Empty:
                        pass
                track_q.put_nowait(item)
                if item is None:
                    break

        try:
            with FrameStream(cfg.source,
                             maxsize=4,
                             target_fps=cfg.target_fps) as stream:

                # ── Measure true fps from first two frames ────────────────
                # stream.fps reflects codec metadata which is often wrong
                # (e.g. ETH hotel AVI reports ~59 fps but content is 25 fps).
                # We sample wall-clock time between the first two frames and
                # use that as the output fps, clamped to [1, 60].
                fanout_thread = threading.Thread(target=_fanout, args=(stream,),
                                                 daemon=True, name="FanOut")
                fanout_thread.start()

                # Grab the first two frames to confirm the stream is live,
                # then re-queue them so the render loop sees them.
                first  = render_q.get(timeout=5.0)
                second = render_q.get(timeout=5.0)
                if first is None or second is None:
                    return  # empty video

                # fps for the output file: use codec metadata clamped to a
                # sane range.  The user can override via cfg.output_fps if the
                # AVI metadata is wrong (e.g. ETH hotel reports ~59 fps).
                raw_fps = stream.fps if stream.fps > 0 else 25.0
                fps_out = float(max(1.0, min(120.0, cfg.output_fps or raw_fps)))

                # Re-queue the two seed frames
                seed_q: queue.Queue = queue.Queue()
                seed_q.put(first)
                seed_q.put(second)

                # ── Initialise video writer ───────────────────────────────
                if cfg.output_path:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(cfg.output_path), fourcc, float(fps_out),
                        (stream.width, stream.height))

                with TrackerThread(
                    frame_q=track_q,
                    obs_q=obs_q,
                    H=cfg.H,
                    obs_len=cfg.obs_len,
                    stride_frames=cfg.stride_frames,
                    detector_backend=cfg.detector_backend,
                    store_frames=cfg.vidtraj_encoder is not None,
                    **cfg.detector_kwargs,
                ) as tracker:

                    with InferenceThread(
                        obs_q=obs_q,
                        pred_q=pred_q,
                        adapter=self._adapter,  
                        H_inv=self._H_inv,
                        vidtraj_encoder=cfg.vidtraj_encoder,
                    ) as inference:

                        while True:
                            # Drain seed frames first, then live render_q.
                            # Keep looping while the fanout thread is alive
                            # (it may not have pushed all frames yet).
                            if not seed_q.empty():
                                item = seed_q.get_nowait()
                            else:
                                try:
                                    item = render_q.get(timeout=0.1)
                                except queue.Empty:
                                    if fanout_thread.is_alive():
                                        continue   # fanout still running
                                    # Fanout done and queue empty → drain any
                                    # remaining sentinel
                                    try:
                                        item = render_q.get_nowait()
                                    except queue.Empty:
                                        break      # truly finished

                            if item is None:
                                break   # genuine EOS sentinel

                            # ── Latest prediction (non-blocking) ─────────
                            try:
                                new_result = pred_q.get_nowait()
                                if new_result is not None:
                                    latest_result = new_result
                                # None means the inference thread finished —
                                # that is fine; keep rendering with the last
                                # known prediction until the video ends.
                            except queue.Empty:
                                pass

                            # ── Render ────────────────────────────────────
                            annotated = draw_overlay(
                                item.bgr, latest_result, agent_trail,
                                cfg.trail_len)

                            lag = (item.frame_idx - latest_result.frame_idx
                                   if latest_result else 0)
                            n_agents = (len(latest_result.ids)
                                        if latest_result else 0)
                            annotated = draw_hud(
                                annotated, item.frame_idx, n_agents,
                                cfg.model_type, lag)

                            if writer is not None:
                                writer.write(annotated)

                            if cfg.display:
                                cv2.imshow("Live trajectory prediction",
                                           annotated)
                                if cv2.waitKey(1) & 0xFF == ord("q"):
                                    break

                            frame_idx = item.frame_idx

        except KeyboardInterrupt:
            pass
        finally:
            if writer is not None:
                writer.release()
            if cfg.display:
                cv2.destroyAllWindows()

        print(f"[LivePredictor] finished at frame {frame_idx}.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run live pedestrian trajectory prediction on a video.")
    p.add_argument("--source", required=True,
                   help="Path to video file, or integer camera index.")
    p.add_argument("--homography", required=True,
                   help="Path to H.txt (3×3 homography, whitespace-separated).")
    p.add_argument("--model", default="cv",
                   choices=["cv", "social_lstm", "stgcnn", "agentformer",
                            "moflow",
                            "group+cv", "group+social_lstm", "group+stgcnn",
                            "group+agentformer", "group+moflow"],
                   help="Trajectory prediction model to use.")
    p.add_argument("--checkpoint", default=None,
                   help="Path to model checkpoint (.pt).  Not needed for --model cv.")
    p.add_argument("--obs-len", type=int, default=8)
    p.add_argument("--pred-len", type=int, default=12)
    p.add_argument("--stride", type=int, default=10,
                   help="Frames between successive prediction triggers.")
    p.add_argument("--detector", default="yolo",
                   choices=["yolo", "fasterrcnn"])
    p.add_argument("--device", default="cpu",
                   help="Torch device, e.g. 'cpu', 'cuda', 'mps'.")
    p.add_argument("--display", action="store_true",
                   help="Open an OpenCV window showing the annotated video.")
    p.add_argument("--output", default=None,
                   help="Write annotated video to this path.")
    p.add_argument("--fps", type=float, default=None,
                   help="Cap playback speed to this fps.")
    p.add_argument("--output-fps", type=float, default=None,
                   help="fps for the output video (overrides codec metadata). "
                        "Use 25 for ETH/UCY AVI files whose metadata is wrong.")
    p.add_argument("--vidtraj-backend", default=None,
               choices=["swin", "dinov2"],
               help="Enable VidTraj video encoding with this backbone.")
    p.add_argument("--vidtraj-d", type=int, default=128,
                help="VidTraj embedding dimension.")
    p.add_argument("--vidtraj-freeze", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Parse source as int (camera index) or string (file path)
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    H = np.loadtxt(args.homography)
    assert H.shape == (3, 3), f"Expected 3×3 homography, got {H.shape}"

    cfg = LiveConfig(
        source=source,
        H=H,
        model_type=args.model,
        checkpoint_path=args.checkpoint,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        stride_frames=args.stride,
        detector_backend=args.detector,
        device=args.device,
        display=args.display,
        output_path=args.output,
        target_fps=args.fps,
        output_fps=args.output_fps,
        detector_kwargs={"conf": 0.15}
    )

    if args.vidtraj_backend:
        from src.video_encoder.vidtraj import build_vidtraj
        cfg.vidtraj_encoder = build_vidtraj(
            backend=args.vidtraj_backend,
            d_vid=args.vidtraj_d,
            freeze_backbone=args.vidtraj_freeze,
            obs_len=args.obs_len,
            device=args.device,
        )

    LivePredictor(cfg).run()


if __name__ == "__main__":
    main()
    