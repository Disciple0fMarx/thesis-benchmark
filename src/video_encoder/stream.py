"""
stream.py
---------
Thread-1: reads raw frames from a video file or camera device and puts them
into a small bounded queue.  The queue size is intentionally tiny (default 2)
so that Thread-2 (tracker) always works on nearly-current frames.  When the
queue is full the oldest frame is silently discarded — we prefer recency over
completeness during live inference.

Usage is always through the context manager so the background thread is
guaranteed to be joined on exit:

    with FrameStream("video.mp4") as stream:
        while stream.is_alive():
            item = stream.get(timeout=0.1)   # (frame_idx, bgr_frame) or None
            if item is not None:
                ...
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np


@dataclass
class FrameItem:
    """Single decoded frame with metadata."""
    frame_idx: int          # absolute frame counter since stream start
    timestamp_sec: float    # wall-clock time the frame was grabbed
    bgr: np.ndarray         # H×W×3 uint8, BGR colour order (OpenCV native)


class FrameStream:
    """
    Wraps a cv2.VideoCapture and reads frames in a daemon thread.

    Parameters
    ----------
    source:
        Path to a video file, or an integer camera index (0 = default webcam).
    maxsize:
        Maximum frames held in the internal queue.  Keep this small (1–4).
        When full, the oldest frame is dropped to make room for the newest.
    target_fps:
        If set, the reader will sleep between grabs to cap the output rate.
        Useful when replaying a pre-recorded file faster than real-time is
        undesirable.  None means read as fast as the decoder allows.
    """

    def __init__(
        self,
        source: Union[str, Path, int],
        maxsize: int = 2,
        target_fps: Optional[float] = None,
    ) -> None:
        self._source = source
        self._maxsize = maxsize
        self._target_fps = target_fps

        self._q: queue.Queue[Optional[FrameItem]] = queue.Queue(maxsize=maxsize)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="FrameStream")
        self._started = False

        # populated once the capture is opened so callers can query video meta
        self.fps: float = 0.0
        self.width: int = 0
        self.height: int = 0
        self.total_frames: int = 0          # -1 for live camera sources

        self._meta_ready = threading.Event()
        self._thread_exc: Optional[BaseException] = None  # set by _run on error

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "FrameStream":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()
        # Block until the capture is open and metadata is available.
        # If the thread crashes (e.g. file not found) _meta_ready is never set;
        # we wait briefly then re-raise the stored exception so the caller sees
        # a clean RuntimeError rather than a silent hang.
        self._meta_ready.wait(timeout=5.0)
        if self._thread_exc is not None:
            raise self._thread_exc

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=3.0)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def get(self, timeout: float = 0.05) -> Optional[FrameItem]:
        """
        Return the next FrameItem, or None on timeout / sentinel.
        A None sentinel value in the queue signals end-of-stream.
        """
        try:
            item = self._q.get(timeout=timeout)
            if item is None:
                return None          # end-of-stream sentinel passed through
            return item
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    # Internal reader loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        cap = cv2.VideoCapture(self._source if isinstance(self._source, int)
                               else str(self._source))
        if not cap.isOpened():
            exc = RuntimeError(f"Cannot open video source: {self._source!r}")
            self._thread_exc = exc
            self._meta_ready.set()   # unblock start() so it can re-raise
            return                   # thread exits cleanly; caller raises

        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.total_frames = int(total) if total > 0 else -1
        self._meta_ready.set()

        frame_interval = (1.0 / self._target_fps) if self._target_fps else 0.0
        frame_idx = 0
        last_grab = 0.0

        while not self._stop_event.is_set():
            now = time.monotonic()
            if frame_interval > 0 and (now - last_grab) < frame_interval:
                time.sleep(0.001)
                continue

            ret, bgr = cap.read()
            if not ret:
                break           # end of file or camera disconnected

            last_grab = time.monotonic()
            item = FrameItem(frame_idx=frame_idx,
                             timestamp_sec=last_grab,
                             bgr=bgr)

            # Non-blocking put: if the queue is full, evict the oldest frame
            # so the tracker always sees the most recent data.
            if self._q.full():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            self._q.put_nowait(item)
            frame_idx += 1

        cap.release()
        # Push sentinel to signal end-of-stream to consumers
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        