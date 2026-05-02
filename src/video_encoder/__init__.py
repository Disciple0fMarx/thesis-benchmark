"""
src/video_encoder
-----------------
Video encoder module for the pedestrian trajectory prediction benchmark.

Batch pipeline
~~~~~~~~~~~~~~
VideoEncoder          — full offline pipeline: detect → track → write obsmat
CalibrationHelper     — homography estimation from clicked correspondences

Live pipeline
~~~~~~~~~~~~~
LivePredictor         — three-thread live inference from file or camera
LiveConfig            — configuration dataclass for LivePredictor

Low-level building blocks (importable directly if needed)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FrameStream           — thread-safe video reader
TrackerThread         — detection + ByteTrack + homography
InferenceThread       — model adapter runner
build_adapter         — factory for model adapters
build_detector        — factory for detector backends
"""

from .live import LiveConfig, LivePredictor
from .stream import FrameStream, FrameItem
from .tracker_thread import (
    TrackerThread,
    ObsWindow,
    build_detector,
    pixel_to_world,
    world_to_pixel,
)
from .inference_thread import (
    InferenceThread,
    PredResult,
    build_adapter,
    ConstantVelocityAdapter,
    SocialLSTMAdapter,
    STGCNNAdapter,
    AgentFormerAdapter,
    MoFlowAdapter,
    GroupInferenceAdapter,
)

__all__ = [
    # Live pipeline
    "LiveConfig",
    "LivePredictor",
    # Threads
    "FrameStream",
    "FrameItem",
    "TrackerThread",
    "InferenceThread",
    # Data types
    "ObsWindow",
    "PredResult",
    # Factories
    "build_adapter",
    "build_detector",
    # Adapters (for direct use or subclassing)
    "ConstantVelocityAdapter",
    "SocialLSTMAdapter",
    "STGCNNAdapter",
    "AgentFormerAdapter",
    "MoFlowAdapter",
    "GroupInferenceAdapter",
    # Geometry helpers
    "pixel_to_world",
    "world_to_pixel",
]
