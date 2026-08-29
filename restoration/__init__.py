"""Unified restoration abstraction for DeepMosaicsPlus.

Layers:
    DeviceManager   - resolves auto/cuda/mps/directml/cpu once for the whole app.
    ModelManager    - model manifest, download, SHA-256 verify, local cache.
    RestorationBackend - BGR uint8 frames in -> BGR uint8 frames out.
    RestorationService - detect -> crop -> backend -> composite, shared by
                         image / video / server paths.
"""

from .device_manager import DeviceManager, DeviceInfo
from .model_manager import ModelManager, ResolvedModel
from .service import RestorationService

__all__ = [
    "DeviceManager",
    "DeviceInfo",
    "ModelManager",
    "ResolvedModel",
    "RestorationService",
]
