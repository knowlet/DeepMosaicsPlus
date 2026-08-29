"""RestorationBackend contract.

Inputs and outputs are plain OpenCV BGR uint8 numpy arrays. Every backend
internally handles RGB conversion, tensor ranges, padding to network-friendly
sizes and mixed precision. Callers never touch tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import cv2
import numpy as np


@dataclass
class RestoreRequest:
    """A batch of aligned crops: one square region tracked over a clip."""

    frames: List[np.ndarray]              # BGR uint8 crops, all same HxW
    masks: Optional[List[np.ndarray]] = None   # uint8 single-channel (optional)
    meta: dict = field(default_factory=dict)   # e.g. {"scene_id": 0}


@dataclass
class BackendCaps:
    """How the service should drive this backend."""

    output_mode: str = "all"      # "all": restore every frame of the clip
                                  # "center": only the middle frame is valid
    max_clip_len: int = 16        # preferred temporal window
    input_size: int = 0           # 0 -> keep caller crop size (backend resizes)


class RestorationBackend:
    """Base class. Subclasses implement _restore_impl."""

    caps: BackendCaps = BackendCaps()
    name: str = "abstract"

    # ------------------------------------------------------------------
    def restore(self, request: RestoreRequest) -> List[np.ndarray]:
        """Return the same number of BGR uint8 frames as request.frames.

        For caps.output_mode == "center", frames other than the middle one are
        echoes of the input; callers must honour caps when compositing.
        """
        if not request.frames:
            return []
        out = self._restore_impl(request)
        assert len(out) == len(request.frames), (
            f"{self.name}: returned {len(out)} frames for {len(request.frames)} inputs"
        )
        for i, frame in enumerate(out):
            h, w = request.frames[i].shape[:2]
            if frame.shape[:2] != (h, w):
                raise ValueError(
                    f"{self.name}: output {i} shape {frame.shape[:2]} != input {(h, w)}"
                )
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
                out[i] = frame
        return out

    def _restore_impl(self, request: RestoreRequest) -> List[np.ndarray]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    def warmup(self) -> None:  # optional hook
        pass

    def close(self) -> None:  # optional hook
        pass


# ----------------------------------------------------------------------
# helpers shared by backends
# ----------------------------------------------------------------------
def pad_to_multiple(img: np.ndarray, multiple: int):
    """Zero-pad H,W up to a multiple; returns (padded, (pad_h, pad_w))."""
    h, w = img.shape[:2]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
    return img, (ph, pw)


def unpad(img: np.ndarray, pads) -> np.ndarray:
    ph, pw = pads
    h, w = img.shape[:2]
    return img[: h - ph or None, : w - pw or None]


def resize_keep_aspect(img: np.ndarray, target: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = target / max(h, w)
    if abs(scale - 1.0) < 1e-3:
        return img
    return cv2.resize(img, (max(4, int(round(w * scale))), max(4, int(round(h * scale)))),
                      interpolation=cv2.INTER_CUBIC)
