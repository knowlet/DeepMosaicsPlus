"""LADA official BasicVSR++ backend (AGPL-3.0 weights/code by ladaapp).

Loads the released lada_mosaic_restoration_model_generic_v*.pth checkpoints
through the vendored thirdparty/lada implementation (no mmengine needed).
Uses torchvision deformable convolution where available and the vendored
MPS-compatible alignment path on Apple Silicon.

Input contract handling: service gives arbitrary-size BGR uint8 crops.  The
network requires at least 256 px per axis, but enlarging a narrow crop to meet
that minimum can make its long edge enormous.  Content is therefore only ever
downscaled to the active memory limit; reflect/replicate padding supplies the
network minimum and alignment.  Padding is removed before every output is
resized back to its exact input size.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import torch

from .base import BackendCaps, RestorationBackend, RestoreRequest


class LadaOfficialBackend(RestorationBackend):
    name = "lada-official-basicvsrpp"
    caps = BackendCaps(output_mode="all", max_clip_len=8, input_size=0)

    def __init__(self, weights_path: str, device_manager, num_blocks: int = 15):
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from thirdparty.lada.mmengine_shim import load_checkpoint
        from thirdparty.lada.basicvsr_plusplus_net import BasicVSRPlusPlusNet

        self.dm = device_manager
        self.net = BasicVSRPlusPlusNet(mid_channels=64, num_blocks=num_blocks,
                                       spynet_pretrained=None)
        load_checkpoint(self.net, weights_path, map_location="cpu", strict=True)
        self.net.eval()
        self.dm.move(self.net)

    def warmup(self):  # pragma: no cover - heavy on CPU, optional
        pass

    # ------------------------------------------------------------------
    @staticmethod
    def _content_limit(max_restore_side: int) -> int:
        """Largest aligned content side that cannot exceed the user limit.

        Limits below the model's 256 px minimum still constrain image content;
        the remaining area is padding rather than an upscaled image.  For
        larger non-multiple-of-four limits, reserving the final 1--3 pixels
        avoids alignment padding crossing the requested memory boundary.
        """
        if max_restore_side <= 0 or max_restore_side < 256:
            return max_restore_side
        return max_restore_side - max_restore_side % 4

    @staticmethod
    def _downscale_only(frame: np.ndarray, max_side: int) -> np.ndarray:
        if max_side <= 0:
            return frame
        h, w = frame.shape[:2]
        longest = max(h, w)
        if longest <= max_side:
            return frame
        scale = max_side / float(longest)
        return cv2.resize(
            frame,
            (max(1, int(round(w * scale))),
             max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _pad(frame: np.ndarray, canvas_h: int, canvas_w: int):
        """Symmetrically pad a frame and return it plus its content origin."""
        h, w = frame.shape[:2]
        pad_h, pad_w = canvas_h - h, canvas_w - w
        top, left = pad_h // 2, pad_w // 2
        bottom, right = pad_h - top, pad_w - left
        # REFLECT_101 gives the least abrupt boundary for normal crops.  A
        # singleton dimension has no neighbour to reflect, so replicate it.
        border = (cv2.BORDER_REFLECT_101
                  if h > 1 and w > 1 else cv2.BORDER_REPLICATE)
        padded = cv2.copyMakeBorder(
            frame, top, bottom, left, right, border)
        return padded, (top, left, h, w)

    def _restore_impl(self, request: RestoreRequest):
        frames = request.frames
        originals = [(f.shape[0], f.shape[1]) for f in frames]

        # The service sets max_restore_side from the selected memory profile.
        # Keeping this guard in the backend also protects direct backend users
        # (tests, integrations, and the static-image path).
        requested_limit = int(getattr(self, "max_restore_side", 0) or 0)
        content_limit = self._content_limit(requested_limit)
        working = [self._downscale_only(f, content_limit) for f in frames]

        # BasicVSR++ downsamples by four and asserts both resulting axes >=64.
        # Padding, not proportional enlargement, meets those constraints.
        max_h = max(f.shape[0] for f in working)
        max_w = max(f.shape[1] for f in working)
        canvas_h = max(256, (max_h + 3) // 4 * 4)
        canvas_w = max(256, (max_w + 3) // 4 * 4)

        padded, placements = [], []
        for frame in working:
            prepared, placement = self._pad(frame, canvas_h, canvas_w)
            padded.append(prepared)
            placements.append(placement)
        arr = np.stack(padded)

        rgb = arr[:, :, :, ::-1].astype(np.float32) / 255.0      # BGR->RGB,[0,1]
        x = torch.from_numpy(rgb).permute(0, 3, 1, 2).unsqueeze(0)  # 1,T,C,H,W
        x = x.to(self.dm.info.device)

        try:
            with torch.inference_mode():
                y = self.net(x)
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" in message or "mps backend out of memory" in message:
                del x
                self.dm.empty_cache()
                raise RuntimeError(
                    "LADA restoration ran out of device memory. Retry with "
                    "--max_restore_side 320 --restore_clip_len 4 (or lower)."
                ) from exc
            raise
        y = y.squeeze(0).clamp(0, 1)

        out = (y * 255.0).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        restored = []
        for frame, placement, (oh, ow) in zip(out, placements, originals):
            top, left, content_h, content_w = placement
            frame = frame[top:top + content_h, left:left + content_w]
            if frame.shape[:2] != (oh, ow):
                frame = cv2.resize(
                    frame, (ow, oh), interpolation=cv2.INTER_CUBIC)
            restored.append(frame[:, :, ::-1].copy())            # RGB->BGR
        return restored
