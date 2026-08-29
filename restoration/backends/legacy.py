"""Legacy DeepMosaics backends behind the unified interface.

Wraps the original pix2pix / pix2pixHD / BVDNet checkpoints so AMD
(DirectML) and low-spec machines keep working. Inference quirks of the old
models (input resize to 128/512, BVDNet N/S/T windowing, previous-frame
feedback) are fully encapsulated here.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import torch

from .base import BackendCaps, RestorationBackend, RestoreRequest


def infer_legacy_kind(path_or_name: str) -> str:
    """Classic behaviour: guess generator family from the filename."""
    base = os.path.basename(str(path_or_name)).lower()
    if "video" in base:
        return "video"
    if "hd" in base:
        return "HD"
    if "resnet_9blocks" in base:
        return "resnet_9blocks"
    if "unet_256" in base:
        return "unet_256"
    if "unet_128" in base or "unet" in base:
        return "unet_128"
    return "unet_128"


class LegacyPix2PixBackend(RestorationBackend):
    """Per-frame restoration with the original pix2pix generators."""

    name = "legacy-pix2pix"
    caps = BackendCaps(output_mode="all", max_clip_len=1, input_size=0)

    def __init__(self, weights_path: str, device_manager, netg: str = "auto"):
        import sys
        sys.path.insert(0, self._repo_root())
        from models.pix2pix_model import define_G          # noqa: E402
        from models.pix2pixHD_model import define_G as define_HD  # noqa: E402

        kind = infer_legacy_kind(weights_path) if netg in ("auto", "", None) else netg
        if kind == "HD":
            net = define_HD(3, 3, 64, "global", 4)
        else:
            net = define_G(3, 3, 64, kind, norm="batch",
                           use_dropout=True, init_type="normal", gpu_ids=[])

        sd = torch.load(weights_path, map_location="cpu", weights_only=True)
        if hasattr(sd, "state_dict"):
            sd = sd.state_dict()
        net.load_state_dict(sd)
        net.eval()
        self.kind = kind
        self.dm = device_manager
        self.net = device_manager.move(net)

    @staticmethod
    def _repo_root():
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def _restore_impl(self, request: RestoreRequest):
        size = 512 if self.kind == "HD" else (256 if self.kind == "unet_256" else 128)
        outs = []
        for frame in request.frames:
            oh, ow = frame.shape[:2]
            img = cv2.resize(frame, (size, size), interpolation=cv2.INTER_CUBIC) \
                if (oh, ow) != (size, size) else frame
            x = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float()
            x = ((x / 255.0 - 0.5) / 0.5).unsqueeze(0).to(self.dm.info.device)
            ctx = self.dm.autocast_context(False)   # legacy nets trained fp32
            with torch.inference_mode(), ctx:
                y = self.net(x)
            y = y[0].float().cpu().clamp(-1, 1)
            y = ((y + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
            out = y[:, :, ::-1].copy()
            if out.shape[:2] != (oh, ow):           # honour the size contract
                out = cv2.resize(out, (ow, oh), interpolation=cv2.INTER_LANCZOS4)
            outs.append(out)
        return outs


class LegacyVideoBackend(RestorationBackend):
    """BVDNet fusion cleaner (N=2, S=3, T=5), same semantics as before.

    The service feeds windows of 2*N*S+1 consecutive frames; only the middle
    output is meaningful (caps.output_mode == 'center').
    """

    name = "legacy-video"
    caps = BackendCaps(output_mode="center", max_clip_len=13, input_size=256)

    def __init__(self, weights_path: str, device_manager, N: int = 2, S: int = 3):
        import sys
        sys.path.insert(0, self._repo_root())
        from models.BVDNet import define_G                   # noqa: E402

        gpu_id = "0" if device_manager.info.type == "cuda" else "-1"
        self.net = define_G(N=N, n_blocks=4, gpu_id=gpu_id)
        sd = torch.load(weights_path, map_location="cpu", weights_only=True)
        if hasattr(sd, "state_dict"):
            sd = sd.state_dict()
        self.net.load_state_dict(sd)
        self.net.eval()
        self.dm = device_manager
        self.net = device_manager.move(self.net)
        self.N, self.S = N, S
        self.T = 2 * N + 1
        self._previous = None

    @staticmethod
    def _repo_root():
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def restore_center(self, pool: list) -> np.ndarray:
        """pool: exactly 2*N*S+1 BGR frames; returns restored middle frame."""
        assert len(pool) == 2 * self.N * self.S + 1, \
            f"BVDNet needs {2*self.N*self.S+1} frames, got {len(pool)}"
        mid = self.N * self.S
        crops = [pool[pos] for pos in range(0, len(pool), self.S)][:self.T]
        arr = np.stack([cv2.resize(c, (256, 256), interpolation=cv2.INTER_CUBIC)
                        for c in crops])                     # T,H,W,3 BGR
        rgb = arr[:, :, :, ::-1].astype(np.float32)
        stream = ((rgb / 255.0 - 0.5) / 0.5).transpose(0, 3, 1, 2)   # T,C,H,W
        stream = torch.from_numpy(stream).unsqueeze(0).permute(0, 2, 1, 3, 4)  # 1,C,T,H,W

        dev = self.dm.info.device
        stream = stream.to(dev)
        if self._previous is None:
            # The sampled centre is crops[N] (pool[N*S]), not crops[N*S].
            prev = cv2.resize(crops[self.N], (256, 256),
                              interpolation=cv2.INTER_CUBIC)
            prev_t = torch.from_numpy(
                (((prev[:, :, ::-1].astype(np.float32) / 255.0 - 0.5) / 0.5))
                .transpose(2, 0, 1)).unsqueeze(0).to(dev)
        else:
            prev_t = self._previous.to(dev)
        with torch.inference_mode():
            pred = self.net(stream, prev_t)
        self._previous = pred.detach()
        out = pred[0].float().cpu().clamp(-1, 1)
        out = ((out + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
        out = out[:, :, ::-1].copy()
        target_h, target_w = pool[mid].shape[:2]
        if out.shape[:2] != (target_h, target_w):
            out = cv2.resize(out, (target_w, target_h),
                             interpolation=cv2.INTER_LANCZOS4)
        return out

    def reset_sequence(self) -> None:
        """Reset autoregressive feedback at each scene/run boundary."""
        self._previous = None

    def _restore_impl(self, request: RestoreRequest):
        """Static/by-frame compatibility using a repeated temporal pool."""
        self.reset_sequence()
        outs = []
        for frame in request.frames:
            outs.append(self.restore_center(
                [frame] * (2 * self.N * self.S + 1)))
        return outs


class TraditionalBackend(RestorationBackend):
    """Blur + downsample + Lanczos upscale; no network at all."""

    name = "traditional"
    caps = BackendCaps(output_mode="all", max_clip_len=1, input_size=0)

    def __init__(self, blur: int = 10, down: int = 10):
        self.blur, self.down = max(1, blur | 1), max(1, down)

    def _restore_impl(self, request: RestoreRequest):
        outs = []
        for frame in request.frames:
            h, w = frame.shape[:2]
            img = cv2.blur(frame, (self.blur, self.blur))
            small = img[::self.down, ::self.down, :]
            outs.append(cv2.resize(small, (w, h), interpolation=cv2.INTER_LANCZOS4))
        return outs


class NullBackend(RestorationBackend):
    """Identity backend (returns inputs unchanged); used for tests/dry runs."""

    name = "null"
    caps = BackendCaps(output_mode="all", max_clip_len=32, input_size=0)

    def __init__(self, device_manager=None):
        pass

    def _restore_impl(self, request: RestoreRequest):
        return [f.copy() for f in request.frames]
