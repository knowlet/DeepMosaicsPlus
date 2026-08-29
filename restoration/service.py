"""RestorationService: one restoration pipeline for images, video, server.

    detect (BiSeNet) -> scene grouping -> unified crop -> RestorationBackend
    -> composite back into the frame

The service never touches ffmpeg or file layout; callers feed BGR uint8
frames and receive BGR uint8 frames. Static images go through the same
backend as videos by using a single-frame clip.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .device_manager import DeviceManager, pick_auto_backend
from .backends import RestorationBackend, build_backend
from .backends.base import RestoreRequest
from .model_manager import ModelManager


class RestorationService:
    def __init__(
        self,
        backend: RestorationBackend,
        device_manager: DeviceManager,
        detector: Optional[Callable] = None,
        model_id: str = "",
        quiet: bool = False,
    ):
        self.backend = backend
        self.dm = device_manager
        self.detector = detector
        self.model_id = model_id
        self.quiet = quiet
        self.max_restore_side = int(
            getattr(backend, "max_restore_side", 0) or 0)

    # ------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        opt,
        quiet: bool = False,
        skip_detector: bool = False,
    ) -> "RestorationService":
        """Build everything from CLI options (--model / --device / manifest)."""
        dm = DeviceManager(getattr(opt, "device", "auto"), quiet=quiet)

        requested = getattr(opt, "model", "auto")
        if requested == "auto":
            requested = pick_auto_backend(dm.info.type)
            if not quiet:
                print(f"[model] auto -> {requested} ({dm.info.type})")

        mm = ModelManager(
            extra_manifests=getattr(opt, "manifest", None),
            quiet=quiet,
        )
        resolved = mm.resolve(
            requested, dm,
            model_path_fallback=getattr(opt, "model_path", None),
        )
        backend = build_backend(resolved, opt)
        configure_memory_profile(opt, dm, backend, quiet=quiet)

        detector = None
        netM = None
        if not skip_detector:
            detector_path = getattr(opt, "mosaic_position_model_path", "auto")
            if detector_path in (None, "", "auto"):
                detector_model = mm.resolve("mosaic-detector", dm)
                opt.mosaic_position_model_path = detector_model.weights_path
                if not quiet:
                    print(f"[detector] {detector_model.entry.id} "
                          f"v{detector_model.entry.version}")
            from models import loadmodel          # project imports stay lazy
            netM = loadmodel.bisenet(opt, "mosaic")
            detector = _make_detector(netM, opt)

        svc = cls(backend=backend, device_manager=dm, detector=detector,
                  model_id=resolved.entry.id, quiet=quiet)
        svc.detector_net = netM
        svc._opt = opt
        if not quiet:
            e = resolved.entry
            print(f"[model] {e.id} v{e.version} | backend={e.backend} | "
                  f"license={e.license}")
            try:
                params = getattr(backend, "net", None)
                if params is not None and hasattr(params, "count_params"):
                    print(f"[model] parameters: {params.count_params():.2f}M")
            except Exception:
                pass
        return svc

    def load_detector(self) -> None:
        """Lazy-load the BiSeNet mosaic-position detector."""
        raise NotImplementedError  # created in factory; kept for API clarity

    # ------------------------------------------------------------------
    # detection
    # ------------------------------------------------------------------
    def detect(self, img_bgr: np.ndarray):
        """mask,x,y,size for the largest mosaic region (or zeros)."""
        if self.detector is None:
            raise RuntimeError("no detector loaded")
        return self.detector(img_bgr)

    def detect_all(self, img_bgr: np.ndarray):
        """Multi-mosaic detection: returns (mask_all, boxes).

        mask_all is the binary mask with all detected components (or None).
        boxes is list of (x, y, size) for each component, sorted by area.
        Falls back to single detection if multi not available.
        """
        if self.detector is None:
            raise RuntimeError("no detector loaded")
        # Try multi first
        try:
            from models import runmodel
            # need netM and opt; detector was built from opt, so we can retrieve via closure
            # But we store netM as detector_net
            netM = getattr(self, 'detector_net', None)
            # Try to get opt from detector closure - fallback to single
            # We store opt in service if available via _opt
            opt = getattr(self, '_opt', None)
            if netM is not None and opt is not None:
                mask_all, boxes = runmodel.get_mosaic_position_multi(img_bgr, netM, opt)
                if mask_all is not None and boxes:
                    return mask_all, boxes
        except Exception as e:
            print(f"detect_all fallback: {e}")
        # Fallback to single
        mask, x, y, s = self.detect(img_bgr)
        if mask is None or s <= 0:
            return None, []
        return mask, [(x, y, s)]

    # ------------------------------------------------------------------
    # single image path (also used by server)
    # ------------------------------------------------------------------
    def clean_image(self, img_bgr: np.ndarray, opt) -> np.ndarray:
        # Prefer multi-mosaic path when available
        try:
            mask_all, boxes = self.detect_all(img_bgr)
            if mask_all is not None and boxes:
                # Filter boxes by size (auto-adapt already handled in detector)
                valid_boxes = []
                min_size = int(getattr(opt, "min_mosaic_size", 40))
                auto = not bool(getattr(opt, "no_auto_adapt", False))
                for x, y, s in boxes:
                    if s <= 18:
                        continue
                    if not auto and s <= min_size:
                        continue
                    valid_boxes.append((x, y, s))
                if not valid_boxes:
                    return img_bgr
                # If single, use fast path
                if len(valid_boxes) == 1:
                    x, y, s = valid_boxes[0]
                    crop = img_bgr[max(0, y - s):y + s, max(0, x - s):x + s]
                    restored = self.restore_region_sequence([crop])[0]
                    return _composite(img_bgr, restored, mask_all, x, y, s, opt)
                # Multi: restore each crop independently and composite sequentially
                result = img_bgr.copy()
                # For compositing, we need per-component mask. We have mask_all,
                # but we will extract per-component masks via connected components
                # for precise feathering.
                try:
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_all.astype(np.uint8), connectivity=8)
                    # Build list of component masks
                    comp_masks = []
                    for i in range(1, num_labels):
                        comp = np.zeros_like(mask_all, dtype=np.uint8)
                        comp[labels == i] = 255
                        # Compute its box to match valid_boxes (nearest)
                        ys, xs = np.where(comp > 127)
                        if len(xs) == 0:
                            continue
                        cx = int(np.mean(xs)); cy = int(np.mean(ys))
                        x_min, x_max = int(np.min(xs)), int(np.max(xs))
                        y_min, y_max = int(np.min(ys)), int(np.max(ys))
                        s = max(x_max - x_min, y_max - y_min)//2
                        # Find closest valid_box
                        best = min(valid_boxes, key=lambda b: np.hypot(b[0]-cx, b[1]-cy)) if valid_boxes else None
                        if best and np.hypot(best[0]-cx, best[1]-cy) < s*1.5:
                            comp_masks.append((best[0], best[1], best[2], comp))
                        else:
                            comp_masks.append((cx, cy, s, comp))
                except Exception:
                    # Fallback: use boxes directly with full mask
                    comp_masks = [(x, y, s, mask_all) for x, y, s in valid_boxes]

                for x, y, s, comp_mask in comp_masks:
                    crop = result[max(0, y - s):y + s, max(0, x - s):x + s]
                    if crop.size == 0:
                        continue
                    restored = self.restore_region_sequence([crop])[0]
                    result = _composite(result, restored, comp_mask, x, y, s, opt)
                return result
        except Exception as e:
            print(f"multi clean fallback: {e}")
        # Fallback to single
        mask, x, y, size = self.detect(img_bgr)
        min_size = int(getattr(opt, "min_mosaic_size", 40))
        auto = not bool(getattr(opt, "no_auto_adapt", False))
        if mask is None or size <= 18:
            return img_bgr
        if not auto and size <= min_size:
            return img_bgr
        crop = img_bgr[max(0, y - size):y + size, max(0, x - size):x + size]
        restored = self.restore_region_sequence([crop])[0]
        return _composite(img_bgr, restored, mask, x, y, size, opt)

    # ------------------------------------------------------------------
    # temporal sequence over a fixed square region
    # ------------------------------------------------------------------
    def restore_region_sequence(
        self,
        crops: Sequence[np.ndarray],
        progress_cb: Optional[Callable[[int], None]] = None,
    ) -> List[np.ndarray]:
        """Restore the same region tracked across consecutive frames.

        Returns len(crops) restored crops. Windowing/blending follows the
        backend caps so every backend shares this code path.
        """
        n = len(crops)
        if n == 0:
            return []
        original_shapes = [crop.shape[:2] for crop in crops]
        working_crops = [
            _downscale_to_side(crop, self.max_restore_side) for crop in crops
        ]
        caps = self.backend.caps
        out: List[Optional[np.ndarray]] = [None] * n

        if caps.output_mode == "center":
            if not hasattr(self.backend, "restore_center"):
                raise RuntimeError(
                    f"backend {self.backend.name} declares output_mode='center' "
                    f"but provides no restore_center(); cannot drive it.")
            if hasattr(self.backend, "reset_sequence"):
                self.backend.reset_sequence()
            out = self._run_center_windows(working_crops, progress_cb=progress_cb)
        elif caps.max_clip_len <= 1:
            for i, c in enumerate(working_crops):
                out[i] = self.backend.restore(RestoreRequest(frames=[c]))[0]
                if progress_cb:
                    progress_cb(i + 1)
        else:
            out = self._run_chunked(caps, working_crops, progress_cb=progress_cb)

        result = []
        for i in range(n):
            restored = out[i] if out[i] is not None else working_crops[i]
            original_h, original_w = original_shapes[i]
            if restored.shape[:2] != (original_h, original_w):
                restored = cv2.resize(
                    restored, (original_w, original_h),
                    interpolation=cv2.INTER_CUBIC)
            result.append(restored)
        return result

    # -- chunked with overlap blending -----------------------------------
    def _run_chunked(self, caps, crops, progress_cb=None):
        n = len(crops)
        L = max(2, min(caps.max_clip_len, n))
        overlap = max(1, L // 4)
        step = max(1, L - overlap)

        if n <= L:
            starts = [0]
        else:
            starts = list(range(0, n - L + 1, step))
            if starts[-1] != n - L:
                starts.append(n - L)

        acc_w = np.zeros(n, np.float32)
        acc = None

        done = set()
        total = len(starts)
        for ci, s in enumerate(starts):
            e = min(s + L, n)
            window = list(range(s, e))
            frames = [crops[i] for i in window]
            restored = self.backend.restore(RestoreRequest(frames=frames))

            center = (s + e - 1) / 2.0
            span = max((e - s - 1) / 2.0, 1.0)
            for j, i in enumerate(window):
                wgt = float(np.clip(1.0 - abs(i - center) / (span + 1.0), 0.05, 1.0))
                f32 = restored[j].astype(np.float32)
                if acc is None:
                    acc = [np.zeros_like(f32) for _ in range(n)]
                acc[i] += f32 * wgt
                acc_w[i] += wgt

            if not self.quiet:
                print(f"\r[clip {ci+1}/{total}]", end="")
            if progress_cb:
                progress_cb(ci + 1)

        if not self.quiet:
            print()
        blended = []
        for i in range(n):
            if acc[i] is None or acc_w[i] <= 0:
                blended.append(crops[i])
            else:
                blended.append(np.clip(acc[i] / acc_w[i] + 0.5, 0, 255).astype(np.uint8))
        return blended

    # -- BVDNet-style sliding center windows ------------------------------
    def _run_center_windows(self, crops, progress_cb=None):
        S = getattr(self.backend, "S", 3)
        N = getattr(self.backend, "N", 2)
        pool_len = 2 * N * S + 1
        n = len(crops)

        def clamp_window(i):
            # The backend performs stride-S sampling inside this consecutive
            # pool. Applying S here as well would spread a 13-frame window over
            # 37 frames and destroy the original BVDNet semantics.
            idx = np.clip(i - N * S + np.arange(pool_len), 0, n - 1)
            return idx.astype(int).tolist()

        out: List[Optional[np.ndarray]] = [None] * n
        for i in range(n):
            win = clamp_window(i)
            pool = [crops[j] for j in win]
            out[i] = self.backend.restore_center(pool)
            if progress_cb:
                progress_cb(i + 1)
            if not self.quiet:
                print(f"\r[frame {i+1}/{n}]", end="")
        if not self.quiet:
            print()
        return out


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _make_detector(netM, opt):
    """Wrap legacy runmodel.get_mosaic_position (returns mask,x,y,size)."""
    from models import runmodel

    def detect(img_bgr):
        return runmodel.get_mosaic_position(img_bgr, netM, opt)

    return detect


def _downscale_to_side(frame: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale only; never enlarge a small or narrow restoration crop."""
    if max_side <= 0:
        return frame
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame
    scale = max_side / float(longest)
    return cv2.resize(
        frame,
        (max(4, int(round(w * scale))), max(4, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def recommended_memory_limits(device_type: str, total_gib: Optional[float] = None):
    """Conservative crop/window defaults for the supported workstation tier."""
    if device_type == "cuda":
        if total_gib is not None and total_gib <= 6.5:
            return 320, 4
        if total_gib is not None and total_gib <= 9.0:
            return 384, 6
        return 512, 8
    if device_type == "mps":
        # Unified memory is shared with macOS and cannot be queried reliably
        # through torch, so use the conservative 6-8 GB working profile.
        return 384, 4
    if device_type == "cpu":
        return 320, 2
    return 384, 1


def configure_memory_profile(opt, dm, backend, quiet=False):
    total_gib = None
    if dm.info.type == "cuda":
        try:
            import torch
            props = torch.cuda.get_device_properties(dm.info.device)
            total_gib = props.total_memory / float(1024 ** 3)
        except Exception:
            total_gib = None
    auto_side, auto_clip = recommended_memory_limits(dm.info.type, total_gib)
    side = int(getattr(opt, "max_restore_side", 0) or auto_side)
    clip = int(getattr(opt, "restore_clip_len", 0) or auto_clip)
    if side < 64:
        raise ValueError("--max_restore_side must be 0 or at least 64")
    if clip < 1:
        raise ValueError("--restore_clip_len must be 0 or at least 1")
    opt.max_restore_side = side
    opt.restore_clip_len = clip
    backend.max_restore_side = side
    if backend.caps.output_mode == "all":
        backend.caps = replace(
            backend.caps, max_clip_len=min(backend.caps.max_clip_len, clip))
    if not quiet:
        memory = f", VRAM={total_gib:.1f} GiB" if total_gib else ""
        print(f"[memory] crop<={side}px, clip<={backend.caps.max_clip_len}{memory}")


def _composite(img_origin, img_fake, mask, x, y, size, opt) -> np.ndarray:
    from util import image_processing as impro

    strength = float(np.clip(getattr(opt, "restore_strength", 1.0), 0.0, 1.0))
    if strength < 1.0:
        h, w = img_origin.shape[:2]
        x0, x1 = max(0, x - size), min(w, x + size)
        y0, y1 = max(0, y - size), min(h, y + size)
        if x1 > x0 and y1 > y0:
            source = img_origin[y0:y1, x0:x1]
            restored = cv2.resize(img_fake, (x1 - x0, y1 - y0),
                                  interpolation=cv2.INTER_CUBIC)
            img_fake = np.clip(
                source.astype(np.float32) * (1.0 - strength)
                + restored.astype(np.float32) * strength + 0.5,
                0, 255).astype(np.uint8)

    kwargs = {}
    if getattr(opt, "luma_sharpen", False):
        kwargs["luma_sharpen_amount"] = getattr(opt, "luma_sharpen_amount", 0.0)
    if getattr(opt, "bilateral_sharpen", False):
        kwargs["bilateral_sharpen_amount"] = getattr(opt, "bilateral_sharpen_amount", 0.0)
    if getattr(opt, "freq_inject", False):
        kwargs["freq_inject_amount"] = getattr(opt, "freq_inject_amount", 0.0)
    return impro.replace_mosaic(
        img_origin, img_fake, mask, x, y, size,
        bool(getattr(opt, "no_feather", False)), **kwargs)


def composite(img_origin, img_fake, mask, x, y, size, opt):
    """Public alias used by cores.clean."""
    return _composite(img_origin, img_fake, mask, x, y, size, opt)
