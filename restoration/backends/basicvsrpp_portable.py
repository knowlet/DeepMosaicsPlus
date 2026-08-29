"""Portable BasicVSR++-style recurrent video restoration.

Quality backend following the LADA recipe (recurrent propagation over a clip,
flow-guided alignment, x4 latent upsampling). Differences from the official
BasicVSR++:

* alignment uses backward warping (grid_sample) instead of DCNv2 so the whole
  network is plain PyTorch ops and runs on CUDA / MPS / CPU;
* optical flow comes from classical OpenCV Farneback computed per clip, which
  removes the dependency on a pretrained SpyNet checkpoint.

Weights are produced by train/mosaicvr/train.py --arch basicvsrpp (LADA-style
online mosaic synthesis + DeepMosaics perceptual / feature-matching /
multi-scale GAN losses) so they always match this implementation.
"""

from __future__ import annotations

import contextlib

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BackendCaps, RestorationBackend, RestoreRequest


# ----------------------------------------------------------------------
# building blocks
# ----------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, n_feats: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.body(x)


def _meshgrid(h, w, device, dtype):
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((xs, ys), dim=-1).unsqueeze(0)      # 1,H,W,2 (x,y)


def backward_warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Sample x at (pixel + flow); flow in pixels (B,2,H,W), same grid."""
    b, _, h, w = x.shape
    base = _meshgrid(h, w, x.device, flow.dtype)
    coords = base + flow.permute(0, 2, 3, 1)
    gx = coords[..., 0] / max(w - 1, 1) * 2.0 - 1.0
    gy = coords[..., 1] / max(h - 1, 1) * 2.0 - 1.0
    grid = torch.stack((gx, gy), dim=-1)
    # Older MPS builds do not implement padding_mode='border'. Clamping the
    # grid then using zeros is equivalent to border sampling.
    if x.device.type == "mps":
        grid = grid.clamp(-1.0, 1.0)
        padding_mode = "zeros"
    else:
        padding_mode = "border"
    return F.grid_sample(x, grid, mode="bilinear", padding_mode=padding_mode,
                         align_corners=True)


def invert_flow(flow: torch.Tensor) -> torch.Tensor:
    """Inverse of a flow field: inv(x) ~= -f(x + f(x)) (f maps A->B)."""
    return -backward_warp(flow, flow)


def compose_flow(f_ij: torch.Tensor, f_jk: torch.Tensor) -> torch.Tensor:
    """Compose i->j with j->k into approximate i->k."""
    return f_ij + backward_warp(f_jk, f_ij)


# ----------------------------------------------------------------------
# core network
# ----------------------------------------------------------------------
class FlowGuidedPropagation(nn.Module):
    """Recurrent second-order flow-guided propagation over a clip.

    flows[i] maps frame i grid -> frame i+1 (pixels). Traversal direction
    decides whether raw or inverted flows are used for alignment.
    """

    def __init__(self, n_feats: int, n_blocks: int = 3):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(n_feats * 3, n_feats, 1, 1, 0),
            nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(*[ResBlock(n_feats) for _ in range(n_blocks)])

    def _align(self, prop: torch.Tensor, f1, f2):
        a1 = backward_warp(prop, f1.to(prop.dtype))
        if f2 is None:
            a2 = a1
        else:
            a2 = backward_warp(prop, f2.to(prop.dtype))
        return a1, a2

    def forward(self, feats, flows, reverse: bool):
        """
        feats: list[T] of (B,C,H,W)
        flows: list[T-1] of (B,2,H,W); flows[i]: frame i -> frame i+1.
        reverse=False -> forward traversal (0..T-1);
        reverse=True  -> backward traversal (T-1..0).
        """
        T = len(feats)
        out = [None] * T
        order = list(range(T - 1, -1, -1)) if reverse else list(range(T))
        out[order[0]] = feats[order[0]]

        for cur in order[1:]:
            nxt = cur + 1 if reverse else cur - 1
            prop = out[nxt]
            if reverse:
                f1 = flows[cur]                                   # cur -> nxt
                f2 = compose_flow(flows[cur], flows[cur + 1]) \
                    if cur + 1 < T - 1 else None                  # cur -> cur+2
            else:
                f1 = invert_flow(flows[nxt])                      # nxt -> cur inverted
                f2 = compose_flow(invert_flow(flows[nxt]), invert_flow(flows[nxt - 1])) \
                    if nxt - 1 >= 0 else None                     # cur -> cur-2
            a1, a2 = self._align(prop, f1, f2)
            fused = self.fusion(torch.cat((feats[cur], a1, a2), dim=1))
            out[cur] = self.refine(fused) + feats[cur]
        return out


class UpsamplePath(nn.Module):
    """1/4-resolution merged features -> full-res residual RGB ([-1,1])."""

    def __init__(self, n_feats: int):
        super().__init__()
        self.conv = nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1)
        self.up1 = nn.Sequential(nn.Conv2d(n_feats, n_feats * 4, 3, 1, 1),
                                 nn.PixelShuffle(2), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.Conv2d(n_feats, n_feats * 4, 3, 1, 1),
                                 nn.PixelShuffle(2), nn.ReLU(inplace=True))
        self.out = nn.Conv2d(n_feats, 3, 3, 1, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.up1(x)
        x = self.up2(x)
        return torch.tanh(self.out(x))


class MosaicVSRNet(nn.Module):
    """Input/output: (T,C,H,W) RGB in [-1,1]."""

    def __init__(self, n_feats: int = 48, n_props_blocks: int = 3):
        super().__init__()
        self.extract = nn.Sequential(
            nn.Conv2d(3, n_feats, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, 2, 1), nn.ReLU(inplace=True),
            ResBlock(n_feats), ResBlock(n_feats),
        )
        self.prop_backward = FlowGuidedPropagation(n_feats, n_props_blocks)
        self.prop_forward = FlowGuidedPropagation(n_feats, n_props_blocks)
        self.upsample = UpsamplePath(n_feats)

    # ------------------------------------------------------------------
    @staticmethod
    def compute_flows(frames_rgb01: torch.Tensor, out_hw=None) -> list:
        """frames: (T,3,H,W) RGB in [0,1]; out_hw: (H,W) grid the flows are
        rescaled to (feature resolution). Returns list[T-1] of (1,2,H,W)
        float32 tensors on the same device; flows[i]: frame i -> frame i+1."""
        import torch.nn.functional as F

        T = frames_rgb01.shape[0]
        if T < 2:
            return []
        if out_hw is not None:
            frames_small = F.interpolate(frames_rgb01.float(), size=out_hw,
                                         mode="bilinear", align_corners=False)
        else:
            frames_small = frames_rgb01.float()
        _, _, fh, fw = frames_small.shape

        win = max(5, min(21, fh if fh % 2 == 1 else fh - 1))
        imgs = frames_small.detach().cpu().numpy()
        flows = []
        for i in range(T - 1):
            prev = (np.transpose(imgs[i], (1, 2, 0)) * 255).astype(np.uint8)
            nxt = (np.transpose(imgs[i + 1], (1, 2, 0)) * 255).astype(np.uint8)
            g_prev = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
            g_nxt = cv2.cvtColor(nxt, cv2.COLOR_RGB2GRAY)
            fl = cv2.calcOpticalFlowFarneback(
                g_prev, g_nxt, None, 0.5, 3, int(win), 3, 5, 1.2, 0)
            flows.append(torch.from_numpy(fl).permute(2, 0, 1).unsqueeze(0))
        return [f.to(frames_rgb01.device) for f in flows]

    # ------------------------------------------------------------------
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        T = frames.shape[0]
        feats = [self.extract(frames[t:t + 1]) for t in range(T)]   # (1,C,h,w)

        if T == 1:                       # static image -> single-frame path
            merged = torch.cat((feats[0], feats[0]), dim=1)          # (1,2C,h,w)
            return frames + self.upsample(merged)

        feat_hw = feats[0].shape[-2:]
        frames01 = (frames.float() * 0.5 + 0.5).clamp(0, 1)
        flows = self.compute_flows(frames01, out_hw=feat_hw)

        back = self.prop_backward(feats, flows, reverse=True)
        fwd = self.prop_forward(feats, flows, reverse=False)

        outs = []
        for t in range(T):
            merged = torch.cat((back[t], fwd[t]), dim=1)             # (1,2C,h,w)
            outs.append(frames[t:t + 1] + self.upsample(merged))
        return torch.cat(outs, dim=0)

    def count_params(self) -> float:
        return sum(p.numel() for p in self.parameters()) / 1e6


# ----------------------------------------------------------------------
# backend wrapper
# ----------------------------------------------------------------------
class BasicVSRPPBackend(RestorationBackend):
    name = "lada-basicvsrpp"
    caps = BackendCaps(output_mode="all", max_clip_len=15, input_size=256)

    def __init__(self, weights_path: str, device_manager, amp: bool = True):
        self.dm = device_manager
        sd = _load_sd(weights_path)
        n_feats = 48
        w = sd.get("extract.0.weight")
        if w is not None:
            n_feats = int(w.shape[0])
        self.net = MosaicVSRNet(n_feats=n_feats)
        missing, unexpected = self.net.load_state_dict(sd, strict=False)
        if len(missing) > 8 or len(unexpected) > 8:
            raise RuntimeError(
                f"Weights {weights_path!r} do not match the portable BasicVSR++ "
                f"(missing={len(missing)}, unexpected={len(unexpected)} keys).\n"
                f"Train matching weights with: train/mosaicvr/train.py --arch basicvsrpp"
            )
        if missing or unexpected:
            print(f"[model] loaded with missing={len(missing)} unexpected={len(unexpected)} keys")
        self.net.eval()
        self.dm.move(self.net)
        self.amp = amp

    def warmup(self):
        self.restore(RestoreRequest(
            frames=[np.zeros((64, 64, 3), np.uint8)] * 2))

    def _restore_impl(self, request: RestoreRequest):
        frames = request.frames
        h, w = frames[0].shape[:2]

        ph = (4 - h % 4) % 4          # two stride-2 convs + two PixelShuffles
        pw = (4 - w % 4) % 4
        arr = np.stack([
            cv2.copyMakeBorder(f, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
            for f in frames
        ])                                            # T,H,W,3 BGR uint8

        rgb = arr[:, :, :, ::-1].copy()               # -> RGB
        x = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        x = x.to(self.dm.info.device)

        ctx = self.dm.autocast_context(self.amp)
        with torch.inference_mode(), ctx:
            y = self.net(x)
        if y.dtype != torch.float32:
            y = y.float()

        y = y.clamp(-1, 1).mul(127.5).add(127.5).round().byte()
        out = y.permute(0, 2, 3, 1).cpu().numpy()     # T,Hp,Wp,3 RGB
        if ph or pw:
            out = out[:, :h, :w]
        return [f[:, :, ::-1].copy() for f in out]    # back to BGR


def _load_sd(path: str) -> dict:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "netG", "generator"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if hasattr(obj, "state_dict"):
        obj = obj.state_dict()
    return {(k[7:] if k.startswith("module.") else k): v for k, v in obj.items()}
