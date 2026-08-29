"""MosaicVR-Lite: compact MPS-first video restoration backbone (v2).

Design goals:
* standard PyTorch ops only (Conv2d / PixelShuffle / grid_sample) so it runs
  everywhere, including Apple Silicon MPS and old GPUs via DirectML;
* small enough (~0.6M params) for interactive GUI use on low-end hardware;
* same clip-based interface as the quality backend, so swapping the default
  model later is a manifest edit, not a code change.

Architecture: shallow RGB encoder -> bidirectional gated recurrent refinement
at 1/2 resolution with warped alignment -> pixel-shuffle decoder with global
residual. Train with train/mosaicvr/train.py --arch lite.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BackendCaps, RestorationBackend, RestoreRequest
from .basicvsrpp_portable import backward_warp, invert_flow


class GatedRecBlock(nn.Module):
    """Cheap recurrent block: convGRU-style gate + flow-warped neighbour."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch * 2, ch, 3, 1, 1)
        self.gate = nn.Sequential(nn.Conv2d(ch * 3, ch, 3, 1, 1), nn.Sigmoid())

    def forward(self, x, state=None):
        if state is None:
            state = torch.zeros_like(x)
        cat = self.conv(torch.cat((x, state), dim=1))
        g = self.gate(torch.cat((x, state, cat), dim=1))
        return torch.tanh(cat) * g + state * (1 - g)


class MosaicVRLiteNet(nn.Module):
    def __init__(self, ch: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, ch, 3, 2, 1), nn.ReLU(inplace=True),
            Res(ch),
        )
        self.rec_b = GatedRecBlock(ch)
        self.rec_f = GatedRecBlock(ch)
        self.mix = nn.Sequential(nn.Conv2d(ch * 2, ch, 1), nn.ReLU(inplace=True))
        self.dec = nn.Sequential(
            Res(ch),
            nn.Conv2d(ch, ch * 4, 3, 1, 1),       # 2x pixel shuffle (enc is 1/2 res)
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 3, 3, 1, 1),
        )

    @staticmethod
    def compute_flows(frames_rgb01: torch.Tensor, out_hw=None) -> list:
        from .basicvsrpp_portable import MosaicVSRNet
        return MosaicVSRNet.compute_flows(frames_rgb01, out_hw)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        T = frames.shape[0]
        f = [self.enc(frames[t:t + 1]) for t in range(T)]          # (1,C,h,w)

        flows = []
        if T > 1:
            frames01 = (frames.float() * 0.5 + 0.5).clamp(0, 1)
            flows = self.compute_flows(frames01, out_hw=f[0].shape[-2:])

        # backward pass states
        states_b = [None] * T
        for t in range(T - 1, -1, -1):
            prev_state = states_b[t + 1] if t + 1 < T else None
            if prev_state is not None and flows:
                fl = flows[t]                       # t -> t+1
                prev_state = backward_warp(prev_state, fl.to(prev_state.dtype))
            states_b[t] = self.rec_b(f[t], prev_state)

        states_f = [None] * T
        for t in range(T):
            prev_state = states_f[t - 1] if t - 1 >= 0 else None
            if prev_state is not None and flows:
                fl = invert_flow(flows[t - 1])       # t -> t-1 sampling grid
                prev_state = backward_warp(prev_state, fl.to(prev_state.dtype))
            states_f[t] = self.rec_f(f[t], prev_state)

        outs = []
        for t in range(T):
            merged = self.mix(torch.cat((states_b[t], states_f[t]), dim=1))
            outs.append(frames[t:t + 1] + torch.tanh(self.dec(merged)))
        return torch.cat(outs, dim=0)

    def count_params(self) -> float:
        return sum(p.numel() for p in self.parameters()) / 1e6


class Res(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1))

    def forward(self, x):
        return x + self.body(x)


class MosaicVRLiteBackend(RestorationBackend):
    name = "mosaicvr-lite"
    caps = BackendCaps(output_mode="all", max_clip_len=8, input_size=192)

    def __init__(self, weights_path: str, device_manager, amp: bool = True):
        self.dm = device_manager
        obj = torch.load(weights_path, map_location="cpu", weights_only=True)
        if isinstance(obj, dict):
            for key in ("state_dict", "netG", "generator", "model"):
                if isinstance(obj.get(key), dict):
                    obj = obj[key]
                    break
        sd = obj if isinstance(obj, dict) else obj.state_dict()
        sd = {(k[7:] if k.startswith("module.") else k): v
              for k, v in sd.items()}
        ch = int(sd["enc.0.weight"].shape[0]) if "enc.0.weight" in sd else 32
        self.net = MosaicVRLiteNet(ch=ch)
        self.net.load_state_dict(sd)
        self.net.eval()
        self.dm.move(self.net)
        self.amp = amp

    def warmup(self):
        self.restore(RestoreRequest(
            frames=[np.zeros((64, 64, 3), np.uint8)] * 2))

    def _restore_impl(self, request: RestoreRequest):
        frames = request.frames
        h, w = frames[0].shape[:2]
        ph = (4 - h % 4) % 4
        pw = (4 - w % 4) % 4
        arr = np.stack([cv2.copyMakeBorder(f, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
                        for f in frames])
        rgb = arr[:, :, :, ::-1].copy()
        x = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        x = x.to(self.dm.info.device)
        ctx = self.dm.autocast_context(self.amp)
        with torch.inference_mode(), ctx:
            y = self.net(x)
        if y.dtype != torch.float32:
            y = y.float()
        y = y.clamp(-1, 1).mul(127.5).add(127.5).round().byte()
        out = y.permute(0, 2, 3, 1).cpu().numpy()
        if ph or pw:
            out = out[:, :h, :w]
        return [f[:, :, ::-1].copy() for f in out]
