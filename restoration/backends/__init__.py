"""Backend factory: manifest entry -> live RestorationBackend."""

from __future__ import annotations

from ..model_manager import ResolvedModel
from .base import BackendCaps, RestoreRequest, RestorationBackend
from .basicvsrpp_portable import BasicVSRPPBackend, MosaicVSRNet
from .mosaicvr_lite import MosaicVRLiteBackend, MosaicVRLiteNet
from .lada_official import LadaOfficialBackend
from .legacy import (
    LegacyPix2PixBackend,
    LegacyVideoBackend,
    NullBackend,
    TraditionalBackend,
)


def build_backend(resolved: ResolvedModel, opt=None) -> RestorationBackend:
    """Create the concrete backend described by a manifest entry."""
    dm = resolved.device_manager
    family = resolved.entry.backend

    if family == "basicvsrpp_portable":
        return BasicVSRPPBackend(resolved.weights_path, dm)

    if family == "basicvsrpp_lada_official":
        return LadaOfficialBackend(resolved.weights_path, dm)

    if family == "mosaicvr_lite":
        return MosaicVRLiteBackend(resolved.weights_path, dm)

    if family == "legacy_pix2pix":
        netg = getattr(opt, "netG", "auto") if opt is not None else "auto"
        if netg == "video":            # mislabelled checkpoint: trust filename
            return LegacyVideoBackend(resolved.weights_path, dm)
        return LegacyPix2PixBackend(resolved.weights_path, dm, netg=netg)

    if family == "legacy_video":
        return LegacyVideoBackend(resolved.weights_path, dm)

    if family == "traditional":
        blur = getattr(opt, "tr_blur", 10) if opt is not None else 10
        down = getattr(opt, "tr_down", 10) if opt is not None else 10
        return TraditionalBackend(blur=blur, down=down)

    raise RuntimeError(f"Unknown backend family '{family}' in manifest entry "
                       f"'{resolved.entry.id}'")


__all__ = [
    "BackendCaps", "RestoreRequest", "RestorationBackend",
    "BasicVSRPPBackend", "MosaicVSRNet",
    "MosaicVRLiteBackend", "MosaicVRLiteNet",
    "LegacyPix2PixBackend", "LegacyVideoBackend", "LadaOfficialBackend",
    "TraditionalBackend", "NullBackend",
    "build_backend",
]
