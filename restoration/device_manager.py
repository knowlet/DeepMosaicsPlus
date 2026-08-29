"""Central device selection.

One place decides the torch device so models never guess on their own.
Supported choices: auto | cuda | mps | directml | cpu.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeviceInfo:
    """Resolved device plus the capabilities that matter to backends."""

    device: object            # torch.device or torch_directml device wrapper
    type: str                 # cuda | mps | directml | cpu
    name: str = ""
    supports_amp: bool = False

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return f"<Device {self.type}:{self.name} amp={self.supports_amp}>"


class DeviceManager:
    """Resolve and cache the active torch device."""

    def __init__(self, requested: str = "auto", quiet: bool = False):
        self.requested = (requested or "auto").lower()
        self.quiet = quiet
        self.info: DeviceInfo = self._resolve(self.requested)

    # ------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        if not self.quiet:
            print(msg)

    def _detect_directml(self):
        try:
            import torch_directml  # type: ignore
            dev = torch_directml.device()
            return dev, torch_directml.device_name(0) if hasattr(torch_directml, "device_name") else "directml"
        except Exception:
            return None, ""

    def _resolve(self, requested: str) -> DeviceInfo:
        import torch

        if requested == "auto":
            if torch.cuda.is_available():
                return self._resolve("cuda")
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                return self._resolve("mps")
            dev, name = self._detect_directml()
            if dev is not None:
                info = DeviceInfo(device=dev, type="directml", name=str(name))
                self._log(f"[device] DirectML detected: {info.name}")
                return info
            return self._resolve("cpu")

        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "--device cuda was requested but CUDA is not available. "
                    "Use --device auto/mps/directml/cpu."
                )
            idx = torch.cuda.current_device()
            info = DeviceInfo(
                device=torch.device("cuda", idx),
                type="cuda",
                name=torch.cuda.get_device_name(idx),
                supports_amp=True,
            )
            self._log(f"[device] CUDA: {info.name}")
            return info

        if requested == "mps":
            mps_ok = (
                getattr(torch.backends, "mps", None) is not None
                and torch.backends.mps.is_available()
            )
            if not mps_ok:
                raise RuntimeError(
                    "--device mps was requested but MPS is not available "
                    "(need Apple Silicon + PyTorch >= 1.12, macOS >= 12.3)."
                )
            info = DeviceInfo(device=torch.device("mps"), type="mps", name="Apple GPU")
            self._log("[device] MPS: Apple GPU")
            return info

        if requested == "directml":
            dev, name = self._detect_directml()
            if dev is None:
                raise RuntimeError(
                    "--device directml was requested but torch-directml is not installed."
                )
            info = DeviceInfo(device=dev, type="directml", name=str(name))
            self._log(f"[device] DirectML: {info.name}")
            return info

        if requested == "cpu":
            info = DeviceInfo(device=torch.device("cpu"), type="cpu", name="CPU")
            self._log("[device] CPU")
            return info

        raise RuntimeError(
            f"Unknown --device '{requested}'. Choices: auto|cuda|mps|directml|cpu."
        )

    # ------------------------------------------------------------------
    def move(self, module):
        """Move a nn.Module to the resolved device (DirectML safe)."""
        if self.info.type == "directml":
            module = module.to(self.info.device)
        else:
            module = module.to(self.info.device)
        return module

    def autocast_context(self, enabled: bool = True):
        """Mixed precision context; disabled where unsupported."""
        import contextlib
        import torch

        if enabled and self.info.supports_amp and self.info.type in ("cuda", "mps"):
            dtype = torch.bfloat16 if self.info.type == "mps" else torch.float16
            return torch.autocast(device_type=self.info.type, dtype=dtype)
        return contextlib.nullcontext()

    def empty_cache(self) -> None:
        import torch

        if self.info.type == "cuda":
            torch.cuda.empty_cache()
        elif (self.info.type == "mps" and hasattr(torch, "mps")
              and hasattr(torch.mps, "empty_cache")):
            torch.mps.empty_cache()


def pick_auto_backend(device_type: str) -> str:
    """Policy: which backend family should `--model auto` choose."""
    if device_type in ("cuda", "mps"):
        return "lada-official-basicvsrpp"  # released author checkpoint
    if device_type == "directml":
        return "legacy"              # AMD fallback
    return "traditional"             # deterministic CPU fallback
