"""ModelManager: manifest resolution, download, SHA-256 verify, cache.

Layout inside the cache root (default: pretrained_models/cache):
    <model_id>/<version>/<filename>

A model is considered installed when its weights file exists at the expected
path; downloads are streamed to <filename>.part and atomically renamed after
the hash check passes.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

import torch

from .device_manager import DeviceManager
from .manifest import Manifest, ManifestEntry


@dataclass
class ResolvedModel:
    """Everything a backend factory needs."""

    entry: ManifestEntry
    weights_path: str
    device_manager: DeviceManager


def _sha256_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        while True:
            block = fp.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


_QUIET_DOWNLOAD = False
_LAST_PROGRESS_BUCKET = -1


def _progress(count: int, block: int, total: int) -> None:
    global _LAST_PROGRESS_BUCKET
    if _QUIET_DOWNLOAD:
        return
    if total <= 0:
        sys.stdout.write(f"\rdownloading... {count * block / 1e6:.1f} MB")
    else:
        pct = min(100.0, 100.0 * count * block / total)
        bucket = int(pct)
        if bucket == _LAST_PROGRESS_BUCKET and pct < 100.0:
            return
        _LAST_PROGRESS_BUCKET = bucket
        bar = "#" * int(pct // 4)
        sys.stdout.write(f"\r[{bar:<25}] {pct:5.1f}%")
    sys.stdout.flush()


class ModelManager:
    def __init__(
        self,
        cache_root: Optional[str] = None,
        extra_manifests: Optional[List[str]] = None,
        quiet: bool = False,
    ):
        self.cache_root = os.environ.get(
            "DEEPMOSAICS_CACHE",
            cache_root or os.path.join("pretrained_models", "cache"),
        )
        self.manifest = Manifest.load(extra_manifests)
        self.quiet = quiet

    # ------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        if not self.quiet:
            print(msg)

    # ------------------------------------------------------------------
    @staticmethod
    def _looks_like_path(name: str) -> bool:
        return name.endswith((".pth", ".pt", ".ckpt")) or os.path.isfile(name)

    @staticmethod
    def _typed_checkpoint(name: str):
        """Return (backend, path) for an explicitly typed local checkpoint."""
        prefixes = {
            "lite": "mosaicvr_lite",
            "portable": "basicvsrpp_portable",
        }
        prefix, separator, path = name.partition(":")
        backend = prefixes.get(prefix.lower())
        if backend and separator:
            return backend, os.path.expanduser(path)
        return None

    def resolve(
        self,
        name: str,
        device_manager: DeviceManager,
        model_path_fallback: Optional[str] = None,
    ) -> ResolvedModel:
        """Resolve --model value (id / alias / checkpoint path).

        Legacy behaviour: an explicit .pth path with no manifest entry is
        wrapped as a legacy pix2pix-style checkpoint (type inferred from the
        filename), exactly like classic DeepMosaics usage.
        """
        entry = self.manifest.get(name)
        typed = self._typed_checkpoint(name)

        # New trainer outputs need an explicit family so they cannot be
        # mistaken for classic pix2pix/BVDNet checkpoints. Bare .pth paths
        # deliberately retain the original DeepMosaics semantics.
        if entry is None and typed is not None:
            return self._resolve_typed_checkpoint(
                typed[0], typed[1], device_manager)

        # explicit checkpoint path wins when it does not shadow an id
        if entry is None and self._looks_like_path(name):
            return self._resolve_local_checkpoint(name, device_manager)
        if entry is None and model_path_fallback and self._looks_like_path(model_path_fallback) \
                and name in ("legacy", "auto"):
            return self._resolve_local_checkpoint(model_path_fallback, device_manager)

        if entry is None:
            known = ", ".join(self.manifest.ids())
            raise RuntimeError(
                f"Unknown --model '{name}'. Known ids/aliases: {known}. "
                f"A .pth path is also accepted."
            )

        if entry.status != "released":
            raise RuntimeError(
                f"Model '{entry.id}' has status '{entry.status}': {entry.notes}"
            )

        weights_file = entry.files.get("weights")
        if weights_file is None or not weights_file.filename:
            if entry.backend.startswith("legacy_") and model_path_fallback \
                    and self._looks_like_path(model_path_fallback):
                return self._resolve_local_checkpoint(model_path_fallback,
                                                      device_manager)
            if entry.backend == "traditional":
                # algorithmic backend, nothing to download
                self._check_device(entry, device_manager)
                return ResolvedModel(entry=entry, weights_path=None,
                                     device_manager=device_manager)
            raise RuntimeError(f"Manifest entry '{entry.id}' declares no weights file.")

        target_dir = os.path.join(self.cache_root, entry.id, entry.version)
        if os.path.basename(weights_file.filename) != weights_file.filename:
            raise RuntimeError(
                f"Manifest entry '{entry.id}' has an unsafe weights filename: "
                f"{weights_file.filename!r}")
        target = os.path.join(target_dir, weights_file.filename)

        if not os.path.isfile(target):
            self._download(entry, weights_file, target_dir)

        assert os.path.isfile(target)
        self._check_hash(entry, weights_file, target)
        self._check_device(entry, device_manager)
        return ResolvedModel(entry=entry, weights_path=target, device_manager=device_manager)

    # ------------------------------------------------------------------
    def _resolve_local_checkpoint(self, path: str, device_manager: DeviceManager) -> ResolvedModel:
        from .manifest import ManifestEntry

        if not os.path.isfile(path):
            raise RuntimeError(
                f"Legacy checkpoint does not exist: {path}\n"
                "Download an original DeepMosaics checkpoint and pass its "
                "path with --model/--model_path, or use --model traditional."
            )
        base = os.path.basename(path).lower()
        if "video" in base:
            backend = "legacy_video"
        else:
            backend = "legacy_pix2pix"
        entry = ManifestEntry(
            id=os.path.splitext(base)[0] or "local-checkpoint",
            backend=backend,
            title=f"user checkpoint: {path}",
            devices=["cuda", "directml", "mps", "cpu"],
            license="user-provided",
            status="released",
            notes="resolved from local file; legacy inference path",
        )
        self._log(f"[model] treating {path!r} as legacy checkpoint ({backend})")
        return ResolvedModel(entry=entry, weights_path=path, device_manager=device_manager)

    def _resolve_typed_checkpoint(
        self, backend: str, path: str, device_manager: DeviceManager,
    ) -> ResolvedModel:
        if not os.path.isfile(path):
            raise RuntimeError(f"Typed checkpoint does not exist: {path}")
        label = "lite" if backend == "mosaicvr_lite" else "portable"
        entry = ManifestEntry(
            id=f"local-{label}",
            backend=backend,
            title=f"user-trained {label} checkpoint: {path}",
            devices=["cuda", "mps", "cpu"],
            license="user-trained",
            status="released",
            notes="explicitly typed local checkpoint",
        )
        self._check_device(entry, device_manager)
        self._log(f"[model] treating {path!r} as {backend}")
        return ResolvedModel(entry=entry, weights_path=path,
                             device_manager=device_manager)

    # ------------------------------------------------------------------
    def _check_device(self, entry: ManifestEntry, dm: DeviceManager) -> None:
        if dm.info.type not in entry.devices:
            raise RuntimeError(
                f"Model '{entry.id}' supports devices {entry.devices} but the active "
                f"device is '{dm.info.type}'. Pick another model or another --device; "
                f"DeepMosaicsPlus will not silently swap models for you."
            )

    def _check_hash(self, entry: ManifestEntry, wf, path: str) -> None:
        if wf.size_bytes is not None and os.path.getsize(path) != int(wf.size_bytes):
            raise RuntimeError(
                f"Size mismatch for {path}. expected {wf.size_bytes} bytes, "
                f"got {os.path.getsize(path)} bytes")
        if not wf.verify_hash():
            self._log(f"[model] WARNING: no sha256 recorded for {entry.id}; skipping verify.")
            return
        digest = _sha256_of(path)
        if digest.lower() != wf.sha256.lower():
            raise RuntimeError(
                f"SHA-256 mismatch for {path}.\n expected {wf.sha256}\n got      {digest}\n"
                f"Delete the cached file to retry the download."
            )
        self._log(f"[model] sha256 ok: {entry.id}/{entry.version}")

    # ------------------------------------------------------------------
    def _download(self, entry: ManifestEntry, wf, target_dir: str) -> None:
        global _QUIET_DOWNLOAD, _LAST_PROGRESS_BUCKET
        _QUIET_DOWNLOAD = self.quiet
        _LAST_PROGRESS_BUCKET = -1
        os.makedirs(target_dir, exist_ok=True)
        target = os.path.join(target_dir, wf.filename)
        tmp = target + ".part"

        last_err = None
        sources = list(wf.urls)
        if not sources and wf.gdrive_id:
            sources.append(wf.gdrive_id)
        for url in sources:
            try:
                self._log(f"[model] downloading {entry.id} v{entry.version}\n         {url}")
                if "drive.google.com" in url or "drive.usercontent.google.com" in url \
                        or (wf.gdrive_id and url == wf.gdrive_id):
                    self._download_gdrive(url or wf.gdrive_id, tmp)
                else:
                    urllib.request.urlretrieve(url, tmp, reporthook=_progress)
                sys.stdout.write("\n")
                # A corrupt or interrupted download must never become a cached
                # model. Verify the temporary file, then publish atomically.
                self._check_hash(entry, wf, tmp)
                os.replace(tmp, target)
                return
            except Exception as e:  # try next mirror
                last_err = e
                self._log(f"[model] download failed: {e}")
                if os.path.isfile(tmp):
                    os.remove(tmp)

        if os.path.isfile(tmp):
            os.remove(tmp)
        manual = (
            f"Place the file manually at:\n  {target}\n"
            f"(source: {entry.homepage or 'see manifest'})"
        )
        if last_err is not None:
            raise RuntimeError(f"Could not download weights for {entry.id}: {last_err}\n{manual}")
        raise RuntimeError(
            f"No download URL configured for '{entry.id}' yet.\n{manual}"
        )

    def _download_gdrive(self, spec: str, dst: str) -> None:
        """Best-effort Google Drive fetch (works for public files)."""
        fid = wf_gdrive_id = spec
        m = re.search(r"id=([\w-]{10,})", spec)
        if m:
            fid = m.group(1)
        elif "/d/" in spec:
            fid = spec.split("/d/")[1].split("/")[0]
        url = f"https://drive.google.com/uc?export=download&confirm=t&id={fid}"
        urllib.request.urlretrieve(url, dst, reporthook=_progress)


# ----------------------------------------------------------------------
def load_state_dict_flex(path: str) -> dict:
    """Load a checkpoint saved in common layouts (raw state_dict / nested)."""
    # Runtime checkpoints are tensor dictionaries. ``weights_only=True`` keeps
    # a downloaded checkpoint from executing arbitrary pickle payloads.
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "netG", "generator", "model", "params"):
            if key in obj and isinstance(obj[key], dict):
                sd = obj[key]
                sd = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
                return sd
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    return obj
