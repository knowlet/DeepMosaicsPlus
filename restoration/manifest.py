"""Model manifest: declarative registry of restoration models.

A manifest entry describes one model: backend family, supported devices,
license, download URLs and integrity data. Built-in entries live in
restoration/manifests/models.yaml. Users can extend or override them with
--manifest my_models.yaml (entries are merged by id).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

BUILTIN_MANIFEST = os.path.join(os.path.dirname(__file__), "manifests", "models.yaml")


@dataclass
class ModelFile:
    filename: str
    urls: List[str] = field(default_factory=list)
    sha256: Optional[str] = None          # None -> warn instead of verify
    size_bytes: Optional[int] = None
    gdrive_id: Optional[str] = None       # alternative to a direct url

    def verify_hash(self) -> bool:
        return self.sha256 is not None


@dataclass
class ManifestEntry:
    id: str
    backend: str                          # basicvsrpp_portable | mosaicvr_lite | legacy_*
    title: str = ""
    version: str = "0.0.0"
    aliases: List[str] = field(default_factory=list)
    devices: List[str] = field(default_factory=lambda: ["cuda", "mps", "cpu"])
    license: str = "unknown"
    homepage: str = ""
    status: str = "released"              # released | planned
    notes: str = ""
    files: Dict[str, ModelFile] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, entry_id: str, raw: dict) -> "ManifestEntry":
        files = {}
        for key, f in (raw.get("files") or {}).items():
            files[key] = ModelFile(
                filename=f.get("filename", ""),
                urls=list(f.get("urls", []) or []),
                sha256=f.get("sha256"),
                size_bytes=f.get("size_bytes"),
                gdrive_id=f.get("gdrive_id"),
            )
        return cls(
            id=entry_id,
            backend=raw.get("backend", ""),
            title=raw.get("title", entry_id),
            version=str(raw.get("version", "0.0.0")),
            aliases=list(raw.get("aliases", []) or []),
            devices=list(raw.get("devices", ["cuda", "mps", "cpu"]) or []),
            license=raw.get("license", "unknown"),
            homepage=raw.get("homepage", ""),
            status=raw.get("status", "released"),
            notes=raw.get("notes", ""),
            files=files,
        )

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "title": self.title,
            "version": self.version,
            "aliases": self.aliases,
            "devices": self.devices,
            "license": self.license,
            "homepage": self.homepage,
            "status": self.status,
            "notes": self.notes,
        }


class Manifest:
    """In-memory registry with alias lookup and file merging."""

    def __init__(self):
        self._entries: Dict[str, ManifestEntry] = {}
        self._alias: Dict[str, str] = {}

    # ------------------------------------------------------------------
    def add(self, entry: ManifestEntry, override: bool = False) -> None:
        if entry.id in self._entries and not override:
            raise ValueError(f"model id already exists: {entry.id}")
        self._entries[entry.id] = entry
        self._alias[entry.id] = entry.id
        for alias in entry.aliases:
            self._alias[alias.lower()] = entry.id

    def get(self, name: str) -> Optional[ManifestEntry]:
        return self._entries.get(self._alias.get(name.lower()))

    def ids(self) -> List[str]:
        return sorted(self._entries.keys())

    def all(self) -> List[ManifestEntry]:
        return [self._entries[k] for k in sorted(self._entries.keys())]

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, extra_paths: Optional[List[str]] = None) -> "Manifest":
        m = cls()
        paths = [p for p in [BUILTIN_MANIFEST] if os.path.isfile(p)]
        paths += list(extra_paths or [])
        for path in paths:
            with open(path, "r", encoding="utf-8") as fp:
                raw = yaml.safe_load(fp) or {}
            for entry_id, body in (raw.get("models") or {}).items():
                m.add(ManifestEntry.from_dict(entry_id, body or {}), override=True)
        return m
