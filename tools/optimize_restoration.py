"""Closed-loop calibration for restoration strength.

The optimizer never rewrites the stable model manifest. It calibrates candidates
on the first scene, evaluates the selected candidate on a disjoint holdout scene,
and writes an auditable JSON report with an actionable ``--restore_strength``
flag. Promotion requires holdout PSNR/SSIM gains without temporal regression.

Example::

    .venv/bin/python tools/optimize_restoration.py --model quality --device mps
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from types import SimpleNamespace

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from restoration.backends import build_backend
from restoration.device_manager import DeviceManager, pick_auto_backend
from restoration.model_manager import ModelManager
from restoration.service import RestorationService
from restoration.service import configure_memory_profile
from tools import closed_loop_test as closed


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _blend(source: np.ndarray, restored: np.ndarray, strength: float) -> np.ndarray:
    return np.clip(
        source.astype(np.float32) * (1.0 - strength)
        + restored.astype(np.float32) * strength + 0.5,
        0, 255).astype(np.uint8)


def _metrics(targets, sources, restored, indices, strength):
    psnr_values, ssim_values = [], []
    candidates = []
    for i in indices:
        candidate = _blend(sources[i], restored[i], strength)
        candidates.append(candidate)
        psnr_values.append(closed.psnr(targets[i], candidate))
        ssim_values.append(closed.ssim(targets[i], candidate))
    selected_targets = [targets[i] for i in indices]
    return {
        "psnr": round(float(np.mean(psnr_values)), 5),
        "ssim": round(float(np.mean(ssim_values)), 7),
        "temporal_l1": round(
            closed.temporal_residual_l1(selected_targets, candidates), 7),
    }


def _score(metrics):
    # PSNR remains primary; SSIM rewards structure and the temporal term
    # discourages a superficially sharp but flickering calibration candidate.
    return float(metrics["psnr"] + 10.0 * metrics["ssim"]
                 - 10.0 * metrics["temporal_l1"])


def _prepare_work(work_dir: str):
    if work_dir:
        root = os.path.abspath(work_dir)
        required = ["original.mp4", "mosaicked.mp4", "gt_boxes.json"]
        missing = [name for name in required
                   if not os.path.isfile(os.path.join(root, name))]
        if missing:
            raise RuntimeError(f"work directory is missing: {', '.join(missing)}")
        return root
    root = tempfile.mkdtemp(prefix="dmp_optimize_")
    closed.WORK = root
    closed.phase_a()
    return root


def _restore_disjoint(service, sources, split):
    """Restore calibration and holdout separately to prevent temporal leak."""
    calibration = service.restore_region_sequence(sources[:split])
    holdout = service.restore_region_sequence(sources[split:])
    return calibration + holdout


def optimize(args):
    root = _prepare_work(args.work_dir)
    closed.WORK = root
    original = closed.read_frames(os.path.join(root, "original.mp4"))
    mosaicked = closed.read_frames(os.path.join(root, "mosaicked.mp4"))
    with open(os.path.join(root, "gt_boxes.json"), "r", encoding="utf-8") as fp:
        gt = json.load(fp)
    n = min(len(original), len(mosaicked), len(gt))
    if n < 4:
        raise RuntimeError("at least four ground-truth frames are required")

    half = closed.MOSAIC_HALF
    targets, sources = [], []
    for frame_idx, tx, ty in gt[:n]:
        targets.append(original[frame_idx][ty-half:ty+half, tx-half:tx+half])
        sources.append(mosaicked[frame_idx][ty-half:ty+half, tx-half:tx+half])

    dm = DeviceManager(args.device, quiet=False)
    model_name = (pick_auto_backend(dm.info.type)
                  if args.model == "auto" else args.model)
    resolved = ModelManager(quiet=False).resolve(model_name, dm)
    profile_opt = SimpleNamespace(
        tr_blur=args.tr_blur, tr_down=args.tr_down,
        max_restore_side=args.max_restore_side,
        restore_clip_len=args.restore_clip_len)
    backend = build_backend(resolved, profile_opt)
    configure_memory_profile(profile_opt, dm, backend, quiet=True)
    service = RestorationService(backend, dm, detector=None,
                                 model_id=resolved.entry.id, quiet=True)
    split = int(round(n * args.calibration_fraction))
    split = min(max(1, split), n - 1)
    calibration_idx = list(range(split))
    holdout_idx = list(range(split, n))
    started = time.time()
    restored = _restore_disjoint(service, sources, split)
    elapsed = time.time() - started
    if len(restored) != n:
        raise RuntimeError(f"backend returned {len(restored)} frames for {n} inputs")

    strengths = sorted(set(float(v) for v in args.strengths.split(",")))
    if not strengths or any(v < 0.0 or v > 1.0 for v in strengths):
        raise ValueError("--strengths must be comma-separated values in [0,1]")

    candidates = []
    for strength in strengths:
        calibration = _metrics(targets, sources, restored,
                               calibration_idx, strength)
        candidates.append({
            "restore_strength": strength,
            "calibration": calibration,
            "objective": round(_score(calibration), 7),
        })
    selected = max(candidates, key=lambda c: c["objective"])
    selected_strength = float(selected["restore_strength"])
    holdout_baseline = _metrics(targets, sources, restored, holdout_idx, 0.0)
    holdout_selected = _metrics(targets, sources, restored, holdout_idx,
                                selected_strength)
    psnr_delta = holdout_selected["psnr"] - holdout_baseline["psnr"]
    ssim_delta = holdout_selected["ssim"] - holdout_baseline["ssim"]
    temporal_delta = (holdout_selected["temporal_l1"]
                      - holdout_baseline["temporal_l1"])
    promotion = {
        "nonzero_restoration": selected_strength > 0.0,
        "holdout_psnr_gain": psnr_delta >= args.min_psnr_gain,
        "holdout_ssim_gain": ssim_delta >= args.min_ssim_gain,
        "holdout_temporal_non_regression": (
            temporal_delta <= args.max_temporal_regression),
        "all_frames_restored": len(restored) == n,
    }

    report = {
        "schema_version": 2,
        "model": {
            "id": resolved.entry.id,
            "version": resolved.entry.version,
            "backend": resolved.entry.backend,
            "device": dm.info.type,
            "checkpoint": resolved.weights_path,
            "sha256": (_sha256(resolved.weights_path)
                       if resolved.weights_path else None),
            "max_restore_side": profile_opt.max_restore_side,
            "restore_clip_len": backend.caps.max_clip_len,
        },
        "dataset": {
            "work_dir": root,
            "frames": n,
            "calibration_frames": len(calibration_idx),
            "holdout_frames": len(holdout_idx),
        },
        "inference_seconds": round(elapsed, 3),
        "candidates": candidates,
        "selected": {
            **selected,
            "holdout_baseline": holdout_baseline,
            "holdout": holdout_selected,
            "holdout_psnr_delta": round(psnr_delta, 5),
            "holdout_ssim_delta": round(ssim_delta, 7),
            "holdout_temporal_l1_delta": round(temporal_delta, 7),
            "recommended_cli": f"--restore_strength {selected_strength:g}",
        },
        "promotion_gates": promotion,
        "passed": all(promotion.values()),
    }
    output = (os.path.abspath(args.output) if args.output
              else os.path.join(root, "optimization_report.json"))
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    print(json.dumps(report["selected"], indent=2))
    print(json.dumps({"promotion_gates": promotion,
                      "passed": report["passed"]}, indent=2))
    print("Report:", output)
    return report


def build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", default="quality")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--work-dir", default="",
                        help="existing closed-loop fixture; generated when omitted")
    parser.add_argument("--output", default="")
    parser.add_argument("--strengths", default=",".join(
        f"{v:.2f}" for v in np.linspace(0.0, 1.0, 21)))
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--min-psnr-gain", type=float, default=0.02)
    parser.add_argument("--min-ssim-gain", type=float, default=0.0001)
    parser.add_argument("--max-temporal-regression", type=float, default=0.001,
                        help="largest allowed holdout temporal L1 increase")
    parser.add_argument("--tr-blur", type=int, default=10)
    parser.add_argument("--tr-down", type=int, default=10)
    parser.add_argument("--max-restore-side", type=int, default=0)
    parser.add_argument("--restore-clip-len", type=int, default=0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not 0.0 < args.calibration_fraction < 1.0:
        raise ValueError("--calibration-fraction must be between 0 and 1")
    if args.max_temporal_regression < 0.0:
        raise ValueError("--max-temporal-regression must be non-negative")
    report = optimize(args)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
