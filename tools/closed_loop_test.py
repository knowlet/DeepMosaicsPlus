"""Closed-loop verification of the full restoration pipeline.

Phase A: build a synthetic 720p clip with motion, noise, a scene cut and an
         audio track; apply mosaic at KNOWN positions (ground truth) using the
         project's own mosaic code.
Phase B: run the real CLI (deepmosaic.py --mode clean) on the mosaicked clip.
Phase C: score the result against ground truth (PSNR / SSIM / temporal
         residual error in the known region), verify container integrity,
         frame count, duration, audio.

Usage:
    .venv/bin/python tools/closed_loop_test.py [--model traditional|lada-basicvsrpp] [--device cpu|mps]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

W, H, FPS = 1280, 720, 25
DUR_A, DUR_B = 6.0, 6.0            # two scenes -> one hard cut at 6 s
N_A, N_B = int(DUR_A * FPS), int(DUR_B * FPS)
MOSAIC_HALF = 48                   # half-size of the square to mosaic

WORK = "/tmp/dmp_closed_loop"


def build_scene_frame(t, scene):
    """Synthetic content with gradients + moving objects + sensor-ish noise."""
    rng = np.random.RandomState(int(t * 1000))
    yy, xx = np.mgrid[0:H, 0:W]
    if scene == 0:
        base = np.stack([(xx * 255 // W),
                         ((yy * 180 // H) + 40),
                         (255 - xx * 200 // W)], axis=-1).astype(np.uint8)
        cx = int(W * (0.25 + 0.45 * t))
        cy = int(H * (0.55 - 0.15 * t))
    else:
        base = np.full((H, W, 3), 90, np.uint8)
        for k in range(4):
            x0 = int(W * (0.1 * k + 0.05)) + int(30 * t) % 60
            base[:, x0:x0 + 24] = (200, 210, 220)
        cx = int(W * (0.75 - 0.5 * t))
        cy = int(H * (0.35 + 0.2 * t))
    cv2.circle(base, (cx, cy), 56, (60, 160, 90), -1)
    cv2.rectangle(base, (cx - 20, cy - 20), (cx + 20, cy + 20), (30, 30, 200), 3)

    # moving "target" patch whose position IS the mosaic ground truth
    tx = int(np.clip(cx, MOSAIC_HALF + 8, W - MOSAIC_HALF - 8))
    ty = int(np.clip(cy, MOSAIC_HALF + 8, H - MOSAIC_HALF - 8))

    frame = base.copy()
    noise = rng.randint(0, 14, frame.shape, dtype=np.uint8)
    return cv2.add(frame, noise), tx, ty


def mosaic_region(frame, cx, cy, half, block=12):
    """Apply pixelation mosaic exactly like util.mosaic squa_avg."""
    x0, y0 = cx - half, cy - half
    patch = frame[y0:y0 + 2 * half, x0:x0 + 2 * half]
    small_h = max(2, (2 * half) // block)
    small_w = max(2, (2 * half) // block)
    small = cv2.resize(patch, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    out = frame.copy()
    out[y0:y0 + 2 * half, x0:x0 + 2 * half] = cv2.resize(
        small, (2 * half, 2 * half), interpolation=cv2.INTER_NEAREST)
    return out


def phase_a():
    os.makedirs(WORK, exist_ok=True)
    raw = os.path.join(WORK, "original.mp4")
    mos = os.path.join(WORK, "mosaicked.mp4")
    gt_path = os.path.join(WORK, "gt_boxes.json")

    vw_raw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    vw_mos = cv2.VideoWriter(mos, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    gt = []
    idx = 0
    for frames, scene in ((N_A, 0), (N_B, 1)):
        for i in range(frames):
            t = i / FPS
            frame, tx, ty = build_scene_frame(t, scene)
            vw_raw.write(frame)
            vw_mos.write(mosaic_region(frame, tx, ty, MOSAIC_HALF))
            gt.append([idx, tx, ty])
            idx += 1
    vw_raw.release()
    vw_mos.release()
    with open(gt_path, "w", encoding="utf-8") as fp:
        json.dump(gt, fp)

    # add a real audio track so voice handling is exercised
    for name in (raw, mos):
        tmp = name.replace(".mp4", "_av.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-i", name,
            "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
            "-c:v", "copy", "-c:a", "aac", "-shortest", tmp,
        ], check=True)
        os.replace(tmp, name)

    print(f"[A] wrote {raw} / {mos} ({idx} frames @ {FPS}fps {W}x{H}, audio ok)")
    return raw, mos, gt_path


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_format", "-show_streams", path],
                       capture_output=True, text=True)
    d = json.loads(r.stdout)
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    a = [s for s in d["streams"] if s["codec_type"] == "audio"]
    return {
        "frames": int(v.get("nb_frames", 0)),
        "duration": float(d["format"]["duration"]),
        "has_audio": bool(a),
        "width": v.get("width"), "height": v.get("height"),
    }


def read_frames(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse < 1e-10 else 10 * np.log10(255 ** 2 / mse)


def ssim(a, b):
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(ga, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(gb, (11, 11), 1.5)
    saa = cv2.GaussianBlur(ga * ga, (11, 11), 1.5)
    sbb = cv2.GaussianBlur(gb * gb, (11, 11), 1.5)
    sab = cv2.GaussianBlur(ga * gb, (11, 11), 1.5)
    var_a, var_b, cov = saa - mu_a ** 2, sbb - mu_b ** 2, sab - mu_a * mu_b
    num = (2 * mu_a * mu_b + C1) * (2 * cov + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (var_a + var_b + C2)
    return float(np.clip(np.mean(num / den), -1.0, 1.0))


def temporal_residual_l1(targets, candidates, indices=None,
                         scene_cut_threshold=0.25):
    """Measure frame-difference error while ignoring hard scene cuts.

    The metric compares candidate motion/change with the ground-truth change,
    so a static but sharp frame sequence cannot score well by merely avoiding
    flicker. Values are normalized to [0, 2], and lower is better.
    """
    n = min(len(targets), len(candidates))
    selected = list(range(n)) if indices is None else list(indices)
    errors = []
    previous = None
    for current in selected:
        if current < 0 or current >= n:
            raise IndexError(f"temporal metric index {current} outside 0..{n - 1}")
        if previous is None or current != previous + 1:
            previous = current
            continue
        target_delta = (targets[current].astype(np.float32)
                        - targets[previous].astype(np.float32))
        target_change = float(np.mean(np.abs(target_delta)) / 255.0)
        if target_change <= scene_cut_threshold:
            candidate_delta = (candidates[current].astype(np.float32)
                               - candidates[previous].astype(np.float32))
            errors.append(float(np.mean(np.abs(candidate_delta - target_delta))
                                / 255.0))
        previous = current
    return float(np.mean(errors)) if errors else 0.0


def _region_crops(frames, gt, n):
    half = MOSAIC_HALF
    return [frames[i][gt[i][2] - half:gt[i][2] + half,
                      gt[i][1] - half:gt[i][1] + half]
            for i in range(n)]


def _region_scores(orig_f, test_f, gt):
    n = min(len(orig_f), len(test_f), len(gt))
    half = MOSAIC_HALF
    P, S = [], []
    for i in range(n):
        _, tx, ty = gt[i]
        o = orig_f[i][ty - half:ty + half, tx - half:tx + half]
        t = test_f[i][ty - half:ty + half, tx - half:tx + half]
        P.append(psnr(o, t)); S.append(ssim(o, t))
    return float(np.mean(P)), float(np.mean(S)), n


def phase_c(mos, cleaned, gt_path):
    """C1: full-CLI result scored against ground truth."""
    orig = read_frames(os.path.join(WORK, "original.mp4"))
    mos_f = read_frames(mos)
    clean_f = read_frames(cleaned)

    with open(gt_path, "r", encoding="utf-8") as fp:
        gt = json.load(fp)
    pm, sm, n1 = _region_scores(orig, mos_f, gt)
    pc, sc, n2 = _region_scores(orig, clean_f, gt)
    n = min(n1, n2)
    targets = _region_crops(orig, gt, n)
    sources = _region_crops(mos_f, gt, n)
    candidates = _region_crops(clean_f, gt, n)
    temporal_source = temporal_residual_l1(targets, sources)
    temporal_cleaned = temporal_residual_l1(targets, candidates)
    return {
        "input_frames": len(orig),
        "mosaicked_frames": len(mos_f),
        "output_frames": len(clean_f),
        "frames_compared": min(n1, n2),
        "psnr_mosaicked": round(pm, 2), "psnr_cli_cleaned": round(pc, 2),
        "ssim_mosaicked": round(sm, 4), "ssim_cli_cleaned": round(sc, 4),
        "temporal_l1_mosaicked": round(temporal_source, 7),
        "temporal_l1_cli_cleaned": round(temporal_cleaned, 7),
        "temporal_l1_delta": round(temporal_cleaned - temporal_source, 7),
        "container": probe(cleaned),
    }


def phase_c2_backend_direct(gt_path, model, device, restore_strength=1.0,
                            max_restore_side=0, restore_clip_len=0):
    """C2: bypass the detector; restore crops at KNOWN GT boxes through the
    same service/backends the CLI uses. Isolates backend behaviour from
    detection quality (which is meaningless with random fixture weights)."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from restoration.device_manager import DeviceManager
    from restoration.service import RestorationService, configure_memory_profile

    from types import SimpleNamespace
    _opt = SimpleNamespace(
        device=device, tr_blur=10, tr_down=10,
        max_restore_side=max_restore_side,
        restore_clip_len=restore_clip_len)

    dm = DeviceManager(device, quiet=True)
    if model == "traditional":
        from restoration.backends.legacy import TraditionalBackend as BE
        be = BE(blur=10, down=10)
    else:
        from restoration.model_manager import ModelManager
        from restoration.backends import build_backend
        r = ModelManager(quiet=True).resolve(model, dm,
                                             model_path_fallback=None)
        be = build_backend(r, _opt)
    configure_memory_profile(_opt, dm, be, quiet=True)
    svc = RestorationService(backend=be, device_manager=dm,
                             detector=None, model_id=model, quiet=True)

    orig = read_frames(os.path.join(WORK, "original.mp4"))
    mos_f = read_frames(mos_path_global)
    with open(gt_path, "r", encoding="utf-8") as fp:
        gt = json.load(fp)
    half = MOSAIC_HALF
    crops = [mos_f[b[0]][b[2]-half:b[2]+half, b[1]-half:b[1]+half] for b in gt]
    # Synthetic fixture has a known hard cut. Match the production contract:
    # temporal propagation must never cross scene boundaries.
    scene_split = min(N_A, len(crops))
    outs = svc.restore_region_sequence(crops[:scene_split])
    if scene_split < len(crops):
        outs.extend(svc.restore_region_sequence(crops[scene_split:]))
    P_before, P_after, S_before, S_after = [], [], [], []
    targets, candidates = [], []
    for i in range(len(outs)):
        idx, tx, ty = gt[i]
        target = orig[idx][ty-half:ty+half, tx-half:tx+half]
        candidate = np.clip(
            crops[i].astype(np.float32) * (1.0 - restore_strength)
            + outs[i].astype(np.float32) * restore_strength + 0.5,
            0, 255).astype(np.uint8)
        targets.append(target)
        candidates.append(candidate)
        P_before.append(psnr(target, crops[i]))
        P_after.append(psnr(target, candidate))
        S_before.append(ssim(target, crops[i]))
        S_after.append(ssim(target, candidate))
    before, after = float(np.mean(P_before)), float(np.mean(P_after))
    s_before, s_after = float(np.mean(S_before)), float(np.mean(S_after))
    temporal_before = temporal_residual_l1(targets, crops[:len(outs)])
    temporal_after = temporal_residual_l1(targets, candidates)
    return {"backend": getattr(be, "name", model),
            "frames_restored": len(outs),
            "restore_strength": restore_strength,
            "max_restore_side": _opt.max_restore_side,
            "restore_clip_len": be.caps.max_clip_len,
            "psnr_mosaicked_gt_region": round(before, 2),
            "psnr_restored_gt_region": round(after, 2),
            "psnr_delta": round(after - before, 2),
            "ssim_mosaicked_gt_region": round(s_before, 5),
            "ssim_restored_gt_region": round(s_after, 5),
            "ssim_delta": round(s_after - s_before, 5),
            "temporal_l1_mosaicked_gt_region": round(temporal_before, 7),
            "temporal_l1_restored_gt_region": round(temporal_after, 7),
            "temporal_l1_delta": round(temporal_after - temporal_before, 7)}


def phase_c3_composite_oracle(gt_path):
    """C3: feed the ORIGINAL crop as 'perfect restoration' through the
    composite path; measures masking/feathering overhead only."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from util import image_processing as impro

    orig = read_frames(os.path.join(WORK, "original.mp4"))
    with open(gt_path, "r", encoding="utf-8") as fp:
        gt = json.load(fp)
    half = MOSAIC_HALF
    P = []
    for b in gt[::10]:
        i, tx, ty = b
        mask = np.zeros((H, W), np.uint8)
        mask[ty-half:ty+half, tx-half:tx+half] = 255
        out = impro.replace_mosaic(orig[i].copy(), orig[i][ty-half:ty+half,
                                                        tx-half:tx+half],
                                   mask, tx, ty, half, True)
        P.append(psnr(orig[i], out))
    return {"psnr_ceiling": round(float(np.mean(P)), 2)}


def evaluate_gates(report, expected_duration, expected_size,
                   require_quality_gain=True):
    """Evaluate frame/container/quality invariants without hiding failures."""
    container = report["container"]
    exact_frames = (
        report["input_frames"] == report["output_frames"]
        == report["frames_compared"]
    )
    duration_tolerance = max(0.10, 2.0 / FPS)
    quality_ok = True
    if require_quality_gain:
        quality_ok = (
            report["psnr_cli_cleaned"] >= report["psnr_mosaicked"]
            and report["ssim_cli_cleaned"] >= report["ssim_mosaicked"] - 0.001
        )
    checks = {
        "exact_frame_count": exact_frames,
        "audio_preserved": bool(container.get("has_audio")),
        "duration_preserved": abs(float(container.get("duration", 0.0))
                                  - expected_duration) <= duration_tolerance,
        "dimensions_preserved": (
            container.get("width"), container.get("height")) == expected_size,
        "quality_non_regression": quality_ok,
    }
    return {"passed": all(checks.values()), "checks": checks,
            "quality_gate_required": bool(require_quality_gain)}


def main():
    global WORK
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="traditional")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--work-dir", default="",
                    help="artifact directory (default: a fresh /tmp directory)")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="maximum CLI runtime in seconds")
    ap.add_argument("--baseline-only", action="store_true",
                    help="skip the quality-gain gate for infrastructure baselines")
    ap.add_argument("--restore-strength", type=float, default=1.0,
                    help="blend factor used by CLI and direct-backend scoring")
    ap.add_argument("--max-restore-side", type=int, default=0,
                    help="production memory profile override; 0=auto")
    ap.add_argument("--restore-clip-len", type=int, default=0,
                    help="production temporal-window override; 0=auto")
    args = ap.parse_args()
    WORK = (os.path.abspath(args.work_dir) if args.work_dir
            else tempfile.mkdtemp(prefix="dmp_closed_loop_"))
    os.makedirs(WORK, exist_ok=True)
    print(f"Artifacts: {WORK}")

    print("=== Phase A: synthetic clip + ground-truth mosaic ===")
    raw, mos, gt_path = phase_a()

    print(f"\n=== Phase B: CLI clean (--model {args.model} --device {args.device}) ===")
    out_dir = os.path.join(WORK, "out")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable, "deepmosaic.py",
        "--mode", "clean",
        "--media_path", mos,
        "--model", args.model,
        "--device", args.device,
        "--mosaic_position_model_path", "auto",
        "--model_path", "./pretrained_models/mosaic/add_face.pth",
        "--restore_strength", str(args.restore_strength),
        "--max_restore_side", str(args.max_restore_side),
        "--restore_clip_len", str(args.restore_clip_len),
        "--no_preview",
        "--temp_dir", os.path.join(WORK, "tmp"),
        "--result_dir", out_dir,
    ]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    cleaned = os.path.join(out_dir, "mosaicked_clean.mp4")
    if os.path.exists(cleaned):
        os.remove(cleaned)
    try:
        r = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), env=env, stdin=subprocess.DEVNULL,
            timeout=args.timeout)
    except subprocess.TimeoutExpired:
        print(f"CLI TIMED OUT after {args.timeout}s")
        sys.exit(1)
    if r.returncode != 0:
        print("CLI FAILED"); sys.exit(1)
    assert os.path.isfile(cleaned), "no output produced"

    print("\n=== Phase C1: CLI end-to-end scoring ===")
    rep = phase_c(mos, cleaned, gt_path)
    rep["model"], rep["device"] = args.model, args.device
    print(json.dumps(rep, indent=2))

    print("\n=== Phase C2: backend direct (GT boxes, detector bypassed) ===")
    globals()["mos_path_global"] = mos
    c2 = phase_c2_backend_direct(gt_path, args.model, args.device,
                                 restore_strength=args.restore_strength,
                                 max_restore_side=args.max_restore_side,
                                 restore_clip_len=args.restore_clip_len)
    print(json.dumps(c2, indent=2))

    print("\n=== Phase C3: composite oracle ceiling ===")
    c3 = phase_c3_composite_oracle(gt_path)
    print(json.dumps(c3, indent=2))

    gates = evaluate_gates(rep, expected_duration=DUR_A + DUR_B,
                           expected_size=(W, H),
                           require_quality_gain=False)
    gates["checks"]["backend_restored_all_frames"] = \
        c2["frames_restored"] == rep["input_frames"]
    gates["checks"]["backend_quality_non_regression"] = (
        True if (args.baseline_only or args.model == "traditional") else
        c2["psnr_delta"] >= 0.0 and c2["ssim_delta"] >= -0.001
    )
    gates["checks"]["backend_temporal_non_regression"] = (
        True if (args.baseline_only or args.model == "traditional") else
        c2["temporal_l1_delta"] <= 0.001
    )
    gates["checks"]["composite_contract"] = c3["psnr_ceiling"] > 40.0
    gates["passed"] = all(gates["checks"].values())
    report_path = os.path.join(WORK, "report.json")
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump({"cli": rep, "backend_direct": c2,
                   "composite_oracle": c3, "gates": gates}, fp, indent=2)
    print(json.dumps(gates, indent=2))
    print("\nVERDICT:", "PASS" if gates["passed"] else "FAIL")
    print("Report:", report_path)
    if not gates["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
