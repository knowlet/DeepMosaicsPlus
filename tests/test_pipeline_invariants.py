"""Regression tests for frame-accurate I/O and moving-region compositing."""

import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFrameExtraction(unittest.TestCase):
    def test_segment_plan_is_exact_and_non_overlapping(self):
        from util.ffmpeg import _plan_frame_segments
        plan = _plan_frame_segments(total_frames=30, fps=10.0,
                                    start_sec=0.0, segments=4)
        self.assertEqual([p[2] for p in plan], [8, 8, 7, 7])
        self.assertEqual([p[1] for p in plan], [1, 9, 17, 24])
        self.assertEqual(sum(p[2] for p in plan), 30)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                         "ffmpeg/ffprobe required")
    def test_parallel_extract_preserves_every_cfr_frame(self):
        from util.ffmpeg import video2image_parallel
        with tempfile.TemporaryDirectory() as root:
            video = os.path.join(root, "input.mp4")
            frames_dir = os.path.join(root, "frames")
            os.makedirs(frames_dir)
            subprocess.run([
                "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                "testsrc=size=64x48:rate=10:duration=3",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", video,
            ], check=True)
            video2image_parallel(
                video, os.path.join(frames_dir, "output_%06d.png"),
                fps=10, segments=4, qv=2)
            names = sorted(n for n in os.listdir(frames_dir) if n.endswith(".png"))
            self.assertEqual(names,
                             [f"output_{i:06d}.png" for i in range(1, 31)])


class TestMovingRegionComposite(unittest.TestCase):
    def test_identity_union_restore_maps_back_without_spatial_warp(self):
        from cores.clean import _crop_for_backend, _extract_member_crop, run_union_box
        from util.image_processing import replace_mosaic

        yy, xx = np.mgrid[0:64, 0:80]
        frame = np.stack((xx, yy, (xx + yy) % 255), axis=-1).astype(np.uint8)
        run = [(0, 24, 28, 8), (1, 36, 30, 8)]
        cx, cy, half = run_union_box(run)
        union = _crop_for_backend(frame, cx, cy, half)

        for _, x, y, size in run:
            restored_member = _extract_member_crop(
                union, cx, cy, half, x, y, size, frame.shape)
            mask = np.zeros(frame.shape[:2], np.uint8)
            mask[y-size:y+size, x-size:x+size] = 255
            out = replace_mosaic(frame.copy(), restored_member, mask,
                                 x, y, size, True)
            np.testing.assert_array_equal(out, frame)


class TestMaskWriteBarrier(unittest.TestCase):
    def test_detection_waits_for_mask_writes(self):
        from cores import clean

        class FakeDetector:
            def to(self, _device):
                return self

            def eval(self):
                return self

            def cpu(self):
                return self

        with tempfile.TemporaryDirectory() as root:
            frame_dir = os.path.join(root, "video2image")
            mask_dir = os.path.join(root, "mosaic_mask")
            os.makedirs(frame_dir)
            os.makedirs(mask_dir)
            names = [f"output_{i:06d}.png" for i in range(1, 3)]
            for name in names:
                cv2.imwrite(os.path.join(frame_dir, name),
                            np.zeros((32, 32, 3), np.uint8))

            opt = SimpleNamespace(
                temp_dir=root, device="cpu", no_preview=True,
                position_batch_size=2, scene_threshold=32,
                medfilt_num=1,
            )
            completed = []

            def slow_write(_path, _mask):
                time.sleep(0.15)
                completed.append(_path)
                return True

            detect_result = (np.full((32, 32), 255, np.uint8), 16, 16, 8)
            with mock.patch.object(clean, "_detect_one", return_value=detect_result), \
                    mock.patch.object(clean.cv2, "imwrite", side_effect=slow_write):
                clean.get_mosaic_positions(opt, FakeDetector(), names, savemask=True)
            self.assertEqual(len(completed), len(names))


class TestClosedLoopMetrics(unittest.TestCase):
    def test_ssim_identity_is_one(self):
        from tools.closed_loop_test import ssim
        rng = np.random.RandomState(7)
        frame = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        self.assertAlmostEqual(ssim(frame, frame), 1.0, places=6)

    def test_temporal_metric_is_zero_for_identity_and_detects_flicker(self):
        from tools.closed_loop_test import temporal_residual_l1
        targets = [np.zeros((8, 8, 3), np.uint8) for _ in range(3)]
        identity = [frame.copy() for frame in targets]
        flicker = [targets[0].copy(),
                   np.full((8, 8, 3), 255, np.uint8),
                   targets[2].copy()]
        self.assertEqual(temporal_residual_l1(targets, identity), 0.0)
        self.assertGreater(temporal_residual_l1(targets, flicker), 0.5)

    def test_temporal_metric_skips_hard_scene_cut(self):
        from tools.closed_loop_test import temporal_residual_l1
        targets = [np.zeros((8, 8, 3), np.uint8),
                   np.full((8, 8, 3), 255, np.uint8)]
        candidates = [np.zeros((8, 8, 3), np.uint8),
                      np.zeros((8, 8, 3), np.uint8)]
        self.assertEqual(temporal_residual_l1(targets, candidates), 0.0)

    def test_gate_rejects_frame_loss_and_quality_regression(self):
        from tools.closed_loop_test import evaluate_gates
        report = {
            "input_frames": 300,
            "output_frames": 297,
            "psnr_mosaicked": 30.66,
            "psnr_cli_cleaned": 27.42,
            "ssim_mosaicked": 0.88,
            "ssim_cli_cleaned": 0.86,
            "container": {"has_audio": True, "duration": 11.88,
                          "width": 1280, "height": 720},
        }
        gates = evaluate_gates(report, expected_duration=12.0,
                               expected_size=(1280, 720),
                               require_quality_gain=True)
        self.assertFalse(gates["passed"])
        self.assertFalse(gates["checks"]["exact_frame_count"])
        self.assertFalse(gates["checks"]["quality_non_regression"])


class TestServerEntrypoint(unittest.TestCase):
    def test_help_works_without_pythonpath_or_interactive_input(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        result = subprocess.run(
            [sys.executable, os.path.join(repo_root, "tools", "server.py"),
             "--help"],
            cwd=tempfile.gettempdir(), env=env, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--source_url", result.stdout)


class TestOptimizerSplit(unittest.TestCase):
    def test_calibration_and_holdout_are_restored_in_separate_calls(self):
        from tools.optimize_restoration import _restore_disjoint

        class RecordingService:
            def __init__(self):
                self.calls = []

            def restore_region_sequence(self, frames):
                self.calls.append(list(frames))
                return list(frames)

        service = RecordingService()
        frames = list(range(10))
        restored = _restore_disjoint(service, frames, 4)
        self.assertEqual(service.calls, [list(range(4)), list(range(4, 10))])
        self.assertEqual(restored, frames)


if __name__ == "__main__":
    unittest.main(verbosity=2)
