"""CPU-only test suite for the restoration abstraction layer.

Run inside the project environment:
    .venv/bin/python -m unittest tests.test_restoration -v
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from restoration.manifest import Manifest  # noqa: E402
from restoration.device_manager import DeviceManager, pick_auto_backend  # noqa: E402
from restoration.model_manager import ModelManager  # noqa: E402
from restoration.service import recommended_memory_limits  # noqa: E402
from restoration.backends.base import RestoreRequest  # noqa: E402
from restoration.backends.basicvsrpp_portable import (  # noqa: E402
    MosaicVSRNet, backward_warp, invert_flow, compose_flow)
from restoration.backends.mosaicvr_lite import MosaicVRLiteNet  # noqa: E402
from restoration.backends import (BackendCaps, NullBackend,  # noqa: E402
                                  RestorationBackend, TraditionalBackend,
                                  build_backend)


def bgr_image(h=64, w=64, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 255, (h, w, 3)).astype(np.uint8)


class TestDeviceManager(unittest.TestCase):
    def test_cpu_explicit(self):
        dm = DeviceManager("cpu", quiet=True)
        self.assertEqual(dm.info.type, "cpu")

    def test_auto_resolves(self):
        dm = DeviceManager("auto", quiet=True)
        self.assertIn(dm.info.type, ("cuda", "mps", "directml", "cpu"))

    def test_invalid_raises(self):
        with self.assertRaises(RuntimeError):
            DeviceManager("tpu", quiet=True)

    def test_policy(self):
        self.assertEqual(pick_auto_backend("cuda"), "lada-official-basicvsrpp")
        self.assertEqual(pick_auto_backend("mps"), "lada-official-basicvsrpp")
        self.assertEqual(pick_auto_backend("directml"), "legacy")
        self.assertEqual(pick_auto_backend("cpu"), "traditional")

    def test_memory_profiles_cover_6_to_12_gib_cuda(self):
        self.assertEqual(recommended_memory_limits("cuda", 6.0), (320, 4))
        self.assertEqual(recommended_memory_limits("cuda", 8.0), (384, 6))
        self.assertEqual(recommended_memory_limits("cuda", 12.0), (512, 8))
        self.assertEqual(recommended_memory_limits("mps"), (384, 4))


class TestManifest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = Manifest.load()

    def test_builtin_entries(self):
        ids = set(self.manifest.ids())
        self.assertIn("deepmosaics-bisenet-detector", ids)
        self.assertIn("lada-basicvsrpp", ids)
        self.assertIn("mosaicvr-lite", ids)

    def test_alias(self):
        e = self.manifest.get("quality")
        self.assertEqual(e.id, "lada-official-basicvsrpp")
        self.assertEqual(e.status, "released")
        self.assertIn("mps", e.devices)

    def test_detector_alias_has_integrity_metadata(self):
        e = self.manifest.get("mosaic-detector")
        self.assertEqual(e.id, "deepmosaics-bisenet-detector")
        self.assertEqual(len(e.files["weights"].sha256), 64)
        self.assertEqual(e.files["weights"].size_bytes, 49704452)

    def test_random_smoke_model_is_not_released(self):
        e = self.manifest.get("lada-basicvsrpp")
        self.assertNotEqual(e.status, "released")
        self.assertNotIn("quality", e.aliases)

    def test_planned_status(self):
        e = self.manifest.get("lite")
        self.assertNotEqual(e.status, "released")

    def test_device_check_message(self):
        from restoration.model_manager import ResolvedModel
        mm = ModelManager(quiet=True)
        dm = DeviceManager("cpu", quiet=True)
        entry = self.manifest.get("quality")
        entry.devices = ["mps"]           # force incompatibility
        resolved = ResolvedModel(entry=entry,
                                 weights_path="dummy.pth",
                                 device_manager=dm)
        with self.assertRaises(RuntimeError) as ctx:
            mm._check_device(entry, dm)
        self.assertIn("will not silently swap", str(ctx.exception))


class TestModelManager(unittest.TestCase):
    def test_unknown_id(self):
        mm = ModelManager(quiet=True)
        dm = DeviceManager("cpu", quiet=True)
        with self.assertRaises(RuntimeError) as ctx:
            mm.resolve("no-such-model", dm)
        self.assertIn("Known ids", str(ctx.exception))

    def test_local_checkpoint_wraps_legacy(self):
        tmp = tempfile.NamedTemporaryFile(suffix="_video_clean.pth", delete=False)
        torch.save({}, tmp.name)
        tmp.close()
        try:
            mm = ModelManager(quiet=True)
            dm = DeviceManager("cpu", quiet=True)
            r = mm.resolve(tmp.name, dm)
            self.assertEqual(r.entry.backend, "legacy_video")
        finally:
            os.remove(tmp.name)

    def test_typed_lite_checkpoint_uses_new_backend(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
        torch.save({}, tmp.name)
        tmp.close()
        try:
            mm = ModelManager(quiet=True)
            resolved = mm.resolve(
                "lite:" + tmp.name, DeviceManager("cpu", quiet=True))
            self.assertEqual(resolved.entry.backend, "mosaicvr_lite")
            self.assertEqual(resolved.weights_path, tmp.name)
        finally:
            os.remove(tmp.name)

    def test_typed_checkpoint_rejects_missing_file(self):
        mm = ModelManager(quiet=True)
        with self.assertRaisesRegex(RuntimeError, "Typed checkpoint does not exist"):
            mm.resolve("portable:/missing/model.pth",
                       DeviceManager("cpu", quiet=True))

    def test_weightless_traditional(self):
        mm = ModelManager(quiet=True)
        dm = DeviceManager("cpu", quiet=True)
        r = mm.resolve("traditional", dm)
        self.assertIsNone(r.weights_path)

    def test_hash_verify(self):
        import tempfile
        data = b"not-a-real-model"
        h = __import__("hashlib").sha256(data).hexdigest()
        d = tempfile.mkdtemp()
        p = os.path.join(d, "weights.bin")
        with open(p, "wb") as fp:
            fp.write(data)
        from restoration.manifest import ManifestEntry, ModelFile
        entry = ManifestEntry(id="t", backend="basicvsrpp_portable",
                              status="released",
                              files={"weights": ModelFile(
                                  filename="weights.bin",
                                  urls=[], sha256=h)})
        mm = ModelManager(cache_root=d, quiet=True)
        wf = entry.files["weights"]
        target = os.path.join(mm.cache_root, entry.id, entry.version, wf.filename)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as fp:
            fp.write(data)
        mm._check_hash(entry, wf, target)          # should pass
        wf.sha256 = "0" * 64
        with self.assertRaises(RuntimeError):
            mm._check_hash(entry, wf, target)

    def test_download_is_verified_before_publish(self):
        import hashlib
        import tempfile
        from restoration.manifest import ManifestEntry, ModelFile

        good = b"expected-author-weights"
        bad = b"truncated-or-corrupt-download"
        entry = ManifestEntry(
            id="atomic-download",
            backend="basicvsrpp_portable",
            version="1",
            files={"weights": ModelFile(
                filename="weights.pth",
                urls=["https://example.invalid/weights.pth"],
                sha256=hashlib.sha256(good).hexdigest(),
            )},
        )
        root = tempfile.mkdtemp()
        mm = ModelManager(cache_root=root, quiet=True)
        target_dir = os.path.join(root, entry.id, entry.version)
        target = os.path.join(target_dir, "weights.pth")

        def fake_download(_url, dst, reporthook=None):
            with open(dst, "wb") as fp:
                fp.write(bad)

        with mock.patch("urllib.request.urlretrieve", side_effect=fake_download):
            with self.assertRaises(RuntimeError):
                mm._download(entry, entry.files["weights"], target_dir)
        self.assertFalse(os.path.exists(target))

    def test_legacy_alias_uses_explicit_fallback_checkpoint(self):
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix="_video_clean.pth", delete=False)
        torch.save({}, tmp.name)
        tmp.close()
        try:
            mm = ModelManager(quiet=True)
            r = mm.resolve("legacy", DeviceManager("cpu", quiet=True),
                           model_path_fallback=tmp.name)
            self.assertEqual(r.entry.backend, "legacy_video")
            self.assertEqual(r.weights_path, tmp.name)
        finally:
            os.remove(tmp.name)

    def test_legacy_alias_rejects_missing_fallback_checkpoint(self):
        mm = ModelManager(quiet=True)
        with self.assertRaisesRegex(RuntimeError, "Legacy checkpoint does not exist"):
            mm.resolve(
                "legacy", DeviceManager("cpu", quiet=True),
                model_path_fallback="/definitely/missing/clean_video.pth",
            )


class TestFlows(unittest.TestCase):
    def test_identity_flow_warp(self):
        x = torch.rand(1, 3, 16, 16)
        zero = torch.zeros(1, 2, 16, 16)
        y = backward_warp(x, zero)
        self.assertTrue(torch.allclose(x, y, atol=1e-5))

    def test_invert_compose(self):
        # smooth low-frequency field (realistic motion) rather than noise
        small = torch.rand(1, 2, 3, 3) * 2 - 1
        f = torch.nn.functional.interpolate(small, size=(16, 16),
                                            mode="bilinear", align_corners=False)
        inv = invert_flow(f)
        comp = compose_flow(f, inv)
        # composing a flow with its inverse should be near zero
        self.assertLess(comp.abs().mean().item(), 0.5)


class TestNets(unittest.TestCase):
    def test_basicvsrpp_shapes(self):
        net = MosaicVSRNet(n_feats=16)
        x = torch.rand(3, 3, 64, 64) * 2 - 1
        with torch.no_grad():
            y = net(x)
        self.assertEqual(y.shape, x.shape)
        self.assertLess(net.count_params(), 10.0)

    def test_lite_shapes(self):
        net = MosaicVRLiteNet(ch=16)
        x = torch.rand(2, 3, 64, 64) * 2 - 1
        with torch.no_grad():
            y = net(x)
        self.assertEqual(y.shape, x.shape)


class TestBackendContract(unittest.TestCase):
    def _check(self, backend, frames):
        out = backend.restore(RestoreRequest(frames=frames))
        self.assertEqual(len(out), len(frames))
        for o, f in zip(out, frames):
            self.assertEqual(o.dtype, np.uint8)
            self.assertEqual(o.shape, f.shape)
        return out

    def test_null(self):
        frames = [bgr_image(seed=i) for i in range(5)]
        out = self._check(NullBackend(), frames)
        for o, f in zip(out, frames):
            np.testing.assert_array_equal(o, f)

    def test_traditional(self):
        frames = [bgr_image(seed=i) for i in range(3)]
        out = self._check(TraditionalBackend(blur=5, down=4), frames)

    def test_basicvsrpp_random_weights(self):
        import tempfile
        net = MosaicVSRNet(n_feats=12)
        sd_path = os.path.join(tempfile.mkdtemp(), "w.pth")
        torch.save(net.state_dict(), sd_path)
        from restoration.backends.basicvsrpp_portable import BasicVSRPPBackend
        be = BasicVSRPPBackend(sd_path, DeviceManager("cpu", quiet=True), amp=False)
        frames = [bgr_image(h=48, w=60, seed=i) for i in range(3)]
        self._check(be, frames)

    @staticmethod
    def _mock_lada_backend(max_restore_side):
        from restoration.backends.lada_official import LadaOfficialBackend

        class RecordingIdentity(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.input_shapes = []

            def forward(self, value):
                self.input_shapes.append(tuple(value.shape))
                return value

        backend = LadaOfficialBackend.__new__(LadaOfficialBackend)
        backend.dm = DeviceManager("cpu", quiet=True)
        backend.net = RecordingIdentity()
        backend.max_restore_side = max_restore_side
        return backend

    def test_lada_narrow_crop_pads_without_proportional_upscale(self):
        backend = self._mock_lada_backend(max_restore_side=384)
        frames = [bgr_image(h=80, w=384, seed=i) for i in range(2)]

        out = self._check(backend, frames)

        # The former min-side scaling path produced a 256x1229-ish tensor.
        # Padding holds the long edge at the 384 px memory profile instead.
        self.assertEqual(backend.net.input_shapes, [(1, 2, 3, 256, 384)])
        for restored, source in zip(out, frames):
            np.testing.assert_array_equal(restored, source)

    def test_lada_sub_256_limit_caps_content_before_minimum_padding(self):
        backend = self._mock_lada_backend(max_restore_side=64)
        frames = [bgr_image(h=80, w=320, seed=i) for i in range(2)]
        real_resize = cv2.resize

        with mock.patch("restoration.backends.lada_official.cv2.resize",
                        wraps=real_resize) as resize:
            out = self._check(backend, frames)

        # The 64 px limit constrains real image content to 16x64.  The 256
        # tensor is unavoidable model-minimum padding, not proportional image
        # enlargement, and every output is mapped back to its original crop.
        self.assertEqual(resize.call_args_list[0].args[1], (64, 16))
        self.assertEqual(backend.net.input_shapes, [(1, 2, 3, 256, 256)])
        self.assertEqual([frame.shape for frame in out],
                         [frame.shape for frame in frames])
        self.assertTrue(all(frame.dtype == np.uint8 for frame in out))

    def test_mismatched_weights_raise(self):
        import tempfile
        net = MosaicVRLiteNet(ch=8)     # wrong arch on purpose
        sd_path = os.path.join(tempfile.mkdtemp(), "w.pth")
        torch.save(net.state_dict(), sd_path)
        from restoration.backends.basicvsrpp_portable import BasicVSRPPBackend
        with self.assertRaises(RuntimeError) as ctx:
            BasicVSRPPBackend(sd_path, DeviceManager("cpu", quiet=True))
        self.assertIn("do not match", str(ctx.exception))

    def test_chunked_sequence_blending(self):
        svc = _make_service(NullBackend())
        crops = [bgr_image(h=40, w=40, seed=i) for i in range(10)]
        out = svc.restore_region_sequence(crops)
        self.assertEqual(len(out), len(crops))

    def test_center_windows_contract(self):
        from restoration.backends.base import BackendCaps, RestorationBackend

        class FakeCenter(RestorationBackend):
            name = "fake-center"
            caps = BackendCaps(output_mode="center", max_clip_len=13)
            window_sizes = []
            windows = []

            def restore_center(self, pool):
                self.window_sizes.append(len(pool))
                self.windows.append([int(f[0, 0, 0]) for f in pool])
                return pool[len(pool)//2].copy()

            def _restore_impl(self, request):   # pragma: no cover
                raise NotImplementedError

        be = FakeCenter()
        svc = _make_service(be)
        crops = [np.full((24, 24, 3), i, np.uint8) for i in range(40)]
        out = svc.restore_region_sequence(crops)
        self.assertEqual(len(out), len(crops))
        # every output frame got its own sliding window
        self.assertEqual(len(be.window_sizes), len(crops))
        self.assertEqual(be.windows[20], list(range(14, 27)))
        self.assertEqual(int(out[20][0, 0, 0]), 20)

    def test_center_without_restore_center_raises(self):
        from restoration.backends.base import BackendCaps, RestorationBackend

        class Bad(RestorationBackend):
            name = "bad"
            caps = BackendCaps(output_mode="center")

            def _restore_impl(self, request):
                return list(request.frames)

        with self.assertRaises(RuntimeError):
            _make_service(Bad()).restore_region_sequence(
                [bgr_image(h=8, w=8) for _ in range(3)])


    def test_legacy_pix2pix_wrapper(self):
        """Random unet_128 checkpoint -> legacy wrapper keeps the size contract."""
        import tempfile
        from models.pix2pix_model import define_G
        net = define_G(3, 3, 64, "unet_128", norm="batch", use_dropout=True,
                       init_type="normal", gpu_ids=[])
        sd_path = os.path.join(tempfile.mkdtemp(), "clean_unet_128_t.pth")
        torch.save(net.state_dict(), sd_path)

        from restoration.backends.legacy import LegacyPix2PixBackend
        be = LegacyPix2PixBackend(sd_path, DeviceManager("cpu", quiet=True))
        frames = [bgr_image(h=96, w=120, seed=i) for i in range(2)]
        out = self._check(be, frames)

    def test_legacy_kind_distinguishes_unet_256(self):
        from restoration.backends.legacy import infer_legacy_kind
        self.assertEqual(infer_legacy_kind("clean_unet_256_face.pth"), "unet_256")

    def test_legacy_video_uses_sampled_center_as_initial_feedback(self):
        from restoration.backends.legacy import LegacyVideoBackend

        class EchoPrevious:
            def __call__(self, stream, previous):
                return previous

        be = LegacyVideoBackend.__new__(LegacyVideoBackend)
        be.N, be.S, be.T = 2, 3, 5
        be.dm = DeviceManager("cpu", quiet=True)
        be.net = EchoPrevious()
        be._previous = None
        pool = [np.full((32, 40, 3), i, np.uint8) for i in range(13)]
        out = be.restore_center(pool)
        self.assertEqual(out.shape, pool[6].shape)
        self.assertLess(abs(float(out.mean()) - 6.0), 1.0)

    def test_lite_checkpoint_saved_by_trainer_loads(self):
        import tempfile
        from restoration.backends.mosaicvr_lite import MosaicVRLiteBackend
        net = MosaicVRLiteNet()
        path = os.path.join(tempfile.mkdtemp(), "lite.pth")
        torch.save({"netG": net.state_dict(), "iter": 1}, path)
        be = MosaicVRLiteBackend(path, DeviceManager("cpu", quiet=True))
        self._check(be, [bgr_image(h=32, w=36, seed=i) for i in range(2)])

    def test_lite_flow_changes_recurrent_result(self):
        net = MosaicVRLiteNet(ch=8).eval()
        x = torch.rand(3, 3, 24, 24) * 2 - 1

        def flows(value):
            return [torch.full((1, 2, 12, 12), value, device=x.device)
                    for _ in range(2)]

        with torch.no_grad(), mock.patch.object(net, "compute_flows",
                                                side_effect=lambda *_a, **_k: flows(0.0)):
            y0 = net(x)
        with torch.no_grad(), mock.patch.object(net, "compute_flows",
                                                side_effect=lambda *_a, **_k: flows(1.0)):
            y1 = net(x)
        self.assertGreater(float((y0 - y1).abs().max()), 1e-6)


class TestAuthorCheckpoints(unittest.TestCase):
    def test_original_deepmosaics_detector_checkpoint_on_mps(self):
        if not (getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()):
            self.skipTest("MPS is not available")
        path = os.path.join(
            "pretrained_models", "cache", "deepmosaics-bisenet-detector",
            "0.5.1", "mosaic_position.pth")
        if not os.path.isfile(path):
            self.skipTest("original DeepMosaics detector is not installed")
        from models.BiSeNet_model import BiSeNet
        net = BiSeNet(num_classes=1, context_path="resnet18", train_flag=False)
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        net.eval().to("mps")
        with torch.inference_mode():
            out = net(torch.zeros(1, 3, 64, 64, device="mps"))
        self.assertEqual(tuple(out.shape), (1, 1, 64, 64))

    def test_official_lada_quality_checkpoint_on_mps(self):
        if not (getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()):
            self.skipTest("MPS is not available")
        dm = DeviceManager("mps", quiet=True)
        mm = ModelManager(quiet=True)
        resolved = mm.resolve("quality", dm)
        if not os.path.isfile(resolved.weights_path):
            self.skipTest("official LADA checkpoint is not installed")
        be = build_backend(resolved)
        frames = [bgr_image(h=48, w=52, seed=i) for i in range(2)]
        out = be.restore(RestoreRequest(frames=frames))
        self.assertEqual([f.shape for f in out], [f.shape for f in frames])
        self.assertTrue(all(f.dtype == np.uint8 for f in out))

    def test_original_deepmosaics_bvdnet_checkpoint_on_mps(self):
        if not (getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()):
            self.skipTest("MPS is not available")
        dm = DeviceManager("mps", quiet=True)
        resolved = ModelManager(quiet=True).resolve("dm-baseline", dm)
        if not os.path.isfile(resolved.weights_path):
            self.skipTest("original DeepMosaics BVDNet checkpoint is not installed")
        be = build_backend(resolved)
        pool = [np.full((40, 44, 3), i, np.uint8) for i in range(13)]
        out = be.restore_center(pool)
        self.assertEqual(out.shape, pool[6].shape)
        self.assertEqual(out.dtype, np.uint8)


class TestGroupRuns(unittest.TestCase):
    def test_gap_merge_and_cuts(self):
        from cores.clean import group_runs
        positions = [
            [10, 10, 120],   # active
            [11, 11, 120],   # active
            [12, 12, 0],     # gap (inactive)
            [13, 13, 0],     # gap (inactive) -> within tolerance 3
            [14, 14, 120],   # active
            [15, 15, 120],   # active
        ]
        runs = group_runs(positions, cuts=[6], min_size=100)
        self.assertEqual(len(runs), 1)
        idxs = [b[0] for b in runs[0]]
        self.assertEqual(idxs, [0, 1, 2, 3, 4, 5])
        # interpolated gap boxes have positive sizes
        self.assertTrue(all(b[3] > 0 for b in runs[0]))

    def test_cut_splits(self):
        from cores.clean import group_runs
        positions = [[i, i, 150] for i in range(6)]
        runs = group_runs(positions, cuts=[3], min_size=100)
        self.assertEqual(len(runs), 2)
        self.assertEqual([b[0] for b in runs[0]], [0, 1, 2])
        self.assertEqual([b[0] for b in runs[1]], [3, 4, 5])

    def test_max_run_split(self):
        from cores.clean import group_runs
        positions = [[i, i, 150] for i in range(20)]
        runs = group_runs(positions, cuts=[], min_size=100, max_run=5)
        self.assertGreaterEqual(len(runs), 4)


class TestServiceEndToEnd(unittest.TestCase):
    """Detector stub + traditional backend over a synthetic mosaicked image."""

    def test_service_bounds_oversized_crop_and_restores_original_geometry(self):
        class RecordingBackend(RestorationBackend):
            name = "recording"
            caps = BackendCaps(output_mode="all", max_clip_len=1)

            def __init__(self):
                self.seen_shapes = []
                self.max_restore_side = 64

            def _restore_impl(self, request):
                self.seen_shapes.extend(frame.shape[:2]
                                        for frame in request.frames)
                return [frame.copy() for frame in request.frames]

        backend = RecordingBackend()
        svc = _make_service(backend)
        frame = bgr_image(120, 240, seed=12)
        restored = svc.restore_region_sequence([frame])
        self.assertEqual(backend.seen_shapes, [(32, 64)])
        self.assertEqual(restored[0].shape, frame.shape)

    def test_clean_image(self):
        img = bgr_image(128, 128, seed=42)
        # draw a fake mosaic block at (70..110)
        block = img[70:110, 30:70]
        img[70:110, 30:70] = (block.reshape(10, 4, 10, 4, 3)
                              .mean(axis=(1, 3)).astype(np.uint8).repeat(4, 0).repeat(4, 1))

        mask = np.zeros((128, 128), np.uint8)
        mask[65:115, 25:75] = 255

        def detect(_img):
            return mask, 50, 90, 30      # mask, x, y, size

        be = TraditionalBackend(blur=5, down=4)
        svc = _make_service(be)
        svc.detector = detect

        opt = _FakeOpt()
        img_before = img.copy()
        out = svc.clean_image(img, opt)
        self.assertEqual(out.shape, img.shape)
        # replace_mosaic composites in place; compare against the snapshot
        self.assertTrue(np.abs(out.astype(int) - img_before.astype(int)).max() > 0)

    def test_zero_restore_strength_is_identity(self):
        from restoration.service import composite
        img = bgr_image(80, 90, seed=9)
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[20:60, 25:65] = 255
        fake = np.zeros((40, 40, 3), np.uint8)
        opt = _FakeOpt()
        opt.restore_strength = 0.0
        out = composite(img.copy(), fake, mask, 45, 40, 20, opt)
        np.testing.assert_array_equal(out, img)


class _FakeOpt:
    min_mosaic_size = 20
    no_feather = True
    luma_sharpen = False
    bilateral_sharpen = False
    freq_inject = False


def _make_service(backend):
    from restoration.device_manager import DeviceManager
    from restoration.service import RestorationService
    return RestorationService(backend=backend,
                              device_manager=DeviceManager("cpu", quiet=True),
                              detector=None, model_id="test", quiet=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
