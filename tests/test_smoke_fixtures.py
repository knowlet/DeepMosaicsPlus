import contextlib
import io
import ntpath
import os
import shlex
import subprocess
import tempfile
import unittest

from tools.make_smoke_fixtures import (
    REPO_ROOT,
    _build_smoke_manifest,
    _format_copy_paste_command as format_smoke_command,
    _is_path_within,
    _resolve_output_dir,
    build_parser,
)
from train.mosaicvr.train import (
    _format_copy_paste_command as format_training_command,
)


class TestSmokeFixtureIsolation(unittest.TestCase):
    def test_default_command_uses_portable_cpu_device(self):
        self.assertEqual(build_parser().parse_args([]).device, "cpu")

    def test_unverified_directml_and_auto_routes_are_rejected(self):
        for device in ("directml", "auto"):
            with self.subTest(device=device), contextlib.redirect_stderr(
                    io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    build_parser().parse_args(["--device", device])
                self.assertEqual(raised.exception.code, 2)

    def test_manifest_claims_only_verified_portable_device_families(self):
        with tempfile.NamedTemporaryFile() as checkpoint:
            checkpoint.write(b"random fixture")
            checkpoint.flush()
            manifest = _build_smoke_manifest(checkpoint.name)
        devices = manifest["models"]["smoke-portable"]["devices"]
        self.assertEqual(devices, ["cuda", "mps", "cpu"])
        self.assertNotIn("directml", devices)

    def test_repo_pretrained_models_rejected_from_foreign_cwd(self):
        previous = os.getcwd()
        foreign = tempfile.mkdtemp()
        try:
            os.chdir(foreign)
            target = os.path.join(REPO_ROOT, "pretrained_models", "random")
            with self.assertRaisesRegex(RuntimeError, "refusing"):
                _resolve_output_dir(target)
        finally:
            os.chdir(previous)

    def test_relative_default_is_anchored_to_repo(self):
        previous = os.getcwd()
        foreign = tempfile.mkdtemp()
        try:
            os.chdir(foreign)
            resolved = _resolve_output_dir(
                "closed_loop_artifacts/random_smoke_fixtures")
            expected = os.path.join(
                REPO_ROOT, "closed_loop_artifacts", "random_smoke_fixtures")
            self.assertEqual(resolved, os.path.realpath(expected))
        finally:
            os.chdir(previous)

    def test_windows_different_drive_is_outside_production_tree(self):
        self.assertFalse(_is_path_within(
            r"D:\Smoke Fixtures", r"C:\DeepMosaicsPlus\pretrained_models",
            path_module=ntpath,
        ))

    def test_windows_same_drive_production_child_is_detected(self):
        self.assertTrue(_is_path_within(
            r"C:\DeepMosaicsPlus\pretrained_models\random",
            r"C:\DeepMosaicsPlus\pretrained_models",
            path_module=ntpath,
        ))


class TestCopyPasteCommands(unittest.TestCase):
    def test_smoke_command_quotes_posix_paths_and_environment(self):
        argv = [
            "/Applications/Python Build/python3",
            "/tmp/Deep Mosaics/deepmosaic.py",
            "--model",
            "smoke-portable",
        ]
        cache = "/tmp/Deep Mosaics/cache dir"
        command = format_smoke_command(
            argv, env={"DEEPMOSAICS_CACHE": cache}, os_name="posix")
        self.assertEqual(
            command,
            "DEEPMOSAICS_CACHE=%s %s" % (
                shlex.quote(cache), shlex.join(argv)),
        )

    def test_smoke_command_quotes_windows_paths_and_environment(self):
        argv = [
            r"C:\Program Files\Python\python.exe",
            r"C:\Deep Mosaics\deepmosaic.py",
            "--model",
            r"portable:C:\Model Builds\portable final.pth",
        ]
        cache = r"C:\Deep Mosaics\cache dir"
        command = format_smoke_command(
            argv, env={"DEEPMOSAICS_CACHE": cache}, os_name="nt")
        self.assertEqual(
            command,
            'set "DEEPMOSAICS_CACHE=%s" && %s' % (
                cache, subprocess.list2cmdline(argv)),
        )

    def test_training_command_quotes_posix_typed_checkpoint(self):
        argv = [
            "/Applications/Python Build/python3",
            "/tmp/Deep Mosaics/deepmosaic.py",
            "--model",
            "lite:/tmp/Model Builds/lite final.pth",
        ]
        self.assertEqual(
            format_training_command(argv, os_name="posix"),
            shlex.join(argv),
        )

    def test_training_command_quotes_windows_typed_checkpoint(self):
        argv = [
            r"C:\Program Files\Python\python.exe",
            r"C:\Deep Mosaics\deepmosaic.py",
            "--model",
            r"lite:C:\Model Builds\lite final.pth",
        ]
        self.assertEqual(
            format_training_command(argv, os_name="nt"),
            subprocess.list2cmdline(argv),
        )

if __name__ == "__main__":
    unittest.main()
