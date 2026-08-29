import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import install_script

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 still runs the application.
    tomllib = None


class BackendDetectionTests(unittest.TestCase):
    def test_apple_silicon_selects_mps_without_external_probe(self):
        probe = mock.Mock(side_effect=AssertionError("probe must not run"))
        self.assertEqual(
            install_script.detect_backend("Darwin", "arm64", probe),
            "mps",
        )

    def test_intel_mac_selects_cpu(self):
        self.assertEqual(install_script.detect_backend("Darwin", "x86_64"), "cpu")

    def test_windows_prefers_nvidia_cuda(self):
        self.assertEqual(
            install_script.detect_backend(
                "Windows",
                "AMD64",
                lambda _command: "AMD Radeon Graphics\nNVIDIA RTX 4070",
            ),
            "cuda",
        )

    def test_windows_amd_selects_directml(self):
        self.assertEqual(
            install_script.detect_backend(
                "Windows", "AMD64", lambda _command: "AMD Radeon RX 7900"
            ),
            "directml",
        )

    def test_linux_nvidia_smi_selects_cuda(self):
        def probe(command):
            return "GeForce RTX 3060" if command[0] == "nvidia-smi" else ""

        self.assertEqual(install_script.detect_backend("Linux", "x86_64", probe), "cuda")

    def test_linux_amd_uses_cpu_fallback(self):
        def probe(command):
            return "AMD Radeon" if command[0] == "lspci" else ""

        self.assertEqual(install_script.detect_backend("Linux", "x86_64", probe), "cpu")


class InstallCommandTests(unittest.TestCase):
    def test_mps_uses_ordinary_macos_wheels(self):
        command = install_script.pytorch_install_command("mps", "Darwin")
        self.assertIn("torch", command)
        self.assertIn("torchvision", command)
        self.assertNotIn("--index-url", command)

    def test_cuda_uses_an_official_cuda_wheel_index(self):
        command = install_script.pytorch_install_command("cuda", "Linux")
        self.assertIn("torch", command)
        self.assertIn("torchvision", command)
        self.assertIn(install_script.PYTORCH_CUDA_INDEX, command)

    def test_intel_mac_cpu_uses_ordinary_macos_wheels(self):
        command = install_script.pytorch_install_command("cpu", "Darwin")
        self.assertNotIn("--index-url", command)

    def test_windows_cpu_uses_cpu_wheel_index(self):
        command = install_script.pytorch_install_command("cpu", "Windows")
        self.assertIn("https://download.pytorch.org/whl/cpu", command)

    def test_windows_directml_owns_its_compatible_torch_dependencies(self):
        command = install_script.pytorch_install_command("directml", "Windows")
        self.assertIn("torch-directml", command)
        self.assertNotIn("torch", command)
        self.assertNotIn("torchvision", command)

    def test_directml_is_rejected_outside_windows(self):
        with self.assertRaises(ValueError):
            install_script.pytorch_install_command("directml", "Linux")

    def test_common_packages_cover_cli_runtime(self):
        command = install_script.common_install_command()
        for package in ("numpy", "opencv-python", "pillow", "pyyaml", "tqdm"):
            self.assertIn(package, command)
        self.assertIn("flask", command)
        self.assertNotIn("scikit-image", command)


class LauncherTests(unittest.TestCase):
    def _create(self, ui, system_name):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        (root / install_script.UI_TARGETS[ui]).touch()
        path = install_script.create_launcher(ui, root, system_name)
        self.addCleanup(temporary.cleanup)
        return path

    def test_windows_stable_launcher_points_to_shipped_ui(self):
        path = self._create("stable", "Windows")
        self.assertEqual(path.name, "Launch DeepMosaicsPlus.bat")
        self.assertIn('"deepmosaicui.pyw"', path.read_text(encoding="utf-8"))

    def test_macos_modern_launcher_is_executable_and_points_to_shipped_ui(self):
        path = self._create("modern", "Darwin")
        self.assertEqual(path.name, "Launch DeepMosaicsPlus.command")
        self.assertIn("deepmosaicui_modern_NEW.pyw", path.read_text(encoding="utf-8"))
        self.assertTrue(path.stat().st_mode & stat.S_IXUSR)

    def test_missing_ui_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                install_script.create_launcher("stable", Path(temporary), "Linux")


@unittest.skipUnless(tomllib is not None, "tomllib is built in on Python 3.11+")
class UvProjectMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        with (root / "pyproject.toml").open("rb") as stream:
            cls.project = tomllib.load(stream)["project"]
        cls.lock_path = root / "uv.lock"

    def test_core_declares_every_quality_cli_dependency(self):
        dependencies = "\n".join(self.project["dependencies"]).lower()
        for package in ("numpy", "opencv-python", "pyyaml", "torch",
                        "torchvision", "tqdm"):
            self.assertIn(package, dependencies)
        self.assertNotIn("flask", dependencies)

    def test_mps_floor_and_directml_fork_are_explicit(self):
        dependencies = self.project["dependencies"]
        self.assertTrue(any(
            item.startswith("torch>=2.8") and "platform_machine == 'arm64'" in item
            for item in dependencies))
        self.assertTrue(any(
            item.startswith("torchvision>=0.23")
            and "platform_machine == 'arm64'" in item
            for item in dependencies))
        self.assertEqual(
            self.project["optional-dependencies"]["directml"],
            ["torch-directml==0.2.5.dev240914; sys_platform == 'win32'"],
        )

    def test_lockfile_is_checked_in_next_to_project_metadata(self):
        self.assertTrue(self.lock_path.is_file())


if __name__ == "__main__":
    unittest.main()
