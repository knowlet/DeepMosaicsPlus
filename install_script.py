"""Cross-platform dependency installer for DeepMosaicsPlus.

The installer deliberately keeps hardware detection separate from model
selection. It installs a PyTorch build that the runtime's ``--device auto``
policy can use, then creates a launcher for one of the UIs that actually ships
in this repository.
"""

from __future__ import annotations

import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence


COMMON_PACKAGES = (
    "flask",
    "numpy",
    "opencv-python",
    "pillow",
    "pyyaml",
    "tqdm",
)

PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PYTORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu118"

UI_PACKAGES = {
    "stable": "customtkinter",
    "modern": "PyQt6",
}

UI_TARGETS = {
    "stable": "deepmosaicui.pyw",
    "modern": "deepmosaicui_modern_NEW.pyw",
}


def cls() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def title() -> None:
    cls()
    print("=" * 55)
    print("              Dependency Installer")
    print("=" * 55)
    print()


def pause() -> None:
    input("\nPress Enter to exit...")


def run(cmd: Sequence[str]) -> None:
    command = [str(part) for part in cmd]
    print(">", " ".join(command))
    subprocess.check_call(command)


def _command_output(command: Sequence[str]) -> str:
    """Return probe output, allowing hardware tools to be absent."""

    try:
        return subprocess.check_output(
            list(command),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        return ""


def detect_backend(
    system_name: Optional[str] = None,
    machine: Optional[str] = None,
    probe: Optional[Callable[[Sequence[str]], str]] = None,
) -> str:
    """Choose the install backend without importing PyTorch.

    Apple Silicon uses the regular macOS wheels, which include MPS support.
    DirectML is intentionally limited to Windows. Linux AMD/Intel installs a
    CPU build because this project does not currently ship a ROCm backend.
    """

    system_name = system_name or platform.system()
    machine = (machine or platform.machine()).lower()
    probe = probe or _command_output

    if system_name == "Darwin":
        return "mps" if machine in {"arm64", "aarch64"} else "cpu"

    if system_name == "Windows":
        output = probe(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | "
                "Select-Object -ExpandProperty Name",
            ]
        ).lower()
        if "nvidia" in output:
            return "cuda"
        if any(name in output for name in ("amd", "radeon", "intel")):
            return "directml"
        return "cpu"

    if system_name == "Linux":
        # A successful nvidia-smi query is stronger evidence than a PCI name:
        # it also proves the NVIDIA driver is visible to this process.
        if probe(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ]
        ):
            return "cuda"
        if "nvidia" in probe(["lspci"]).lower():
            return "cuda"

    return "cpu"


def detect_gpu() -> str:
    """Interactive wrapper retained for compatibility with the old script."""

    print("Detecting graphics hardware...")
    backend = detect_backend()
    labels = {
        "cuda": "NVIDIA CUDA GPU",
        "directml": "Windows DirectML GPU",
        "mps": "Apple Silicon MPS GPU",
        "cpu": "CPU fallback",
    }
    print(f"✓ {labels[backend]} selected.\n")
    return backend


def pytorch_install_command(
    backend: str,
    system_name: Optional[str] = None,
) -> list[str]:
    """Build the pip command for a resolved hardware backend."""

    system_name = system_name or platform.system()
    command = [sys.executable, "-m", "pip", "install"]

    if backend == "cuda":
        return command + [
            "torch",
            "torchvision",
            "--index-url",
            PYTORCH_CUDA_INDEX,
        ]

    if backend == "mps":
        # The ordinary macOS wheels contain MPS support. A CPU-only wheel index
        # would break Apple Silicon acceleration.
        return command + ["torch", "torchvision"]

    if backend == "directml":
        if system_name != "Windows":
            raise ValueError("DirectML installation is supported only on Windows")
        # torch-directml declares exact compatible torch and torchvision
        # dependencies. Asking pip for an unrelated latest torch in the same
        # transaction can produce an unsatisfiable environment.
        return command + ["torch-directml"]

    if backend != "cpu":
        raise ValueError(f"Unknown backend: {backend}")

    if system_name == "Darwin":
        # The PyTorch CPU wheel index has no macOS wheels. Intel Macs also use
        # the ordinary platform wheel, simply without an available MPS device.
        return command + ["torch", "torchvision"]

    return command + [
        "torch",
        "torchvision",
        "--index-url",
        PYTORCH_CPU_INDEX,
    ]


def install_pytorch(backend: str) -> None:
    print("Installing PyTorch...")
    run(pytorch_install_command(backend))


def common_install_command() -> list[str]:
    return [sys.executable, "-m", "pip", "install", *COMMON_PACKAGES]


def install_common() -> None:
    print("\nInstalling common packages...\n")
    run(common_install_command())


def install_ui(choice: str) -> str:
    print("\nInstalling UI...\n")
    ui = "stable" if choice == "1" else "modern"
    run([sys.executable, "-m", "pip", "install", UI_PACKAGES[ui]])
    return ui


def check_ffmpeg() -> bool:
    print("\nChecking FFmpeg...")
    missing = [tool for tool in ("ffmpeg", "ffprobe") if not shutil.which(tool)]
    if not missing:
        print("✓ FFmpeg and ffprobe found.")
        return True

    print()
    print("WARNING")
    print("-------------------------------------")
    print(f"{', '.join(missing)} was not found in PATH.")
    print("Install FFmpeg (including ffprobe) before processing video.")
    print()
    return False


def launcher_name(system_name: Optional[str] = None) -> str:
    system_name = system_name or platform.system()
    if system_name == "Windows":
        return "Launch DeepMosaicsPlus.bat"
    if system_name == "Darwin":
        return "Launch DeepMosaicsPlus.command"
    return "launch_deepmosaicsplus.sh"


def launcher_contents(
    ui: str,
    system_name: Optional[str] = None,
    executable: Optional[str] = None,
) -> str:
    """Return launcher text using a shipped UI filename."""

    if ui not in UI_TARGETS:
        raise ValueError(f"Unknown UI: {ui}")
    system_name = system_name or platform.system()
    executable = executable or sys.executable
    target = UI_TARGETS[ui]

    if system_name == "Windows":
        return (
            "@echo off\r\n"
            "cd /d \"%~dp0\"\r\n"
            f'"{executable}" "{target}" %*\r\n'
        )

    return (
        "#!/bin/sh\n"
        'cd -- "$(dirname -- "$0")"\n'
        f"exec {shlex.quote(executable)} {shlex.quote(target)} \"$@\"\n"
    )


def create_launcher(
    ui: str,
    root_dir: Optional[Path] = None,
    system_name: Optional[str] = None,
) -> Path:
    print("\nCreating launcher...")
    system_name = system_name or platform.system()
    root_dir = Path(root_dir) if root_dir is not None else Path(__file__).resolve().parent
    target = root_dir / UI_TARGETS[ui]
    if not target.is_file():
        raise FileNotFoundError(f"UI entry point not found: {target}")

    path = root_dir / launcher_name(system_name)
    path.write_text(
        launcher_contents(ui, system_name=system_name),
        encoding="utf-8",
        newline="",
    )
    if system_name != "Windows":
        path.chmod(path.stat().st_mode | 0o111)

    print(f"✓ Launcher created: {path.name}")
    return path


def main() -> None:
    title()
    print("Python:", sys.version.split()[0])
    print()
    print("Select interface\n")
    print("1) Stable UI (CustomTkinter)")
    print("2) Modern UI (PyQt6)")
    print()

    while True:
        choice = input("Selection: ").strip()
        if choice in ("1", "2"):
            break

    print()
    backend = detect_gpu()
    install_pytorch(backend)
    install_common()
    ui = install_ui(choice)
    check_ffmpeg()
    launcher = create_launcher(ui)

    print()
    print("=" * 55)
    print("Installation Complete!")
    print("=" * 55)
    print(f"\nLaunch with: {launcher.name}")
    pause()


if __name__ == "__main__":
    main()
