"""Generate isolated throwaway fixtures for pipeline smoke tests.

These checkpoints have RANDOM weights - detection/restoration quality is
garbage. They exist only to exercise the full pipeline end-to-end without
downloading real models:

    python tools/make_smoke_fixtures.py

The script prints an invocation using its generated manifest and isolated
``DEEPMOSAICS_CACHE``. A portable checkpoint must not be passed as a bare
``.pth`` because bare paths intentionally mean legacy DeepMosaics models.

The default output is intentionally outside ``pretrained_models`` so random
fixtures can never shadow verified production weights.
"""

import argparse
import hashlib
import os
import shlex
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml


REPO_ROOT = os.path.realpath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SMOKE_DEVICES = ("cuda", "mps", "cpu")


def _format_copy_paste_command(argv, env=None, os_name=None):
    """Render an argv (and optional environment) for the target shell.

    Windows launchers are consumed by ``cmd.exe`` and therefore need the same
    quoting rules as ``subprocess``. POSIX shells use ``shlex.join`` for argv,
    while environment values are quoted separately from their ``NAME=``
    prefix so the shell still recognizes them as assignments.
    """
    target_os = os.name if os_name is None else os_name
    args = [os.fspath(value) for value in argv]
    environment = {
        str(key): os.fspath(value) for key, value in (env or {}).items()
    }
    if target_os == "nt":
        command = subprocess.list2cmdline(args)
        assignments = [
            'set "%s=%s"' % (key, value)
            for key, value in environment.items()
        ]
        return " && ".join(assignments + [command])

    command = shlex.join(args)
    assignments = [
        "%s=%s" % (key, shlex.quote(value))
        for key, value in environment.items()
    ]
    return " ".join(assignments + [command])


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_smoke_manifest(portable_path):
    """Describe only the portable device routes covered by this project."""
    return {
        "models": {
            "smoke-portable": {
                "backend": "basicvsrpp_portable",
                "title": "RANDOM portable smoke fixture (NO QUALITY)",
                "version": "0.0.0",
                "devices": list(SMOKE_DEVICES),
                "license": "test-only",
                "status": "released",
                "notes": "Random deterministic weights for pipeline tests only.",
                "files": {
                    "weights": {
                        "filename": os.path.basename(portable_path),
                        "urls": [],
                        "sha256": _sha256(portable_path),
                        "size_bytes": os.path.getsize(portable_path),
                    }
                },
            }
        }
    }


def _is_path_within(path, directory, path_module=os.path):
    """Return whether ``path`` is under ``directory`` on that path flavor.

    ``ntpath.commonpath`` raises ``ValueError`` for paths on different drives;
    an output on another drive is simply outside the production model tree.
    """
    try:
        common = path_module.commonpath((path, directory))
    except ValueError:
        return False
    return path_module.normcase(common) == path_module.normcase(directory)


def _resolve_output_dir(value):
    candidate = os.path.expanduser(value)
    if not os.path.isabs(candidate):
        candidate = os.path.join(REPO_ROOT, candidate)
    output_dir = os.path.realpath(candidate)
    production_dir = os.path.realpath(os.path.join(REPO_ROOT,
                                                   "pretrained_models"))
    if _is_path_within(output_dir, production_dir):
        raise RuntimeError(
            "refusing to place random fixtures under pretrained_models; "
            "choose an isolated --output-dir")
    return output_dir


def build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--output-dir",
        default=os.path.join("closed_loop_artifacts", "random_smoke_fixtures"),
        help="isolated destination; pretrained_models is deliberately rejected",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--device",
        choices=SMOKE_DEVICES,
        default="cpu",
        help="device written into the generated smoke-test command",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    import warnings
    warnings.warn("Generating RANDOM-WEIGHT fixtures for smoke tests only!")
    torch.manual_seed(args.seed)

    output_dir = _resolve_output_dir(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    from models.BiSeNet_model import BiSeNet
    net = BiSeNet(num_classes=1, context_path="resnet18", train_flag=False)
    detector_path = os.path.join(output_dir, "detector_random.pth")
    torch.save(net.state_dict(), detector_path)
    print("wrote RANDOM detector fixture:", detector_path)

    from restoration.backends.basicvsrpp_portable import MosaicVSRNet
    qnet = MosaicVSRNet(n_feats=24)
    cache_dir = os.path.join(output_dir, "cache")
    portable_dir = os.path.join(cache_dir, "smoke-portable", "0.0.0")
    os.makedirs(portable_dir, exist_ok=True)
    portable_path = os.path.join(portable_dir, "portable_random.pth")
    torch.save(qnet.state_dict(), portable_path)
    print("wrote RANDOM restoration fixture:", portable_path)

    manifest_path = os.path.join(output_dir, "smoke_manifest.yaml")
    manifest = _build_smoke_manifest(portable_path)
    with open(manifest_path, "w", encoding="utf-8") as fp:
        yaml.safe_dump(manifest, fp, sort_keys=False)
    print("wrote smoke manifest:", manifest_path)
    print("These files are test fixtures, not pretrained models.")
    quoted = [
        sys.executable, os.path.join(REPO_ROOT, "deepmosaic.py"),
        "--mode", "clean", "--media_path",
        "clip.mp4", "--manifest", manifest_path, "--model",
        "smoke-portable", "--device", args.device,
        "--mosaic_position_model_path", detector_path, "--no_preview",
    ]
    command = _format_copy_paste_command(
        quoted, env={"DEEPMOSAICS_CACHE": cache_dir})
    print("Run:")
    print(command)


if __name__ == "__main__":
    main()
