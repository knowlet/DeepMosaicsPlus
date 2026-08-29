<div align="center">
  <img src="./imgs/logo_new.png" width="400" alt="DeepMosaicsPlus"><br>

  # DeepMosaicsPlus

  **Mosaic-aware video restoration with verified pretrained models**

  [![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
  [![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
  [![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)](#hardware-support)
</div>

DeepMosaicsPlus combines the original DeepMosaics mosaic-specific detector and
GAN workflow with LADA's dataset/restoration approach and a unified video
restoration interface. The first usable quality backend is LADA's official
BasicVSR++ checkpoint; a compact MPS-friendly replacement remains a training
target and is not presented as a pretrained model yet.

> Restoration is generative: it estimates plausible detail and cannot recover
> information that the mosaic destroyed. Use it only on media you are allowed
> to process.

## Quick start

Requirements: Python 3.10+ (3.11 recommended), a current PyTorch/torchvision
pair, OpenCV, PyYAML, and FFmpeg on `PATH`.

With `uv`, the checked-in project metadata and lockfile install the complete
CLI/runtime environment automatically:

```bash
uv sync
uv run python deepmosaic.py --help
```

The first `uv run` also performs the sync, so `uv sync` may be omitted.
Optional entry points use extras: `uv sync --extra stable-ui` for the stable
GUI, `--extra modern-ui` for the PyQt GUI, and `--extra server` for HTTP mode.

```bash
git clone https://github.com/foooooooooooooooooooooooooootw/DeepMosaicsPlus.git
cd DeepMosaicsPlus
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python install_script.py
```

Run the quality backend on NVIDIA CUDA or Apple Silicon MPS:

```bash
python deepmosaic.py \
  --mode clean \
  --media_path input.mp4 \
  --model quality \
  --device auto \
  --no_preview
```

The first run downloads both author-provided checkpoints into
`pretrained_models/cache/`, verifies their SHA-256 hashes, and then loads them:

- DeepMosaics BiSeNet `mosaic_position.pth` for mosaic localization.
- LADA generic v1.2 BasicVSR++ for restoration.

For the graphical interface, run the UI that exposes the unified model/device
selectors:

```bash
python deepmosaicui.pyw
```

## Hardware support

| Hardware | Inference | Training scope | Default policy |
|---|---|---|---|
| NVIDIA, 6–12 GB | CUDA quality backend | Fine-tune or train from scratch | official LADA |
| Apple Silicon | MPS quality/legacy inference | Single-machine compact-model fine-tune | official LADA |
| AMD on Windows | DirectML legacy checkpoint | Not currently supported by the new trainer | user-supplied legacy model |
| CPU | Traditional fallback; neural models are possible but slow | Smoke tests only | traditional |

The automatic memory profile uses 320px/4-frame restoration calls around 6 GB
CUDA, 384px/6 frames around 8 GB, 512px/8 frames around 12 GB, and a conservative
384px/4-frame MPS profile. Override it when needed:

```bash
python deepmosaic.py --mode clean --media_path input.mp4 --model quality \
  --device cuda --max_restore_side 320 --restore_clip_len 4 --no_preview
```

Explicit device requests fail clearly when unavailable; the program does not
silently move a named model to another backend. On AMD/DirectML, pass an
original DeepMosaics pix2pix checkpoint explicitly:

```bash
uv sync --extra directml
```

```bash
python deepmosaic.py --mode clean --media_path input.mp4 \
  --device directml --model legacy \
  --model_path /path/to/clean_unet_128_youknow.pth --no_preview
```

The official DeepMosaics BVDNet video checkpoint is also available as the
`dm-baseline` model on CUDA, MPS, and CPU:

```bash
python deepmosaic.py --mode clean --media_path input.mp4 \
  --device mps --model dm-baseline --no_preview
```

## Models

All model selection goes through
[`restoration/manifests/models.yaml`](restoration/manifests/models.yaml).

| CLI name | Backend | Weights | Status |
|---|---|---|---|
| `quality`, `lada` | official LADA BasicVSR++ | auto-download, hash verified | released baseline |
| `dm-baseline` | original DeepMosaics BVDNet | auto-download, hash verified | released baseline |
| `traditional` | blur/down/up | none | released fallback |
| `lite` | MosaicVR-Lite | none published | training target |
| `portable-dev` | compact BasicVSR++-style research model | none published | development only |
| `legacy` or `path.pth` | original pix2pix family | user supplied | compatibility path |

`--model auto` selects `quality` on CUDA/MPS, `legacy` on DirectML, and
`traditional` on CPU. DirectML therefore requires a valid `--model_path`.
See [the model manifest guide](docs/model_manifest.md) for cache layout and
custom manifests.

## Quality calibration and closed-loop tests

The closed loop generates a deterministic two-scene video, mosaics known
regions, runs the real CLI, and checks frame count, audio, duration, dimensions,
backend coverage, composite geometry, PSNR, SSIM, and scene-cut-aware temporal
residual error:

```bash
python tools/closed_loop_test.py \
  --model quality --device mps --restore-strength 0.15 \
  --work-dir /tmp/dmp_closed_loop_quality
```

Calibrate a blend strength on one split and require PSNR/SSIM gains without
temporal regression on a disjoint holdout:

```bash
python tools/optimize_restoration.py \
  --model quality --device mps \
  --work-dir /tmp/dmp_closed_loop_quality \
  --output /tmp/dmp_closed_loop_quality/optimization_report.json
```

`--restore_strength` ranges from `0` (source mosaic) to `1` (full model). A
synthetic calibration result is evidence about the test fixture, not a universal
default. Promote a value only after it passes a representative, licensed real
holdout set.

Random-weight fixtures are isolated from production caches:

```bash
python tools/make_smoke_fixtures.py
```

They are intentionally unusable as quality models and cannot be written under
`pretrained_models/` by that tool.

## Training your own compact model

Dataset layout:

```text
datasets/face/<clip>/origin_image/00001.jpg
datasets/face/<clip>/mask/00001.png
```

The trainer performs online mosaic/degradation synthesis and combines pixel,
optional VGG perceptual, multi-scale GAN, and feature-matching losses. Start a
compact CUDA run:

```bash
python train/mosaicvr/train.py \
  --arch lite --dataset ./datasets/face --device cuda \
  --batchsize 1 --finesize 256 --T 5 \
  --val_dataset ./datasets/face_val --val_steps 20
```

After you have a compatible self-trained compact checkpoint, fine-tune it on
Apple Silicon. No pretrained compact starting checkpoint is published yet;
the official LADA checkpoint is a different architecture. Smaller crops and
three-frame clips are the practical starting point for unified memory:

```bash
python train/mosaicvr/train.py \
  --arch lite --dataset ./datasets/face --device mps \
  --continue_train checkpoints/lite_256/lite_final.pth \
  --batchsize 1 --loadsize 160 --finesize 128 --T 3 \
  --lambda_VGG 0
```

Full from-scratch BasicVSR++ training remains CUDA-oriented:

```bash
python train/mosaicvr/train.py \
  --arch basicvsrpp --dataset ./datasets/face --device cuda
```

Checkpoint files contain generator/discriminator and optimizer state for
resuming. A trained generator can be published through a custom manifest
without changing runtime code, or run directly with an explicit backend prefix:

```bash
python deepmosaic.py --mode clean --media_path input.mp4 \
  --model lite:checkpoints/lite_256/lite_final.pth \
  --device mps --no_preview
```

## Useful CLI options

```text
--model auto|quality|dm-baseline|traditional|legacy|legacy.pth
        |lite:/path.pth|portable:/path.pth
--device auto|cuda|mps|directml|cpu
--restore_strength 0.0..1.0
--max_restore_side 0     # source-content cap before required padding; >=64
--restore_clip_len 0     # auto by device memory
--mask_threshold 48      # auto-adapted; use --no_auto_adapt to fix
--min_mosaic_area 150    # auto-lowered; use --no_auto_adapt to fix
--min_mosaic_size 40     # auto-lowered; use --no_auto_adapt to fix
--all_mosaic_area        # detect all regions per frame (multi-mosaic, auto-enabled for multi)
--no_auto_adapt          # disable automatic threshold/size adaptation
--start_time HH:MM:SS
--last_time HH:MM:SS
--fps 0                 # preserve source rate
--no_preview
```

Automatic mosaic detection adapts `mask_threshold`/`min_mosaic_area`/`min_mosaic_size`
per image/video (`--no_auto_adapt` disables it). Multi-mosaic is handled
simultaneously: a 1280×720 clip with 1–3 mosaics is verified in closed-loop
(single `+4.62 dB`, dual `+4.24/+4.85 dB`, triple `+4.16/+3.48/+4.24 dB` PSNR
on the mosaicked crops).

See [all options](docs/options_introduction.md).

For the optional HTTP service, provide the complete corresponding-source URL
for the exact deployed revision; every response exposes it:

```bash
python tools/server.py --model quality --device auto \
  --source_url https://example.org/source/deepmosaicsplus-exact-revision.tar.gz
```

## Architecture

```text
CLI / GUI / server
        |
RestorationService  detect -> track/group -> crop -> restore -> composite
        |
RestorationBackend  BGR uint8 frames in -> same count and size out
        |
ModelManager        manifest -> download -> SHA-256 -> atomic cache publish
        |
DeviceManager       CUDA / MPS / DirectML / CPU
```

Temporal sequence handling, model-specific padding/color conversion, and
compositing are centralized so legacy BVDNet and newer restoration backbones
obey the same frame/geometry contract.

## License and provenance

The original DeepMosaics-derived portions remain under GPL-3.0. The integrated
LADA code and official LADA restoration checkpoint are AGPL-3.0; consequently,
distribution or network use of the combined quality path must satisfy the
applicable AGPL-3.0 source-availability terms. See [`NOTICE.md`](NOTICE.md),
[`LICENSE`](LICENSE), and [`thirdparty/lada/LICENSE`](thirdparty/lada/LICENSE).

Principal upstream projects:

- [DeepMosaics](https://github.com/HypoX64/DeepMosaics)
- [LADA](https://github.com/ladaapp/lada)
- [MMagic / BasicVSR++](https://github.com/open-mmlab/mmagic)
- [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [pix2pixHD](https://github.com/NVIDIA/pix2pixHD)

## Known boundaries

- The current quality release is the official LADA baseline, not the planned
  compact replacement.
- MPS inference and a one-iteration compact fine-tune have hardware tests; long
  training stability still needs a representative dataset run.
- CUDA and DirectML paths require hardware-specific release validation before a
  binary release is called production-ready.
- Detector quality must be evaluated on curated real examples; synthetic
  gradients are useful for pipeline invariants but are out of distribution for
  the author detector.
