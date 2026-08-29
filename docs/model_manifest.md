# Model manifest and unified restoration interface

Every image, video, GUI, and server restoration request uses the same contract:

```text
RestorationService        detect -> group -> crop -> backend -> composite
RestorationBackend        BGR uint8 frames in -> same count/size out
ModelManager              resolve -> download -> verify -> atomic cache publish
DeviceManager             auto / cuda / mps / directml / cpu
```

## Built-in entries

The source of truth is
[`restoration/manifests/models.yaml`](../restoration/manifests/models.yaml).

| id | aliases | backend | devices | status |
|---|---|---|---|---|
| `deepmosaics-bisenet-detector` | `mosaic-detector`, `detector` | official DeepMosaics BiSeNet detector | CUDA, MPS, DirectML, CPU | released; verified download |
| `lada-official-basicvsrpp` | `quality`, `lada`, `lada-baseline` | official LADA generic v1.2 BasicVSR++ | CUDA, MPS, CPU | released; verified download |
| `deepmosaics-video-baseline` | `dm-baseline` | original DeepMosaics BVDNet | CUDA, MPS, CPU | released; verified download |
| `legacy-pix2pix` | `legacy` | original pix2pix/pix2pixHD family | CUDA, DirectML, MPS, CPU | released; checkpoint required |
| `traditional` | `classic`, `none` | deterministic blur/down/up | all | released; no weights |
| `mosaicvr-lite` | `lite` | compact MPS-first backbone | CUDA, MPS, CPU | planned; no published weights |
| `lada-basicvsrpp` | `portable-dev` | compact research backbone | CUDA, MPS, CPU | planned; no published weights |

`planned` entries cannot be selected as production models. In particular,
random checkpoints produced by `tools/make_smoke_fixtures.py` are test-only and
are isolated from the production cache.

## Selection policy

- `--model auto` selects official LADA on CUDA/MPS, legacy on DirectML, and
  traditional on CPU.
- DirectML legacy selection requires an existing `--model_path`.
- An explicit incompatible device/model pair fails instead of silently swapping
  the named model.
- Auto memory profiles bound restoration crop/window sizes for 6-12 GB CUDA and
  Apple Silicon; `--max_restore_side` and `--restore_clip_len` override them.
- A bare `.pth` supplied through `--model` is interpreted as an original
  DeepMosaics legacy checkpoint. New backend checkpoints should be registered in
  a manifest so the architecture is unambiguous, or use `lite:/path.pth` /
  `portable:/path.pth` for a local trainer output.

## Cache and integrity

```text
pretrained_models/cache/<model-id>/<version>/<filename>
```

Set `DEEPMOSAICS_CACHE` to override the root. Downloads go to `filename.part`,
are checked for declared byte size and SHA-256, and are atomically renamed only
after verification. A corrupt existing file is rejected.

The built-in released neural models have pinned hashes. A custom entry without
a hash is allowed for local development but prints a warning and should not be
published as a release.

## Register a custom model

```yaml
# my_models.yaml
models:
  my-lite-v2:
    backend: mosaicvr_lite
    version: 0.2.0
    aliases: [my-lite]
    devices: [cuda, mps, cpu]
    license: AGPL-3.0
    status: released
    files:
      weights:
        filename: my_lite_v2.pth
        urls: ["https://example.org/my_lite_v2.pth"]
        sha256: <64 lowercase hex characters>
        size_bytes: <exact byte count>
```

```bash
python deepmosaic.py --mode clean --media_path in.mp4 \
  --model my-lite --manifest my_models.yaml --device auto
```

Calculate a digest with:

```bash
shasum -a 256 my_lite_v2.pth        # macOS
sha256sum my_lite_v2.pth            # Linux
```

## Training contract

```bash
# Compact model, practical for CUDA training or MPS fine-tuning.
python train/mosaicvr/train.py --arch lite --dataset ./datasets/face

# Portable research backbone; full training remains CUDA-oriented.
python train/mosaicvr/train.py --arch basicvsrpp \
  --dataset ./datasets/face --device cuda
```

Dataset layout:

```text
<dataset>/<video>/origin_image/00001.jpg
<dataset>/<video>/mask/00001.png
```

Online synthesis keeps one mosaic parameter set and one spatial crop per
temporal clip. The loss combines L2 with optional VGG perceptual and
DeepMosaics-style multi-scale GAN/feature matching. Checkpoints include model,
optimizer, iteration, epoch, and configuration state so a fine-tune can resume.

The official LADA checkpoint is used by its original architecture and is not a
drop-in initialization for `lite` or `portable-dev`. Publish a compatible
compact checkpoint before advertising one-click MPS fine-tuning.

The trainer prints a ready-to-run command at completion. For example:

```bash
python deepmosaic.py --mode clean --media_path in.mp4 \
  --model lite:checkpoints/lite_256/lite_final.pth --device mps
```
