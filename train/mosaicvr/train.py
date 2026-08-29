"""Train the MosaicVR restoration backends.

The trainer keeps the useful parts of the original DeepMosaics/LADA recipe
(online mosaic synthesis and the optional multi-scale GAN), but deliberately
keeps the command line small enough for a single workstation. It supports
CUDA training from scratch and MPS fine-tuning without making either backend
guess where its parameters or losses live.

Dataset layout per video folder::

    <dataset>/<video>/origin_image/%05d.jpg   (clean frames)
    <dataset>/<video>/mask/%05d.png           (region to mosaic)

Run from the repository root, for example::

    python train/mosaicvr/train.py --arch lite --dataset ./datasets/face

Checkpoints contain generator/discriminator state, optimizer state, the
parsed configuration, and the exact iteration/epoch at which they were
written. A legacy bare ``netG.state_dict()`` checkpoint can still be passed
to ``--continue_train``.
"""

from __future__ import annotations

import argparse
import os
import random
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, Optional, Tuple

# This file is often invoked from a checkout's parent directory. Resolve
# imports from the file location rather than the caller's current directory.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import cv2
import numpy as np
import torch
import torch.nn as nn

from restoration.backends.basicvsrpp_portable import MosaicVSRNet
from restoration.backends.mosaicvr_lite import MosaicVRLiteNet
from restoration.device_manager import DeviceManager
from models import BVDNet, model_util
from util import degradater
from util import image_processing as impro
from util import mosaic as mosaic_util


def _format_copy_paste_command(argv, os_name=None):
    """Render argv with the quoting rules of the target command shell."""
    target_os = os.name if os_name is None else os_name
    args = [os.fspath(value) for value in argv]
    if target_os == "nt":
        return subprocess.list2cmdline(args)
    return shlex.join(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # data
    parser.add_argument("--dataset", type=str, default="./datasets/face/")
    parser.add_argument("--val_dataset", type=str, default="",
                        help="optional held-out dataset with the same layout")
    parser.add_argument("--val_steps", type=int, default=0,
                        help="validation clips per checkpoint; 0 disables validation")
    parser.add_argument("--M", type=int, default=100,
                        help="frames used per video (kept for dataset compatibility)")
    parser.add_argument("--T", type=int, default=5, help="clip length")
    parser.add_argument("--S", type=int, default=1,
                        help="temporal stride between clip frames")
    parser.add_argument("--finesize", type=int, default=256,
                        help="training crop size")
    parser.add_argument("--loadsize", type=int, default=286)
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--degradate_p", type=float, default=0.5,
                        help="probability of extra degradations (LADA recipe)")
    parser.add_argument("--mosaic_mods", type=str,
                        default="squa_mid,squa_avg,rect_avg",
                        help="online mosaic styles sampled during training")
    # model
    parser.add_argument("--arch", type=str, default="basicvsrpp",
                        choices=["basicvsrpp", "lite"])
    parser.add_argument("--n_feats", type=int, default=48,
                        help="width for basicvsrpp")
    parser.add_argument("--lite_ch", type=int, default=32,
                        help="width for lite")
    # optim
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--n_epoch", type=int, default=200)
    parser.add_argument("--save_freq", type=int, default=10000,
                        help="checkpoint interval in iterations")
    parser.add_argument("--log_freq", type=int, default=100,
                        help="log interval in iterations")
    parser.add_argument("--lambda_L2", type=float, default=100.0)
    parser.add_argument("--lambda_GAN", type=float, default=0.01,
                        help="multi-scale GAN weight; set 0 or --no_gan to disable")
    parser.add_argument("--lambda_VGG", type=float, default=1.0,
                        help="VGG perceptual weight; 0 avoids constructing/downloading VGG")
    parser.add_argument("--no_gan", action="store_true",
                        help="disable the DeepMosaics multi-scale GAN")
    parser.add_argument("--savename", type=str, default="")
    parser.add_argument("--device", type=str, default="auto",
                        help="auto | cuda | mps | directml | cpu")
    parser.add_argument("--seed", type=int, default=1337,
                        help="random seed for Python, NumPy and PyTorch")
    parser.add_argument("--max_iters", type=int, default=0,
                        help="stop after this many iterations; 0 means n_epoch schedule")
    parser.add_argument("--continue_train", type=str, default="",
                        help="checkpoint path to resume from")
    return parser


def seed_everything(seed: int) -> None:
    """Make data synthesis and model initialization repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class OnlineMosaicClipDataset:
    """Read clean/mask frame pairs and synthesize one mosaic clip on demand."""

    def __init__(self, root: str, opt: argparse.Namespace):
        self.opt = opt
        self.root = os.path.abspath(os.path.expanduser(root))
        self.videos = []
        required = (opt.T - 1) * opt.S + 1
        for name in sorted(os.listdir(self.root)):
            vdir = os.path.join(self.root, name)
            odir = os.path.join(vdir, "origin_image")
            mdir = os.path.join(vdir, "mask")
            if os.path.isdir(odir) and os.path.isdir(mdir):
                origin = [p for p in os.listdir(odir)
                          if p.lower().endswith((".jpg", ".jpeg", ".png"))]
                masks = [p for p in os.listdir(mdir)
                         if p.lower().endswith((".jpg", ".jpeg", ".png"))]
                n = min(len(origin), len(masks))
                if n >= required:
                    self.videos.append((vdir, n))
        if not self.videos:
            raise RuntimeError("no usable videos under %s" % self.root)
        self.mods = [m.strip() for m in opt.mosaic_mods.split(",") if m.strip()]
        if not self.mods:
            raise ValueError("--mosaic_mods must contain at least one style")

    def __len__(self):
        return sum(v[1] for v in self.videos)

    def _load(self, vdir: str, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        oimg = impro.imread(
            os.path.join(vdir, "origin_image", "%05d.jpg" % (idx + 1)),
            loadsize=self.opt.loadsize, rgb=True,
        )
        mask = impro.imread(
            os.path.join(vdir, "mask", "%05d.png" % (idx + 1)),
            mod="gray", loadsize=self.opt.loadsize,
        )
        if oimg is None or mask is None:
            raise RuntimeError("could not read frame %d from %s" % (idx + 1, vdir))
        return oimg, mask

    def get_clip(self) -> Tuple[np.ndarray, np.ndarray]:
        opt = self.opt
        vdir, count = random.choice(self.videos)
        max_start = max(0, count - (opt.T - 1) * opt.S - 1)
        start = random.randint(0, max_start)

        # Keep one mosaic parameter set for a clip so temporal changes are
        # caused by the source video, not by a changing synthetic mask.
        first_o, first_m = self._load(vdir, start)
        msize, mod, rect_rat, feather = mosaic_util.get_random_parameter(
            first_o, first_m
        )
        msize = max(1, int(msize))
        if mod not in self.mods:
            mod = random.choice(self.mods)
        startpos = [random.randint(0, max(1, int(msize))),
                    random.randint(0, max(1, int(msize)))]

        # A temporal clip is one aligned sample. Re-sampling the spatial crop
        # for every frame creates artificial camera jumps and teaches a video
        # backbone to model augmentation noise as motion.
        h, w = first_o.shape[:2]
        fh = fw = opt.finesize
        ym = random.randint(0, max(0, h - fh))
        xm = random.randint(0, max(0, w - fw))

        ori_clip, mos_clip = [], []
        for t in range(opt.T):
            idx = min(start + t * opt.S, count - 1)
            oimg, mask = self._load(vdir, idx)
            mimg = mosaic_util.addmosaic_base(
                oimg, mask, msize, 0, mod, rect_rat, feather, startpos
            )
            if random.random() < opt.degradate_p:
                params = degradater.get_random_degenerate_params("weaker_2")
                mimg = degradater.degradate(mimg, params)
            oimg = oimg[ym:ym + fh, xm:xm + fw]
            mimg = mimg[ym:ym + fh, xm:xm + fw]
            if oimg.shape[0] != fh or oimg.shape[1] != fw:
                oimg = cv2.resize(oimg, (fw, fh))
                mimg = cv2.resize(mimg, (fw, fh))
            ori_clip.append(oimg)
            mos_clip.append(mimg)

        ori = np.stack(ori_clip).astype(np.float32)  # T,H,W,3 RGB
        mos = np.stack(mos_clip).astype(np.float32)
        ori = ((ori / 255.0) - 0.5) / 0.5
        mos = ((mos / 255.0) - 0.5) / 0.5
        return mos.transpose(0, 3, 1, 2), ori.transpose(0, 3, 1, 2)


def clips_loader(dataset: OnlineMosaicClipDataset,
                 batchsize: int) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    while True:
        batch_m, batch_o = [], []
        for _ in range(batchsize):
            m, o = dataset.get_clip()
            batch_m.append(m)
            batch_o.append(o)
        yield (
            torch.from_numpy(np.stack(batch_m)),
            torch.from_numpy(np.stack(batch_o)),
        )


def _safe_load(path: str) -> Any:
    """Load tensor-only checkpoint data without executing arbitrary pickles."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "This trainer requires a current PyTorch build with safe "
            "weights_only checkpoint loading. Upgrade torch before resuming."
        ) from exc


def _state_dict(obj: Any, preferred: str = "netG") -> Optional[Dict[str, torch.Tensor]]:
    """Extract a plain state_dict from a new or legacy checkpoint."""
    if not isinstance(obj, dict):
        return None
    # Do not fall back from ``netD`` to ``netG``: a checkpoint without a
    # discriminator (for example a --no_gan fine-tune) must not accidentally
    # load generator tensors into D.
    candidates = [preferred, "state_dict", "model"]
    if preferred == "netG":
        candidates.append("netG")
    for key in candidates:
        value = obj.get(key)
        if isinstance(value, dict) and value and all(
            isinstance(k, str) and torch.is_tensor(v) for k, v in value.items()
        ):
            return value
    if obj and all(isinstance(k, str) and torch.is_tensor(v)
                   for k, v in obj.items()):
        return obj
    return None


def _cpu_state_dict(module: Optional[nn.Module]) -> Optional[Dict[str, torch.Tensor]]:
    if module is None:
        return None
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def _optimizer_to(optimizer: Optional[torch.optim.Optimizer], device: object) -> None:
    """Move optimizer tensors after loading a CPU checkpoint."""
    if optimizer is None:
        return
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    # Network values are [-1, 1]; convert the squared error to [0, 1] units.
    mse = torch.mean(((pred.detach().float() - target.detach().float()) / 2.0) ** 2)
    return float((-10.0 * torch.log10(mse.clamp_min(1e-12))).item())


def feature_matching(fake_d_out, real_d_out):
    """L1 matching over intermediate features, excluding PatchGAN logits."""
    total = None
    for fake_scale, real_scale in zip(fake_d_out, real_d_out):
        if not isinstance(fake_scale, list) or len(fake_scale) < 2:
            raise RuntimeError(
                "feature matching requires discriminator intermediate features")
        for fake, real in zip(fake_scale[:-1], real_scale[:-1]):
            value = torch.mean(torch.abs(fake - real.detach()))
            total = value if total is None else total + value
    if total is not None:
        return total
    return torch.zeros((), dtype=torch.float32)


def _set_requires_grad(module: Optional[nn.Module], enabled: bool) -> None:
    if module is not None:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)


def _make_models(opt: argparse.Namespace, dm: DeviceManager):
    if opt.arch == "basicvsrpp":
        netG = MosaicVSRNet(n_feats=opt.n_feats)
    else:
        netG = MosaicVRLiteNet(ch=opt.lite_ch)
    dm.move(netG)
    netG.train()
    print("[%s] parameters: %.2fM" % (opt.arch, netG.count_params()))

    optimizer_G = torch.optim.Adam(
        netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2)
    )
    lossfun_L2 = dm.move(nn.MSELoss())

    # VGGLoss used to instantiate VGG unconditionally and then only move it
    # for CUDA. Construct it lazily and use DeviceManager for every device.
    lossfun_VGG = None
    if opt.lambda_VGG > 0:
        lossfun_VGG = model_util.VGGLoss("-1")
        dm.move(lossfun_VGG)
        lossfun_VGG.eval()
        for parameter in lossfun_VGG.parameters():
            parameter.requires_grad_(False)

    use_gan = (not opt.no_gan) and opt.lambda_GAN > 0
    netD = optimizer_D = lossfun_GAND = lossfun_GANG = None
    if use_gan:
        # BVDNet's legacy gpu_id helper only knows CUDA; instantiate on CPU,
        # then move the complete discriminator/loss stack together.
        netD = BVDNet.define_D(
            n_layers_D=2, num_D=3, gpu_id="-1", return_features=True)
        dm.move(netD)
        netD.train()
        optimizer_D = torch.optim.Adam(
            netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2)
        )
        lossfun_GAND = dm.move(BVDNet.GANLoss("D"))
        lossfun_GANG = dm.move(BVDNet.GANLoss("G"))
    return (netG, optimizer_G, lossfun_L2, lossfun_VGG,
            use_gan, netD, optimizer_D, lossfun_GAND, lossfun_GANG)


def _checkpoint_payload(opt, netG, netD, optimizer_G, optimizer_D, it, epoch):
    return {
        "netG": _cpu_state_dict(netG),
        "netD": _cpu_state_dict(netD),
        "optimizer_G": optimizer_G.state_dict(),
        "optimizer_D": optimizer_D.state_dict() if optimizer_D is not None else None,
        "config": vars(opt).copy(),
        "iter": int(it),
        "epoch": int(epoch),
        "arch": opt.arch,
    }


def _save_checkpoint(path, opt, netG, netD, optimizer_G, optimizer_D, it, epoch):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(_checkpoint_payload(opt, netG, netD, optimizer_G, optimizer_D, it, epoch), path)
    print("saved ->", path)


def _resume(path, opt, netG, netD, optimizer_G, optimizer_D, dm):
    obj = _safe_load(path)
    sd_g = _state_dict(obj, "netG")
    if sd_g is None:
        raise RuntimeError("checkpoint has no tensor netG/state_dict: %s" % path)
    missing, unexpected = netG.load_state_dict(sd_g, strict=False)
    if missing or unexpected:
        print("[resume] netG missing=%d unexpected=%d" %
              (len(missing), len(unexpected)))
    it = int(obj.get("iter", 0)) if isinstance(obj, dict) else 0
    epoch = int(obj.get("epoch", 0)) if isinstance(obj, dict) else 0
    if isinstance(obj, dict):
        if isinstance(obj.get("optimizer_G"), dict):
            optimizer_G.load_state_dict(obj["optimizer_G"])
            _optimizer_to(optimizer_G, dm.info.device)
        if netD is not None:
            sd_d = _state_dict(obj, "netD")
            if sd_d is not None:
                netD.load_state_dict(sd_d, strict=False)
            if optimizer_D is not None and isinstance(obj.get("optimizer_D"), dict):
                optimizer_D.load_state_dict(obj["optimizer_D"])
                _optimizer_to(optimizer_D, dm.info.device)
    print("resumed from %s (epoch=%d iter=%d)" % (path, epoch, it))
    return epoch, it


def _run_validation(dataset, steps, netG, dm, opt) -> float:
    """Evaluate held-out synthesized clips and return average PSNR."""
    if dataset is None or steps <= 0:
        return float("nan")
    was_training = netG.training
    netG.eval()
    values = []
    with torch.inference_mode():
        for _ in range(steps):
            mosaic, target = dataset.get_clip()
            mosaic = torch.from_numpy(mosaic).unsqueeze(0).to(dm.info.device)
            target = torch.from_numpy(target).unsqueeze(0).to(dm.info.device)
            predictions = []
            # Keep the same per-clip execution contract used by training.
            for clip in mosaic:
                predictions.append(netG(clip))
            pred = torch.stack(predictions, dim=0)
            values.append(_psnr(pred, target))
    if was_training:
        netG.train()
    return float(np.mean(values))


def train(opt: argparse.Namespace) -> str:
    if opt.T < 1 or opt.S < 1 or opt.batchsize < 1:
        raise ValueError("T, S and batchsize must be positive")
    if opt.max_iters < 0 or opt.val_steps < 0:
        raise ValueError("max_iters and val_steps must be non-negative")
    seed_everything(opt.seed)
    dm = DeviceManager(opt.device, quiet=False)
    models = _make_models(opt, dm)
    (netG, optimizer_G, lossfun_L2, lossfun_VGG, use_gan,
     netD, optimizer_D, lossfun_GAND, lossfun_GANG) = models

    dataset = OnlineMosaicClipDataset(opt.dataset, opt)
    val_dataset = None
    if opt.val_dataset:
        val_dataset = OnlineMosaicClipDataset(opt.val_dataset, opt)
    clips = clips_loader(dataset, opt.batchsize)
    print("train videos:", len(dataset.videos))
    if val_dataset is not None:
        print("val videos:", len(val_dataset.videos))

    checkpoint_dir = os.path.join(
        REPO_ROOT, "checkpoints", opt.savename or ("%s_%d" % (opt.arch, opt.finesize))
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    try:
        from tensorboardX import SummaryWriter
        tb = SummaryWriter(os.path.join(
            checkpoint_dir, "tensorboard", time.strftime("%Y-%m-%d_%H-%M-%S")
        ))
    except Exception:
        tb = None

    start_epoch = 0
    it = 0
    if opt.continue_train:
        start_epoch, it = _resume(
            opt.continue_train, opt, netG, netD, optimizer_G, optimizer_D, dm
        )

    best_psnr = -float("inf")
    last_val_it = None
    last_epoch = start_epoch
    stop = False
    for epoch in range(start_epoch, opt.n_epoch):
        last_epoch = epoch
        steps = max(50, len(dataset.videos) * 4)
        for _ in range(steps):
            if opt.max_iters and it >= opt.max_iters:
                stop = True
                break
            it += 1
            mosaic_batch, ori_batch = next(clips)
            mosaic_batch = mosaic_batch.to(dm.info.device)
            ori_batch = ori_batch.to(dm.info.device)

            optimizer_G.zero_grad(set_to_none=True)
            predictions = []
            # MosaicVSRNet/MosaicVRLiteNet consume one T,C,H,W clip. Do not
            # flatten B and T: flattening makes temporal propagation cross
            # sample boundaries and gives the centre frame the wrong index.
            for clip in mosaic_batch:
                predictions.append(netG(clip))
            pred_batch = torch.stack(predictions, dim=0)
            mid = pred_batch.shape[1] // 2
            loss_l2 = lossfun_L2(pred_batch, ori_batch)
            gan_g = fm = loss_d = loss_l2.detach().new_zeros(())

            if use_gan:
                # D supplies gradients to G through its input, but its own
                # parameters do not need gradients until the D step. Freezing
                # them and evaluating the real branch without a graph saves a
                # material amount of unified/VRAM memory.
                _set_requires_grad(netD, False)
                d_in_fake = torch.cat(
                    (mosaic_batch[:, mid], pred_batch[:, mid]), dim=1
                )
                d_in_real = torch.cat(
                    (mosaic_batch[:, mid], ori_batch[:, mid]), dim=1
                )
                fake_d = netD(d_in_fake)
                with torch.no_grad():
                    real_d = netD(d_in_real)
                gan_g = lossfun_GANG(fake_d)
                fm = feature_matching(fake_d, real_d)
            loss_g = opt.lambda_L2 * loss_l2
            if use_gan:
                loss_g = loss_g + 10.0 * fm + opt.lambda_GAN * gan_g
            if lossfun_VGG is not None and opt.lambda_VGG > 0:
                flat_pred = pred_batch.reshape(-1, *pred_batch.shape[2:])
                flat_ori = ori_batch.reshape(-1, *ori_batch.shape[2:])
                loss_g = loss_g + opt.lambda_VGG * lossfun_VGG(flat_pred, flat_ori)
            loss_g.backward()
            optimizer_G.step()

            if use_gan:
                # Recompute both branches after G's backward pass; reusing
                # fake_d/real_d would attempt to traverse freed autograd graphs.
                _set_requires_grad(netD, True)
                optimizer_D.zero_grad(set_to_none=True)
                d_fake = netD(d_in_fake.detach())
                d_real = netD(d_in_real.detach())
                loss_d = lossfun_GAND(d_fake, d_real)
                loss_d.backward()
                optimizer_D.step()

            if it % max(1, opt.log_freq) == 0 or it == 1:
                msg = "epoch %d iter %d | L2 %.4f" % (epoch, it, loss_l2.item())
                if use_gan:
                    msg += " | GAN_G %.4f FM %.4f | D %.4f" % (
                        float(gan_g.detach()), float(fm.detach()), float(loss_d.detach())
                    )
                print(msg)
                if tb:
                    tb.add_scalar("loss/L2", loss_l2.item(), it)
                    if use_gan:
                        tb.add_scalar("loss/GAN_G", float(gan_g.detach()), it)
                        tb.add_scalar("loss/fm", float(fm.detach()), it)
                        tb.add_scalar("loss/D", float(loss_d.detach()), it)

            if it % max(1, opt.save_freq) == 0:
                _save_checkpoint(
                    os.path.join(checkpoint_dir, "%s_latest.pth" % opt.arch),
                    opt, netG, netD, optimizer_G, optimizer_D, it, epoch,
                )

                if val_dataset is not None and opt.val_steps > 0:
                    last_val_it = it
                    score = _run_validation(
                        val_dataset, opt.val_steps, netG, dm, opt
                    )
                    print("validation iter %d | PSNR %.4f dB" % (it, score))
                    if tb:
                        tb.add_scalar("val/PSNR", score, it)
                    if score > best_psnr:
                        best_psnr = score
                        _save_checkpoint(
                            os.path.join(checkpoint_dir, "%s_best.pth" % opt.arch),
                            opt, netG, netD, optimizer_G, optimizer_D, it, epoch,
                        )
        if stop:
            break

    # A short bounded run often ends before save_freq. Still perform one
    # holdout pass and materialize *_best.pth when validation was requested.
    if val_dataset is not None and opt.val_steps > 0 and it > 0 and last_val_it != it:
        score = _run_validation(val_dataset, opt.val_steps, netG, dm, opt)
        print("validation iter %d | PSNR %.4f dB" % (it, score))
        if tb:
            tb.add_scalar("val/PSNR", score, it)
        if score > best_psnr:
            best_psnr = score
            _save_checkpoint(
                os.path.join(checkpoint_dir, "%s_best.pth" % opt.arch),
                opt, netG, netD, optimizer_G, optimizer_D, it, last_epoch,
            )

    final_path = os.path.join(checkpoint_dir, "%s_final.pth" % opt.arch)
    _save_checkpoint(
        final_path, opt, netG, netD, optimizer_G, optimizer_D, it, last_epoch
    )
    if tb:
        tb.close()
    print("done ->", final_path)
    prefix = "lite" if opt.arch == "lite" else "portable"
    inference_cmd = [
        sys.executable, os.path.join(REPO_ROOT, "deepmosaic.py"),
        "--mode", "clean", "--media_path", "input.mp4",
        "--model", "%s:%s" % (prefix, final_path),
        "--device", "auto", "--no_preview",
    ]
    print("inference ->", _format_copy_paste_command(inference_cmd))
    return final_path


def main(argv=None) -> int:
    opt = build_parser().parse_args(argv)
    train(opt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
