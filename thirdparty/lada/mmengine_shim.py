# SPDX-FileCopyrightText: DeepMosaicsPlus contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Added by DeepMosaicsPlus; updated through 2026-08-24.

"""Minimal mmengine stand-in so LADA vendored nets run without mmengine.

Only what inference needs: BaseModule (plain nn.Module), init helpers,
a no-op MMLogger, and a load_checkpoint that unwraps common key layouts.
"""

import torch
import torch.nn as nn


class MMLogger:
    @staticmethod
    def get_current_instance():
        return MMLogger()

    def warning(self, *a, **k):
        pass

    def info(self, *a, **k):
        pass


class BaseModule(nn.Module):
    def __init__(self, init_cfg=None):
        super().__init__()
        self.init_cfg = init_cfg


def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module, a=0, mode="fan_out", nonlinearity="relu",
                 bias=0, distribution="normal"):
    assert distribution in ["normal", "uniform"]
    if distribution == "normal":
        nn.init.kaiming_normal_(module.weight, a=a, mode=mode,
                                nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_uniform_(module.weight, a=a, mode=mode,
                                 nonlinearity=nonlinearity)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def _unwrap_sd(checkpoint):
    """Return a flat state_dict from common checkpoint layouts."""
    sd = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "params_ema", "params", "netG",
                    "generator", "model"):
            v = checkpoint.get(key) if hasattr(checkpoint, "get") else None
            if isinstance(v, dict):
                sd = v
                break
    # nested wrapper: prefer EMA weights when present
    if isinstance(sd, dict):
        ema = {k[len("generator_ema."):]: v for k, v in sd.items()
               if k.startswith("generator_ema.")}
        if ema:
            return ema
        gen = {k[len("generator."):]: v for k, v in sd.items()
               if k.startswith("generator.")}
        if gen:
            return gen
    return sd


def load_checkpoint(model, filename, map_location=None, strict=False, logger=None):
    checkpoint = torch.load(filename, map_location=map_location, weights_only=True)
    sd = _unwrap_sd(checkpoint)
    sd = {(k[7:] if k.startswith("module.") else k): v
          for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    return checkpoint


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution="normal"):
    assert distribution in ["normal", "uniform"]
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "normal":
            nn.init.xavier_normal_(module.weight, gain)
        else:
            nn.init.xavier_uniform_(module.weight, gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def update_init_info(module, init_info):  # logging-only in upstream
    pass


class Registry(dict):
    """Tiny stand-in: register_module stores classes by name."""

    def __init__(self, *a, **k):
        super().__init__()
        self._name = k.get("name", "registry")

    def register_module(self, name=None, **k):
        def deco(cls):
            self[name or cls.__name__] = cls
            return cls
        return deco

    def get(self, key):
        return super().get(key)

    def build(self, cfg, *a, **k):  # pragma: no cover - not used here
        raise NotImplementedError


class _BatchNorm(nn.modules.batchnorm._BatchNorm):
    pass
