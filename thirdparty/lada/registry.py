# SPDX-FileCopyrightText: DeepMosaicsPlus contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Replaced the upstream registry dependency through 2026-08-24.

from .mmengine_shim import Registry

MODELS = Registry(name="models")
