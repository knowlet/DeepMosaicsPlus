# Vendored LADA inference subset

Source reference:
[`ladaapp/lada@20cb34a20a83c72c87a991d2c949032c70085b16`](https://github.com/ladaapp/lada/tree/20cb34a20a83c72c87a991d2c949032c70085b16)
(`0.11.1-dev`). Vendored and modified through 2026-08-24.

| Local file | Provenance and local change |
|---|---|
| `basicvsr_plusplus_net.py` | LADA/MMagic-derived; imports redirected to the local mmengine shim |
| `model_utils.py` | LADA/MMagic-derived; imports redirected to the local registry/shim |
| `deformconv.py` | LADA implementation retained for its MPS-compatible fallback |
| `flow_warp.py` | LADA/MMagic-derived implementation retained |
| `registry.py` | Upstream registry dependency replaced by minimal local export |
| `mmengine_shim.py` | New inference-only compatibility implementation |
| `__init__.py` | New local package glue |

LADA-covered code is AGPL-3.0-only. Files retaining OpenMMLab notices also
carry Apache-2.0 provenance. Full terms are available at
[`../../LICENSE-AGPL-3.0`](../../LICENSE-AGPL-3.0) and
[`../../LICENSE-APACHE-2.0`](../../LICENSE-APACHE-2.0).
