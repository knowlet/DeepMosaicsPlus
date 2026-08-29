# Licensing and provenance notice

This repository combines components with compatible copyleft terms; it does
not erase the notices or licensing of the individual upstream works.

## DeepMosaics base

- Upstream: https://github.com/HypoX64/DeepMosaics
- Reference revision: `b311aaac319439a0a1e8004b4a64c4222c55f38c`
- Upstream code license: GPL-3.0-only; see [`LICENSE`](LICENSE).
- DeepMosaicsPlus has modified the runtime, model loading, device handling,
  training, video pipeline, UI, and tests. The modernization work in this
  checkout was updated through 2026-08-24.

The original BiSeNet and BVDNet checkpoints are fetched at runtime from the
author's published download locations and are SHA-256 verified. They are not to
be bundled into a binary/source release unless their separate checkpoint
redistribution terms have been confirmed. The upstream repository's GPL notice
is not, by itself, treated here as proof of a separate weight license.

## LADA and MMagic-derived restoration code

- Upstream: https://github.com/ladaapp/lada
- Vendored source reference: `20cb34a20a83c72c87a991d2c949032c70085b16`
  (`0.11.1-dev`).
- Official model-card/weight revision:
  `bcf461d46d9a98981fc64b815df5178f42215cdf`.
- LADA terms: AGPL-3.0-only; see
  [`LICENSE-AGPL-3.0`](LICENSE-AGPL-3.0).
- OpenMMLab/MMagic-derived portions retain their Apache-2.0 notices; see
  [`LICENSE-APACHE-2.0`](LICENSE-APACHE-2.0).

Files under `thirdparty/lada/` were vendored for the official BasicVSR++
inference backend. DeepMosaicsPlus changed imports to local shims, replaced the
registry dependency with a minimal implementation, and added an inference-only
`mmengine_shim.py` plus package glue. Those changes were made for portable
CUDA/MPS/CPU inference and updated through 2026-08-24. Exact file-level details
are in [`thirdparty/lada/README.md`](thirdparty/lada/README.md).

The official LADA checkpoint remains under its upstream AGPL terms. Its pinned
download and SHA-256 appear in `restoration/manifests/models.yaml`.

## Combined application

The original GPL-3.0-only portions remain under GPL. The LADA-integrated quality
path is a combined work whose AGPL-3.0-only terms apply to that covered portion
and to remote-network source availability as specified by AGPL section 13.
Operators exposing `tools/server.py` must prominently offer the complete
corresponding source for the exact deployed version to remote users.

This notice is an engineering provenance record, not legal advice.
