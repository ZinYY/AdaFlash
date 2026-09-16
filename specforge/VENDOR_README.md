# Vendored SpecForge (`specforge`)

This directory contains a **minimal subset** of [SpecForge](https://github.com/sgl-project/SpecForge)
copied into `asyn_train` so the async training pipeline does not depend on a separate
`SpecForge/` checkout on `PYTHONPATH`.

## License

The upstream project is distributed under the **MIT License** (Copyright (c) 2025 sgl-project).
The full license text is in [`LICENSE`](LICENSE) (copied from the upstream repository).

## Files vendored from upstream

| Local path | Upstream path |
|------------|---------------|
| `args.py` | `SpecForge/specforge/args.py` |
| `utils.py` | subset of `SpecForge/specforge/utils.py` (`print_with_rank` only) |
| `distributed.py` | `SpecForge/specforge/distributed.py` |
| `core/dflash.py` | `SpecForge/specforge/core/dflash.py` |
| `modeling/draft/dflash.py` | `SpecForge/specforge/modeling/draft/dflash.py` |
| `modeling/target/dflash_target_model.py` | `SpecForge/specforge/modeling/target/dflash_target_model.py` |
| `modeling/target/target_utils.py` | `SpecForge/specforge/modeling/target/target_utils.py` |
| `modeling/target/sglang_backend/*` | `SpecForge/specforge/modeling/target/sglang_backend/*` |

Each `.py` file begins with a short vendoring notice pointing here.

## Updating from upstream

When syncing with a newer SpecForge release, re-copy the files listed above from
`SpecForge/specforge/` and re-apply the header block from `asyn_train/scripts/tools/sync_vendored_specforge.py`
(if present) or manually preserve the notice at the top of each file.

## Runtime

Entry scripts call `pipeline.bootstrap.ensure_vendored_specforge()` so `import specforge` resolves
to this tree (`asyn_train/` on `sys.path`) before any repo-root `SpecForge/` install.
