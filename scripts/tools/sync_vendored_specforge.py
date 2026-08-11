#!/usr/bin/env python3
"""Re-copy SpecForge modules into asyn_train/specforge (preserves vendoring headers)."""
from __future__ import annotations

import argparse
from pathlib import Path

HEADER_TEMPLATE = """# ------------------------------------------------------------------------------
# Vendored from SpecForge (sgl-project), MIT License.
# Upstream: SpecForge/specforge/{rel}
# Full attribution and license: asyn_train/specforge/VENDOR_README.md
# ------------------------------------------------------------------------------

"""

FILES = [
    "args.py",
    "distributed.py",
    "core/dflash.py",
    "modeling/draft/dflash.py",
    "modeling/target/dflash_target_model.py",
    "modeling/target/target_utils.py",
    "modeling/target/sglang_backend/__init__.py",
    "modeling/target/sglang_backend/model_runner.py",
    "modeling/target/sglang_backend/patch.py",
    "modeling/target/sglang_backend/utils.py",
]

UTILS_SUBSET = HEADER_TEMPLATE.format(rel="utils.py (subset)") + """import logging

import torch.distributed as dist

logger = logging.getLogger(__name__)


def print_with_rank(message):
    if dist.is_available() and dist.is_initialized():
        logger.info("rank %s: %s", dist.get_rank(), message)
    else:
        logger.info("non-distributed: %s", message)
"""


def prepend_header(rel: str, text: str) -> str:
    header = HEADER_TEMPLATE.format(rel=rel)
    if text.startswith("#!"):
        first, rest = text.split("\n", 1)
        return first + "\n" + header + rest
    if text.startswith("# coding"):
        lines = text.split("\n", 2)
        if len(lines) >= 3:
            return "\n".join(lines[:2]) + "\n" + header + lines[2]
    return header + text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="dFlash_wu repo root (parent of SpecForge and asyn_train)",
    )
    args = parser.parse_args()
    src_root = args.repo_root / "SpecForge" / "specforge"
    dst_root = args.repo_root / "asyn_train" / "specforge"
    if not src_root.is_dir():
        raise SystemExit(f"upstream not found: {src_root}")

    for rel in FILES:
        src = src_root / rel
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(prepend_header(rel, src.read_text(encoding="utf-8")), encoding="utf-8")
        print("synced", rel)

    (dst_root / "utils.py").write_text(UTILS_SUBSET, encoding="utf-8")
    print("synced utils.py (subset)")
    license_src = args.repo_root / "SpecForge" / "LICENSE"
    if license_src.is_file():
        import shutil

        shutil.copy2(license_src, dst_root / "LICENSE")
        print("copied LICENSE")


if __name__ == "__main__":
    main()
