# coding=utf-8
"""Ensure ``import specforge`` resolves to ``asyn_train/specforge``, not repo ``SpecForge/``."""

from __future__ import annotations

import sys

from pipeline.paths import ASYN_TRAIN_ROOT, REPO_ROOT

_ASYN_TRAIN_STR = str(ASYN_TRAIN_ROOT)
_REPO_SPEC_FORGE = REPO_ROOT / "SpecForge"


def ensure_vendored_specforge(*, prefer_vendored: bool = True) -> None:
    """
    Put ``asyn_train`` on ``sys.path`` so ``import specforge`` loads
    ``asyn_train/specforge``.

    When *prefer_vendored* is true, remove repo-root ``SpecForge`` from ``sys.path``
    so an editable install does not shadow the vendored tree.
    """
    if prefer_vendored:
        repo_sf = str(_REPO_SPEC_FORGE)
        sys.path[:] = [p for p in sys.path if p != repo_sf]

    if _ASYN_TRAIN_STR not in sys.path:
        sys.path.insert(0, _ASYN_TRAIN_STR)
