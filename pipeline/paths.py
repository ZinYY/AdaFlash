# coding=utf-8
"""Filesystem roots for the asyn_train package."""

from __future__ import annotations

from pathlib import Path

# ``asyn_train/`` (parent of ``pipeline/``)
ASYN_TRAIN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ASYN_TRAIN_ROOT.parent

CONFIGS_DIR = ASYN_TRAIN_ROOT / "configs"
CACHE_DIR = ASYN_TRAIN_ROOT / "cache"
TEST_DATA_DIR = ASYN_TRAIN_ROOT / "test_data"
BIN_DIR = ASYN_TRAIN_ROOT / "bin"
