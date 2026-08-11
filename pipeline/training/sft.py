# coding=utf-8
"""SFT / teacher-KD async training worker."""

from __future__ import annotations

import argparse
import logging
import os

import torch

from pipeline.bootstrap import ensure_vendored_specforge

ensure_vendored_specforge()

from specforge.args import SGLangBackendArgs
from specforge.core.dflash import OnlineDFlashModel
from specforge.distributed import destroy_distributed, get_dp_group, init_distributed

from pipeline.training.args_common import (
    add_distributed_args,
    add_model_args,
    add_swap_pipeline_args,
    add_training_hyper_args,
    prepare_sglang_target_train_args,
    validate_common_training_args,
)
from pipeline.training.loop import run_buffer_training_loop
from pipeline.training.setup import (
    build_sft_optimizer,
    load_draft_stack,
    load_target_embeddings,
    load_target_model,
)
from pipeline.training.utils import _log_rank0, _setup_training_logging, _wrap_fsdp_if_distributed

logger = logging.getLogger(__name__)


def main() -> None:
    _setup_training_logging()

    parser = argparse.ArgumentParser(description="Async pipeline: SFT training worker")
    add_model_args(parser)
    parser.add_argument(
        "--teacher-kd-temperature",
        type=float,
        default=None,
        metavar="T",
        help="Optional logits KD when set > 0. Teacher logits from target hidden + lm_head.",
    )
    parser.add_argument(
        "--teacher-kd-alpha",
        type=float,
        default=0.5,
        help="When --teacher-kd-temperature set: alpha*CE_hard + (1-alpha)*KD_soft.",
    )
    add_training_hyper_args(parser)
    parser.add_argument(
        "--from-ground-truth",
        action="store_true",
        default=False,
        help="Supervised chat from question + ground_truth (see training utils dataset).",
    )
    add_swap_pipeline_args(parser)
    add_distributed_args(parser)
    SGLangBackendArgs.add_args(parser)
    args = parser.parse_args()
    prepare_sglang_target_train_args(args)

    if not args.from_ground_truth:
        gt_env = os.environ.get("FROM_GROUND_TRUTH", "").strip().lower()
        if gt_env in ("1", "true", "yes"):
            args.from_ground_truth = True
    validate_common_training_args(parser, args)

    dist_inited = False
    try:
        if args.target_model_backend == "sglang":
            init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
            dist_inited = True
            _setup_training_logging()
        _run(args)
    finally:
        if dist_inited:
            destroy_distributed()


def _run(args: argparse.Namespace) -> None:
    target_model = load_target_model(args)
    draft_model, draft_config, tokenizer, mask_token_id = load_draft_stack(
        args, target_model
    )
    target_components = load_target_embeddings(args)

    kd_kw: dict = {}
    if args.teacher_kd_temperature is not None:
        kd_kw["teacher_kd_temperature"] = args.teacher_kd_temperature
        kd_kw["teacher_kd_alpha"] = args.teacher_kd_alpha
    dflash_model = OnlineDFlashModel(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        block_size=draft_model.block_size,
        mask_token_id=mask_token_id,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
        **kd_kw,
    )
    if kd_kw:
        _log_rank0(
            "[training] Teacher logits KD: T=%s alpha=%s (alpha=0 => soft KL only)",
            args.teacher_kd_temperature,
            args.teacher_kd_alpha,
        )

    optimizer = build_sft_optimizer(args, draft_model)

    dflash_model = _wrap_fsdp_if_distributed(
        dflash_model, draft_model, tp_size=getattr(args, "tp_size", 1)
    )

    run_buffer_training_loop(
        args,
        mode_label="SFT",
        draft_model=draft_model,
        draft_config=draft_config,
        target_model=target_model,
        dflash_model=dflash_model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        from_ground_truth=args.from_ground_truth,
        dp_group=get_dp_group(),
    )


if __name__ == "__main__":
    main()
