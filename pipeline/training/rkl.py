# coding=utf-8
"""RKL + optional thresh-head async training worker."""

from __future__ import annotations

import argparse
import logging
from typing import Optional

import torch

from pipeline.bootstrap import ensure_vendored_specforge

ensure_vendored_specforge()

from specforge.args import SGLangBackendArgs
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
from pipeline.training.rkl_model import OnlineDFlashModelReverseKL
from pipeline.training.setup import (
    build_rkl_optimizer,
    load_draft_stack,
    load_target_embeddings,
    load_target_model,
    log_rkl_thresh_heads,
)
from pipeline.training.utils import _log_rank0, _setup_training_logging, _wrap_fsdp_if_distributed

logger = logging.getLogger(__name__)


def _rkl_round_log(
    train_step: int,
    train_wall_s: float,
    total_round_s: float,
    loss: float,
    acc: float,
    clip_ratio: Optional[float],
    thresh_mae: Optional[float],
    current_version: int,
    consumed_n: int,
    consumed_summary: str,
    optimizer: torch.optim.Optimizer,
    _consumed: set,
) -> None:
    lr_draft = optimizer.param_groups[0]["lr"]
    lr_log = (
        f"lr_draft={lr_draft:.2e} lr_thresh={optimizer.param_groups[1]['lr']:.2e}"
        if len(optimizer.param_groups) > 1
        else f"lr={lr_draft:.2e}"
    )
    parts = [
        f"[training] Buffer round {train_step} complete | train_s={train_wall_s:.2f} total_s={total_round_s:.2f} | ",
        f"loss={loss:.4f} acc={acc:.4f}",
    ]
    if clip_ratio is not None:
        parts.append(f" clip_ratio={clip_ratio:.6f}")
    else:
        parts.append(" clip_ratio=n/a")
    if thresh_mae is not None:
        parts.append(f" thresh_mae={thresh_mae:.6f}")
    parts.append(f" | {lr_log} | draft_version={current_version} | ")
    parts.append(f"consumed_jsonl={consumed_n} {consumed_summary}")
    _log_rank0("".join(parts))


def main() -> None:
    _setup_training_logging()

    parser = argparse.ArgumentParser(description="Async pipeline: RKL training worker")
    add_model_args(parser)
    parser.add_argument(
        "--rkl-temperature",
        type=float,
        default=1.0,
        metavar="T",
        help="Temperature for student/teacher softmax in KL(p_S||p_T).",
    )
    parser.add_argument(
        "--rkl-alpha",
        type=float,
        default=0.5,
        metavar="A",
        help="loss = alpha*CE_hard(teacher argmax) + (1-alpha)*RKL.",
    )
    parser.add_argument(
        "--rkl-div-clip-tau",
        type=float,
        default=None,
        metavar="TAU",
        help="Per-token RKL cap before weighted mean; omit for no clip.",
    )
    parser.add_argument(
        "--train-thresh-head",
        action="store_true",
        help="Train adaptive length head (detached; does not affect draft grads).",
    )
    parser.add_argument(
        "--thresh-head-loss-type",
        default="mse",
        choices=["mse", "mae"],
    )
    parser.add_argument("--thresh-label-lookahead", type=int, default=1)
    parser.add_argument("--detailed-debug-print", action="store_true")
    parser.add_argument(
        "--thresh-head-learning-rate",
        type=float,
        default=None,
        metavar="LR",
        help="LR for thresh_head* param group when --train-thresh-head.",
    )
    add_training_hyper_args(parser)
    add_swap_pipeline_args(parser)
    add_distributed_args(parser)
    SGLangBackendArgs.add_args(parser)
    args = parser.parse_args()
    args.use_draft_mask_token_id = True
    prepare_sglang_target_train_args(args)

    validate_common_training_args(parser, args)
    if args.rkl_temperature <= 0:
        parser.error("--rkl-temperature must be > 0")
    if not 0.0 <= args.rkl_alpha <= 1.0:
        parser.error("--rkl-alpha must be in [0, 1]")
    if args.rkl_div_clip_tau is not None and args.rkl_div_clip_tau <= 0:
        parser.error("--rkl-div-clip-tau must be > 0 when set")

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

    dflash_model = OnlineDFlashModelReverseKL(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        block_size=draft_model.block_size,
        mask_token_id=mask_token_id,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
        rkl_temperature=args.rkl_temperature,
        rkl_alpha=args.rkl_alpha,
        rkl_div_clip_tau=args.rkl_div_clip_tau,
        train_thresh_head=args.train_thresh_head,
        thresh_head_loss_type=args.thresh_head_loss_type,
        thresh_label_lookahead=args.thresh_label_lookahead,
        detailed_debug_print=args.detailed_debug_print,
    )
    if args.rkl_div_clip_tau is not None:
        _log_rank0(
            "[training] RKL mix: alpha=%.4f (CE) + (1-alpha)*KL(p_S||p_T), T=%s, div_clip_tau=%s",
            args.rkl_alpha,
            args.rkl_temperature,
            args.rkl_div_clip_tau,
        )
    else:
        _log_rank0(
            "[training] RKL mix: alpha=%.4f (CE) + (1-alpha)*KL(p_S||p_T), T=%s",
            args.rkl_alpha,
            args.rkl_temperature,
        )

    optimizer, _draft_params, thresh_params = build_rkl_optimizer(args, draft_model)
    log_rkl_thresh_heads(args, draft_model)

    dflash_model = _wrap_fsdp_if_distributed(
        dflash_model, draft_model, tp_size=getattr(args, "tp_size", 1)
    )

    run_buffer_training_loop(
        args,
        mode_label="RKL",
        draft_model=draft_model,
        draft_config=draft_config,
        target_model=target_model,
        dflash_model=dflash_model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        thresh_head_active=bool(args.train_thresh_head and thresh_params),
        dp_group=get_dp_group(),
        on_round_complete=_rkl_round_log,
    )


if __name__ == "__main__":
    main()
