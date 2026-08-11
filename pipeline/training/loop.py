# coding=utf-8
"""Shared buffer polling / train / save loop for SFT and RKL workers."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

import torch

from pipeline.training.utils import (
    TRAINING_STATE_FILENAME,
    _collect_samples,
    _dist_barrier,
    _dist_rank,
    _fsdp_save_draft,
    _load_training_state,
    _log_rank0,
    _read_version,
    _save_draft,
    _save_training_state,
    _summarize_consumed_files,
    _swap_paths,
    _train_step,
    clear_stop_pipeline_marker,
    snapshot_model_weights_dir,
    stop_pipeline_marker_exists,
    write_stop_pipeline_marker,
)

logger = logging.getLogger(__name__)

RoundLogFn = Callable[
    [
        int,
        float,
        float,
        float,
        float,
        float,
        Optional[float],
        Optional[float],
        int,
        int,
        torch.optim.Optimizer,
        set,
    ],
    None,
]


def _default_round_log(
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
    lr = optimizer.param_groups[0]["lr"]
    _log_rank0(
        "[training] Buffer round %d complete | train_s=%.2f total_s=%.2f | "
        "loss=%.4f acc=%.4f | lr=%.2e | draft_version=%d | consumed_jsonl=%d %s",
        train_step,
        train_wall_s,
        total_round_s,
        loss,
        acc,
        lr,
        current_version,
        consumed_n,
        consumed_summary,
    )


def run_buffer_training_loop(
    args,
    *,
    mode_label: str,
    draft_model,
    draft_config,
    target_model,
    dflash_model,
    optimizer: torch.optim.Optimizer,
    tokenizer,
    thresh_head_active: bool = False,
    from_ground_truth: bool = False,
    on_round_complete: Optional[RoundLogFn] = None,
    dp_group=None,  # dist.ProcessGroup for DistributedSampler in _train_step
) -> None:
    data_buffer, model_weights = _swap_paths(args.swap_dir)
    current_version = _read_version(model_weights)
    train_step = 0
    thresh_lr = getattr(args, "thresh_head_learning_rate", None) if thresh_head_active else None

    resumed = _load_training_state(
        model_weights,
        optimizer,
        apply_train_step=True,
        learning_rate=args.learning_rate,
        thresh_head_learning_rate=thresh_lr,
    )
    if resumed is not None:
        train_step, ckpt_dv = resumed
        if ckpt_dv is not None and ckpt_dv != current_version:
            _log_rank0(
                "[training] Note: training_state draft_version=%s vs version.txt=%s",
                ckpt_dv,
                current_version,
            )
        _log_rank0(
            "[training] Resumed train_step=%d from %s",
            train_step,
            TRAINING_STATE_FILENAME,
        )
    _log_rank0(
        "[training] Ready (%s). poll=%.1fs threshold=%d buffer_epochs=%d grad_accum=%d draft_version=%d",
        mode_label,
        args.poll_interval,
        args.train_threshold,
        args.buffer_epochs,
        args.gradient_accumulation_steps,
        current_version,
    )

    if _dist_rank() == 0:
        clear_stop_pipeline_marker(args.swap_dir)

    log_fn = on_round_complete or _default_round_log

    while True:
        samples, files = _collect_samples(data_buffer, args.train_threshold)

        if len(samples) < args.train_threshold:
            if stop_pipeline_marker_exists(args.swap_dir):
                _log_rank0(
                    "[training] stop_pipeline set and buffer %d/%d; stopping "
                    "(inference ended without --loop or pipeline shutdown).",
                    len(samples),
                    args.train_threshold,
                )
                break
            if _dist_rank() == 0:
                logger.debug(
                    "[training] Buffer has %d/%d samples, waiting...",
                    len(samples),
                    args.train_threshold,
                )
            time.sleep(args.poll_interval)
            continue

        _load_training_state(
            model_weights,
            optimizer,
            apply_train_step=False,
            learning_rate=args.learning_rate,
            thresh_head_learning_rate=thresh_lr,
        )
        _dist_barrier()

        _log_rank0(
            "[training] Starting buffer round %d | samples=%d buffer_epochs=%d "
            "grad_accum=%d batch_size=%d max_length=%d lr=%.2e",
            train_step + 1,
            len(samples),
            args.buffer_epochs,
            args.gradient_accumulation_steps,
            args.batch_size,
            args.max_length,
            optimizer.param_groups[0]["lr"],
        )
        if len(optimizer.param_groups) > 1:
            _log_rank0(
                "[training]   thresh_head param_group lr=%.2e",
                optimizer.param_groups[1]["lr"],
            )

        t_round = time.perf_counter()
        t_train = time.perf_counter()
        loss, acc, clip_ratio, thresh_mae = _train_step(
            samples,
            draft_model=draft_model,
            target_model=target_model,
            dflash_model=dflash_model,
            optimizer=optimizer,
            tokenizer=tokenizer,
            max_length=args.max_length,
            batch_size=args.batch_size,
            chat_template=args.chat_template,
            enable_thinking=args.enable_thinking,
            buffer_epochs=args.buffer_epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            from_ground_truth=from_ground_truth,
            log_interval=getattr(args, "log_interval", 0),
            log_prefix="[training]",
            dp_group=dp_group,
        )
        train_wall_s = time.perf_counter() - t_train
        train_step += 1

        current_version += 1
        draft_models_root = Path(args.swap_dir) / args.draft_models_subdir
        # stop_pipeline marker is a simple file touch – no FSDP involvement.
        if _dist_rank() == 0:
            if args.max_draft_version > 0 and current_version >= args.max_draft_version:
                write_stop_pipeline_marker(args.swap_dir)
                logger.info(
                    "[training] Reached max_draft_version=%d (about to save v=%d); wrote stop_pipeline under %s before save.",
                    args.max_draft_version,
                    current_version,
                    args.swap_dir,
                )
        _fsdp_save_draft(
            dflash_model,
            model_weights,
            draft_model,
            draft_config,
            current_version,
            optimizer=optimizer,
            train_step=train_step,
            draft_version=current_version,
            log_fn=lambda: _log_rank0(
                "[training] Saved draft v%d -> %s", current_version, model_weights
            ),
        )
        if _dist_rank() == 0 and (
            args.draft_snapshot_interval > 0
            and current_version > 0
            and current_version % args.draft_snapshot_interval == 0
        ):
            snapshot_model_weights_dir(model_weights, draft_models_root, current_version)

        consumed = set(files)
        if _dist_rank() == 0:
            for f in consumed:
                try:
                    f.unlink()
                except FileNotFoundError:
                    pass
        _dist_barrier()

        total_round_s = time.perf_counter() - t_round
        log_fn(
            train_step,
            train_wall_s,
            total_round_s,
            loss,
            acc,
            clip_ratio,
            thresh_mae,
            current_version,
            len(consumed),
            _summarize_consumed_files(consumed),
            optimizer,
            consumed,
        )

        if args.max_draft_version > 0 and current_version >= args.max_draft_version:
            _log_rank0(
                "[training] Stopping after max_draft_version=%d (current_version=%d).",
                args.max_draft_version,
                current_version,
            )
            break

        if args.max_steps > 0 and train_step >= args.max_steps:
            _log_rank0("[training] Reached max_steps=%d, stopping.", args.max_steps)
            break

    _log_rank0("[training] Done.")
