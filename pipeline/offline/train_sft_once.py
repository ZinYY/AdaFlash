# coding=utf-8
"""One-shot offline SFT training on a static JSONL dataset.

Matches a single buffer round of ``run_async_pipeline_osd.sh``:
load JSONL, train for ``--buffer-epochs`` full passes, save draft weights to ``--output-dir``.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _ensure_repo_pythonpath() -> None:
    from pipeline.bootstrap import ensure_vendored_specforge
    from pipeline.paths import REPO_ROOT

    ensure_vendored_specforge()
    sp = str(REPO_ROOT / "sglang_dflash/python")
    if Path(sp).is_dir() and sp not in sys.path:
        sys.path.insert(0, sp)


_ensure_repo_pythonpath()

from specforge.args import SGLangBackendArgs
from specforge.core.dflash import OnlineDFlashModel
from specforge.distributed import destroy_distributed, get_dp_group, init_distributed

from pipeline.offline.train_rkl_once import _prepare_offline_train_args
from pipeline.paths import ASYN_TRAIN_ROOT, REPO_ROOT
from pipeline.training.args_common import add_distributed_args, add_model_args
from pipeline.training.setup import (
    build_sft_optimizer,
    load_draft_stack,
    load_target_embeddings,
    load_target_model,
)
from pipeline.training.utils import (
    _dist_barrier,
    _dist_rank,
    _fsdp_save_draft,
    _log_rank0,
    _setup_training_logging,
    _train_step,
    _wrap_fsdp_if_distributed,
    load_jsonl_records,
)

logger = logging.getLogger(__name__)


def _copy_dflash_remote_code(output_dir: Path) -> None:
    candidates = [
        REPO_ROOT / "dflash" / "dflash" / "model.py",
        Path(__file__).resolve().parents[3] / "dflash" / "dflash" / "model.py",
    ]
    dst = output_dir / "dflash.py"
    if dst.exists():
        return
    for src in candidates:
        if src.is_file():
            shutil.copy2(src, dst)
            _log_rank0("[offline_sft] copied remote code -> %s", dst)
            return


def _row_usable(row: Dict[str, Any], *, from_ground_truth: bool) -> bool:
    if from_ground_truth:
        q = row.get("question")
        gt = row.get("ground_truth")
        q_ok = isinstance(q, str) and q.strip()
        gt_ok = isinstance(gt, str) and gt.strip()
        return bool(q_ok and gt_ok)

    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        response = row.get("response", "")
        return isinstance(response, str) and bool(response.strip())

    conv = row.get("conversations")
    if isinstance(conv, list) and conv:
        return True

    q = row.get("question", "")
    r = row.get("response", "")
    q_ok = isinstance(q, str) and q.strip()
    r_ok = isinstance(r, str) and r.strip()
    return bool(q_ok and r_ok)


def _filter_jsonl_samples(
    rows: List[Dict[str, Any]], *, from_ground_truth: bool
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        if _row_usable(row, from_ground_truth=from_ground_truth):
            out.append(row)
        else:
            logger.warning("[offline_sft] skip row %d: unusable for from_ground_truth=%s", i, from_ground_truth)
    return out


def offline_train_sft_once(args: argparse.Namespace) -> Path:
    """Run one offline SFT training pass and save weights."""
    _prepare_offline_train_args(args)

    out_path = Path(args.output_dir).resolve()
    if _dist_rank() == 0:
        out_path.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl_records(Path(args.dataset_path))
    samples = _filter_jsonl_samples(rows, from_ground_truth=args.from_ground_truth)
    if args.max_samples is not None and args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        raise ValueError(f"No usable samples in {args.dataset_path}")

    _log_rank0(
        "[offline_sft] dataset=%s usable_samples=%d buffer_epochs=%d batch_size=%d "
        "grad_accum=%d max_length=%d from_ground_truth=%s sglang_context_length=%s",
        args.dataset_path,
        len(samples),
        args.buffer_epochs,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.max_length,
        args.from_ground_truth,
        args.sglang_context_length,
    )

    target_model = load_target_model(args)
    draft_model, draft_config, tokenizer, mask_token_id = load_draft_stack(args, target_model)
    target_components = load_target_embeddings(args)

    kd_kw: Dict[str, Any] = {}
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
            "[offline_sft] Teacher logits KD: T=%s alpha=%s (alpha=0 => soft KL only)",
            args.teacher_kd_temperature,
            args.teacher_kd_alpha,
        )
    else:
        _log_rank0("[offline_sft] Teacher KD disabled (hard CE only)")

    optimizer = build_sft_optimizer(args, draft_model)

    dflash_model = _wrap_fsdp_if_distributed(
        dflash_model, draft_model, tp_size=getattr(args, "tp_size", 1)
    )

    checkpoint_save_fn = None
    if args.save_interval > 0:

        def _save_checkpoint(global_opt_step: int) -> None:
            with _fsdp_summon_ctx(dflash_model):
                if _dist_rank() == 0:
                    _save_draft(out_path, draft_model, draft_config, args.draft_version)
                    _save_training_state(
                        out_path,
                        optimizer,
                        train_step=global_opt_step,
                        draft_version=args.draft_version,
                    )
                    if not args.no_copy_remote_code:
                        _copy_dflash_remote_code(out_path)
                    logger.info(
                        "[offline_sft] checkpoint opt_step=%d -> %s",
                        global_opt_step,
                        out_path,
                    )

        checkpoint_save_fn = _save_checkpoint
        _log_rank0(
            "[offline_sft] save_interval=%d (overwrite checkpoint -> %s)",
            args.save_interval,
            out_path,
        )

    _dist_barrier()
    t0 = time.perf_counter()
    loss, acc, _clip_ratio, _thresh_mae = _train_step(
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
        from_ground_truth=args.from_ground_truth,
        log_interval=args.log_interval,
        log_prefix="[offline_sft]",
        save_interval=args.save_interval,
        checkpoint_save_fn=checkpoint_save_fn,
        dp_group=get_dp_group(),
    )
    train_s = time.perf_counter() - t0

    if _dist_rank() == 0:
        logger.info(
            "[offline_sft] done train_s=%.2f loss=%.4f acc=%.4f",
            train_s,
            loss,
            acc,
        )

    _fsdp_save_draft(
        dflash_model,
        out_path,
        draft_model,
        draft_config,
        args.draft_version,
        optimizer=optimizer,
        train_step=1,
        draft_version=args.draft_version,
        copy_remote_code_fn=_copy_dflash_remote_code if not args.no_copy_remote_code else None,
        log_fn=lambda: logger.info(
            "[offline_sft] saved -> %s (version=%d)", out_path, args.draft_version
        ),
    )
    return out_path


def main() -> None:
    _setup_training_logging()

    parser = argparse.ArgumentParser(
        description="Offline one-shot SFT training (single async buffer round)."
    )
    add_model_args(parser)
    parser.set_defaults(
        draft_config_path=str(ASYN_TRAIN_ROOT / "configs" / "qwen3-8b-dflash.json"),
        attention_backend="flex_attention",
        num_anchors=512,
        loss_decay_gamma=7.0,
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="JSONL dataset (conversations, prompt+response, or question+response).",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--buffer-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=25,
        metavar="N",
        help="Log every N optimizer steps; 0 disables.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=0,
        metavar="N",
        help="Overwrite-save checkpoint to --output-dir every N optimizer steps; 0 disables.",
    )
    parser.add_argument("--chat-template", default="qwen")
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    parser.add_argument(
        "--from-ground-truth",
        action="store_true",
        default=False,
        help="Supervise on question+ground_truth (FROM_GROUND_TRUTH=1 in async pipeline).",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--draft-version", type=int, default=1)
    parser.add_argument("--no-copy-remote-code", action="store_true")
    parser.add_argument("--no-trust-remote-code", action="store_true")
    parser.add_argument(
        "--teacher-kd-temperature",
        type=float,
        default=None,
        metavar="T",
        help="Optional logits KD; omit for hard CE only (default, matches TEACHER_KD_DISABLE=1).",
    )
    parser.add_argument(
        "--teacher-kd-alpha",
        type=float,
        default=0.0,
        help="When --teacher-kd-temperature set: alpha*CE_hard + (1-alpha)*KD_soft.",
    )
    add_distributed_args(parser)
    SGLangBackendArgs.add_args(parser)
    args = parser.parse_args()

    if not args.from_ground_truth:
        gt_env = os.environ.get("FROM_GROUND_TRUTH", "").strip().lower()
        if gt_env in ("1", "true", "yes"):
            args.from_ground_truth = True

    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient-accumulation-steps must be >= 1")
    if args.log_interval < 0:
        parser.error("--log-interval must be >= 0")
    if args.save_interval < 0:
        parser.error("--save-interval must be >= 0")
    if args.buffer_epochs < 1:
        parser.error("--buffer-epochs must be >= 1")

    dist_inited = False
    try:
        if args.target_model_backend == "sglang":
            init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
            dist_inited = True
            _setup_training_logging()

        offline_train_sft_once(args)
    finally:
        if dist_inited:
            destroy_distributed()


if __name__ == "__main__":
    main()
