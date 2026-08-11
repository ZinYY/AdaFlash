# coding=utf-8
"""One-shot offline RKL + optional thresh-head training on a static JSONL dataset.

Matches a single buffer round of ``run_async_pipeline_adaflash.sh``:
load JSONL rows (``prompt`` + ``response``), train for ``--buffer-epochs`` full passes,
save draft weights to ``--output-dir``.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def _ensure_repo_pythonpath() -> None:
    from pipeline.bootstrap import ensure_vendored_specforge
    from pipeline.paths import REPO_ROOT

    ensure_vendored_specforge()
    sp = str(REPO_ROOT / "sglang_dflash/python")
    if Path(sp).is_dir() and sp not in sys.path:
        sys.path.insert(0, sp)


_ensure_repo_pythonpath()

from specforge.args import SGLangBackendArgs
from specforge.distributed import destroy_distributed, get_dp_group, init_distributed

from pipeline.paths import ASYN_TRAIN_ROOT, REPO_ROOT
from pipeline.training.args_common import add_distributed_args, add_model_args
from pipeline.training.rkl_model import OnlineDFlashModelReverseKL
from pipeline.training.setup import (
    build_rkl_optimizer,
    load_draft_stack,
    load_target_embeddings,
    load_target_model,
    log_rkl_thresh_heads,
)
from pipeline.training.utils import (
    _dist_barrier,
    _dist_rank,
    _fsdp_save_draft,
    _log_rank0,
    _save_draft,
    _save_training_state,
    _setup_training_logging,
    _train_step,
    _wrap_fsdp_if_distributed,
    load_jsonl_records,
)

logger = logging.getLogger(__name__)


def _copy_dflash_remote_code(output_dir: Path) -> None:
    """Copy ``dflash.py`` for ``trust-remote-code`` loading (same as offline thresh init)."""
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
            _log_rank0("[offline_rkl] copied remote code -> %s", dst)
            return


def _filter_jsonl_samples(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep rows compatible with async inference buffer / ``_TextDataset``."""
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        prompt = row.get("prompt")
        response = row.get("response")
        if not isinstance(prompt, str) or not prompt.strip():
            logger.warning("[offline_rkl] skip row %d: missing/empty prompt", i)
            continue
        if not isinstance(response, str) or not response.strip():
            logger.warning("[offline_rkl] skip row %d: missing/empty response", i)
            continue
        sample: Dict[str, Any] = {
            "prompt": prompt,
            "response": response,
        }
        if "question" in row:
            sample["question"] = row["question"]
        if "ground_truth" in row:
            sample["ground_truth"] = row["ground_truth"]
        out.append(sample)
    return out


def _prepare_offline_train_args(args: argparse.Namespace) -> None:
    """Fields expected by ``load_target_model`` / ``SGLangBackendArgs.from_args``."""
    args.use_draft_mask_token_id = True
    args.trust_remote_code = not getattr(args, "no_trust_remote_code", False)
    args.target_batch_size = args.batch_size
    if getattr(args, "sglang_context_length", None) is None:
        args.sglang_context_length = args.max_length
    if (
        hasattr(args, "rkl_div_clip_tau")
        and args.rkl_div_clip_tau is not None
        and args.rkl_div_clip_tau <= 0
    ):
        args.rkl_div_clip_tau = None


def offline_train_rkl_once(args: argparse.Namespace) -> Path:
    """Run one offline RKL (+ adaptive length head) training pass and save weights."""
    _prepare_offline_train_args(args)

    out_path = Path(args.output_dir).resolve()
    if _dist_rank() == 0:
        out_path.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl_records(Path(args.dataset_path))
    samples = _filter_jsonl_samples(rows)
    if args.max_samples is not None and args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        raise ValueError(f"No usable samples in {args.dataset_path}")

    _log_rank0(
        "[offline_rkl] dataset=%s usable_samples=%d buffer_epochs=%d batch_size=%d "
        "grad_accum=%d max_length=%d sglang_context_length=%s lazy_dataset=%s",
        args.dataset_path,
        len(samples),
        args.buffer_epochs,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.max_length,
        args.sglang_context_length,
        args.lazy_dataset,
    )

    target_model = load_target_model(args)
    draft_model, draft_config, tokenizer, _mask_token_id = load_draft_stack(args, target_model)
    target_components = load_target_embeddings(args)

    dflash_model = OnlineDFlashModelReverseKL(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        block_size=draft_model.block_size,
        mask_token_id=_mask_token_id,
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

    dflash_model = _wrap_fsdp_if_distributed(
        dflash_model, draft_model, tp_size=getattr(args, "tp_size", 1)
    )

    if args.rkl_div_clip_tau is not None:
        _log_rank0(
            "[offline_rkl] RKL mix: alpha=%.4f (CE) + (1-alpha)*KL(p_S||p_T), T=%s, div_clip_tau=%s",
            args.rkl_alpha,
            args.rkl_temperature,
            args.rkl_div_clip_tau,
        )
    else:
        _log_rank0(
            "[offline_rkl] RKL mix: alpha=%.4f (CE) + (1-alpha)*KL(p_S||p_T), T=%s",
            args.rkl_alpha,
            args.rkl_temperature,
        )

    optimizer, _draft_params, thresh_params = build_rkl_optimizer(args, draft_model)
    log_rkl_thresh_heads(args, draft_model)
    if args.train_thresh_head and thresh_params:
        _log_rank0(
            "[offline_rkl] thresh_head active | loss=%s lookahead=%d lr=%.2e",
            args.thresh_head_loss_type,
            args.thresh_label_lookahead,
            args.thresh_head_learning_rate or args.learning_rate,
        )

    checkpoint_save_fn = None
    if args.save_interval > 0:

        def _save_checkpoint(global_opt_step: int) -> None:
            _fsdp_save_draft(
                dflash_model,
                out_path,
                draft_model,
                draft_config,
                args.draft_version,
                optimizer=optimizer,
                train_step=global_opt_step,
                draft_version=args.draft_version,
                copy_remote_code_fn=_copy_dflash_remote_code if not args.no_copy_remote_code else None,
                log_fn=lambda: logger.info(
                    "[offline_rkl] checkpoint opt_step=%d -> %s",
                    global_opt_step,
                    out_path,
                ),
            )

        checkpoint_save_fn = _save_checkpoint
        _log_rank0(
            "[offline_rkl] save_interval=%d (overwrite checkpoint -> %s)",
            args.save_interval,
            out_path,
        )

    _dist_barrier()
    t0 = time.perf_counter()
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
        from_ground_truth=False,
        log_interval=args.log_interval,
        log_prefix="[offline_rkl]",
        save_interval=args.save_interval,
        checkpoint_save_fn=checkpoint_save_fn,
        lazy_dataset=args.lazy_dataset,
        dp_group=get_dp_group(),
    )
    train_s = time.perf_counter() - t0

    if _dist_rank() == 0:
        parts = [
            f"[offline_rkl] done train_s={train_s:.2f} loss={loss:.4f} acc={acc:.4f}",
        ]
        if clip_ratio is not None:
            parts.append(f" clip_ratio={clip_ratio:.6f}")
        if thresh_mae is not None:
            parts.append(f" thresh_mae={thresh_mae:.6f}")
        logger.info("".join(parts))

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
            "[offline_rkl] saved -> %s (version=%d)", out_path, args.draft_version
        ),
    )
    return out_path


def main() -> None:
    _setup_training_logging()

    parser = argparse.ArgumentParser(
        description="Offline one-shot RKL + thresh-head training (single async buffer round)."
    )
    add_model_args(parser)
    parser.set_defaults(
        draft_config_path=str(ASYN_TRAIN_ROOT / "configs" / "qwen3-8b-dflash-thresh-head.json"),
        attention_backend="flex_attention",
        num_anchors=512,
        loss_decay_gamma=7.0,
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="JSONL with prompt+response per line (e.g. regenerate_dataset output).",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for model.safetensors + config.")
    parser.add_argument(
        "--buffer-epochs",
        type=int,
        default=2,
        help="Full passes over the dataset (same as TRAIN_BUFFER_EPOCHS in async pipeline).",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=2,
        help="Micro-batches per optimizer.step (TRAIN_GRADIENT_ACCUMULATION_STEPS).",
    )
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
    parser.add_argument(
        "--lazy-dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tokenize samples on demand during training (default). "
        "Use --no-lazy-dataset to preprocess the full dataset up front.",
    )
    parser.add_argument("--chat-template", default="qwen")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Qwen3: pass enable_thinking to apply_chat_template.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--draft-version",
        type=int,
        default=1,
        help="Written to version.txt after save.",
    )
    parser.add_argument(
        "--no-copy-remote-code",
        action="store_true",
        help="Do not copy dflash.py into output-dir.",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disable trust_remote_code on HF loads.",
    )
    parser.add_argument(
        "--rkl-temperature",
        type=float,
        default=1.0,
        metavar="T",
    )
    parser.add_argument(
        "--rkl-alpha",
        type=float,
        default=0.8,
        metavar="A",
        help="loss = alpha*CE_hard + (1-alpha)*RKL.",
    )
    parser.add_argument(
        "--rkl-div-clip-tau",
        type=float,
        default=0.01,
        metavar="TAU",
        help="Per-token RKL cap; use 0 or negative to disable clip.",
    )
    parser.add_argument(
        "--train-thresh-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train adaptive length head (default: on, same as rkl_thresh_head pipeline).",
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
        default=2e-4,
        metavar="LR",
    )
    add_distributed_args(parser)
    SGLangBackendArgs.add_args(parser)
    args = parser.parse_args()

    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient-accumulation-steps must be >= 1")
    if args.log_interval < 0:
        parser.error("--log-interval must be >= 0")
    if args.save_interval < 0:
        parser.error("--save-interval must be >= 0")
    if args.buffer_epochs < 1:
        parser.error("--buffer-epochs must be >= 1")
    if args.rkl_temperature <= 0:
        parser.error("--rkl-temperature must be > 0")
    if not 0.0 <= args.rkl_alpha <= 1.0:
        parser.error("--rkl-alpha must be in [0, 1]")
    dist_inited = False
    try:
        if args.target_model_backend == "sglang":
            init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
            dist_inited = True
            _setup_training_logging()

        offline_train_rkl_once(args)
    finally:
        if dist_inited:
            destroy_distributed()


if __name__ == "__main__":
    main()
