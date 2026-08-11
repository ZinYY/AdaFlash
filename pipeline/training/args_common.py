# coding=utf-8
"""Shared argparse groups for SFT and RKL training workers."""

from __future__ import annotations

import argparse


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target-model-path", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--target-model-backend",
        default="sglang",
        choices=["hf", "sglang"],
        help="sglang target uses SGLangBackendArgs (--sglang-*).",
    )
    parser.add_argument(
        "--draft-config-path",
        required=True,
        help="Path to the DFlash draft config JSON.",
    )
    parser.add_argument(
        "--initial-draft-path",
        default=None,
        help="Optional: pre-trained draft weights to start from.",
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--attention-backend",
        default="flex_attention",
        choices=["eager", "sdpa", "flex_attention"],
    )
    parser.add_argument("--num-anchors", type=int, default=512)
    parser.add_argument("--loss-decay-gamma", type=float, default=7.0)
    parser.add_argument("--trust-remote-code", action="store_true")


def add_training_hyper_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--chat-template", default="qwen")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Qwen3: pass enable_thinking to apply_chat_template (match inference_worker).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=25,
        metavar="N",
        help="Log every N optimizer steps during training; 0 disables.",
    )


def add_swap_pipeline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--swap-dir", default="./swap")
    parser.add_argument(
        "--train-threshold",
        type=int,
        default=128,
        help="Trigger a training step when this many samples are buffered.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between data_buffer polls.",
    )
    parser.add_argument(
        "--buffer-epochs",
        type=int,
        default=1,
        help="Full passes over the current buffer batch before save.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Micro-batches per optimizer.step (loss scaled by 1/accum).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Stop after this many buffer rounds (save cycles); 0 = forever.",
    )
    parser.add_argument(
        "--draft-snapshot-interval",
        type=int,
        default=50,
        metavar="N",
        help="Copy model_weights to swap/<draft-models-subdir>/<v> when version%%N==0; 0=off.",
    )
    parser.add_argument(
        "--draft-models-subdir",
        type=str,
        default="draft_models",
        help="Subdirectory under --swap-dir for versioned weight snapshots.",
    )
    parser.add_argument(
        "--max-draft-version",
        type=int,
        default=0,
        metavar="V",
        help="Save draft v=V then stop; writes stop_pipeline before final version.txt. 0=no limit.",
    )


def add_distributed_args(parser: argparse.ArgumentParser) -> None:
    dist_group = parser.add_argument_group("distributed (sglang target)")
    dist_group.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size; WORLD_SIZE must be divisible by this.",
    )
    dist_group.add_argument(
        "--dist-timeout",
        type=int,
        default=30,
        help="NCCL collective timeout in minutes (init_distributed).",
    )


def prepare_sglang_target_train_args(args: argparse.Namespace) -> None:
    """Cap SGLang KV pool to the training batch (same as offline once scripts)."""
    args.target_batch_size = args.batch_size
    if getattr(args, "sglang_context_length", None) is None:
        args.sglang_context_length = args.max_length


def validate_common_training_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient-accumulation-steps must be >= 1")
    if getattr(args, "log_interval", 0) < 0:
        parser.error("--log-interval must be >= 0")
    if args.draft_snapshot_interval < 0:
        parser.error("--draft-snapshot-interval must be >= 0")
    if args.max_draft_version < 0:
        parser.error("--max-draft-version must be >= 0")
