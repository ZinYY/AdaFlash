# coding=utf-8
"""Load target/draft models and optimizers for async training workers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import AutoConfig, AutoTokenizer

from specforge.args import SGLangBackendArgs
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.target.dflash_target_model import get_dflash_target_model
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead

from pipeline.training.utils import _log_rank0

logger = logging.getLogger(__name__)


def load_target_model(args) -> object:
    logger.info("[training] Loading target model (%s)...", args.target_model_backend)
    target_kw = {}
    if args.target_model_backend == "sglang":
        target_kw = SGLangBackendArgs.from_args(args).to_kwargs()
    return get_dflash_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=torch.bfloat16,
        device="cuda" if args.target_model_backend == "hf" else None,
        trust_remote_code=args.trust_remote_code,
        **target_kw,
    )


def load_draft_stack(args, target_model) -> Tuple[DFlashDraftModel, object, AutoTokenizer, int]:
    draft_config = AutoConfig.from_pretrained(args.draft_config_path)
    if not hasattr(draft_config, "dflash_config") or draft_config.dflash_config is None:
        draft_config.dflash_config = {}
    draft_config._attn_implementation = args.attention_backend

    draft_model = DFlashDraftModel(draft_config).cuda().to(torch.bfloat16)

    if args.initial_draft_path:
        load_id = (
            str(Path(args.initial_draft_path).resolve())
            if Path(args.initial_draft_path).is_dir()
            else args.initial_draft_path
        )
        # Architecture from local draft_config; checkpoint supplies weights only.
        loaded = DFlashDraftModel.from_pretrained(
            load_id,
            config=draft_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=getattr(args, "trust_remote_code", True),
        )
        incomp = draft_model.load_state_dict(loaded.state_dict(), strict=False)
        del loaded
        if incomp.missing_keys:
            logger.warning(
                "[training] draft load missing keys (%d): %s",
                len(incomp.missing_keys),
                incomp.missing_keys[:12],
            )
        if incomp.unexpected_keys:
            logger.info(
                "[training] draft load ignored keys (%d): %s",
                len(incomp.unexpected_keys),
                incomp.unexpected_keys[:12],
            )
        logger.info("[training] Loaded initial draft from %s", load_id)

    target_model.set_capture_layers(draft_model.target_layer_ids)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)

    if (
        getattr(args, "use_draft_mask_token_id", False)
        and getattr(draft_config, "dflash_config", None)
        and draft_config.dflash_config.get("mask_token_id") is not None
    ):
        mask_token_id = int(draft_config.dflash_config["mask_token_id"])
    else:
        if tokenizer.mask_token_id is None:
            tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        mask_token_id = tokenizer.mask_token_id
        draft_config.dflash_config["mask_token_id"] = mask_token_id

    draft_model.mask_token_id = mask_token_id
    draft_config.dflash_config["target_layer_ids"] = draft_model.target_layer_ids
    return draft_model, draft_config, tokenizer, mask_token_id


def load_target_embeddings(args) -> TargetEmbeddingsAndHead:
    return TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path,
        device="cuda",
        trust_remote_code=args.trust_remote_code,
    )


def build_sft_optimizer(args, draft_model: DFlashDraftModel) -> torch.optim.Optimizer:
    return torch.optim.AdamW(draft_model.parameters(), lr=args.learning_rate)


def build_rkl_optimizer(
    args,
    draft_model: DFlashDraftModel,
) -> Tuple[torch.optim.Optimizer, List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    draft_params = [
        p for n, p in draft_model.named_parameters() if not n.startswith("thresh_head")
    ]
    thresh_params = [
        p for n, p in draft_model.named_parameters() if n.startswith("thresh_head")
    ]
    lr_thresh = (
        args.thresh_head_learning_rate
        if args.thresh_head_learning_rate is not None
        else args.learning_rate
    )
    if args.train_thresh_head and thresh_params:
        optimizer = torch.optim.AdamW(
            [
                {"params": draft_params, "lr": args.learning_rate},
                {"params": thresh_params, "lr": lr_thresh, "is_thresh_head": True},
            ]
        )
        _log_rank0(
            "[training] AdamW (shared): draft params lr=%.2e | thresh_head* lr=%.2e",
            args.learning_rate,
            lr_thresh,
        )
    else:
        optimizer = torch.optim.AdamW(draft_model.parameters(), lr=args.learning_rate)
    return optimizer, draft_params, thresh_params


def log_rkl_thresh_heads(args, draft_model: DFlashDraftModel) -> None:
    if not args.train_thresh_head:
        return
    if hasattr(draft_model, "thresh_head_two_model") and draft_model.thresh_head_two_model is not None:
        _log_rank0("[training] adaptive length head (two_model) will be trained")
    elif hasattr(draft_model, "thresh_head_subsequent") and draft_model.thresh_head_subsequent is not None:
        _log_rank0("[training] adaptive length head (subsequent) will be trained")
    else:
        _log_rank0("[training] WARNING: --train-thresh-head set but no adaptive length head found in model")
