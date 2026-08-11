#!/usr/bin/env python3
# coding=utf-8
"""Shared helpers for SFT / on-policy / once-off training scripts (logging, swap paths, buffer IO, dataset, train step, draft save)."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# Optimizer + buffer-round counter; lives under model_weights/ (SGLang ignores .pt).
TRAINING_STATE_FILENAME = "training_state.pt"
# Training touches this at max draft version; inference_worker exits when present.
# inference_worker also writes it after a single dataset pass when ``--loop`` is off,
# so training can exit the buffer wait instead of blocking forever.
STOP_PIPELINE_FILENAME = "stop_pipeline"


def stop_pipeline_marker_path(swap_dir: str | Path) -> Path:
    return Path(swap_dir) / STOP_PIPELINE_FILENAME


def clear_stop_pipeline_marker(swap_dir: str | Path) -> None:
    p = stop_pipeline_marker_path(swap_dir)
    try:
        p.unlink(missing_ok=True)
    except OSError as e:
        logger.warning("[training] could not remove stop marker %s: %s", p, e)


def write_stop_pipeline_marker(swap_dir: str | Path) -> None:
    p = stop_pipeline_marker_path(swap_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()


def stop_pipeline_marker_exists(swap_dir: str | Path) -> bool:
    """True if ``swap_dir/stop_pipeline`` exists (shutdown or inference single-pass done)."""
    return stop_pipeline_marker_path(swap_dir).exists()


def snapshot_model_weights_dir(weights_dir: Path, dest_parent: Path, version: int) -> None:
    """Copy the entire ``weights_dir`` tree to ``dest_parent / str(version)`` (full checkpoint)."""
    dest_parent.mkdir(parents=True, exist_ok=True)
    dest = dest_parent / str(version)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(weights_dir, dest, symlinks=False)
    logger.info("[training] Snapshot draft weights v%d -> %s", version, dest)


def _setup_training_logging() -> None:
    """Configure root logging so INFO lines appear under torchrun (root may exist already)."""
    fmt = "%(asctime)s %(levelname)s %(name)s - %(message)s"
    datefmt = "%H:%M:%S"
    kwargs = {"format": fmt, "datefmt": datefmt, "level": logging.INFO}
    if sys.version_info >= (3, 8):
        kwargs["force"] = True
    try:
        logging.basicConfig(**kwargs)
    except TypeError:
        root = logging.getLogger()
        for h in list(root.handlers):
            root.removeHandler(h)
        logging.basicConfig(format=fmt, datefmt=datefmt, level=logging.INFO)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in root.handlers:
        h.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)


def _dist_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _dist_barrier() -> None:
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()


def _log_rank0(msg: str, *args) -> None:
    if _dist_rank() == 0:
        logger.info(msg, *args)


def _summarize_consumed_files(consumed: set) -> str:
    names = sorted(p.name for p in consumed)
    if not names:
        return "[]"
    joined = ", ".join(names)
    if len(joined) <= 240:
        return f"[{joined}]"
    return f"[{joined[:237]}...] ({len(names)} files)"


def _training_state_path(weights_dir: Path) -> Path:
    return weights_dir / TRAINING_STATE_FILENAME


def _apply_optimizer_learning_rates(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    *,
    thresh_head_learning_rate: Optional[float] = None,
) -> None:
    """Override param group ``lr`` from CLI after loading a checkpoint.

    Groups with ``is_thresh_head: True`` use ``thresh_head_learning_rate`` when set,
    else ``learning_rate``. All other groups use ``learning_rate``.
    """
    for g in optimizer.param_groups:
        if g.get("is_thresh_head"):
            g["lr"] = float(
                thresh_head_learning_rate
                if thresh_head_learning_rate is not None
                else learning_rate
            )
        else:
            g["lr"] = float(learning_rate)


def _load_training_state(
    weights_dir: Path,
    optimizer: torch.optim.Optimizer,
    *,
    apply_train_step: bool,
    learning_rate: float,
    thresh_head_learning_rate: Optional[float] = None,
) -> Optional[Tuple[int, Optional[int]]]:
    """
    Load ``training_state.pt`` if present: restores AdamW state from file, then
    applies CLI learning rates (draft ``learning_rate``; optional adaptive length head group).
    """
    path = _training_state_path(weights_dir)
    if not path.is_file():
        return None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        logger.warning("[training] Could not read %s: %s", path, e)
        return None
    if not isinstance(ckpt, dict) or "optimizer" not in ckpt:
        logger.warning("[training] Invalid checkpoint (missing optimizer): %s", path)
        return None
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
    except Exception as e:
        logger.warning("[training] Optimizer state_dict mismatch, ignoring file: %s", e)
        return None
    _apply_optimizer_learning_rates(
        optimizer,
        learning_rate,
        thresh_head_learning_rate=thresh_head_learning_rate,
    )
    if apply_train_step:
        ts = int(ckpt.get("train_step", 0))
        dv = ckpt.get("draft_version")
        dv_i = int(dv) if dv is not None else None
        return ts, dv_i
    return None


def _save_training_state(
    weights_dir: Path,
    optimizer: torch.optim.Optimizer,
    train_step: int,
    draft_version: int,
) -> None:
    """Atomically write ``training_state.pt`` (AdamW moments + counters)."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    path = _training_state_path(weights_dir)
    tmp = weights_dir / ".training_state.pt.tmp"
    ckpt = {
        "optimizer": optimizer.state_dict(),
        "train_step": train_step,
        "draft_version": draft_version,
    }
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def _swap_paths(swap_dir: str) -> Tuple[Path, Path]:
    swap = Path(swap_dir)
    data_buffer = swap / "data_buffer"
    model_weights = swap / "model_weights"
    data_buffer.mkdir(parents=True, exist_ok=True)
    model_weights.mkdir(parents=True, exist_ok=True)
    return data_buffer, model_weights


def _read_version(weights_dir: Path) -> int:
    try:
        return int((weights_dir / "version.txt").read_text().strip())
    except Exception:
        return 0


def _write_version_atomic(weights_dir: Path, version: int) -> None:
    """Write version.txt via a temp file + rename for atomicity."""
    tmp = weights_dir / ".version.tmp"
    tmp.write_text(str(version))
    tmp.rename(weights_dir / "version.txt")


def load_jsonl_records(path: Path) -> list[dict]:
    """Load JSONL records one physical line at a time.

    Uses newline-only iteration (``for line in f``), not ``str.splitlines()``, so
    Unicode line/paragraph separators (U+2028 / U+2029) inside JSON strings are not
    treated as record boundaries.
    """
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _remove_malformed_buffer_file(path: Path) -> None:
    """Delete a corrupt buffer file so polling does not re-warn every round."""
    try:
        path.unlink(missing_ok=True)
    except OSError as e:
        logger.warning("[training] could not remove malformed buffer file %s: %s", path, e)
    else:
        logger.info("[training] removed malformed buffer file %s", path)


def _collect_samples(data_buffer: Path, threshold: int) -> Tuple[List[dict], List[Path]]:
    """Return up to `threshold` samples and the paths of their source files.

    Unparseable JSONL files are removed from ``data_buffer`` after a decode
    error so they are not retried indefinitely.
    """
    files = sorted(data_buffer.glob("*.jsonl"))
    samples, paths = [], []
    for f in files:
        try:
            with f.open(encoding="utf-8") as fp:
                for line in fp:
                    if line.strip():
                        samples.append(json.loads(line))
                        paths.append(f)
        except json.JSONDecodeError as e:
            logger.warning("Skipping malformed buffer file %s: %s", f, e)
            _remove_malformed_buffer_file(f)
        except UnicodeDecodeError as e:
            logger.warning("Skipping malformed buffer file %s: %s", f, e)
            _remove_malformed_buffer_file(f)
        except Exception as e:
            logger.warning("Skipping buffer file %s: %s", f, e)
        if len(samples) >= threshold:
            break
    return samples, paths


def _normalize_chat_messages(record: dict) -> Optional[List[Dict[str, str]]]:
    """Return ShareGPT-style messages from ``conversations`` or ``messages``."""
    for col in ("conversations", "messages"):
        conv = record.get(col)
        if not isinstance(conv, list) or not conv:
            continue
        messages: List[Dict[str, str]] = []
        for turn in conv:
            if not isinstance(turn, dict):
                continue
            role = turn.get("role")
            content = turn.get("content", "")
            if not isinstance(role, str) or not role.strip():
                continue
            if not isinstance(content, str):
                content = str(content) if content is not None else ""
            if not content.strip():
                continue
            messages.append({"role": role.strip(), "content": content})
        if messages:
            return messages
    return None


def _last_assistant_index(messages: List[Dict[str, str]]) -> Optional[int]:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            return i
    return None


def _supervised_chat_from_messages(
    tokenizer,
    messages: List[Dict[str, str]],
    *,
    enable_thinking: bool,
) -> Optional[Tuple[str, str]]:
    """
    Build full chat text and loss prefix for multi-turn ``conversations``.

    ``convo``: full history including the last assistant reply.
    ``split_prompt``: prefix before the last assistant answer (with generation prompt).
    Loss is applied only to tokens after ``split_prompt`` (the last assistant reply).
    """
    last_asst_idx = _last_assistant_index(messages)
    if last_asst_idx is None:
        return None

    convo = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    split_prompt = tokenizer.apply_chat_template(
        messages[:last_asst_idx],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return convo, split_prompt


def _preview_chat_messages(messages: List[Dict[str, str]], limit: int = 120) -> str:
    roles = "→".join(m["role"] for m in messages)
    last_user = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"),
        messages[-1]["content"],
    )
    text = f"[{roles}] {last_user}"
    return (text[: limit - 1] + "…") if len(text) > limit else text


def _loss_mask_for_supervised_chat(
    tokenizer,
    q: str,
    convo: str,
    split_prompt: str,
    attention_mask: torch.Tensor,
    offset_mapping: torch.Tensor | None,
) -> Tuple[torch.Tensor, Optional[str]]:
    """
    Mark positions where we want loss: assistant *answer* tokens only.

    Uses ``split_prompt``: everything before the supervised assistant reply.
    For single-turn data this is ``apply_chat_template([user], add_generation_prompt=True)``;
    for multi-turn ``conversations`` it is the prefix before the *last* assistant turn.
    Uses ``offset_mapping`` when available.

    For legacy fallback token counting, ``q`` is either the raw user string
    (single-turn reconstructed chat), the full saved inference ``prompt`` string
    (buffer rows written by inference_worker), or ``split_prompt`` for multi-turn
    conversations; all are only used when offsets are unavailable.

    Returns ``(loss_mask, None)`` on success, or ``(loss_mask, reason)`` when using
    the legacy q_len heuristic (caller should aggregate-log warnings).
    """
    attn_f = attention_mask.float()
    if (
        offset_mapping is not None
        and offset_mapping.dim() == 2
        and convo.startswith(split_prompt)
    ):
        split = len(split_prompt)
        loss_mask = torch.zeros_like(attn_f)
        for i in range(offset_mapping.size(0)):
            if attn_f[i] <= 0:
                continue
            s = int(offset_mapping[i, 0].item())
            e = int(offset_mapping[i, 1].item())
            if s == 0 and e == 0:
                continue
            if s >= split:
                loss_mask[i] = 1.0
        return loss_mask, None
    # Fallback: legacy heuristic (wrong under chat templates; kept if no offsets).
    parts = []
    if not convo.startswith(split_prompt):
        parts.append("convo does not start with add_generation_prompt(user) prefix")
    if offset_mapping is None:
        parts.append("no offset_mapping (need fast tokenizer / return_offsets_mapping)")
    elif offset_mapping.dim() != 2:
        parts.append(f"offset_mapping dim={offset_mapping.dim()} (expected 2)")
    reason = "; ".join(parts) if parts else "unknown"
    q_enc = tokenizer(q, add_special_tokens=False)
    q_len = len(q_enc["input_ids"])
    loss_mask = attn_f.clone()
    loss_mask[: q_len + 2] = 0
    return loss_mask, reason


def _dflash_supervised_in_anchor_prefix(
    loss_mask: torch.Tensor, seq_len: int, block_size: int
) -> int:
    """
    Count positions i with loss_mask[i] > 0.5 for i in [0, seq_len - block_size].
    Matches ``OnlineDFlashModel._sample_anchor_positions``; need count >= 2 to avoid
    ``ValueError: should preprocess the data.``
    """
    bs = block_size
    max_anchor = max(seq_len - bs, 0)
    return int((loss_mask[: max_anchor + 1] > 0.5).sum().item())


def _encode_text_record(
    r: dict,
    *,
    tokenizer,
    max_length: int,
    enable_thinking: bool,
    block_size: int,
    from_ground_truth: bool,
) -> Tuple[Optional[dict], Optional[str], Optional[str], bool]:
    """
    Tokenize one JSONL row on demand.

    Returns ``(item, fallback_reason, preview_src, skipped_anchor)``.
    ``item`` is None when the row should be skipped.
    """
    if from_ground_truth:
        q_raw = r.get("question")
        gt_raw = r.get("ground_truth")
        q = q_raw if isinstance(q_raw, str) else (str(q_raw) if q_raw is not None else "")
        if isinstance(gt_raw, str):
            a = gt_raw
        else:
            a = str(gt_raw) if gt_raw is not None else ""
        if not q.strip() or not a.strip():
            return None, None, None, False
        convo = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
        split_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        legacy_q_for_fallback = q
        preview_src = q
    else:
        saved_prompt = r.get("prompt")
        if isinstance(saved_prompt, str) and saved_prompt:
            split_prompt = saved_prompt
            a = r.get("response", "")
            if not isinstance(a, str):
                a = str(a) if a is not None else ""
            convo = split_prompt + a
            legacy_q_for_fallback = split_prompt
            preview_src = split_prompt
        else:
            messages = _normalize_chat_messages(r)
            if messages is not None:
                built = _supervised_chat_from_messages(
                    tokenizer,
                    messages,
                    enable_thinking=enable_thinking,
                )
                if built is None:
                    return None, None, None, False
                convo, split_prompt = built
                legacy_q_for_fallback = split_prompt
                preview_src = _preview_chat_messages(messages)
            else:
                q = r.get("question", "")
                if not isinstance(q, str):
                    q = str(q) if q is not None else ""
                a = r.get("response", "")
                if not isinstance(a, str):
                    a = str(a) if a is not None else ""
                if not q.strip():
                    return None, None, None, False
                convo = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=enable_thinking,
                )
                split_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
                legacy_q_for_fallback = q
                preview_src = q

    enc_kw: dict = dict(
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    if convo.startswith(split_prompt):
        enc_kw["return_offsets_mapping"] = True
    try:
        enc = tokenizer(convo, **enc_kw)
    except (TypeError, ValueError):
        enc_kw.pop("return_offsets_mapping", None)
        enc = tokenizer(convo, **enc_kw)

    input_ids = enc["input_ids"].squeeze(0)
    attention_mask = enc["attention_mask"].squeeze(0)
    om = enc["offset_mapping"].squeeze(0) if "offset_mapping" in enc else None
    loss_mask, fb_reason = _loss_mask_for_supervised_chat(
        tokenizer, legacy_q_for_fallback, convo, split_prompt, attention_mask, om
    )
    seq_len = int(loss_mask.shape[0])
    n_anchor_sup = _dflash_supervised_in_anchor_prefix(loss_mask, seq_len, block_size)
    if n_anchor_sup < 2:
        preview = (preview_src[:120] + "…") if preview_src and len(preview_src) > 120 else preview_src
        return None, None, preview, True

    item = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
    }
    return item, fb_reason, preview_src, False


def _text_dataset_collate_fn(batch: List[Optional[dict]]) -> Optional[dict]:
    """Drop skipped rows (``None``) before default collate."""
    kept = [row for row in batch if row is not None]
    if not kept:
        return None
    return torch.utils.data.dataloader.default_collate(kept)


class _TextDataset(Dataset):
    def __init__(
        self,
        records: List[dict],
        tokenizer,
        max_length: int,
        chat_template: str,
        enable_thinking: bool,
        block_size: int,
        from_ground_truth: bool = False,
        lazy: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chat_template = chat_template
        self.enable_thinking = enable_thinking
        self.block_size = block_size
        self.from_ground_truth = from_ground_truth
        self.lazy = lazy
        self.records = records
        self.items: List[dict] = []
        self._cache: Dict[int, Optional[dict]] = {}
        self.n_fallback = 0
        self.first_fallback_reason: Optional[str] = None
        self.first_q_preview: Optional[str] = None
        self.n_skipped_anchor = 0
        self.first_anchor_skip_preview: Optional[str] = None

        if not lazy:
            for r in records:
                item, fb_reason, preview_src, skipped_anchor = _encode_text_record(
                    r,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    enable_thinking=enable_thinking,
                    block_size=block_size,
                    from_ground_truth=from_ground_truth,
                )
                self._note_encode_stats(
                    item,
                    fb_reason=fb_reason,
                    preview_src=preview_src,
                    skipped_anchor=skipped_anchor,
                )
                if item is not None:
                    self.items.append(item)
            self._log_encode_stats(len(records))

    def _note_encode_stats(
        self,
        item: Optional[dict],
        *,
        fb_reason: Optional[str],
        preview_src: Optional[str],
        skipped_anchor: bool,
    ) -> None:
        if item is None:
            if skipped_anchor:
                self.n_skipped_anchor += 1
                if self.first_anchor_skip_preview is None and preview_src is not None:
                    self.first_anchor_skip_preview = (
                        (preview_src[:120] + "…") if len(preview_src) > 120 else preview_src
                    )
            return
        if fb_reason is not None:
            self.n_fallback += 1
            if self.first_fallback_reason is None:
                self.first_fallback_reason = fb_reason
                if preview_src is not None:
                    self.first_q_preview = (
                        (preview_src[:120] + "…") if len(preview_src) > 120 else preview_src
                    )

    def _log_encode_stats(self, n_records: int) -> None:
        if self.n_skipped_anchor:
            logger.warning(
                "[training] skipped %d/%d samples (need >=2 loss positions in "
                "i<=seq_len-block_size for DFlash anchors; seq_len=%d block_size=%d; "
                "first preview: %r)",
                self.n_skipped_anchor,
                n_records,
                self.max_length,
                self.block_size,
                self.first_anchor_skip_preview,
            )
        if self.n_fallback:
            logger.warning(
                "[training] loss_mask used legacy q_len heuristic for %d/%d samples "
                "(first reason: %s; first q preview: %r). "
                "Prefer offset_mapping + prefix alignment for correct assistant masking.",
                self.n_fallback,
                n_records,
                self.first_fallback_reason,
                self.first_q_preview,
            )

    def __len__(self):
        return len(self.records) if self.lazy else len(self.items)

    def __getitem__(self, idx):
        if not self.lazy:
            return self.items[idx]
        if idx in self._cache:
            return self._cache[idx]
        item, fb_reason, preview_src, skipped_anchor = _encode_text_record(
            self.records[idx],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            enable_thinking=self.enable_thinking,
            block_size=self.block_size,
            from_ground_truth=self.from_ground_truth,
        )
        self._note_encode_stats(
            item,
            fb_reason=fb_reason,
            preview_src=preview_src,
            skipped_anchor=skipped_anchor,
        )
        self._cache[idx] = item
        return item


def _wrap_fsdp_if_distributed(dflash_model, draft_model, tp_size: int = 1):
    """Wrap ``dflash_model`` with FSDP when running under torchrun (world_size > 1).

    Returns the (possibly wrapped) ``dflash_model``.  The caller should replace its
    local ``dflash_model`` reference with the return value and keep ``draft_model``
    unchanged – FSDP ``use_orig_params=True`` allows the AdamW optimizer to
    reference the original parameter objects inside ``draft_model`` directly.

    ``ShardingStrategy.SHARD_GRAD_OP``: full parameters on every rank at all
    times (no all-gather overhead in forward), gradients reduced and sharded
    after backward.  This guarantees consistent weight updates regardless of the
    tp/dp split:

    * tp=N, dp=1  – FSDP averages the N identical gradient copies → correct,
                    though draft forward is still computed N times.
    * tp=1, dp=N  – FSDP averages N independent gradient shards → true data
                    parallelism with 1/N data per rank.
    * tp=K, dp=M  – mixed: each of the M dp replicas computes forward on the
                    same K-card TP batch; FSDP merges gradients across all K×M
                    ranks.
    """
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return dflash_model

    import functools

    from torch.distributed.fsdp import (
        BackwardPrefetch,
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    fsdp_kwargs = dict(
        use_orig_params=True,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )
    block_names = set(getattr(draft_model, "_no_split_modules", None) or [])
    block_classes = {
        type(m) for m in dflash_model.modules()
        if type(m).__name__ in block_names
    }
    if block_classes:
        fsdp_kwargs["auto_wrap_policy"] = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=block_classes,
        )

    wrapped = FSDP(dflash_model, **fsdp_kwargs)
    world = dist.get_world_size()
    dp = world // max(tp_size, 1)
    _log_rank0(
        "[training] FSDP initialized: world=%d tp=%d dp=%d (SHARD_GRAD_OP, use_orig_params=True)",
        world,
        tp_size,
        dp,
    )
    return wrapped


from contextlib import contextmanager


@contextmanager
def _fsdp_summon_ctx(model):
    """Collective context that gathers full FSDP params on all ranks before saving.

    With ``SHARD_GRAD_OP`` each rank only updates its own 1/N parameter slice
    after ``optimizer.step()``.  Accessing parameters for saving outside of a
    forward pass triggers an implicit FSDP all-gather – but only on the rank that
    calls ``save_pretrained``.  The other ranks are left waiting at the barrier,
    causing a 30-minute NCCL timeout and crash.

    ``FSDP.summon_full_params`` is the explicit collective that *all* ranks must
    enter together; within it every rank has the complete, consistent parameter
    values so rank 0 can safely write them to disk.

    For non-FSDP models the context is a plain no-op so call sites are uniform.
    """
    _is_fsdp = False
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        _is_fsdp = isinstance(model, FSDP)
    except ImportError:
        pass

    if _is_fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(model, writeback=False):
            yield
    else:
        yield


def _fsdp_save_draft(
    dflash_model,
    weights_dir: Path,
    draft_model,
    draft_config,
    version: int,
    *,
    optimizer=None,
    train_step: int | None = None,
    draft_version: int | None = None,
    copy_remote_code_fn=None,
    log_fn=None,
) -> None:
    """FSDP-safe checkpoint write: all ranks enter/leave ``summon_full_params`` together.

    Callers must not perform rank-0-only work *between* a prior barrier and this
    function, or non-zero ranks will enter the FSDP collective early and hang.
    """
    with _fsdp_summon_ctx(dflash_model):
        if _dist_rank() == 0:
            _save_draft(weights_dir, draft_model, draft_config, version)
            if optimizer is not None and train_step is not None:
                _save_training_state(
                    weights_dir,
                    optimizer,
                    train_step=train_step,
                    draft_version=draft_version if draft_version is not None else version,
                )
            if copy_remote_code_fn is not None:
                copy_remote_code_fn(weights_dir)
            if log_fn is not None:
                log_fn()
    _dist_barrier()


def _clip_grad_norm(dflash_model, draft_model, max_norm: float) -> None:
    """Clip gradients correctly whether ``dflash_model`` is FSDP-wrapped or plain.

    With FSDP the shard-local gradient norms must be reduced across ranks before
    clipping; ``FullyShardedDataParallel.clip_grad_norm_`` does this.  For a plain
    (non-FSDP) model, fall back to the standard ``torch.nn.utils`` helper operating
    on ``draft_model``'s original parameters.
    """
    if hasattr(dflash_model, "clip_grad_norm_"):
        dflash_model.clip_grad_norm_(max_norm)
    else:
        torch.nn.utils.clip_grad_norm_(draft_model.parameters(), max_norm)


def _maybe_save_checkpoint(
    global_opt_step: int,
    *,
    save_interval: int,
    save_fn: Optional[Callable[[int], None]],
) -> None:
    """Overwrite-save checkpoint every ``save_interval`` optimizer steps.

    ``save_fn`` is called on **all ranks** so that FSDP-wrapped models can use
    ``_fsdp_summon_ctx`` inside the closure to gather full params collectively.
    File writes should be gated on ``_dist_rank() == 0`` inside ``save_fn``.
    """
    if save_interval <= 0 or save_fn is None:
        return
    if global_opt_step % save_interval != 0:
        return
    save_fn(global_opt_step)   # all ranks participate (FSDP needs collective)
    _dist_barrier()


def _train_step(
    samples: List[dict],
    draft_model,  # DFlashDraftModel
    target_model,  # DFlashTargetModel
    dflash_model,  # OnlineDFlashModel (wraps both, may be FSDP-wrapped)
    optimizer: torch.optim.Optimizer,
    tokenizer,
    max_length: int,
    batch_size: int,
    chat_template: str,
    enable_thinking: bool,
    buffer_epochs: int,
    gradient_accumulation_steps: int = 1,
    from_ground_truth: bool = False,
    log_interval: int = 0,
    log_prefix: str = "[training]",
    save_interval: int = 0,
    checkpoint_save_fn: Optional[Callable[[int], None]] = None,
    lazy_dataset: bool = True,
    dp_group=None,  # dist.ProcessGroup for data-parallel shard; None = no sharding
) -> Tuple[float, float, Optional[float], Optional[float]]:
    """Train on the current buffer batch for ``buffer_epochs`` full passes (epochs).

    Within each pass: ``(loss / accum).backward()`` with ``accum = max(1, gradient_accumulation_steps)``;
    ``optimizer.step()`` every ``accum`` micro-batches (same pattern as ``training_worker_on_policy``).
    Default ``accum=1`` => one step per DataLoader batch (previous behavior).

    ``from_ground_truth``: if True, each JSONL row uses ``question`` + ``ground_truth`` to build
    the supervised chat (see ``_TextDataset``); otherwise the usual ``prompt``/``conversations``/flat paths apply.

    ``save_interval``: if > 0, call ``checkpoint_save_fn(global_opt_step)`` on rank 0 every N
    optimizer steps (overwrite checkpoint; 0 disables).

    ``lazy_dataset``: if True (default), tokenize each sample in ``__getitem__`` instead of
    preprocessing the full buffer up front.

    ``dp_group``: when provided, each epoch uses ``DistributedSampler`` scoped to this group so
    that each data-parallel replica sees a disjoint shard of ``samples``.  All ranks within the
    same TP group share the same dp-group rank and therefore receive the same batch, which is
    required for tensor-parallel target inference.  When ``None`` (single-GPU or not yet
    initialised), a plain shuffled DataLoader is used unchanged.

    Returns ``(mean_loss, mean_acc, rkl_clip_ratio, thresh_mae)``. The third entry is ``None`` unless
    ``dflash_model`` implements ``reset_rkl_clip_stats`` / ``consume_rkl_clip_ratio``
    (``OnlineDFlashModelReverseKL`` with ``--rkl-div-clip-tau`` set).
    The fourth entry is ``None`` unless adaptive length head training is enabled.
    """
    draft_model.train()
    reset_clip = getattr(dflash_model, "reset_rkl_clip_stats", None)
    if callable(reset_clip):
        reset_clip()
    accum = max(1, int(gradient_accumulation_steps))
    log_iv = max(0, int(log_interval))
    sum_loss, sum_acc, sum_thresh_mae, denom = 0.0, 0.0, 0.0, 0
    global_opt_step = 0

    def _maybe_log_opt_step(
        *,
        epoch_idx: int,
        micro_idx: int,
        batch_loss: float,
        batch_acc: float,
        batch_thresh_mae: Optional[float],
        ep_loss_sum: float,
        ep_acc_sum: float,
    ) -> None:
        if log_iv <= 0 or _dist_rank() != 0:
            return
        if global_opt_step % log_iv != 0:
            return
        parts = [
            f"{log_prefix} opt_step={global_opt_step} epoch={epoch_idx + 1}/{buffer_epochs} ",
            f"micro={micro_idx} loss={batch_loss:.4f} acc={batch_acc:.4f} ",
            f"ep_avg_loss={ep_loss_sum / max(micro_idx, 1):.4f} ",
            f"ep_avg_acc={ep_acc_sum / max(micro_idx, 1):.4f}",
        ]
        if batch_thresh_mae is not None:
            parts.append(f" thresh_mae={batch_thresh_mae:.6f}")
        _log_rank0("".join(parts))

    for _ep in range(buffer_epochs):
        if lazy_dataset and _ep == 0:
            shared_dataset = _TextDataset(
                samples,
                tokenizer,
                max_length,
                chat_template,
                enable_thinking,
                block_size=draft_model.block_size,
                from_ground_truth=from_ground_truth,
                lazy=True,
            )
        dataset = shared_dataset if lazy_dataset else _TextDataset(
            samples,
            tokenizer,
            max_length,
            chat_template,
            enable_thinking,
            block_size=draft_model.block_size,
            from_ground_truth=from_ground_truth,
            lazy=False,
        )
        loader_kwargs: dict = {
            "batch_size": batch_size,
            "drop_last": False,
        }
        if lazy_dataset:
            loader_kwargs["collate_fn"] = _text_dataset_collate_fn

        # When a data-parallel process group is supplied, shard the dataset so
        # that each dp replica sees a disjoint subset.  All ranks belonging to
        # the same TP group share the same dp-group rank and therefore iterate
        # the same sample ordering, satisfying the TP constraint.
        if dp_group is not None and dist.is_initialized():
            from torch.utils.data import DistributedSampler
            dp_size = dist.get_world_size(dp_group)
            dp_rank = dist.get_rank(dp_group)
            if dp_size > 1:
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=dp_size,
                    rank=dp_rank,
                    shuffle=True,
                    drop_last=True,
                )
                sampler.set_epoch(_ep)
                loader_kwargs["sampler"] = sampler
                loader_kwargs["drop_last"] = True
                _log_rank0(
                    "%s epoch %d/%d: DistributedSampler dp_size=%d dp_rank=%d "
                    "total_samples=%d shard_samples=%d",
                    log_prefix,
                    _ep + 1,
                    buffer_epochs,
                    dp_size,
                    dp_rank,
                    len(dataset),
                    len(sampler),
                )
            else:
                loader_kwargs["shuffle"] = True
        else:
            loader_kwargs["shuffle"] = True

        loader = DataLoader(dataset, **loader_kwargs)
        optimizer.zero_grad()
        ep_loss, ep_acc, n_batch = 0.0, 0.0, 0
        micro_in_window = 0

        for batch in loader:
            if batch is None:
                continue
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            loss_mask = batch["loss_mask"].cuda()

            target_out = target_model.generate_dflash_data(
                input_ids, attention_mask, loss_mask
            )
            hidden_states = target_out.hidden_states.cuda()
            teacher_h = target_out.teacher_hidden_states
            if teacher_h is not None:
                teacher_h = teacher_h.cuda()

            loss, acc, *extra = dflash_model(
                input_ids=input_ids,
                hidden_states=hidden_states,
                loss_mask=loss_mask,
                teacher_hidden_states=teacher_h,
            )
            (loss / accum).backward()
            batch_loss = loss.item()
            batch_acc = acc.item()
            batch_thresh_mae: Optional[float] = None
            ep_loss += batch_loss
            ep_acc += batch_acc
            # Extract thresh_mae if available (3rd return value)
            if len(extra) >= 2 and extra[1] is not None:
                batch_thresh_mae = (
                    extra[1].item() if hasattr(extra[1], "item") else float(extra[1])
                )
                sum_thresh_mae += batch_thresh_mae
            n_batch += 1
            micro_in_window += 1
            if micro_in_window >= accum:
                _clip_grad_norm(dflash_model, draft_model, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                micro_in_window = 0
                global_opt_step += 1
                _maybe_log_opt_step(
                    epoch_idx=_ep,
                    micro_idx=n_batch,
                    batch_loss=batch_loss,
                    batch_acc=batch_acc,
                    batch_thresh_mae=batch_thresh_mae,
                    ep_loss_sum=ep_loss,
                    ep_acc_sum=ep_acc,
                )
                _maybe_save_checkpoint(
                    global_opt_step,
                    save_interval=save_interval,
                    save_fn=checkpoint_save_fn,
                )

        if micro_in_window > 0:
            _clip_grad_norm(dflash_model, draft_model, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_opt_step += 1
            _maybe_log_opt_step(
                epoch_idx=_ep,
                micro_idx=n_batch,
                batch_loss=batch_loss,
                batch_acc=batch_acc,
                batch_thresh_mae=batch_thresh_mae,
                ep_loss_sum=ep_loss,
                ep_acc_sum=ep_acc,
            )
            _maybe_save_checkpoint(
                global_opt_step,
                save_interval=save_interval,
                save_fn=checkpoint_save_fn,
            )

        if lazy_dataset and _ep == buffer_epochs - 1:
            dataset._log_encode_stats(len(samples))

        if n_batch == 0:
            logger.warning(
                "[training] no batches after DFlash anchor filter (epoch pass empty); "
                "skipping optimizer.step for this buffer_epochs slice"
            )

        sum_loss += ep_loss / max(n_batch, 1)
        sum_acc += ep_acc / max(n_batch, 1)
        denom += 1
        if log_iv > 0:
            _log_rank0(
                "%s epoch %d/%d done | micro_batches=%d opt_steps=%d | "
                "loss_avg=%.4f acc_avg=%.4f",
                log_prefix,
                _ep + 1,
                buffer_epochs,
                n_batch,
                global_opt_step,
                ep_loss / max(n_batch, 1),
                ep_acc / max(n_batch, 1),
            )

    clip_ratio: Optional[float] = None
    consume_clip = getattr(dflash_model, "consume_rkl_clip_ratio", None)
    if callable(consume_clip):
        clip_ratio = consume_clip()

    thresh_mae = sum_thresh_mae / max(denom, 1) if sum_thresh_mae > 0 else None
    return sum_loss / max(denom, 1), sum_acc / max(denom, 1), clip_ratio, thresh_mae


def _save_draft(weights_dir: Path, draft_model, draft_config, version: int) -> None:
    """
    Save draft model weights + config to weights_dir.
    version.txt is written LAST to signal completion to the inference worker.
    """
    # Save to a temp dir first, then atomically move.
    tmp_dir = weights_dir.parent / ".model_weights_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    draft_model.save_pretrained(str(tmp_dir))

    # Copy any existing non-weight files (e.g. dflash.py) if present.
    skip_names = frozenset({"version.txt", TRAINING_STATE_FILENAME})
    for f in weights_dir.iterdir():
        if f.name in skip_names or (tmp_dir / f.name).exists():
            continue
        shutil.copy2(f, tmp_dir / f.name)

    # Swap: remove old, rename tmp -> weights_dir contents.
    for f in weights_dir.iterdir():
        if f.name != "version.txt":
            f.unlink() if f.is_file() else shutil.rmtree(f)
    for f in tmp_dir.iterdir():
        shutil.move(str(f), str(weights_dir / f.name))
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Write version.txt last.
    _write_version_atomic(weights_dir, version)
    logger.info("[training] Saved draft v%d -> %s", version, weights_dir)
