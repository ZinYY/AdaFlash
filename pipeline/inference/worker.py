#!/usr/bin/env python3
# coding=utf-8
"""
Inference worker for the async training-inference pipeline.

Responsibilities:
  - Load base model + DFLASH draft via SGLang Engine. Use ``CUDA_VISIBLE_DEVICES`` /
    ``--tp-size`` in the shell or CLI to control which GPU(s) are visible (default
    ``--tp-size`` is 1).
  - Iterate over a question dataset and run inference.
  - Save each sample as JSONL into swap/data_buffer/: ``prompt`` is the exact
    ``apply_chat_template`` string passed to the engine; ``response`` is the
    completion plus ``<|im_end|>`` when the run did not stop on
    ``max_new_tokens`` (SGLang ``finish_reason.type == length``). Training can
    concatenate ``prompt`` + ``response`` for loss masking aligned with inference.
    Optional ``question`` keeps the raw user text for debugging.
    When the dataset row carries a gold answer (e.g. ShareGPT-style ``conversations``
    assistant turn, or flat ``response`` / ``answer`` / ``output``), the saved row may
    also include ``ground_truth`` (verbatim reference text; not model output).
    For ``conversations`` / ``messages``, the prompt is built from the full history
    before the last assistant turn (same prefix as training loss masking).
  - Each sample: check ``swap/stop_pipeline`` first, then ``version.txt`` / hot-swap
    draft weights (so training can write ``stop_pipeline`` before the final
    ``version.txt`` bump and inference never loads that last draft).
  - Optionally append per-sample timing/token stats to a JSONL file for plotting.
    Each row's ``draft_version`` is 0 before the first hot-swap from ``swap/model_weights/``,
    then matches ``version.txt`` (1 after first training save, etc.).
  - Optional ``--dataset-start-line`` (1-based) to skip the first N-1 JSONL rows.
  - Without ``--loop``: after one full dataset pass, writes ``swap/stop_pipeline`` so the
    training worker can exit instead of waiting forever for a buffer that will not grow.

Requires sglang installed from this repo (e.g. pip install -e); no PYTHONPATH hack.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sglang as sgl
from transformers import AutoTokenizer

from pipeline.training.utils import (
    STOP_PIPELINE_FILENAME,
    write_stop_pipeline_marker,
    _last_assistant_index,
    _normalize_chat_messages,
    _preview_chat_messages,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_qwen35_model_path(model_path: str) -> bool:
    """True for HF ids like ``Qwen/Qwen3.5-9B`` and local dirs containing ``qwen3.5`` (any case)."""
    return "qwen3.5" in (model_path or "").lower()


def qwen35_dflash_engine_compat_extras(
    model_path: str,
    *,
    speculative: bool,
    mamba_scheduler_strategy: str = "extra_buffer",
) -> Dict[str, Any]:
    """Return ``sgl.Engine`` kwargs and env defaults only for Qwen3.5 + speculative DFLASH.

    ``Qwen3_5ForConditionalGeneration`` + radix + default ``no_buffer`` is rejected by
    ``ServerArgs._handle_mamba_radix_cache``; SGLang expects ``mamba_scheduler_strategy=extra_buffer``
    and ``SGLANG_ENABLE_SPEC_V2=1``. Pure Qwen3 (e.g. ``Qwen3-8B``) does not need this path.
    """
    if not speculative or not _is_qwen35_model_path(model_path):
        return {}
    os.environ.setdefault("SGLANG_DISABLE_CUDNN_CHECK", "1")
    os.environ.setdefault("SGLANG_ENABLE_SPEC_V2", "1")
    return {"mamba_scheduler_strategy": mamba_scheduler_strategy}


def _read_version(version_file: Path) -> int:
    """Return the int in ``version.txt``, or -1 if missing/unreadable (internal sentinel only).

    The main loop uses ``current_draft_version=0`` for the initial speculative draft
    (``--initial-draft-path``) before any hot-swap; that aligns with ``version.txt``
    starting at 1 after the first training save (init = first_saved - 1).
    """
    try:
        return int(version_file.read_text().strip())
    except Exception:
        return -1


def _maybe_update_draft(engine, weights_dir: Path, current_version: int) -> int:
    """
    Check version.txt under weights_dir; if newer, call update_weights_from_disk
    targeting only the draft model.  Returns the (possibly updated) version.
    """
    version_file = weights_dir / "version.txt"
    latest = _read_version(version_file)
    if latest <= current_version:
        return current_version

    logger.info(
        "[inference] New draft version %d detected (current=%d). Updating draft weights...",
        latest,
        current_version,
    )
    t0 = time.perf_counter()
    ret = engine.update_weights_from_disk(
        str(weights_dir), update_speculative_draft=True
    )
    ok = ret[0] if isinstance(ret, (list, tuple)) else ret
    elapsed = time.perf_counter() - t0
    if ok:
        logger.info(
            "[inference] Draft weights updated to version %d in %.2fs.", latest, elapsed
        )
        return latest
    else:
        detail = ret[1] if isinstance(ret, (list, tuple)) and len(ret) > 1 else ret
        logger.warning(
            "[inference] Draft weight update failed (version=%d): %s", latest, detail
        )
        return current_version


def _append_inference_stats(path: Path, row: dict) -> None:
    """One JSON object per line; append + flush for crash-safe incremental logs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as fp:
        fp.write(line)
        fp.flush()


def _completion_text(out: Dict[str, Any]) -> str:
    t = out.get("text", "")
    if isinstance(t, list):
        return "".join(str(x) for x in t)
    return str(t)


def _meta_int(meta: Dict[str, Any], key: str) -> int | None:
    if key not in meta:
        return None
    try:
        return int(meta[key])
    except (TypeError, ValueError):
        return None


def _meta_float(meta: Dict[str, Any], key: str) -> float | None:
    if key not in meta:
        return None
    try:
        return float(meta[key])
    except (TypeError, ValueError):
        return None


def _token_counts(
    raw_out: Dict[str, Any], prompt: str, reply: str, tokenizer
) -> Tuple[int, int]:
    """(prompt_tokens, completion_tokens); fall back to local tokenize if meta missing."""
    meta = raw_out.get("meta_info") if isinstance(raw_out, dict) else None
    meta = meta if isinstance(meta, dict) else {}
    n_in = _meta_int(meta, "prompt_tokens")
    if n_in is None:
        n_in = len(tokenizer.encode(prompt, add_special_tokens=False))
    n_out = _meta_int(meta, "completion_tokens")
    if n_out is None:
        n_out = len(tokenizer.encode(reply, add_special_tokens=False))
    return n_in, n_out


def _run_inference(
    engine, tokenizer, prompt_text: str, sampling_params: dict
) -> Tuple[str, Dict[str, Any]]:
    out = engine.generate(prompt_text, sampling_params)
    if not isinstance(out, dict):
        out = {"text": str(out), "meta_info": {}}
    text = _completion_text(out)
    return text, out


async def _run_inference_async(
    engine, tokenizer, prompt_text: str, sampling_params: dict
) -> Tuple[str, Dict[str, Any]]:
    """Async counterpart of ``_run_inference`` for concurrent in-flight requests on one Engine."""
    out = await engine.async_generate(prompt_text, sampling_params)
    if not isinstance(out, dict):
        out = {"text": str(out), "meta_info": {}}
    text = _completion_text(out)
    return text, out


def run_speculative_inference_bench(
    *,
    model_path: str,
    draft_model_path: str,
    records: List[dict],
    tokenizer: Any,
    question_key: str = "question",
    max_samples: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 1,
    tp_size: int = 1,
    dtype: str = "bfloat16",
    attention_backend: str = "fa3",
    mem_fraction_static: float = 0.4,
    enable_thinking: bool = False,
    no_speculative: bool = False,
    trust_remote_code: bool = True,
    log_level: str = "error",
    mamba_scheduler_strategy: str = "extra_buffer",
    extra_engine_kwargs: Optional[Dict[str, Any]] = None,
    speculative_num_draft_tokens: Optional[int] = None,
    thresh_head_threshold_rate: Optional[float] = None,
    conf_verify_len: Optional[int] = None,
    prob_head_mul_threshold: Optional[float] = None,
    prob_head_candidate_len_min: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run ``sgl.Engine`` speculative decoding (same stack as this module's main loop)
    on the first ``max_samples`` JSONL rows and return aggregate timing / spec stats.

    Call after any prior SGLang / distributed training stack has been torn down so
    VRAM is free. Intended for rank-0-only post-train smoke benchmarks.

    Default ``log_level="error"`` keeps the scheduler quiet; per-sample lines go to stdout.

    ``extra_engine_kwargs``: forwarded verbatim into ``sgl.Engine(...)``; use this to pass
    DFlash ablation controls such as ``thresh_head_threshold_rate``, ``conf_verify_len``,
    ``prob_head_mul_threshold``, etc.

    ``speculative_num_draft_tokens``: SGLang DFlash verify window length (same as draft
    ``block_size`` when aligned). Omit to infer from draft config (fallback 16 in SGLang).

    ``thresh_head_threshold_rate``: Passed to SGLang Engine as ``thresh_head_threshold_rate``
    (direct_len path: scales predicted ratio before ``candidate_len``); omit for SGLang default.

    ``conf_verify_len`` / ``prob_head_mul_threshold`` / ``prob_head_candidate_len_min``: SGLang
    ServerArgs equivalents. ``conf_verify_len=-1`` enables training-free adaptive verify length
    from draft LM softmax confidence (see ``dflash_worker``); ``0`` = off; ``>0`` = fixed cap.
    """
    n_take = min(max_samples, len(records))
    subset = records[:n_take]
    sampling_params = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_new_tokens": max_new_tokens,
    }
    engine_kw: dict = {
        "model_path": model_path,
        "tp_size": tp_size,
        "dtype": dtype,
        "attention_backend": attention_backend,
        "mem_fraction_static": mem_fraction_static,
        "trust_remote_code": trust_remote_code,
        "skip_server_warmup": True,
        "log_level": log_level,
    }
    if not no_speculative:
        engine_kw["speculative_algorithm"] = "DFLASH"
        engine_kw["speculative_draft_model_path"] = draft_model_path
    q35_extras = qwen35_dflash_engine_compat_extras(
        model_path,
        speculative=not no_speculative,
        mamba_scheduler_strategy=mamba_scheduler_strategy,
    )
    engine_kw.update(q35_extras)
    if q35_extras:
        logger.info(
            "[inference_bench] Qwen3.5 DFLASH compat: SGLANG_ENABLE_SPEC_V2=%s mamba_scheduler_strategy=%s",
            os.environ.get("SGLANG_ENABLE_SPEC_V2"),
            q35_extras.get("mamba_scheduler_strategy"),
        )
    if extra_engine_kwargs:
        engine_kw.update(extra_engine_kwargs)
    if speculative_num_draft_tokens is not None:
        ntok = int(speculative_num_draft_tokens)
        if ntok <= 0:
            raise ValueError(f"speculative_num_draft_tokens must be positive, got {ntok}")
        engine_kw["speculative_num_draft_tokens"] = ntok
        logger.info("[inference_bench] speculative_num_draft_tokens=%s (DFlash block / verify window)", ntok)
    if thresh_head_threshold_rate is not None:
        engine_kw["thresh_head_threshold_rate"] = float(thresh_head_threshold_rate)
        logger.info(
            "[inference_bench] thresh_head_threshold_rate=%s",
            thresh_head_threshold_rate,
        )
    if conf_verify_len is not None:
        engine_kw["conf_verify_len"] = int(conf_verify_len)
        logger.info("[inference_bench] conf_verify_len=%s", conf_verify_len)
    if prob_head_mul_threshold is not None:
        engine_kw["prob_head_mul_threshold"] = float(prob_head_mul_threshold)
        logger.info(
            "[inference_bench] prob_head_mul_threshold=%s",
            prob_head_mul_threshold,
        )
    if prob_head_candidate_len_min is not None:
        engine_kw["prob_head_candidate_len_min"] = int(prob_head_candidate_len_min)
        logger.info(
            "[inference_bench] prob_head_candidate_len_min=%s",
            prob_head_candidate_len_min,
        )

    per_sample: List[Dict[str, Any]] = []
    t_wall0 = time.perf_counter()
    with sgl.Engine(**engine_kw) as engine:
        for idx, record in enumerate(subset):
            try:
                prompt, question = _build_inference_prompt(
                    tokenizer,
                    record,
                    question_key=question_key,
                    enable_thinking=enable_thinking,
                )
            except KeyError as e:
                logger.warning("[inference_bench] skip sample %d: %s", idx, e)
                continue
            t_inf = time.perf_counter()
            response, raw_out = _run_inference(engine, tokenizer, prompt, sampling_params)
            response_save = _response_for_buffer(response, raw_out)
            elapsed = time.perf_counter() - t_inf
            n_in, n_out = _token_counts(raw_out, prompt, response_save, tokenizer)
            tps = (n_out / elapsed) if elapsed > 0 else 0.0
            meta = raw_out.get("meta_info") if isinstance(raw_out, dict) else None
            meta = meta if isinstance(meta, dict) else {}
            accept_len = _meta_float(meta, "spec_accept_length")
            accept_rate = _meta_float(meta, "spec_accept_rate")
            verify_len = _meta_float(meta, "spec_verify_len")
            row: Dict[str, Any] = {
                "index": idx,
                "in_tokens": n_in,
                "out_tokens": n_out,
                "time_s": round(elapsed, 6),
                "tps": round(tps, 6),
            }
            if accept_len is not None:
                row["avg_accept_len"] = round(accept_len, 6)
            if accept_rate is not None:
                row["avg_accept_rate"] = round(accept_rate, 6)
            if verify_len is not None:
                row["avg_verify_len"] = round(verify_len, 6)
            per_sample.append(row)
            al_s = f"{accept_len:.4f}" if accept_len is not None else "n/a"
            ar_s = f"{accept_rate:.4f}" if accept_rate is not None else "n/a"
            vl_s = f"{verify_len:.4f}" if verify_len is not None else "n/a"
            # One line per sample on stdout (same format as ``main()`` + spec meta); SGLang decode lines suppressed via log_level=error.
            print(
                f"[inference] sample={len(per_sample)}/{n_take} draft_v=0 "
                f"in_tokens={n_in} out_tokens={n_out} "
                f"time_s={elapsed:.2f} TPS={tps:.1f} "
                f"avg_accept_len={al_s} avg_accept_rate={ar_s} avg_verify_len={vl_s}",
                flush=True,
            )

    wall_s = time.perf_counter() - t_wall0
    if not per_sample:
        out = {
            "n_samples": 0,
            "n_ok": 0,
            "wall_time_s": round(wall_s, 6),
            "avg_tps": 0.0,
            "avg_accept_rate": None,
            "avg_accept_len": None,
            "total_out_tokens": 0,
            "per_sample": [],
        }
        # stdout: SGLang may raise root log level so ``logger.info`` after Engine is invisible.
        print(
            f"[inference_bench] n_ok=0/{n_take} wall_s={wall_s:.2f} avg_TPS=0.0000 "
            f"avg_accept_rate=n/a avg_accept_len=n/a total_out_tokens=0",
            flush=True,
        )
        return out

    total_out = sum(int(r["out_tokens"]) for r in per_sample)
    total_gen_time = sum(float(r["time_s"]) for r in per_sample)
    avg_tps = (total_out / total_gen_time) if total_gen_time > 0 else 0.0
    rates = [float(r["avg_accept_rate"]) for r in per_sample if r.get("avg_accept_rate") is not None]
    lens = [float(r["avg_accept_len"]) for r in per_sample if r.get("avg_accept_len") is not None]
    vlens = [float(r["avg_verify_len"]) for r in per_sample if r.get("avg_verify_len") is not None]
    avg_accept_rate = round(sum(rates) / len(rates), 6) if rates else None
    avg_accept_len = round(sum(lens) / len(lens), 6) if lens else None
    avg_verify_len = round(sum(vlens) / len(vlens), 6) if vlens else None
    out = {
        "n_samples": n_take,
        "n_ok": len(per_sample),
        "wall_time_s": round(wall_s, 6),
        "avg_tps": round(avg_tps, 4),
        "avg_accept_rate": avg_accept_rate,
        "avg_accept_len": avg_accept_len,
        "avg_verify_len": avg_verify_len,
        "total_out_tokens": total_out,
        "per_sample": per_sample,
    }
    ar_sum = f"{avg_accept_rate:.6f}" if avg_accept_rate is not None else "n/a"
    al_sum = f"{avg_accept_len:.6f}" if avg_accept_len is not None else "n/a"
    vl_sum = f"{avg_verify_len:.6f}" if avg_verify_len is not None else "n/a"
    print(
        f"[inference_bench] n_ok={out['n_ok']}/{n_take} wall_s={wall_s:.2f} avg_TPS={out['avg_tps']:.4f} "
        f"avg_accept_rate={ar_sum} avg_accept_len={al_sum} avg_verify_len={vl_sum} total_out_tokens={total_out}",
        flush=True,
    )
    return out


# Matches Qwen3-style chat templates used with ``apply_chat_template`` (see saved ``prompt``).
_BUFFER_ASSISTANT_END = "<|im_end|>"


def _response_for_buffer(response: str, raw_out: Dict[str, Any]) -> str:
    """
    Append the assistant-turn end marker for JSONL training alignment, except when
    SGLang stopped because ``max_new_tokens`` was hit (``meta_info.finish_reason.type``
    is ``length``, same convention as the OpenAI API).
    """
    if not isinstance(response, str):
        response = str(response)
    meta = raw_out.get("meta_info") if isinstance(raw_out, dict) else None
    meta = meta if isinstance(meta, dict) else {}
    fr = meta.get("finish_reason")
    if isinstance(fr, dict) and fr.get("type") == "length":
        return response
    mark = _BUFFER_ASSISTANT_END
    if response.endswith(mark):
        return response
    return response + mark


def _build_inference_prompt(
    tokenizer,
    record: dict,
    *,
    question_key: str,
    enable_thinking: bool,
) -> Tuple[str, str]:
    """
    Build the prompt passed to SGLang, aligned with ``_TextDataset`` loss prefix.

    For ``conversations`` / ``messages``: apply chat template to the full history
    *before* the last assistant turn (regenerate that reply). For flat rows, keep
    the legacy single-user path.

    Returns ``(prompt, question_label)`` where ``question_label`` is stored in
    data_buffer ``question`` for debugging.
    """
    messages = _normalize_chat_messages(record)
    if messages is not None:
        last_asst_idx = _last_assistant_index(messages)
        prefix = messages[:last_asst_idx] if last_asst_idx is not None else messages
        prompt = tokenizer.apply_chat_template(
            prefix,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return prompt, _preview_chat_messages(messages)

    question = _extract_user_question(record, question_key)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return prompt, question


def _extract_user_question(record: dict, question_key: str) -> str:
    """Flat user text: ``question_key``, dflash ``turns`` cache, or first user in conversations."""
    if question_key in record:
        v = record[question_key]
        if isinstance(v, str) and v.strip():
            return v
    turns = record.get("turns")
    if isinstance(turns, list) and turns:
        first = turns[0]
        if isinstance(first, str) and first.strip():
            return first
    for k in ("query", "instruction", "input", "prompt"):
        if k in record and isinstance(record[k], str) and record[k].strip():
            return record[k]
    for col in ("conversations", "messages"):
        conv = record.get(col)
        if not isinstance(conv, list):
            continue
        for turn in conv:
            if not isinstance(turn, dict) or turn.get("role") != "user":
                continue
            content = turn.get("content", "")
            if isinstance(content, str) and content.strip():
                return content
    raise KeyError(
        f"no user question (tried key={question_key!r}, turns, conversations/messages); "
        f"keys={list(record)}"
    )


def _extract_reference_assistant(record: dict) -> Optional[str]:
    """
    Gold assistant text from a dataset JSONL row (for ``ground_truth`` in data_buffer).

    Prefer the last assistant turn in ``conversations`` or ``messages`` (same rule as
    ``training_utils._TextDataset``). Otherwise accept common flat labels: ``response``,
    ``answer``, ``output``, ``completion`` (reference only — not the completion from
    the engine).
    """
    for col in ("conversations", "messages"):
        conv_list = record.get(col)
        if not isinstance(conv_list, list) or not conv_list:
            continue
        for turn in reversed(conv_list):
            if not isinstance(turn, dict) or turn.get("role") != "assistant":
                continue
            c = turn.get("content", "")
            if isinstance(c, str):
                return c
            return str(c) if c is not None else ""

    for key in ("response", "answer", "output", "completion"):
        v = record.get(key)
        if isinstance(v, str) and v.strip():
            return v

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(description="Async pipeline: inference worker")

    # Model
    parser.add_argument("--model-path", default="Qwen/Qwen3-8B")
    parser.add_argument("--initial-draft-path", default="z-lab/Qwen3-8B-DFlash-b16")
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="SGLang tensor parallel size for the Engine (single-GPU pipeline default: 1).",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="fa3")
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--no-speculative", action="store_true")
    parser.add_argument(
        "--mamba-scheduler-strategy",
        type=str,
        default="extra_buffer",
        choices=("auto", "no_buffer", "extra_buffer"),
        help="Only when --model-path looks like Qwen3.5 and speculative DFLASH is on: passed as "
        "mamba_scheduler_strategy (default extra_buffer). Same as bin/regenerate_dataset.py.",
    )
    parser.add_argument(
        "--thresh-head-threshold-rate",
        type=float,
        default=None,
        metavar="RATE",
        help="DFlash adaptive length head (direct_len): multiply predicted ratio before "
        "candidate_len = round(ratio * block_size * rate). Passed to sgl.Engine as "
        "thresh_head_threshold_rate; omit to use SGLang default (1.0).",
    )
    # Dataset
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="JSONL: one object per line. Use --question-key for a string field, "
        "or ShareGPT-style conversations/messages (full history before last assistant).",
    )
    parser.add_argument(
        "--question-key",
        default="question",
        help="If set and present on a flat row, use this string field as the user turn; "
        "otherwise use conversations/messages (prefix before last assistant).",
    )
    parser.add_argument(
        "--dataset-start-line",
        type=int,
        default=1,
        metavar="N",
        help="1-based line index in the JSONL to start from (1 = first record). "
        "Earlier lines are skipped each time the dataset is reloaded.",
    )

    # Sampling
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Qwen3: pass enable_thinking to apply_chat_template.",
    )
    parser.add_argument("--chat-template", default="qwen",
                        help="For prompt formatting (unused if dataset already formatted).")

    # Swap dir
    parser.add_argument("--swap-dir", default="./swap",
                        help="Root of the shared swap directory.")

    # Loop control
    parser.add_argument("--loop", action="store_true",
                        help="Keep cycling over the dataset indefinitely.")
    parser.add_argument("--poll-interval", type=float, default=0.0,
                        help="Sleep (s) between samples (0 = no sleep).")
    parser.add_argument(
        "--inference-stats-jsonl",
        type=str,
        default="auto",
        metavar="PATH",
        help="Append one JSON line per completed sample (for plotting). "
        "Use 'auto' for <swap-dir>/inference_stats.jsonl; empty string to disable.",
    )
    args = parser.parse_args()

    # Setup paths
    swap = Path(args.swap_dir)
    data_buffer = swap / "data_buffer"
    model_weights = swap / "model_weights"
    stop_pipeline_path = swap / STOP_PIPELINE_FILENAME
    data_buffer.mkdir(parents=True, exist_ok=True)
    model_weights.mkdir(parents=True, exist_ok=True)

    stats_jsonl: Optional[Path] = None
    raw_stats = args.inference_stats_jsonl.strip()
    if not raw_stats:
        pass  # disabled
    elif raw_stats.lower() == "auto":
        stats_jsonl = swap / "inference_stats.jsonl"
    else:
        stats_jsonl = Path(raw_stats).expanduser().resolve()
    if stats_jsonl is not None:
        logger.info("[inference] Per-sample stats -> %s", stats_jsonl)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
    }

    engine_kw: dict = {
        "model_path": args.model_path,
        "tp_size": args.tp_size,
        "dtype": args.dtype,
        "attention_backend": args.attention_backend,
        "mem_fraction_static": args.mem_fraction_static,
        "trust_remote_code": True,
        "skip_server_warmup": True,
    }
    if not args.no_speculative:
        engine_kw["speculative_algorithm"] = "DFLASH"
        engine_kw["speculative_draft_model_path"] = args.initial_draft_path
    q35_extras = qwen35_dflash_engine_compat_extras(
        args.model_path,
        speculative=not args.no_speculative,
        mamba_scheduler_strategy=args.mamba_scheduler_strategy,
    )
    engine_kw.update(q35_extras)
    if q35_extras:
        logger.info(
            "[inference] Qwen3.5 DFLASH compat: SGLANG_ENABLE_SPEC_V2=%s mamba_scheduler_strategy=%s",
            os.environ.get("SGLANG_ENABLE_SPEC_V2"),
            q35_extras.get("mamba_scheduler_strategy"),
        )
    if args.thresh_head_threshold_rate is not None:
        engine_kw["thresh_head_threshold_rate"] = args.thresh_head_threshold_rate

    logger.info("[inference] Loading SGLang Engine...")
    t0 = time.perf_counter()
    with sgl.Engine(**engine_kw) as engine:
        logger.info("[inference] Engine ready in %.1fs.", time.perf_counter() - t0)

        # 0 = engine still on --initial-draft-path; 1+ = hot-swapped from swap/model_weights (version.txt).
        current_draft_version = 0

        while True:
            if stop_pipeline_path.exists():
                logger.info(
                    "[inference] %s found under %s; exiting before next dataset pass.",
                    STOP_PIPELINE_FILENAME,
                    swap,
                )
                break
            # Load dataset fresh each cycle so new samples written by training
            # are automatically picked up when looping.
            with open(args.dataset_path) as f:
                records = [json.loads(line) for line in f if line.strip()]

            n_all = len(records)
            skip = max(0, args.dataset_start_line - 1)
            if skip >= n_all:
                logger.error(
                    "[inference] dataset-start-line=%d skips all %d lines; fix or lower.",
                    args.dataset_start_line,
                    n_all,
                )
                records = []
            elif skip > 0:
                logger.info(
                    "[inference] Skipped first %d JSONL line(s); using lines %d-%d (%d samples).",
                    skip,
                    args.dataset_start_line,
                    n_all,
                    n_all - skip,
                )
                records = records[skip:]
            logger.info("[inference] Starting over dataset (%d samples).", len(records))

            stop_requested = False
            for idx, record in enumerate(records):
                if stop_pipeline_path.exists():
                    logger.info(
                        "[inference] %s found under %s; exiting inference loop.",
                        STOP_PIPELINE_FILENAME,
                        swap,
                    )
                    stop_requested = True
                    break
                try:
                    prompt, question = _build_inference_prompt(
                        tokenizer,
                        record,
                        question_key=args.question_key,
                        enable_thinking=args.enable_thinking,
                    )
                except KeyError as e:
                    logger.error("[inference] skip sample %d: %s", idx, e)
                    continue

                # Check for newer draft weights before inference.
                if not args.no_speculative:
                    current_draft_version = _maybe_update_draft(
                        engine, model_weights, current_draft_version
                    )

                t_inf = time.perf_counter()
                response, raw_out = _run_inference(
                    engine, tokenizer, prompt, sampling_params
                )
                response_save = _response_for_buffer(response, raw_out)
                elapsed = time.perf_counter() - t_inf
                n_in, n_out = _token_counts(raw_out, prompt, response_save, tokenizer)
                tps = (n_out / elapsed) if elapsed > 0 else 0.0

                meta = raw_out.get("meta_info") if isinstance(raw_out, dict) else None
                meta = meta if isinstance(meta, dict) else {}
                accept_len = _meta_float(meta, "spec_accept_length")
                accept_rate = _meta_float(meta, "spec_accept_rate")
                verify_len = _meta_float(meta, "spec_verify_len")
                al_s = f"{accept_len:.4f}" if accept_len is not None else "n/a"
                ar_s = f"{accept_rate:.4f}" if accept_rate is not None else "n/a"
                vl_s = f"{verify_len:.4f}" if verify_len is not None else "n/a"

                # stdout (line-buffered with python -u) so tee + terminal see one line per sample.
                print(
                    f"[inference] sample={idx + 1}/{len(records)} draft_v={current_draft_version} "
                    f"in_tokens={n_in} out_tokens={n_out} "
                    f"time_s={elapsed:.2f} TPS={tps:.1f} "
                    f"avg_accept_len={al_s} avg_accept_rate={ar_s} avg_verify_len={vl_s}",
                    flush=True,
                )

                if stats_jsonl is not None:
                    # Same fields as dflash benchmark (HTTP meta): per-request speculative metrics.
                    stats_row: Dict[str, Any] = {
                        "dataset_index": idx,
                        "draft_version": current_draft_version,
                        "in_tokens": n_in,
                        "out_tokens": n_out,
                        "time_s": round(elapsed, 6),
                        "tps": round(tps, 6),
                    }
                    if accept_len is not None:
                        stats_row["avg_accept_len"] = round(accept_len, 6)
                    if accept_rate is not None:
                        stats_row["avg_accept_rate"] = round(accept_rate, 6)
                    if verify_len is not None:
                        stats_row["avg_verify_len"] = round(verify_len, 6)
                    _append_inference_stats(stats_jsonl, stats_row)

                # Save result to data_buffer as a timestamped JSONL file.
                ts = int(time.time() * 1000)
                out_file = data_buffer / f"{ts}_{idx}.jsonl"
                payload: Dict[str, Any] = {
                    "prompt": prompt,
                    "response": response_save,
                    "question": question,
                }
                gt = _extract_reference_assistant(record)
                if gt is not None and str(gt).strip():
                    payload["ground_truth"] = gt if isinstance(gt, str) else str(gt)
                out_file.write_text(json.dumps(payload, ensure_ascii=False) + "\n")

                if args.poll_interval > 0:
                    time.sleep(args.poll_interval)

            if stop_requested:
                break
            if not args.loop:
                # Training polls the buffer forever if inference exits without this signal.
                write_stop_pipeline_marker(swap)
                logger.info(
                    "[inference] Dataset pass finished without --loop; wrote %s under %s "
                    "so training can stop when the buffer is below threshold.",
                    STOP_PIPELINE_FILENAME,
                    swap,
                )
                break

    logger.info("[inference] Done.")


if __name__ == "__main__":
    main()
