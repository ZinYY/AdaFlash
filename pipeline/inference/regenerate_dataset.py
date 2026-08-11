#!/usr/bin/env python3
# coding=utf-8
"""
Regenerate training JSONL by running SGLang DFlash inference over an input dataset.

Writes one JSONL (``prompt``, ``response``, ``question`` per line), same shape as
``pipeline.inference.worker`` data_buffer records. Use for offline regen before
thresh-head init or custom training corpora.

Example::

  cd asyn_train
  CUDA_VISIBLE_DEVICES=0 python bin/regenerate_dataset.py \\
      --model-path Qwen/Qwen3-8B \\
      --initial-draft-path z-lab/Qwen3-8B-DFlash-b16 \\
      --dataset-path cache/dataset/perfectblend_train.jsonl \\
      --output-jsonl cache/dataset/perfectblend_qwen3_8b_regen.jsonl

  # Optional: cap how many rows to run after --dataset-start-line (default: all remaining).
  # --num-samples 4096 \\
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sglang as sgl
from transformers import AutoTokenizer

from pipeline.inference.worker import (
    _build_inference_prompt,
    _maybe_update_draft,
    _meta_float,
    _response_for_buffer,
    _run_inference,
    _run_inference_async,
    _token_counts,
    qwen35_dflash_engine_compat_extras,
)
from pipeline.paths import ASYN_TRAIN_ROOT

logger = logging.getLogger(__name__)


def _resolve_asyn_path(path: str) -> Path:
    """Resolve relative paths under ``asyn_train`` (e.g. ``cache/dataset/foo.jsonl``)."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = ASYN_TRAIN_ROOT / p
    return p.resolve()


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _prepare_sample(
    idx: int,
    record: dict,
    *,
    tokenizer: Any,
    question_key: str,
    enable_thinking: bool,
) -> Tuple[int, str, str]:
    """Return ``(idx, question, prompt)`` for one dataset row."""
    prompt, question = _build_inference_prompt(
        tokenizer,
        record,
        question_key=question_key,
        enable_thinking=enable_thinking,
    )
    return idx, question, prompt


def _maybe_update_draft_locked(
    engine: Any,
    model_weights: Path,
    draft_state: Dict[str, int],
    draft_lock: Optional[threading.Lock],
) -> None:
    if draft_lock is not None:
        with draft_lock:
            draft_state["version"] = _maybe_update_draft(
                engine, model_weights, draft_state["version"]
            )
    else:
        draft_state["version"] = _maybe_update_draft(
            engine, model_weights, draft_state["version"]
        )


def _pack_infer_result(
    idx: int,
    question: str,
    prompt: str,
    response: str,
    raw_out: Dict[str, Any],
    tokenizer: Any,
    elapsed: float,
    draft_version: str,
) -> Tuple[int, Dict[str, Any], float, int, int, float, str, str, str, str]:
    response_save = _response_for_buffer(response, raw_out)
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

    row = {
        "prompt": prompt,
        "response": response_save,
        "question": question,
    }
    return (
        idx,
        row,
        elapsed,
        n_in,
        n_out,
        tps,
        al_s,
        ar_s,
        vl_s,
        draft_version,
    )


def _infer_one(
    engine: Any,
    tokenizer: Any,
    *,
    idx: int,
    question: str,
    prompt: str,
    sampling_params: Dict[str, Any],
    model_weights: Optional[Path],
    no_speculative: bool,
    draft_state: Dict[str, int],
    draft_lock: Optional[threading.Lock],
) -> Tuple[int, Dict[str, Any], float, int, int, float, str, str, str, str]:
    """Serial path: ``engine.generate`` (must run on the main thread)."""
    if model_weights is not None and not no_speculative:
        _maybe_update_draft_locked(engine, model_weights, draft_state, draft_lock)

    t_inf = time.perf_counter()
    response, raw_out = _run_inference(engine, tokenizer, prompt, sampling_params)
    return _pack_infer_result(
        idx,
        question,
        prompt,
        response,
        raw_out,
        tokenizer,
        time.perf_counter() - t_inf,
        str(draft_state["version"]),
    )


async def _infer_one_async(
    engine: Any,
    tokenizer: Any,
    *,
    idx: int,
    question: str,
    prompt: str,
    sampling_params: Dict[str, Any],
    model_weights: Optional[Path],
    no_speculative: bool,
    draft_state: Dict[str, int],
    draft_lock: Optional[threading.Lock],
) -> Tuple[int, Dict[str, Any], float, int, int, float, str, str, str, str]:
    """Concurrent path: ``engine.async_generate`` on the Engine event loop."""
    if model_weights is not None and not no_speculative:
        _maybe_update_draft_locked(engine, model_weights, draft_state, draft_lock)

    t_inf = time.perf_counter()
    response, raw_out = await _run_inference_async(engine, tokenizer, prompt, sampling_params)
    return _pack_infer_result(
        idx,
        question,
        prompt,
        response,
        raw_out,
        tokenizer,
        time.perf_counter() - t_inf,
        str(draft_state["version"]),
    )


def _print_regen_progress(
    done: int,
    total: int,
    result: Tuple[int, Dict[str, Any], float, int, int, float, str, str, str, str],
) -> None:
    idx, _row, elapsed, n_in, n_out, tps, al_s, ar_s, vl_s, draft_v = result
    print(
        f"[regen] sample={done}/{total} idx={idx} draft_v={draft_v} "
        f"in_tokens={n_in} out_tokens={n_out} time_s={elapsed:.2f} TPS={tps:.1f} "
        f"avg_accept_len={al_s} avg_accept_rate={ar_s} avg_verify_len={vl_s}",
        flush=True,
    )


async def _run_prepared_concurrent(
    engine: Any,
    tokenizer: Any,
    prepared: List[Tuple[int, str, str]],
    *,
    concurrency: int,
    sampling_params: Dict[str, Any],
    model_weights: Optional[Path],
    no_speculative: bool,
    draft_state: Dict[str, int],
    draft_lock: Optional[threading.Lock],
) -> List[Tuple[int, Dict[str, Any], float, int, int, float, str, str, str, str]]:
    sem = asyncio.Semaphore(concurrency)
    total = len(prepared)

    async def _one(item: Tuple[int, str, str]) -> Tuple[int, Dict[str, Any], float, int, int, float, str, str, str, str]:
        async with sem:
            idx, question, prompt = item
            return await _infer_one_async(
                engine,
                tokenizer,
                idx=idx,
                question=question,
                prompt=prompt,
                sampling_params=sampling_params,
                model_weights=model_weights,
                no_speculative=no_speculative,
                draft_state=draft_state,
                draft_lock=draft_lock,
            )

    tasks = [asyncio.create_task(_one(item)) for item in prepared]
    results: List[Tuple[int, Dict[str, Any], float, int, int, float, str, str, str, str]] = []
    done = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        done += 1
        _print_regen_progress(done, total, result)
        results.append(result)
    return sorted(results, key=lambda r: r[0])


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    p = argparse.ArgumentParser(
        description="Regenerate JSONL via single-process SGLang inference (DFlash by default)."
    )
    p.add_argument("--dataset-path", type=str, required=True, help="Input JSONL (one object per line).")
    p.add_argument(
        "--dataset-start-line",
        type=int,
        default=1,
        metavar="N",
        help="1-based first line to use (same semantics as inference_worker). Default 1.",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=None,
        metavar="N",
        help="Maximum JSONL rows to run after --dataset-start-line (in order). "
        "Omit to regenerate all remaining rows.",
    )
    p.add_argument(
        "--output-jsonl",
        type=str,
        required=True,
        help="Write all results here (one JSON object per line). Parent dirs are created.",
    )

    # Model / Engine — defaults aligned with run_async_pipeline_osd.sh inference + inference_worker fallbacks
    p.add_argument("--model-path", default="Qwen/Qwen3.5-9B")
    p.add_argument("--initial-draft-path", default="z-lab/Qwen3.5-9B-DFlash")
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attention-backend", default="fa3")
    p.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.80,
        help="Default 0.80 to match INFER_MEM_FRACTION in run_async_pipeline_osd.sh.",
    )
    p.add_argument("--no-speculative", action="store_true")
    p.add_argument(
        "--mamba-scheduler-strategy",
        type=str,
        default="extra_buffer",
        choices=("auto", "no_buffer", "extra_buffer"),
        help="Only when --model-path looks like Qwen3.5 (substring qwen3.5, any case) and speculative DFLASH is on: passed as "
        "mamba_scheduler_strategy (default extra_buffer). Ignored for Qwen3-only paths and with --no-speculative.",
    )

    # Sampling — defaults aligned with pipeline (2048, 0.0); top_p/top_k match inference_worker defaults
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=1)

    p.add_argument(
        "--question-key",
        default="question",
        help="Same as inference_worker: field for user text when present.",
    )
    p.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Qwen3: pass enable_thinking to apply_chat_template (match pipeline ENABLE_THINKING=1).",
    )
    p.add_argument(
        "--swap-dir",
        type=str,
        default=None,
        help="Optional: if set, reload draft from swap/model_weights/version.txt like inference_worker.",
    )
    p.add_argument(
        "--poll-interval",
        type=float,
        default=0.0,
        help="Seconds to sleep between samples (0 = none), same option as inference_worker.",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        metavar="N",
        help="Max in-flight requests via engine.async_generate + asyncio (main thread). Default 1 (serial generate).",
    )

    args = p.parse_args()
    if args.concurrency < 1:
        p.error("--concurrency must be >= 1")
    if args.num_samples is not None and args.num_samples < 1:
        p.error("--num-samples must be >= 1")
    if args.dataset_start_line < 1:
        p.error("--dataset-start-line must be >= 1 (1-based)")

    dataset_path = _resolve_asyn_path(args.dataset_path)
    if not dataset_path.is_file():
        logger.error("Dataset not found: %s", dataset_path)
        sys.exit(1)

    out_path = _resolve_asyn_path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = _load_jsonl(dataset_path)
    n_all = len(all_rows)
    skip = max(0, args.dataset_start_line - 1)
    if skip >= n_all:
        logger.error(
            "dataset-start-line=%d skips all %d lines; nothing to run.",
            args.dataset_start_line,
            n_all,
        )
        sys.exit(1)

    if args.num_samples is None:
        args.num_samples = n_all - skip

    subset = all_rows[skip : skip + args.num_samples]
    if not subset:
        logger.error("Empty slice after start line and num-samples.")
        sys.exit(1)

    logger.info(
        "[regen] dataset=%s total_lines=%d start_line=%d num_samples=%d -> run %d row(s)",
        dataset_path,
        n_all,
        args.dataset_start_line,
        args.num_samples,
        len(subset),
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    sampling_params: Dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
    }

    engine_kw: Dict[str, Any] = {
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
            speculative=True,
            mamba_scheduler_strategy=args.mamba_scheduler_strategy,
        )
        if q35_extras:
            engine_kw.update(q35_extras)
            logger.info(
                "[regen] Qwen3.5 DFLASH compat: SGLANG_ENABLE_SPEC_V2=%s mamba_scheduler_strategy=%s",
                os.environ.get("SGLANG_ENABLE_SPEC_V2"),
                q35_extras.get("mamba_scheduler_strategy"),
            )

    if args.concurrency > 1:
        engine_kw["max_running_requests"] = max(
            args.concurrency,
            int(engine_kw.get("max_running_requests") or 0),
        )
        logger.info("[regen] max_running_requests=%s", engine_kw["max_running_requests"])

    swap: Path | None = None
    model_weights: Path | None = None
    if args.swap_dir:
        swap = Path(args.swap_dir).expanduser().resolve()
        model_weights = swap / "model_weights"
        model_weights.mkdir(parents=True, exist_ok=True)

    # 0 = initial speculative draft; matches inference_worker before first version.txt hot-swap.
    draft_state: Dict[str, int] = {"version": 0}
    draft_lock = threading.Lock() if args.concurrency > 1 and model_weights is not None else None

    prepared: List[Tuple[int, str, str]] = []
    for idx, record in enumerate(subset):
        try:
            prepared.append(
                _prepare_sample(
                    idx,
                    record,
                    tokenizer=tokenizer,
                    question_key=args.question_key,
                    enable_thinking=args.enable_thinking,
                )
            )
        except KeyError as e:
            logger.error("[regen] skip relative index %d: %s", idx, e)

    if not prepared:
        logger.error("[regen] no valid samples to run.")
        sys.exit(1)

    logger.info(
        "[regen] concurrency=%d valid_samples=%d",
        args.concurrency,
        len(prepared),
    )

    def _run_prepared(item: Tuple[int, str, str]) -> Tuple[int, Dict[str, Any], float, int, int, float, str, str, str, str]:
        idx, question, prompt = item
        return _infer_one(
            engine,
            tokenizer,
            idx=idx,
            question=question,
            prompt=prompt,
            sampling_params=sampling_params,
            model_weights=model_weights,
            no_speculative=args.no_speculative,
            draft_state=draft_state,
            draft_lock=draft_lock,
        )

    t0 = time.perf_counter()
    with sgl.Engine(**engine_kw) as engine:
        logger.info("[regen] Engine ready in %.1fs.", time.perf_counter() - t0)

        results_by_idx: Dict[int, Dict[str, Any]] = {}

        if args.concurrency <= 1:
            with out_path.open("w", encoding="utf-8") as out_fp:
                for item in prepared:
                    result = _run_prepared(item)
                    idx, row = result[0], result[1]
                    results_by_idx[idx] = row
                    out_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_fp.flush()
                    _print_regen_progress(idx + 1, len(prepared), result)
                    if args.poll_interval > 0:
                        time.sleep(args.poll_interval)
        else:
            # ThreadPool + engine.generate() breaks: generate() calls run_until_complete()
            # on a shared uvloop and raises "this event loop is already running".
            completed = engine.loop.run_until_complete(
                _run_prepared_concurrent(
                    engine,
                    tokenizer,
                    prepared,
                    concurrency=args.concurrency,
                    sampling_params=sampling_params,
                    model_weights=model_weights,
                    no_speculative=args.no_speculative,
                    draft_state=draft_state,
                    draft_lock=draft_lock,
                )
            )
            for idx, row, *_rest in completed:
                results_by_idx[idx] = row

            with out_path.open("w", encoding="utf-8") as out_fp:
                for idx, _, _ in prepared:
                    out_fp.write(json.dumps(results_by_idx[idx], ensure_ascii=False) + "\n")

            if args.poll_interval > 0:
                logger.warning("[regen] --poll-interval ignored when concurrency > 1")

    logger.info("[regen] Wrote %s (done).", out_path)


if __name__ == "__main__":
    main()
