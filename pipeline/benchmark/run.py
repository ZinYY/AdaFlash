#!/usr/bin/env python3
"""HTTP benchmark against a running SGLang server (``/generate``).

Loads prompts from ``test_data/<dataset>_test.jsonl`` (ShareGPT-style ``conversations``
or preformatted ``{"turns": [...]}`` rows).

Example usage:
python benchmark.py \
    --base-url "http://127.0.0.1:6784" \
    --model "Qwen/Qwen3-8B" \
    --dataset "math_qa" \
    --max-samples 1024 \
    --num-prompts 1024 \
    --concurrency 32 \
    --eval-accuracy
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

random.seed(42)

from pipeline.paths import TEST_DATA_DIR

LOCAL_DATASETS: dict[str, Path] = {
    "math_qa": TEST_DATA_DIR / "math_qa_test.jsonl",
    "math500": TEST_DATA_DIR / "math500_test.jsonl",
    "aime25": TEST_DATA_DIR / "aime25_test.jsonl",
    "gsm8k": TEST_DATA_DIR / "gsm8k_test.jsonl",
    "opencodeinstruct": TEST_DATA_DIR / "opencodeinstruct_test.jsonl",
    "codealpaca": TEST_DATA_DIR / "codealpaca-20k_test.jsonl",
    "sharegpt": TEST_DATA_DIR / "sharegpt_test.jsonl",
    "myblend": TEST_DATA_DIR / "myblend_test.jsonl",
    "perfectblend": TEST_DATA_DIR / "perfectblend_test.jsonl",
    "humaneval": TEST_DATA_DIR / "humaneval_test.jsonl",
    "mbpp": TEST_DATA_DIR / "mbpp_test.jsonl",
    "mt-bench": TEST_DATA_DIR / "mt-bench_test.jsonl",
}

ACCURACY_DATASETS = frozenset({"gsm8k", "math_qa", "math500", "aime25"})

GSM8K_REASONING_INSTRUCTION = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)


def _extract_user_turns(row: dict) -> list[str]:
    if isinstance(row.get("turns"), list) and row["turns"]:
        out: list[str] = []
        for t in row["turns"]:
            if isinstance(t, str) and t.strip():
                out.append(t)
        if out:
            return out

    conv = row.get("conversations")
    if isinstance(conv, list) and conv:
        out = []
        for turn in conv:
            if not isinstance(turn, dict) or turn.get("role") != "user":
                continue
            content = turn.get("content", "")
            if isinstance(content, str) and content.strip():
                out.append(content)
        if out:
            return out

    for key in ("question", "query", "instruction", "input", "prompt"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return [v]

    raise ValueError(
        "JSONL row has no user text (expected conversations[user], turns, or question/...); "
        f"keys={list(row)}"
    )


def _extract_assistant_content(row: dict) -> str:
    conv = row.get("conversations")
    if isinstance(conv, list):
        for turn in conv:
            if isinstance(turn, dict) and turn.get("role") == "assistant":
                content = turn.get("content", "")
                if isinstance(content, str):
                    return content
    return ""


def _extract_math_qa_letter(text: str) -> str:
    text = text.lower().strip().strip('"')
    patterns = (
        r"answer(?:\s+is)?(?:\s*:\s*|\s+)(?:option\s+)?\(?([a-e])\)?\b",
        r"answer\s*=\s*\(?([a-e])\)?\b",
        r"answer\s+choice\s+([a-e])\b",
        r"the answer is\s+\(?([a-e])\)?\b",
        r"option\s+\(?([a-e])\)?(?:\s+is\s+correct)?\b",
        r"\b(?:ans|answer)\s+(?:is\s+)?([a-e])\b",
        r"\b([a-e])\s+is\s+the\s+answer\b",
        r"answer will be\s+([a-e])\b",
        r"correct answer\s*[:\s]+\(?([a-e])\)?\b",
        r"\b([a-e])\s+is\s+correct\b",
    )
    for pat in patterns:
        matches = re.findall(pat, text)
        if matches:
            return matches[-1]
    m = re.search(r"\b([a-e])\s*[\.\)]?\s*$", text)
    if m:
        return m.group(1)
    return ""


def _extract_reference_answer(dataset: str, row: dict) -> str:
    if dataset == "gsm8k":
        content = _extract_assistant_content(row)
        if "####" in content:
            return content.split("####")[-1].strip().replace(",", "")
        return ""
    if dataset == "math_qa":
        return _extract_math_qa_letter(_extract_assistant_content(row))
    if dataset == "math500":
        return _extract_boxed_answer(_extract_assistant_content(row))
    return str(row.get("answer", "")).strip()


def _extract_boxed_answer(text: str) -> str:
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    depth, start = 0, idx + len("\\boxed{")
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i].strip().replace(",", "")
            depth -= 1
    return ""


def _normalize_number(s: str) -> str:
    s = s.strip().replace(",", "").rstrip(".")
    try:
        return str(float(s))
    except ValueError:
        return s


def _extract_gsm8k_prediction(text: str) -> str:
    pred = _extract_boxed_answer(text)
    if pred:
        return pred
    if "####" in text:
        return text.split("####")[-1].strip().replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return numbers[-1] if numbers else ""


def _extract_math_qa_prediction(text: str) -> str:
    m = re.search(r"Answer:\s*\(?([A-Ea-e])\)?\b", text)
    if m:
        return m.group(1).lower()
    return _extract_math_qa_letter(text)


def _answers_match(pred: str, ref: str, dataset: str) -> bool:
    if not ref:
        return False
    if dataset == "math_qa":
        return pred.lower() == ref.lower()
    return _normalize_number(pred) == _normalize_number(ref)


def _row_to_benchmark_entry(row: dict, *, dataset: str = "", include_answer: bool = False) -> dict:
    entry: dict = {"turns": _extract_user_turns(row)}
    if include_answer and dataset in ACCURACY_DATASETS:
        ref = _extract_reference_answer(dataset, row)
        if ref:
            entry["answer"] = ref
    return entry


def _load_jsonl_entries(path: Path, *, dataset: str, need_answers: bool) -> list[dict]:
    entries: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
            if "turns" in row and isinstance(row["turns"], list):
                entry = dict(row)
                if need_answers and dataset in ACCURACY_DATASETS:
                    ref = _extract_reference_answer(dataset, row)
                    if ref:
                        entry["answer"] = ref
            else:
                entry = _row_to_benchmark_entry(
                    row, dataset=dataset, include_answer=need_answers
                )
            entries.append(entry)
    return entries


def load_and_process_dataset(data_name: str, *, need_answers: bool = False) -> list[dict]:
    if data_name not in LOCAL_DATASETS:
        raise ValueError(
            f"Unknown dataset '{data_name}'. Available: {sorted(LOCAL_DATASETS.keys())}"
        )

    path = LOCAL_DATASETS[data_name]
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return _load_jsonl_entries(path, dataset=data_name, need_answers=need_answers)


def _limit_dataset(dataset: list[dict], max_samples: int | None) -> list[dict]:
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    random.shuffle(dataset)
    return dataset[:max_samples]


def _apply_chat_template(tokenizer, messages: list[dict], enable_thinking: bool) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def _send_sglang(
    base_url: str,
    text: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    timeout_s: int,
) -> dict:
    resp = requests.post(
        base_url.rstrip("/") + "/generate",
        json={
            "text": text,
            "sampling_params": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_new_tokens": max_new_tokens,
            },
        },
        timeout=timeout_s,
    )
    resp.raise_for_status()
    out = resp.json()
    return out if isinstance(out, dict) else out[0]


def run_benchmark(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    eval_acc = getattr(args, "eval_accuracy", False)
    if eval_acc and args.dataset not in ACCURACY_DATASETS:
        print(
            f"Warning: --eval-accuracy is only supported for {sorted(ACCURACY_DATASETS)}; "
            f"skipping accuracy for '{args.dataset}'."
        )
        eval_acc = False

    dataset = load_and_process_dataset(args.dataset, need_answers=eval_acc)
    dataset = _limit_dataset(dataset, args.max_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Only reserve an extra concurrency-sized slice when it will actually be
    # consumed by the warmup pass.  Otherwise --no-warmup would benchmark
    # args.num_prompts + args.concurrency requests while reporting only
    # args.num_prompts.
    warmup_count = 0 if args.no_warmup else max(args.concurrency, 1)
    num_prompts = args.num_prompts + warmup_count
    prompts: list[str] = []
    ref_answers: list[str] = []
    for i in range(num_prompts):
        item = dataset[i % len(dataset)]
        user_content = item["turns"][0]
        if args.dataset == "gsm8k":
            user_content += GSM8K_REASONING_INSTRUCTION
        prompts.append(
            _apply_chat_template(
                tokenizer,
                [{"role": "user", "content": user_content}],
                args.enable_thinking,
            )
        )
        if eval_acc:
            ref_answers.append(item.get("answer", ""))

    def send_one(prompt: str, *, max_new_tokens: int) -> dict:
        return _send_sglang(
            args.base_url,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            timeout_s=args.timeout_s,
        )

    try:
        requests.get(args.base_url.rstrip("/") + "/flush_cache", timeout=60).raise_for_status()
    except Exception:
        print("Warning: /flush_cache failed. Continuing.")

    bs = max(args.concurrency, 1)
    if not args.no_warmup and len(prompts) > bs:
        print(
            f"[warmup] {bs} requests, max_new_tokens={args.warmup_max_new_tokens} ..."
        )
        with ThreadPoolExecutor(max_workers=bs) as pool:
            list(
                pool.map(
                    lambda p: send_one(p, max_new_tokens=args.warmup_max_new_tokens),
                    prompts[:bs],
                )
            )
        prompts = prompts[bs:]
        if eval_acc:
            ref_answers = ref_answers[bs:]

    print(f"Running benchmark: {args.num_prompts} prompts, concurrency={args.concurrency} ...")
    start = time.perf_counter()
    total_tokens = 0
    spec_verify_ct_sum = 0
    spec_accept_lengths: list[float] = []
    spec_accept_rates: list[float] = []
    spec_verify_lens: list[float] = []
    spec_draft_times: list[float] = []
    spec_target_times: list[float] = []
    per_req_latencies: list[float] = []
    correct, total_eval = 0, 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(send_one, p, max_new_tokens=args.max_new_tokens): (i, time.perf_counter())
            for i, p in enumerate(prompts)
        }
        for fut in tqdm(as_completed(futures), total=len(prompts), desc="Benchmarking"):
            idx, t0 = futures[fut]
            per_req_latencies.append(time.perf_counter() - t0)
            out = fut.result()
            meta = out.get("meta_info", {}) or {}
            total_tokens += int(meta.get("completion_tokens", 0))
            spec_verify_ct_sum += int(meta.get("spec_verify_ct", 0))
            for key, lst in [
                ("spec_accept_length", spec_accept_lengths),
                ("spec_accept_rate", spec_accept_rates),
                ("spec_verify_len", spec_verify_lens),
                ("spec_draft_time_ms", spec_draft_times),
                ("spec_verify_time_ms", spec_target_times),
            ]:
                if key in meta:
                    try:
                        lst.append(float(meta[key]))
                    except (TypeError, ValueError):
                        pass
            if eval_acc and idx < len(ref_answers):
                text = out.get("text", "")
                if args.dataset == "gsm8k":
                    pred = _extract_gsm8k_prediction(text)
                elif args.dataset in ("math500", "aime25"):
                    pred = _extract_boxed_answer(text) or _extract_gsm8k_prediction(text)
                else:
                    pred = _extract_math_qa_prediction(text)
                if _answers_match(pred, ref_answers[idx], args.dataset):
                    correct += 1
                total_eval += 1

    latency = time.perf_counter() - start
    toks_per_s = total_tokens / max(latency, 1e-6)

    print(f"\n{'=' * 50}")
    print("Backend:          sglang")
    print(f"Dataset:          {args.dataset}")
    print(f"Num prompts:      {args.num_prompts}")
    print(f"Concurrency:      {args.concurrency}")
    print(f"Total latency:    {latency:.1f}s")
    if per_req_latencies:
        print(f"Avg req latency:  {statistics.mean(per_req_latencies):.2f}s")
    print(f"Output tokens:    {total_tokens}")
    print(f"Avg TPS:          {toks_per_s:,.2f} tok/s")
    if spec_accept_lengths:
        print(f"Avg Accept Len:   {statistics.mean(spec_accept_lengths):.3f}")
    if spec_accept_rates:
        print(f"Avg Accept Rate:  {statistics.mean(spec_accept_rates):.3f}")
    if spec_verify_lens:
        print(f"Avg Verify Len:   {statistics.mean(spec_verify_lens):.3f}")
    if spec_verify_ct_sum > 0:
        print(f"Spec verify ct:   {spec_verify_ct_sum}")
    if spec_draft_times:
        print(f"Draft time:       {sum(spec_draft_times):.1f} ms")
    if spec_target_times:
        print(f"Target time:      {sum(spec_target_times):.1f} ms")
    if eval_acc and total_eval > 0:
        print(f"Accuracy:         {correct}/{total_eval} ({correct / total_eval * 100:.1f}%)")
    print(f"{'=' * 50}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DFlash SGLang HTTP benchmark")
    parser.add_argument("--model", type=str, required=True, help="Target model id (for chat template).")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=sorted(LOCAL_DATASETS.keys()),
        help="Local test split under test_data/.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--warmup-max-new-tokens",
        type=int,
        default=64,
        help="max_new_tokens for pre-benchmark warmup requests (default: 64).",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the concurrency-sized warmup pass before timing.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, default=None, help="Cap rows loaded from JSONL.")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:30000")
    parser.add_argument("--num-prompts", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--eval-accuracy",
        action="store_true",
        help="Evaluate answer accuracy (supported: gsm8k, math500, aime25, math_qa).",
    )
    parser.add_argument("--timeout-s", type=int, default=3600)
    args = parser.parse_args()

    if args.enable_thinking and any(x in args.model.lower() for x in ["qwen3-4b", "qwen3-8b"]):
        print(
            "Thinking mode is enabled.",
            flush=True,
        )

    run_benchmark(args)


if __name__ == "__main__":
    main()
