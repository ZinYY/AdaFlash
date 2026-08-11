#!/usr/bin/env python3
# coding=utf-8
"""Download HF datasets and write train/test JSONL for async training and benchmarks.

Train output: ``asyn_train/cache/dataset/<dataset>_train.jsonl``
Test output:  ``asyn_train/test_data/<dataset>_test.jsonl``

Output schema (one JSON object per line):
{
    "id": str,
    "conversations": [{"role": "user"|"assistant", "content": str}, ...]
}
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from tqdm import tqdm

from datasets import Dataset, load_dataset

from pipeline.paths import CACHE_DIR, TEST_DATA_DIR

# `datasets`>=3: use Parquet-only snapshot for repos that dropped loading scripts.
_HF_PARQUET_ONLY_REV = "refs/convert/parquet"

DATASET_CHOICES = [
    "math_qa",
    "gsm8k",
    "opencodeinstruct",
    "codealpaca-20k",
    "math500",
    "sharegpt",
    "perfectblend",
]

# Datasets whose train/test split is taken from contiguous index ranges on a single split.
_INDEX_SPLIT_DATASETS = frozenset({"opencodeinstruct", "sharegpt", "perfectblend"})
_TRAIN_SIZE = 30_000
_TEST_SIZE = 1_024

ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "bing": "assistant",
    "bard": "assistant",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare train/test JSONL under cache/dataset/ and test_data/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bin/prepare_data.py
  python bin/prepare_data.py --dataset math_qa
  python bin/prepare_data.py --force

Split rules:
  If HuggingFace provides both train and test splits, use them as-is.
  Otherwise, split from the available data:
    opencodeinstruct, sharegpt, perfectblend:
        train = rows 1-30000, test = rows 30001-31024
    codealpaca-20k: last 1024 rows as test, the rest as train
    math500: train = rows 1-450, test = rows 451-500
        """,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=DATASET_CHOICES,
        help="Dataset to prepare (default: all supported datasets)",
    )
    parser.add_argument(
        "--train-output-path",
        type=str,
        default=None,
        help=f"Train JSONL directory (default: {CACHE_DIR / 'dataset'})",
    )
    parser.add_argument(
        "--test-output-path",
        type=str,
        default=None,
        help=f"Test JSONL directory (default: {TEST_DATA_DIR})",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Custom dataset path (sharegpt only)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output JSONL files if present",
    )
    return parser.parse_args()


def load_dataset_from_path(data_path: Path) -> Dataset:
    suffix = data_path.suffix.lstrip(".")
    return load_dataset(suffix, data_files=str(data_path), split="train")


def split_dataset(ds: Dataset, dataset_name: str) -> Tuple[Dataset, Dataset]:
    n = len(ds)
    if dataset_name in _INDEX_SPLIT_DATASETS:
        train_end = min(_TRAIN_SIZE, n)
        test_start = _TRAIN_SIZE
        test_end = min(_TRAIN_SIZE + _TEST_SIZE, n)
        train_ds = ds.select(range(train_end))
        test_ds = ds.select(range(test_start, test_end)) if test_end > test_start else ds.select([])
    elif dataset_name == "codealpaca-20k":
        test_size = min(_TEST_SIZE, n)
        train_ds = ds.select(range(n - test_size))
        test_ds = ds.select(range(n - test_size, n))
    elif dataset_name == "math500":
        train_ds = ds.select(range(min(450, n)))
        test_ds = ds.select(range(450, min(500, n))) if n > 450 else ds.select([])
    else:
        raise ValueError(f"No custom split rule for dataset '{dataset_name}'")

    print(
        f"Custom split {dataset_name}: {len(train_ds)} train, {len(test_ds)} test "
        f"(from {n} total rows)"
    )
    return train_ds, test_ds


def resolve_train_test_splits(
    hf_ds, dataset_name: str
) -> Tuple[Dataset, Dataset]:
    """Use official HF train/test when both exist; otherwise apply custom rules."""
    if "train" in hf_ds and "test" in hf_ds:
        train_ds = hf_ds["train"]
        test_ds = hf_ds["test"]
        print(
            f"Using {dataset_name} official HF split: "
            f"{len(train_ds)} train, {len(test_ds)} test"
        )
        return train_ds, test_ds

    if "train" in hf_ds:
        source = hf_ds["train"]
    elif "test" in hf_ds:
        source = hf_ds["test"]
    else:
        split_name = next(iter(hf_ds.keys()))
        source = hf_ds[split_name]
        print(f"Using {dataset_name} HF split '{split_name}' as source")

    return split_dataset(source, dataset_name)


def _write_jsonl(
    ds: Dataset,
    output_path: Path,
    proc_fn: Optional[Callable],
    dataset_name: str,
    desc: str,
) -> int:
    skipped_count = 0
    with open(output_path, "w") as f:
        for item in tqdm(ds, desc=desc):
            if proc_fn is not None:
                row, item_skipped = proc_fn(item, dataset_name)
                if row is None:
                    continue
                skipped_count += item_skipped
            else:
                row = item
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return skipped_count


def process_and_save_ds(
    train_ds: Dataset,
    test_ds: Dataset,
    train_output_path: Path,
    test_output_path: Path,
    proc_fn: Optional[Callable],
    dataset_name: str,
    *,
    force: bool = False,
) -> None:
    train_output_jsonl_path = train_output_path / f"{dataset_name}_train.jsonl"
    test_output_jsonl_path = test_output_path / f"{dataset_name}_test.jsonl"

    if (
        train_output_jsonl_path.exists()
        and test_output_jsonl_path.exists()
        and not force
    ):
        print(
            f"Dataset {dataset_name} already exists at\n"
            f"  train: {train_output_jsonl_path}\n"
            f"  test:  {test_output_jsonl_path}\n"
            f"Skipping... (use --force to overwrite)"
        )
        return

    train_output_path.mkdir(parents=True, exist_ok=True)
    test_output_path.mkdir(parents=True, exist_ok=True)

    total_skipped_count = 0
    total_skipped_count += _write_jsonl(
        train_ds,
        train_output_jsonl_path,
        proc_fn,
        dataset_name,
        f"Writing {dataset_name} train",
    )
    total_skipped_count += _write_jsonl(
        test_ds,
        test_output_jsonl_path,
        proc_fn,
        dataset_name,
        f"Writing {dataset_name} test",
    )

    print(f"Wrote train JSONL to {train_output_jsonl_path}")
    print(f"Wrote test JSONL to {test_output_jsonl_path}")

    if total_skipped_count > 0:
        total_messages = len(train_ds) + len(test_ds)
        print(
            f"Skipped {total_skipped_count}/{total_messages} messages for {dataset_name}"
        )


def process_sharegpt_row(row: Dict, dataset_name: str = None) -> Tuple[Dict, int]:
    conversations = row["conversations"]
    formatted_conversations = []
    skipped_count = 0
    for message in conversations:
        if message["from"] not in ROLE_MAPPING:
            skipped_count += 1
            continue
        new_role = ROLE_MAPPING[message["from"]]
        content = message["value"]
        formatted_conversations.append({"role": new_role, "content": content})

    row = {"id": row["id"], "conversations": formatted_conversations}
    return row, skipped_count


def process_codealpaca_row(row: Dict, dataset_name: str = None) -> Tuple[Dict, int]:
    row_id = hashlib.md5((row["instruction"] + row["output"]).encode()).hexdigest()
    processed_row = {
        "id": row_id,
        "conversations": [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["output"]},
        ],
    }
    return processed_row, 0


def process_opencodeinstruct_row(
    row: Dict, dataset_name: str = None
) -> Tuple[Dict, int]:
    row_id = row.get("id")
    if row_id is None:
        row_id = hashlib.md5((row["input"] + row["output"]).encode()).hexdigest()

    processed_row = {
        "id": row_id,
        "conversations": [
            {"role": "user", "content": row["input"]},
            {"role": "assistant", "content": row["output"]},
        ],
    }
    return processed_row, 0


def process_gsm8k_row(row: Dict, dataset_name: str = None) -> Tuple[Dict, int]:
    row_id = hashlib.md5((row["question"] + row["answer"]).encode()).hexdigest()
    processed_row = {
        "id": row_id,
        "conversations": [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ],
    }
    return processed_row, 0


def process_hendrycks_math_row(row: Dict, dataset_name: str = None) -> Tuple[Dict, int]:
    row_id = hashlib.md5((row["problem"] + row["solution"]).encode()).hexdigest()
    processed_row = {
        "id": row_id,
        "conversations": [
            {"role": "user", "content": row["problem"]},
            {"role": "assistant", "content": row["solution"]},
        ],
    }
    return processed_row, 0


def process_math_qa_row(row: Dict, dataset_name: str = None) -> Tuple[Dict, int]:
    problem = row["Problem"]
    options = row["options"]
    user_content = f"{problem}\n{options}"
    rationale = row["Rationale"]

    row_id = hashlib.md5((user_content + rationale).encode()).hexdigest()
    processed_row = {
        "id": row_id,
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": rationale},
        ],
    }
    return processed_row, 0


def add_index(row, idx) -> Dict:
    row["id"] = idx
    return row


def load_raw_dataset(
    dataset_name: str,
    data_path: Optional[str] = None,
) -> Tuple[Dataset, Dataset, Optional[Callable]]:
    """Return (train_ds, test_ds, proc_fn)."""
    if dataset_name == "sharegpt":
        if data_path is None:
            hf_ds = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered")
        else:
            print(f"Loading sharegpt from custom data path: {data_path}")
            hf_ds = {"train": load_dataset_from_path(Path(data_path))}
        train_ds, test_ds = resolve_train_test_splits(hf_ds, dataset_name)
        return train_ds, test_ds, process_sharegpt_row

    if dataset_name == "perfectblend":
        ds = load_dataset("mlabonne/open-perfectblend")["train"]
        ds = ds.map(add_index, with_indices=True)
        train_ds, test_ds = resolve_train_test_splits({"train": ds}, dataset_name)
        return train_ds, test_ds, process_sharegpt_row

    if dataset_name == "gsm8k":
        hf_ds = load_dataset("openai/gsm8k", "main")
        train_ds, test_ds = resolve_train_test_splits(hf_ds, dataset_name)
        return train_ds, test_ds, process_gsm8k_row

    if dataset_name == "math500":
        hf_ds = load_dataset("HuggingFaceH4/MATH-500")
        train_ds, test_ds = resolve_train_test_splits(hf_ds, dataset_name)
        return train_ds, test_ds, process_hendrycks_math_row

    if dataset_name == "math_qa":
        hf_ds = load_dataset("allenai/math_qa", revision=_HF_PARQUET_ONLY_REV)
        train_ds, test_ds = resolve_train_test_splits(hf_ds, dataset_name)
        return train_ds, test_ds, process_math_qa_row

    if dataset_name == "codealpaca-20k":
        hf_ds = load_dataset("sahil2801/CodeAlpaca-20k", revision=_HF_PARQUET_ONLY_REV)
        train_ds, test_ds = resolve_train_test_splits(hf_ds, dataset_name)
        return train_ds, test_ds, process_codealpaca_row

    if dataset_name == "opencodeinstruct":
        hf_ds = load_dataset("nvidia/OpenCodeInstruct", revision=_HF_PARQUET_ONLY_REV)
        train_ds, test_ds = resolve_train_test_splits(hf_ds, dataset_name)
        return train_ds, test_ds, process_opencodeinstruct_row

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def main() -> None:
    args = parse_args()
    datasets = [args.dataset] if args.dataset else DATASET_CHOICES

    train_output_path = (
        Path(args.train_output_path) if args.train_output_path else CACHE_DIR / "dataset"
    )
    test_output_path = (
        Path(args.test_output_path) if args.test_output_path else TEST_DATA_DIR
    )

    if len(datasets) > 1:
        print(f"Preparing {len(datasets)} datasets: {', '.join(datasets)}")

    for dataset_name in datasets:
        if len(datasets) > 1:
            print(f"\n=== {dataset_name} ===")
        train_ds, test_ds, proc_fn = load_raw_dataset(dataset_name, args.data_path)
        process_and_save_ds(
            train_ds,
            test_ds,
            train_output_path,
            test_output_path,
            proc_fn,
            dataset_name,
            force=args.force,
        )


if __name__ == "__main__":
    main()
