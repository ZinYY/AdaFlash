# Data Preparation

AdaFlash uses normalized JSONL files for asynchronous inference, training, and HTTP benchmarking. The preparation utility downloads supported datasets from Hugging Face and writes both a training split and a benchmark split.

All commands below assume that the current working directory is the repository root.

## Prepare a Dataset

```bash
python bin/prepare_data.py --dataset <name>
```

By default, the command writes:

```text
cache/dataset/<name>_train.jsonl
test_data/<name>_test.jsonl
```

Use `--force` to overwrite both files. For a quick pipeline check, use `--sample-size N` to keep only the first `N` training examples.

```bash
python bin/prepare_data.py --dataset gsm8k
python bin/prepare_data.py --dataset math500
python bin/prepare_data.py --dataset perfectblend --force
python bin/prepare_data.py --dataset math_qa --sample-size 128
```

Run the following command for the authoritative list of dataset names and dataset-specific options:

```bash
python bin/prepare_data.py --help
```

## Supported Dataset Groups

The preparation code includes loaders for the following workload families.

### General instruction and chat

- `ultrachat`
- `sharegpt`
- `eaglechat`
- `magpie-qwen2.5-pro-1m-v0.1`

### Mathematics and reasoning

- `gsm8k`
- `math500` (`HuggingFaceH4/MATH-500`, 500 test examples)
- `hendrycks_math`
- `math_qa`

### Science

- `sciq`
- `camel`

### Code

- `codealpaca-20k`
- `opencodeinstruct`
- `magicoder-evol-instruct`
- `opc` with the dataset-specific `--opc-subset` option

### Mixed workloads

- `perfectblend`
- the `perfectblend-*` variants exposed by `--help`

Dataset downloading and normalization are implemented in `pipeline/data/prepare_data.py`, based on the corresponding SpecForge data-preparation workflow.

## JSONL Format

Prepared rows use a conversation-oriented schema whenever the source dataset supports it:

```json
{
  "id": "example-id",
  "conversations": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

Some dataset-specific fields may be preserved when they are needed for evaluation. The benchmark loader also accepts rows with `turns`, `question`, `query`, `instruction`, `input`, or `prompt` fields.

## Regenerate Training Responses

`regenerate_dataset` runs an existing question set through a target model and, by default, a DFlash drafter. It writes a training JSONL file containing `prompt`, `response`, and `question` fields compatible with the asynchronous training buffer.

```bash
CUDA_VISIBLE_DEVICES=0 python bin/regenerate_dataset.py \
  --model-path Qwen/Qwen3-8B \
  --initial-draft-path z-lab/Qwen3-8B-DFlash-b16 \
  --dataset-path cache/dataset/perfectblend_train.jsonl \
  --dataset-start-line 1 \
  --num-samples 4096 \
  --output-jsonl cache/dataset/perfectblend_qwen3_8b_regen_4096.jsonl
```

### Main regeneration options

| Option | Description |
|---|---|
| `--model-path` | Target autoregressive model or local checkpoint |
| `--initial-draft-path` | Initial DFlash drafter or local checkpoint |
| `--dataset-path` | Input JSONL file, usually a prepared `*_train.jsonl` file |
| `--dataset-start-line` | One-based input line at which processing begins |
| `--num-samples` | Maximum number of examples to process |
| `--output-jsonl` | Destination JSONL path, relative to the repository root unless absolute |
| `--no-speculative` | Disable DFlash and generate with the target model only |
| `--swap-dir` | Optionally hot-reload draft weights using `model_weights/version.txt` |
| `--concurrency` | Number of in-flight `async_generate` requests on one engine |

The default generation settings match the asynchronous OSD inference path:

```text
max_new_tokens=2048
temperature=0.0
mem_fraction_static=0.80
concurrency=1
```

Increase `--concurrency` only after confirming that the target and draft models leave enough device memory for the expected KV cache and verification tensors.

The implementation lives in `pipeline/inference/regenerate_dataset.py`.

## Using Prepared Data

Use the training split with an async launcher:

```bash
export DATASET_PATH="$PWD/cache/dataset/gsm8k_train.jsonl"
bash scripts/pipeline/run_async_pipeline_adaflash.sh "0" "1,2"
```

Use the test split with the HTTP benchmark client:

```bash
python bin/benchmark.py \
  --base-url http://127.0.0.1:6784 \
  --model Qwen/Qwen3-8B \
  --dataset gsm8k \
  --num-prompts 1024 \
  --concurrency 64 \
  --eval-accuracy
```

For the complete serving and benchmark workflow, see [benchmark_experiments.md](benchmark_experiments.md).
