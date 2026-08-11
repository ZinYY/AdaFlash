#!/usr/bin/env bash
# One-shot offline SFT training on a static JSONL dataset.
# Matches a single buffer round of run_async_pipeline_osd.sh
# (same hyperparameters; no inference worker / data_buffer polling).
#
# Usage:
#   bash scripts/offline/train_sft_once.sh [TRAINING_GPUS]
#
# Example (regen JSONL, prompt+response):
#   export DATASET_PATH=cache/dataset/math500_train_qwen3_8b_regen.jsonl
#   export OUTPUT_DIR=temp/qwen3/math500/offline_sft_once/model_weights
#   bash scripts/offline/train_sft_once.sh 2,3
#
# Example (raw train JSONL with conversations):
#   export DATASET_PATH=cache/dataset/perfectblend_train.jsonl
#   export MAX_SAMPLES=450
#   bash scripts/offline/train_sft_once.sh 2,3
#
# Optional env:
#   MODEL_PATH, INITIAL_DRAFT_PATH (unset/empty = from scratch), DRAFT_CONFIG_PATH, DATASET_PATH, OUTPUT_DIR
#   TRAIN_BUFFER_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_LR, TRAIN_MAX_LENGTH
#   TRAIN_GRADIENT_ACCUMULATION_STEPS, TRAIN_ATTENTION_BACKEND
#   TARGET_MODEL_BACKEND, SGLANG_MEM_FRACTION_STATIC, TRAIN_TP_SIZE, TRAIN_DIST_TIMEOUT
#   FROM_GROUND_TRUTH, TEACHER_KD_DISABLE, TEACHER_KD_TEMPERATURE, TEACHER_KD_ALPHA
#   ENABLE_THINKING, MAX_SAMPLES, MASTER_PORT, TRAIN_LOG_INTERVAL

set -euo pipefail

_LIB_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/../lib" &>/dev/null && pwd)
# shellcheck source=common.sh
source "$_LIB_DIR/common.sh"
cd "$ASYN_TRAIN_ROOT"

TRAINING_GPUS="${1:-2,3}"
NUM_TRAIN_GPUS=$(echo "$TRAINING_GPUS" | tr ',' '\n' | wc -l)

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
INITIAL_DRAFT_PATH="${INITIAL_DRAFT_PATH-}"
DRAFT_CONFIG_PATH="${DRAFT_CONFIG_PATH:-$ASYN_TRAIN_ROOT/configs/qwen3-8b-dflash.json}"
DATASET_PATH="${DATASET_PATH:-$ASYN_TRAIN_ROOT/cache/dataset/math500_train_qwen3_8b_regen.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ASYN_TRAIN_ROOT/temp/qwen3/math500/offline_sft_once/model_weights}"

TRAIN_BUFFER_EPOCHS="${TRAIN_BUFFER_EPOCHS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
TRAIN_LR="${TRAIN_LR:-3e-4}"
TRAIN_MAX_LENGTH="${TRAIN_MAX_LENGTH:-2048}"
TRAIN_GRADIENT_ACCUMULATION_STEPS="${TRAIN_GRADIENT_ACCUMULATION_STEPS:-2}"
TRAIN_LOG_INTERVAL="${TRAIN_LOG_INTERVAL:-25}"
TRAIN_SAVE_INTERVAL="${TRAIN_SAVE_INTERVAL:-0}"
TRAIN_ATTENTION_BACKEND="${TRAIN_ATTENTION_BACKEND:-flex_attention}"

TARGET_MODEL_BACKEND="${TARGET_MODEL_BACKEND:-sglang}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.40}"
TRAIN_TP_SIZE="${TRAIN_TP_SIZE:-1}"
TRAIN_DIST_TIMEOUT="${TRAIN_DIST_TIMEOUT:-30}"

TEACHER_KD_DISABLE="${TEACHER_KD_DISABLE:-1}"
TEACHER_KD_TEMPERATURE="${TEACHER_KD_TEMPERATURE:-1.0}"
TEACHER_KD_ALPHA="${TEACHER_KD_ALPHA:-0}"

FROM_GROUND_TRUTH="${FROM_GROUND_TRUTH:-0}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"

THINK_FLAG=()
if [ "$ENABLE_THINKING" = "1" ]; then
  THINK_FLAG=(--enable-thinking)
fi

GROUND_TRUTH_FLAG=()
if [ "$FROM_GROUND_TRUTH" = "1" ]; then
  GROUND_TRUTH_FLAG=(--from-ground-truth)
fi

TEACHER_KD_FLAGS=()
if [ "$TEACHER_KD_DISABLE" != "1" ]; then
  TEACHER_KD_FLAGS+=(--teacher-kd-temperature "$TEACHER_KD_TEMPERATURE" --teacher-kd-alpha "$TEACHER_KD_ALPHA")
fi

MAX_SAMPLES_ARGS=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  MAX_SAMPLES_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

INITIAL_DRAFT_ARGS=()
if [[ -n "$INITIAL_DRAFT_PATH" ]]; then
  INITIAL_DRAFT_ARGS+=(--initial-draft-path "$INITIAL_DRAFT_PATH")
fi

mkdir -p "$(dirname "$OUTPUT_DIR")"

echo "============================================================"
echo "  Offline SFT (single pass)"
echo "  training GPUs  : $TRAINING_GPUS (${NUM_TRAIN_GPUS} GPU(s))"
echo "  dataset        : $DATASET_PATH"
echo "  output         : $OUTPUT_DIR"
echo "  draft config   : $DRAFT_CONFIG_PATH"
echo "  initial draft  : ${INITIAL_DRAFT_PATH:-<from scratch>}"
echo "  buffer epochs  : $TRAIN_BUFFER_EPOCHS"
echo "  grad accum     : $TRAIN_GRADIENT_ACCUMULATION_STEPS"
echo "  log interval   : $TRAIN_LOG_INTERVAL (optimizer steps; 0=off)"
echo "  save interval  : $TRAIN_SAVE_INTERVAL (optimizer steps; 0=off, overwrite -> output)"
echo "  from_ground_truth: $FROM_GROUND_TRUTH"
if [ "$TEACHER_KD_DISABLE" = "1" ]; then
  echo "  teacher KD      : disabled (hard CE only)"
else
  echo "  teacher KD      : T=$TEACHER_KD_TEMPERATURE alpha=$TEACHER_KD_ALPHA"
fi
echo "============================================================"

TORCHRUN=(torchrun --standalone --nproc_per_node "$NUM_TRAIN_GPUS" --master_port="${MASTER_PORT:-29523}")

CUDA_VISIBLE_DEVICES="$TRAINING_GPUS" "${TORCHRUN[@]}" \
  "$BIN_DIR/offline_train_sft_once.py" \
  --target-model-path "$MODEL_PATH" \
  --target-model-backend "$TARGET_MODEL_BACKEND" \
  --draft-config-path "$DRAFT_CONFIG_PATH" \
  "${INITIAL_DRAFT_ARGS[@]}" \
  --dataset-path "$DATASET_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --buffer-epochs "$TRAIN_BUFFER_EPOCHS" \
  --batch-size "$TRAIN_BATCH_SIZE" \
  --learning-rate "$TRAIN_LR" \
  --max-length "$TRAIN_MAX_LENGTH" \
  --gradient-accumulation-steps "$TRAIN_GRADIENT_ACCUMULATION_STEPS" \
  --log-interval "$TRAIN_LOG_INTERVAL" \
  --save-interval "$TRAIN_SAVE_INTERVAL" \
  --attention-backend "$TRAIN_ATTENTION_BACKEND" \
  --tp-size "$TRAIN_TP_SIZE" \
  --dist-timeout "$TRAIN_DIST_TIMEOUT" \
  --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC" \
  "${THINK_FLAG[@]}" \
  "${GROUND_TRUTH_FLAG[@]}" \
  "${TEACHER_KD_FLAGS[@]}" \
  "${MAX_SAMPLES_ARGS[@]}"

echo "[offline_sft] Finished. Weights -> $OUTPUT_DIR"
