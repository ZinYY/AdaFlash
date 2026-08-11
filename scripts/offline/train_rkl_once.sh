#!/usr/bin/env bash
# One-shot offline RKL + thresh-head training on a static JSONL dataset.
# Matches a single buffer round of run_async_pipeline_adaflash.sh
# (same hyperparameters; no inference worker / data_buffer polling).
#
# Usage:
#   bash scripts/offline/train_rkl_once.sh [TRAINING_GPUS]
#
# Example (math500 regen, 2 epochs):
#   export DATASET_PATH=cache/dataset/math500_train_qwen3_8b_regen.jsonl
#   export OUTPUT_DIR=temp/qwen3/math500/offline_rkl_once/model_weights
#   bash scripts/offline/train_rkl_once.sh 3,4
#
# Optional env:
#   MODEL_PATH, INITIAL_DRAFT_PATH (unset/empty = from scratch), DRAFT_CONFIG_PATH, DATASET_PATH, OUTPUT_DIR
#   TRAIN_BUFFER_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_LR, TRAIN_MAX_LENGTH
#   TRAIN_GRADIENT_ACCUMULATION_STEPS, TRAIN_ATTENTION_BACKEND, TRAIN_NUM_ANCHORS
#   TRAIN_LOSS_DECAY_GAMMA, RKL_TEMPERATURE, RKL_ALPHA, RKL_DIV_CLIP_TAU
#   TRAIN_THRESH_HEAD, THRESH_HEAD_LOSS_TYPE, THRESH_LABEL_LOOKAHEAD, THRESH_HEAD_LR
#   TARGET_MODEL_BACKEND, SGLANG_MEM_FRACTION_STATIC, TRAIN_TP_SIZE, TRAIN_DIST_TIMEOUT
#   ENABLE_THINKING, MAX_SAMPLES, MASTER_PORT, TRAIN_LOG_INTERVAL, TRAIN_SAVE_INTERVAL
#   TRAIN_LAZY_DATASET (default true; set false for eager full-dataset preprocess)

set -euo pipefail

_LIB_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/../lib" &>/dev/null && pwd)
# shellcheck source=common.sh
source "$_LIB_DIR/common.sh"
cd "$ASYN_TRAIN_ROOT"

TRAINING_GPUS="${1:-3,4}"
NUM_TRAIN_GPUS=$(echo "$TRAINING_GPUS" | tr ',' '\n' | wc -l)

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
INITIAL_DRAFT_PATH="${INITIAL_DRAFT_PATH-}"
DRAFT_CONFIG_PATH="${DRAFT_CONFIG_PATH:-$ASYN_TRAIN_ROOT/configs/qwen3-8b-dflash-thresh-head.json}"
DATASET_PATH="${DATASET_PATH:-$ASYN_TRAIN_ROOT/cache/dataset/math500_train_qwen3_8b_regen.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ASYN_TRAIN_ROOT/temp/qwen3/math500/offline_rkl_once/model_weights}"

TRAIN_BUFFER_EPOCHS="${TRAIN_BUFFER_EPOCHS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
TRAIN_LR="${TRAIN_LR:-3e-4}"
TRAIN_MAX_LENGTH="${TRAIN_MAX_LENGTH:-2048}"
TRAIN_GRADIENT_ACCUMULATION_STEPS="${TRAIN_GRADIENT_ACCUMULATION_STEPS:-2}"
TRAIN_LOG_INTERVAL="${TRAIN_LOG_INTERVAL:-25}"
TRAIN_SAVE_INTERVAL="${TRAIN_SAVE_INTERVAL:-0}"
TRAIN_LAZY_DATASET="${TRAIN_LAZY_DATASET:-true}"
TRAIN_ATTENTION_BACKEND="${TRAIN_ATTENTION_BACKEND:-flex_attention}"
TRAIN_NUM_ANCHORS="${TRAIN_NUM_ANCHORS:-512}"
TRAIN_LOSS_DECAY_GAMMA="${TRAIN_LOSS_DECAY_GAMMA:-7.0}"

RKL_TEMPERATURE="${RKL_TEMPERATURE:-1.0}"
RKL_ALPHA="${RKL_ALPHA:-0.8}"
RKL_DIV_CLIP_TAU="${RKL_DIV_CLIP_TAU-0.01}"

TRAIN_THRESH_HEAD="${TRAIN_THRESH_HEAD:-true}"
THRESH_HEAD_LOSS_TYPE="${THRESH_HEAD_LOSS_TYPE:-mse}"
THRESH_LABEL_LOOKAHEAD="${THRESH_LABEL_LOOKAHEAD:-1}"
THRESH_HEAD_LR="${THRESH_HEAD_LR:-2e-4}"
DETAILED_DEBUG_PRINT="${DETAILED_DEBUG_PRINT:-false}"

TARGET_MODEL_BACKEND="${TARGET_MODEL_BACKEND:-sglang}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.40}"
TRAIN_TP_SIZE="${TRAIN_TP_SIZE:-1}"
TRAIN_DIST_TIMEOUT="${TRAIN_DIST_TIMEOUT:-30}"

ENABLE_THINKING="${ENABLE_THINKING:-0}"
THINK_FLAG=()
if [ "$ENABLE_THINKING" = "1" ]; then
  THINK_FLAG=(--enable-thinking)
fi

RKL_CLIP_ARGS=()
if [[ -n "$RKL_DIV_CLIP_TAU" ]]; then
  RKL_CLIP_ARGS+=(--rkl-div-clip-tau "$RKL_DIV_CLIP_TAU")
fi

THRESH_HEAD_ARGS=()
if [ "$TRAIN_THRESH_HEAD" = "true" ]; then
  THRESH_HEAD_ARGS+=(--train-thresh-head)
  THRESH_HEAD_ARGS+=(--thresh-head-loss-type "$THRESH_HEAD_LOSS_TYPE")
  THRESH_HEAD_ARGS+=(--thresh-label-lookahead "$THRESH_LABEL_LOOKAHEAD")
  THRESH_HEAD_ARGS+=(--thresh-head-learning-rate "$THRESH_HEAD_LR")
else
  THRESH_HEAD_ARGS+=(--no-train-thresh-head)
fi
if [ "$DETAILED_DEBUG_PRINT" = "true" ]; then
  THRESH_HEAD_ARGS+=(--detailed-debug-print)
fi

MAX_SAMPLES_ARGS=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  MAX_SAMPLES_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

INITIAL_DRAFT_ARGS=()
if [[ -n "$INITIAL_DRAFT_PATH" ]]; then
  INITIAL_DRAFT_ARGS+=(--initial-draft-path "$INITIAL_DRAFT_PATH")
fi

LAZY_DATASET_ARGS=(--lazy-dataset)
if [ "$TRAIN_LAZY_DATASET" = "false" ]; then
  LAZY_DATASET_ARGS=(--no-lazy-dataset)
fi

mkdir -p "$(dirname "$OUTPUT_DIR")"

echo "============================================================"
echo "  Offline RKL + adaptive length head (single pass)"
echo "  training GPUs  : $TRAINING_GPUS (${NUM_TRAIN_GPUS} GPU(s))"
echo "  dataset        : $DATASET_PATH"
echo "  output         : $OUTPUT_DIR"
echo "  draft config   : $DRAFT_CONFIG_PATH"
echo "  initial draft  : ${INITIAL_DRAFT_PATH:-<from scratch>}"
echo "  buffer epochs  : $TRAIN_BUFFER_EPOCHS"
echo "  grad accum     : $TRAIN_GRADIENT_ACCUMULATION_STEPS"
echo "  log interval   : $TRAIN_LOG_INTERVAL (optimizer steps; 0=off)"
echo "  save interval  : $TRAIN_SAVE_INTERVAL (optimizer steps; 0=off, overwrite -> output)"
echo "  lazy dataset   : $TRAIN_LAZY_DATASET (on-demand tokenize; false=eager preprocess)"
echo "  RKL            : T=$RKL_TEMPERATURE alpha=$RKL_ALPHA div_clip_tau=${RKL_DIV_CLIP_TAU:-<off>}"
echo "  adaptive length head    : train=$TRAIN_THRESH_HEAD loss=$THRESH_HEAD_LOSS_TYPE lr=$THRESH_HEAD_LR"
echo "============================================================"

TORCHRUN=(torchrun --standalone --nproc_per_node "$NUM_TRAIN_GPUS" --master_port="${MASTER_PORT:-29522}")

CUDA_VISIBLE_DEVICES="$TRAINING_GPUS" "${TORCHRUN[@]}" \
  "$BIN_DIR/offline_train_rkl_once.py" \
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
  "${LAZY_DATASET_ARGS[@]}" \
  --attention-backend "$TRAIN_ATTENTION_BACKEND" \
  --num-anchors "$TRAIN_NUM_ANCHORS" \
  --loss-decay-gamma "$TRAIN_LOSS_DECAY_GAMMA" \
  --rkl-temperature "$RKL_TEMPERATURE" \
  --rkl-alpha "$RKL_ALPHA" \
  "${RKL_CLIP_ARGS[@]}" \
  "${THRESH_HEAD_ARGS[@]}" \
  --tp-size "$TRAIN_TP_SIZE" \
  --dist-timeout "$TRAIN_DIST_TIMEOUT" \
  --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC" \
  --sglang-context-length "$TRAIN_MAX_LENGTH" \
  "${THINK_FLAG[@]}" \
  "${MAX_SAMPLES_ARGS[@]}"

echo "[offline_rkl] Finished. Weights -> $OUTPUT_DIR"
