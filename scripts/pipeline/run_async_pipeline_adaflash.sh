#!/bin/bash
# Async training-inference pipeline launcher (RKL training worker with adaptive length head).
#
# Same layout as run_async_pipeline_rkl.sh but with adaptive length head training enabled:
#   1. inference_worker.py  – SGLang Engine on INFERENCE_GPUS
#   2. training_worker_rkl.py – CE + reverse-KL mix + adaptive length head on TRAINING_GPUS (torchrun)
#
# Usage:
#   bash run_async_pipeline_adaflash.sh [INFERENCE_GPUS] [TRAINING_GPUS]
#
# Defaults:
#   INFERENCE_GPUS = "4"
#   TRAINING_GPUS  = "5,6,7"
#
# Optional env:
#   TRAIN_TP_SIZE               tensor parallel for training target (default: NUM_TRAIN_GPUS)
#   TRAIN_MAX_LENGTH              also passed as --sglang-context-length to cap KV pool
#   THRESH_HEAD_THRESHOLD_RATE  DFlash direct_len: passed to inference_worker as
#                               --thresh-head-threshold-rate (default 1.3)
#   THRESH_HEAD_LR              training: AdamW lr for thresh_head* only (default 2e-4; draft uses TRAIN_LR)
#   MAX_DRAFT_VERSION           see run_async_pipeline_rkl.sh (stop_pipeline before final version.txt)

set -euo pipefail

_LIB_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/../lib" &>/dev/null && pwd)
# shellcheck source=common.sh
source "$_LIB_DIR/common.sh"
# shellcheck source=pipeline_env.sh
source "$_LIB_DIR/pipeline_env.sh"
cd "$ASYN_TRAIN_ROOT"

# ---------------------------------------------------------------------------
# GPU assignment
# ---------------------------------------------------------------------------
INFERENCE_GPUS="${1:-2}"
TRAINING_GPUS="${2:-3,4}"
NUM_TRAIN_GPUS=$(echo "$TRAINING_GPUS" | tr ',' '\n' | wc -l)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SWAP_DIR="${SWAP_DIR:-$ASYN_TRAIN_ROOT/swap_draft_only}"
DATA_BUFFER="$SWAP_DIR/data_buffer"
MODEL_WEIGHTS="$SWAP_DIR/model_weights"
LOG_DIR="${LOG_DIR:-$ASYN_TRAIN_ROOT/logs_draft_only}"
mkdir -p "$DATA_BUFFER" "$MODEL_WEIGHTS" "$LOG_DIR"
rm -f "$SWAP_DIR/stop_pipeline"

# ---------------------------------------------------------------------------
# Configurable parameters
# ---------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
INITIAL_DRAFT_PATH="${INITIAL_DRAFT_PATH:-models/Qwen3-8B-DFlash-two-model-256-perfectblend-4096}"
INITIAL_DRAFT="$INITIAL_DRAFT_PATH"
# Use adaptive length head config
DRAFT_CONFIG_PATH="${DRAFT_CONFIG_PATH:-$ASYN_TRAIN_ROOT/configs/qwen3-8b-dflash-thresh-head.json}"
# DATASET_PATH="${DATASET_PATH:-$ASYN_TRAIN_ROOT/cache/dataset/perfectblend_train.jsonl}"
# DATASET_PATH="${DATASET_PATH:-$ASYN_TRAIN_ROOT/cache/dataset/gsm8k_train.jsonl}"
DATASET_PATH="${DATASET_PATH:-$ASYN_TRAIN_ROOT/cache/dataset/math_qa_train.jsonl}"
DATASET_START_LINE="${DATASET_START_LINE:-1}"

# Inference settings
INFER_MAX_NEW_TOKENS="${INFER_MAX_NEW_TOKENS:-2048}"
INFER_TEMPERATURE="${INFER_TEMPERATURE:-0.0}"
INFER_MEM_FRACTION="${INFER_MEM_FRACTION:-0.80}"
THRESH_HEAD_THRESHOLD_RATE="${THRESH_HEAD_THRESHOLD_RATE:-1.3}" # for adaptive length head inference

# Training settings
TRAIN_THRESHOLD="${TRAIN_THRESHOLD:-128}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
TRAIN_LR="${TRAIN_LR:-3e-4}"
TRAIN_MAX_LENGTH="${TRAIN_MAX_LENGTH:-2048}"
TRAIN_POLL_INTERVAL="${TRAIN_POLL_INTERVAL:-5}"
TRAIN_BUFFER_EPOCHS="${TRAIN_BUFFER_EPOCHS:-2}"
TRAIN_GRADIENT_ACCUMULATION_STEPS="${TRAIN_GRADIENT_ACCUMULATION_STEPS:-2}"
TRAIN_LOG_INTERVAL="${TRAIN_LOG_INTERVAL:-64}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-0}"

# Draft / attention
TRAIN_ATTENTION_BACKEND="${TRAIN_ATTENTION_BACKEND:-flex_attention}"
TRAIN_NUM_ANCHORS="${TRAIN_NUM_ANCHORS:-512}"
TRAIN_LOSS_DECAY_GAMMA="${TRAIN_LOSS_DECAY_GAMMA:-7.0}"

# RKL (defaults match train_rkl.sh)
# RKL_DIV_CLIP_TAU: unset → 0.01; export RKL_DIV_CLIP_TAU= → omit clip (no --rkl-div-clip-tau).
RKL_TEMPERATURE="${RKL_TEMPERATURE:-1.0}"
RKL_ALPHA="${RKL_ALPHA:-0.8}"
RKL_DIV_CLIP_TAU="${RKL_DIV_CLIP_TAU-0.01}"

# adaptive length head training
TRAIN_THRESH_HEAD="${TRAIN_THRESH_HEAD:-true}"
THRESH_HEAD_LOSS_TYPE="${THRESH_HEAD_LOSS_TYPE:-mse}"
THRESH_LABEL_LOOKAHEAD="${THRESH_LABEL_LOOKAHEAD:-1}"
THRESH_HEAD_LR="${THRESH_HEAD_LR:-2e-4}"
DETAILED_DEBUG_PRINT="${DETAILED_DEBUG_PRINT:-false}"

TARGET_MODEL_BACKEND="${TARGET_MODEL_BACKEND:-sglang}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.40}"
# Default tp=1 (dp=NUM_TRAIN_GPUS); override with TRAIN_TP_SIZE if target needs tensor parallelism.
TRAIN_TP_SIZE="${TRAIN_TP_SIZE:-1}"
TRAIN_DIST_TIMEOUT="${TRAIN_DIST_TIMEOUT:-30}"

if (( NUM_TRAIN_GPUS % TRAIN_TP_SIZE != 0 )); then
  echo "[launcher] ERROR: training GPU count ($NUM_TRAIN_GPUS) must be divisible by TRAIN_TP_SIZE ($TRAIN_TP_SIZE)" >&2
  exit 1
fi

ENABLE_THINKING="${ENABLE_THINKING:-0}"
THINK_FLAG=""
if [ "$ENABLE_THINKING" = "1" ]; then THINK_FLAG="--enable-thinking"; fi

RKL_CLIP_ARGS=()
if [[ -n "$RKL_DIV_CLIP_TAU" ]]; then
  RKL_CLIP_ARGS+=(--rkl-div-clip-tau "$RKL_DIV_CLIP_TAU")
fi

# adaptive length head args
THRESH_HEAD_ARGS=()
if [ "$TRAIN_THRESH_HEAD" = "true" ]; then
  THRESH_HEAD_ARGS+=(--train-thresh-head)
  THRESH_HEAD_ARGS+=(--thresh-head-loss-type "$THRESH_HEAD_LOSS_TYPE")
  THRESH_HEAD_ARGS+=(--thresh-label-lookahead "$THRESH_LABEL_LOOKAHEAD")
  THRESH_HEAD_ARGS+=(--thresh-head-learning-rate "$THRESH_HEAD_LR")
fi
if [ "$DETAILED_DEBUG_PRINT" = "true" ]; then
  THRESH_HEAD_ARGS+=(--detailed-debug-print)
fi

# Versioned draft snapshots + optional stop after reaching a draft version
DRAFT_SNAPSHOT_INTERVAL="${DRAFT_SNAPSHOT_INTERVAL:-0}"
DRAFT_MODELS_SUBDIR="${DRAFT_MODELS_SUBDIR:-draft_models}"
MAX_DRAFT_VERSION="${MAX_DRAFT_VERSION:-0}"
mkdir -p "$SWAP_DIR/$DRAFT_MODELS_SUBDIR"

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------
REPO_DIR=$(dirname "$ASYN_TRAIN_ROOT")

echo "============================================================"
echo "  Async DFlash Pipeline (RKL + adaptive length head worker)"
echo "  inference GPUs : $INFERENCE_GPUS"
echo "  training  GPUs : $TRAINING_GPUS  (${NUM_TRAIN_GPUS} GPU(s))"
echo "  swap dir       : $SWAP_DIR"
echo "  dataset        : $DATASET_PATH"
echo "  dataset start  : line $DATASET_START_LINE (1-based, inference only)"
echo "  draft config   : $DRAFT_CONFIG_PATH"
echo "  initial draft  : $INITIAL_DRAFT_PATH"
echo "  buffer epochs  : $TRAIN_BUFFER_EPOCHS (passes per buffer batch)"
echo "  grad accum     : $TRAIN_GRADIENT_ACCUMULATION_STEPS (micro-batches per optimizer.step)"
echo "  max buffer rnd : $TRAIN_MAX_STEPS (0 = no limit)"
echo "  train max len  : $TRAIN_MAX_LENGTH (sglang-context-length + max_total_tokens cap)"
echo "  train target   : $TARGET_MODEL_BACKEND (sglang-mem-fraction-static=$SGLANG_MEM_FRACTION_STATIC)"
echo "  train tp-size  : $TRAIN_TP_SIZE (default=NUM_TRAIN_GPUS; dp=$((NUM_TRAIN_GPUS / TRAIN_TP_SIZE)))"
echo "  RKL params     : T=$RKL_TEMPERATURE alpha=$RKL_ALPHA div_clip_tau=${RKL_DIV_CLIP_TAU:-<off>}"
echo "  adaptive length head    : train=$TRAIN_THRESH_HEAD loss=$THRESH_HEAD_LOSS_TYPE lookahead=$THRESH_LABEL_LOOKAHEAD"
echo "                   train lr: draft=$TRAIN_LR thresh_head=$THRESH_HEAD_LR debug=$DETAILED_DEBUG_PRINT"
echo "  infer thresh rate: $THRESH_HEAD_THRESHOLD_RATE (--thresh-head-threshold-rate)"
echo "  snapshot int   : $DRAFT_SNAPSHOT_INTERVAL (0=off)"
echo "  max draft ver  : $MAX_DRAFT_VERSION (0=no limit; stop_pipeline before final version.txt)"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. Inference worker (SGLang Engine)
# ---------------------------------------------------------------------------
# Match run_async_pipeline_rkl.sh: tee -> terminal + log. ``--loop`` re-reads the dataset;
# without it, inference writes stop_pipeline after one pass so training does not hang on the buffer.
INFER_LOG="$LOG_DIR/inference.log"
: >"$INFER_LOG"
setsid bash -c "CUDA_VISIBLE_DEVICES=\"$INFERENCE_GPUS\" python -u \"$BIN_DIR/inference_worker.py\" \
    --model-path          \"$MODEL_PATH\"           \
    --initial-draft-path  \"$INITIAL_DRAFT\"         \
    --dataset-path        \"$DATASET_PATH\"          \
    --dataset-start-line  \"$DATASET_START_LINE\"    \
    --swap-dir            \"$SWAP_DIR\"              \
    --max-new-tokens      \"$INFER_MAX_NEW_TOKENS\"  \
    --temperature         \"$INFER_TEMPERATURE\"     \
    --mem-fraction-static \"$INFER_MEM_FRACTION\"    \
    --thresh-head-threshold-rate \"$THRESH_HEAD_THRESHOLD_RATE\" \
    $THINK_FLAG \
    2>&1 | tee -a \"$INFER_LOG\"" &
INFER_PID=$!
echo "[launcher] inference_worker PID=$INFER_PID  (tee -> terminal + $INFER_LOG)"

sleep 5

# ---------------------------------------------------------------------------
# 2. Training worker (torchrun)
# ---------------------------------------------------------------------------
# Use same pattern as run_async_pipeline_rkl.sh: --standalone avoids a fixed TCPStore
# port (run_async_pipeline_adaflash used to hardcode --master_port=29501, which
# fails with EADDRINUSE if another job holds that port).
TRAIN_CMD=(torchrun --standalone --nproc_per_node "$NUM_TRAIN_GPUS")
CUDA_VISIBLE_DEVICES="$TRAINING_GPUS" "${TRAIN_CMD[@]}" \
  "$BIN_DIR/training_worker_rkl.py" \
  --target-model-path "$MODEL_PATH" \
  --target-model-backend "$TARGET_MODEL_BACKEND" \
  --draft-config-path "$DRAFT_CONFIG_PATH" \
  --initial-draft-path "$INITIAL_DRAFT_PATH" \
  --swap-dir "$SWAP_DIR" \
  --train-threshold "$TRAIN_THRESHOLD" \
  --batch-size "$TRAIN_BATCH_SIZE" \
  --learning-rate "$TRAIN_LR" \
  --max-length "$TRAIN_MAX_LENGTH" \
  --poll-interval "$TRAIN_POLL_INTERVAL" \
  --buffer-epochs "$TRAIN_BUFFER_EPOCHS" \
  --gradient-accumulation-steps "$TRAIN_GRADIENT_ACCUMULATION_STEPS" \
  --log-interval "$TRAIN_LOG_INTERVAL" \
  --max-steps "$TRAIN_MAX_STEPS" \
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
  --draft-snapshot-interval "$DRAFT_SNAPSHOT_INTERVAL" \
  --draft-models-subdir "$DRAFT_MODELS_SUBDIR" \
  --max-draft-version "$MAX_DRAFT_VERSION" \
  $THINK_FLAG \
  >"$LOG_DIR/training.log" 2>&1 &
TRAIN_PID=$!
echo "[launcher] training_worker_rkl PID=$TRAIN_PID  (log: $LOG_DIR/training.log)"

# ---------------------------------------------------------------------------
# Wait and forward Ctrl-C to both children
# ---------------------------------------------------------------------------
_launcher_kill_tree() {
  local _pid=${1:-} _sig=${2:-TERM}
  [ -z "$_pid" ] && return 0
  kill -0 "$_pid" 2>/dev/null || return 0
  local _c
  for _c in $(pgrep -P "$_pid" 2>/dev/null); do
    _launcher_kill_tree "$_c" "$_sig"
  done
  kill -s "$_sig" "$_pid" 2>/dev/null || true
}

cleanup() {
  echo ""
  echo "[launcher] Stopping workers (inference=$INFER_PID, training=$TRAIN_PID)..."
  touch "$SWAP_DIR/stop_pipeline" 2>/dev/null || true
  _launcher_kill_tree "$INFER_PID" TERM
  _launcher_kill_tree "$TRAIN_PID" TERM
  sleep 2
  _launcher_kill_tree "$INFER_PID" KILL
  _launcher_kill_tree "$TRAIN_PID" KILL
  wait "$INFER_PID" 2>/dev/null || true
  wait "$TRAIN_PID" 2>/dev/null || true
  echo "[launcher] All workers stopped."
}
trap cleanup INT TERM

echo "[launcher] Both workers running. Press Ctrl-C to stop."
echo "[launcher] Inference logs also on this terminal; training -> $LOG_DIR/training.log"
set +e
wait "$INFER_PID"
infer_ec=$?
wait "$TRAIN_PID"
train_ec=$?
set -e
if [ "$infer_ec" -ne 0 ] || [ "$train_ec" -ne 0 ]; then
  echo "[launcher] A worker exited: inference=$infer_ec training=$train_ec (see logs above)"
fi
echo "[launcher] All workers finished."
