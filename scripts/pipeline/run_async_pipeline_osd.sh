#!/bin/bash
# Async training-inference pipeline launcher.
#
# Usage:
#   bash run_pipeline.sh [INFERENCE_GPUS] [TRAINING_GPUS]
#
# Defaults:
#   INFERENCE_GPUS = "0"          (one GPU for inference)
#   TRAINING_GPUS  = "1,2,3"      (three GPUs for training; adjust freely)
#
# The script launches two background processes:
#   1. inference_worker.py  – SGLang Engine on INFERENCE_GPUS
#   2. training_worker_sft.py   – vendored specforge fine-tuning on TRAINING_GPUS
#
# Both processes share SWAP_DIR (default: swap_sft under this folder); logs under LOG_DIR (default: logs_sft).
# Press Ctrl-C to stop both. Workers are started under ``setsid`` so each tree
# has its own process group; cleanup kills by **group** so SGLang/torchrun
# children are not left behind after ``kill`` on the parent PID only.
#
# Optional env (training_worker_sft, same as run_async_pipeline_rkl):
#   TRAIN_TP_SIZE            tensor parallel for training target (default: 1, dp=NUM_TRAIN_GPUS)
#   TRAIN_MAX_LENGTH           also passed as --sglang-context-length to cap KV pool
#   SWAP_DIR                 swap root for inference_worker + training_worker_sft (default: $ASYN_TRAIN_ROOT/swap_sft)
#   LOG_DIR                  directory for launcher logs inference.log / training.log (default: $ASYN_TRAIN_ROOT/logs_sft)
#   FROM_GROUND_TRUTH        set to 1 to pass --from-ground-truth (see training worker).
#
# inference_stats.jsonl: draft_version 0 = --initial-draft-path; N = engine draft after hot-swap.

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
INFERENCE_GPUS="${1:-1}"
TRAINING_GPUS="${2:-2,3}"
NUM_TRAIN_GPUS=$(echo "$TRAINING_GPUS" | tr ',' '\n' | wc -l)

# ---------------------------------------------------------------------------
# Paths (override SWAP_DIR / LOG_DIR via env before invoking this script)
# ---------------------------------------------------------------------------
SWAP_DIR="${SWAP_DIR:-$ASYN_TRAIN_ROOT/swap_sft}"
LOG_DIR="${LOG_DIR:-$ASYN_TRAIN_ROOT/logs_sft}"
DATA_BUFFER="$SWAP_DIR/data_buffer"
MODEL_WEIGHTS="$SWAP_DIR/model_weights"
mkdir -p "$DATA_BUFFER" "$MODEL_WEIGHTS" "$LOG_DIR"
rm -f "$SWAP_DIR/stop_pipeline"

# ---------------------------------------------------------------------------
# Configurable parameters – edit here or pass as env vars before calling
# ---------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
INITIAL_DRAFT_PATH="${INITIAL_DRAFT_PATH:-${INITIAL_DRAFT:-z-lab/Qwen3-8B-DFlash-b16}}"
INITIAL_DRAFT="$INITIAL_DRAFT_PATH"
DRAFT_CONFIG_PATH="${DRAFT_CONFIG_PATH:-$ASYN_TRAIN_ROOT/configs/qwen3-8b-dflash.json}"
DATASET_PATH="${DATASET_PATH:-$ASYN_TRAIN_ROOT/cache/dataset/perfectblend_train.jsonl}"
DATASET_START_LINE="${DATASET_START_LINE:-1}"

# Inference settings
INFER_MAX_NEW_TOKENS="${INFER_MAX_NEW_TOKENS:-2048}"
INFER_TEMPERATURE="${INFER_TEMPERATURE:-0.0}"
INFER_MEM_FRACTION="${INFER_MEM_FRACTION:-0.80}"

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
TRAIN_ATTENTION_BACKEND="${TRAIN_ATTENTION_BACKEND:-flex_attention}"
TARGET_MODEL_BACKEND="${TARGET_MODEL_BACKEND:-sglang}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.40}"
# Default tp=1 (dp=NUM_TRAIN_GPUS); override with TRAIN_TP_SIZE if target needs tensor parallelism.
TRAIN_TP_SIZE="${TRAIN_TP_SIZE:-1}"
TRAIN_DIST_TIMEOUT="${TRAIN_DIST_TIMEOUT:-30}"

if (( NUM_TRAIN_GPUS % TRAIN_TP_SIZE != 0 )); then
  echo "[launcher] ERROR: training GPU count ($NUM_TRAIN_GPUS) must be divisible by TRAIN_TP_SIZE ($TRAIN_TP_SIZE)" >&2
  exit 1
fi
# Teacher logits KD in training_worker_sft (OnlineDFlashModel): T and alpha for soft vs hard CE.
# alpha=0 => loss is only soft KL to teacher distribution (no hard CE term).
# Set TEACHER_KD_DISABLE=1 to omit flags (legacy weighted hard CE only).
TEACHER_KD_DISABLE="${TEACHER_KD_DISABLE:-1}"
TEACHER_KD_TEMPERATURE="${TEACHER_KD_TEMPERATURE:-1.0}"
TEACHER_KD_ALPHA="${TEACHER_KD_ALPHA:-0}"

ENABLE_THINKING="${ENABLE_THINKING:-0}"
THINK_FLAG=""
if [ "$ENABLE_THINKING" = "1" ]; then THINK_FLAG="--enable-thinking"; fi

# Train on dataset reference (question + ground_truth) instead of prompt + model response.
FROM_GROUND_TRUTH="${FROM_GROUND_TRUTH:-0}"
GROUND_TRUTH_FLAG=""
if [ "$FROM_GROUND_TRUTH" = "1" ]; then GROUND_TRUTH_FLAG="--from-ground-truth"; fi

TEACHER_KD_FLAGS=()
if [ "$TEACHER_KD_DISABLE" != "1" ]; then
  TEACHER_KD_FLAGS+=(--teacher-kd-temperature "$TEACHER_KD_TEMPERATURE" --teacher-kd-alpha "$TEACHER_KD_ALPHA")
fi

DRAFT_SNAPSHOT_INTERVAL="${DRAFT_SNAPSHOT_INTERVAL:-0}"
DRAFT_MODELS_SUBDIR="${DRAFT_MODELS_SUBDIR:-draft_models}"
MAX_DRAFT_VERSION="${MAX_DRAFT_VERSION:-201}"
mkdir -p "$SWAP_DIR/$DRAFT_MODELS_SUBDIR"

# ---------------------------------------------------------------------------
# Env (sglang from pip install -e; specforge via pipeline.bootstrap in Python)
# ---------------------------------------------------------------------------
# shellcheck source=shell/lib/pipeline_env.sh
source "$ASYN_TRAIN_ROOT/shell/lib/pipeline_env.sh"

echo "============================================================"
echo "  Async DFlash Pipeline"
echo "  inference GPUs : $INFERENCE_GPUS"
echo "  training  GPUs : $TRAINING_GPUS  (${NUM_TRAIN_GPUS} GPU(s))"
echo "  swap dir       : $SWAP_DIR"
echo "  log dir        : $LOG_DIR"
echo "  dataset        : $DATASET_PATH"
echo "  dataset start  : line $DATASET_START_LINE (1-based, inference only)"
echo "  initial draft  : $INITIAL_DRAFT_PATH"
echo "  buffer epochs  : $TRAIN_BUFFER_EPOCHS (passes per buffer batch)"
echo "  grad accum      : $TRAIN_GRADIENT_ACCUMULATION_STEPS (micro-batches per optimizer.step)"
echo "  max buffer rnd : $TRAIN_MAX_STEPS (0 = no limit)"
echo "  train max len  : $TRAIN_MAX_LENGTH (sglang-context-length + max_total_tokens cap)"
echo "  train target   : $TARGET_MODEL_BACKEND (sglang-mem-fraction-static=$SGLANG_MEM_FRACTION_STATIC)"
echo "  train tp-size  : $TRAIN_TP_SIZE (default=NUM_TRAIN_GPUS; dp=$((NUM_TRAIN_GPUS / TRAIN_TP_SIZE)))"
echo "  attention      : $TRAIN_ATTENTION_BACKEND"
echo "  enable thinking : $ENABLE_THINKING (1 = --enable-thinking on both workers)"
echo "  from_ground_truth: $FROM_GROUND_TRUTH (1 = --from-ground-truth; SFT uses question+ground_truth in buffer)"
if [ "$TEACHER_KD_DISABLE" = "1" ]; then
  echo "  teacher KD      : disabled (hard CE only)"
else
  echo "  teacher KD      : T=$TEACHER_KD_TEMPERATURE alpha=$TEACHER_KD_ALPHA (alpha=0 => soft only)"
fi
echo "  draft snapshots : every ${DRAFT_SNAPSHOT_INTERVAL} versions -> $SWAP_DIR/$DRAFT_MODELS_SUBDIR/<v> (0=off)"
echo "  max draft ver   : ${MAX_DRAFT_VERSION} (0=no auto-stop; stop_pipeline before final version.txt)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Launch inference worker (terminal + tee to log; -u for line-wise output)
# ---------------------------------------------------------------------------
# Without ``--loop``, inference exits after one pass and writes stop_pipeline so
# training does not wait forever on an empty buffer (see inference_worker + training_worker_sft).
INFER_LOG="$LOG_DIR/inference.log"
: > "$INFER_LOG"
setsid bash -c "CUDA_VISIBLE_DEVICES=\"$INFERENCE_GPUS\" python -u \"$BIN_DIR/inference_worker.py\" \
    --model-path          \"$MODEL_PATH\"           \
    --initial-draft-path  \"$INITIAL_DRAFT\"         \
    --dataset-path        \"$DATASET_PATH\"          \
    --dataset-start-line  \"$DATASET_START_LINE\"    \
    --swap-dir            \"$SWAP_DIR\"              \
    --max-new-tokens      \"$INFER_MAX_NEW_TOKENS\"  \
    --temperature         \"$INFER_TEMPERATURE\"     \
    --mem-fraction-static \"$INFER_MEM_FRACTION\"    \
    $THINK_FLAG \
    2>&1 | tee -a \"$INFER_LOG\"" &
INFER_PID=$!
echo "[launcher] inference_worker PID=$INFER_PID  (tee -> terminal + $INFER_LOG)"

# ---------------------------------------------------------------------------
# Launch training worker (always torchrun: sglang target needs init_process_group env)
# ---------------------------------------------------------------------------
TRAIN_CMD="torchrun --standalone --nproc_per_node $NUM_TRAIN_GPUS"

# Unbuffered Python so training.log gets INFO lines as soon as they are emitted.
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$TRAINING_GPUS" setsid $TRAIN_CMD "$BIN_DIR/training_worker_sft.py" \
    --target-model-path  "$MODEL_PATH"           \
    --target-model-backend "$TARGET_MODEL_BACKEND" \
    --tp-size            "$TRAIN_TP_SIZE"         \
    --dist-timeout        "$TRAIN_DIST_TIMEOUT"   \
    --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC" \
    --sglang-context-length "$TRAIN_MAX_LENGTH" \
    --draft-config-path  "$DRAFT_CONFIG_PATH"    \
    --initial-draft-path "$INITIAL_DRAFT"         \
    --attention-backend  "$TRAIN_ATTENTION_BACKEND" \
    --swap-dir           "$SWAP_DIR"             \
    --train-threshold    "$TRAIN_THRESHOLD"       \
    --batch-size         "$TRAIN_BATCH_SIZE"      \
    --learning-rate      "$TRAIN_LR"              \
    --max-length         "$TRAIN_MAX_LENGTH"      \
    --poll-interval      "$TRAIN_POLL_INTERVAL"   \
    --buffer-epochs      "$TRAIN_BUFFER_EPOCHS"   \
    --gradient-accumulation-steps "$TRAIN_GRADIENT_ACCUMULATION_STEPS" \
    --log-interval       "$TRAIN_LOG_INTERVAL"    \
    --max-steps          "$TRAIN_MAX_STEPS"       \
    --draft-snapshot-interval "$DRAFT_SNAPSHOT_INTERVAL" \
    --draft-models-subdir "$DRAFT_MODELS_SUBDIR" \
    --max-draft-version "$MAX_DRAFT_VERSION" \
    $THINK_FLAG \
    $GROUND_TRUTH_FLAG \
    "${TEACHER_KD_FLAGS[@]}" \
    > "$LOG_DIR/training.log" 2>&1 &
TRAIN_PID=$!
echo "[launcher] training_worker_sft  PID=$TRAIN_PID  (log: $LOG_DIR/training.log)"

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
    wait "$INFER_PID"  2>/dev/null || true
    wait "$TRAIN_PID" 2>/dev/null || true
    echo "[launcher] All workers stopped."
}
trap cleanup INT TERM

echo "[launcher] Both workers running. Press Ctrl-C to stop."
echo "[launcher] Inference logs also on this terminal; training -> $LOG_DIR/training.log"
# wait returns non-zero if a child failed; with set -e that would exit this script early.
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
