#!/usr/bin/env bash
# DFLASH thresh-head serving notes:
# - The adaptive length head is only used for inference when both conditions hold:
#   1. --dflash-dynamic-verify-len is enabled (DYNAMIC_VERIFY_LEN=1 here).
#   2. The final SGLang max_running_requests is > 48.
# - If dynamic verify len is off, or max_running_requests <= 48, SGLang ignores
#   thresh-head weights and runs plain fixed-window DFlash (verify len = block_size).

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

PORT=${PORT:-6784}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH:-}
TP_SIZE=${TP_SIZE:-1}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.80}

CANDIDATE_LEN_MIN=${1:-1}
SPEC_TIMING=${2:-0}
THRESHOLD_RATE=${3:-1.0}
DYNAMIC_VERIFY_LEN=${4:-0}
DYNAMIC_VERIFY_EMA_ALPHA=${5:-0.3}
HARD_CLIP_SCALE=${HARD_CLIP_SCALE:-1.0}
SPECULATIVE_NUM_DRAFT_TOKENS=${SPECULATIVE_NUM_DRAFT_TOKENS:-16}
DFLASH_DEBUG_PRINT=${DFLASH_DEBUG_PRINT:-0}

QWEN35_COMPAT_ARGS=()
if [[ "${MODEL_PATH,,}" == *"qwen3.5"* ]]; then
    export SGLANG_DISABLE_CUDNN_CHECK="${SGLANG_DISABLE_CUDNN_CHECK:-1}"
    QWEN35_COMPAT_ARGS+=(--mamba-scheduler-strategy "${MAMBA_SCHEDULER_STRATEGY:-extra_buffer}")
fi

${PYTHON:-python} -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path "${DRAFT_MODEL_PATH}" \
    --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tp-size "${TP_SIZE}" \
    --dtype bfloat16 \
    --attention-backend fa3 \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    $([[ -n "${MAX_RUNNING_REQUESTS:-}" ]] && echo "--max-running-requests ${MAX_RUNNING_REQUESTS}") \
    --trust-remote-code \
    --skip-server-warmup \
    --prob-head-candidate-len-min "${CANDIDATE_LEN_MIN}" \
    --thresh-head-threshold-rate "${THRESHOLD_RATE}" \
    "${QWEN35_COMPAT_ARGS[@]}" \
    $([ "${DYNAMIC_VERIFY_LEN}" = "1" ] && echo "--dflash-dynamic-verify-len --dflash-dynamic-verify-ema-alpha ${DYNAMIC_VERIFY_EMA_ALPHA} --dflash-hard-clip-scale ${HARD_CLIP_SCALE}") \
    $([ "${SPEC_TIMING}" = "1" ] && echo "--spec-timing") \
    $([ "${DFLASH_DEBUG_PRINT}" = "1" ] && echo "--dflash-debug-print")
