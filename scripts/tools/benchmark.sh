#!/usr/bin/env bash
set -euo pipefail
_LIB_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/../lib" &>/dev/null && pwd)
# shellcheck source=common.sh
source "$_LIB_DIR/common.sh"
export PYTHONPATH="${REPO_ROOT}/sglang_dflash/python${PYTHONPATH:+:$PYTHONPATH}"
cd "$ASYN_TRAIN_ROOT"

export PORT=$(( ( RANDOM % 10000 )  + 20000 ))
BASE_URL="http://127.0.0.1:${PORT}"
CANDIDATE_LEN_MIN=${CANDIDATE_LEN_MIN:-1}
SERVER_STARTUP_TIMEOUT_S=${SERVER_STARTUP_TIMEOUT_S:-180}
DYNAMIC_VERIFY_LEN=${DYNAMIC_VERIFY_LEN:-1}
DYNAMIC_VERIFY_EMA_ALPHA=${DYNAMIC_VERIFY_EMA_ALPHA:-0.3}
NUM_SAMPLES=${NUM_SAMPLES:-1024}

DATASET=${DATASET:-perfectblend}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH:-${ASYN_TRAIN_ROOT}/models/Qwen3-8B-DFlash-two-model-256-perfectblend-4096}
CONCURRENCY=${CONCURRENCY:-64}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.80}
THRESHOLD_RATE=${THRESHOLD_RATE:-1.3}

if [[ -n "${MAX_RUNNING_REQUESTS:-}" ]]; then
    echo "MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS} (manual override)"
    export MAX_RUNNING_REQUESTS
elif [[ "${CONCURRENCY}" -lt 48 ]]; then
    MAX_RUNNING_REQUESTS=48
    echo "MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS} (CONCURRENCY=${CONCURRENCY} < 48)"
    export MAX_RUNNING_REQUESTS
elif [[ "${MODEL_PATH,,}" == *"qwen3.5"* ]]; then
    MAX_RUNNING_REQUESTS=96
    echo "MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS} (Qwen3.5, CONCURRENCY=${CONCURRENCY})"
    export MAX_RUNNING_REQUESTS
else
    echo "MAX_RUNNING_REQUESTS=<sglang auto> (CONCURRENCY=${CONCURRENCY})"
    unset MAX_RUNNING_REQUESTS
fi

if [[ "${MODEL_PATH,,}" == *"qwen3.5"* ]] && [[ "${CONCURRENCY}" -gt 16 ]]; then
    SPECULATIVE_NUM_DRAFT_TOKENS=12
    export SPECULATIVE_NUM_DRAFT_TOKENS
fi

lsof -ti :"${PORT}" | xargs -r kill -9 2>/dev/null || true
sleep 2

SPEC_TIMING=${SPEC_TIMING:-0}
DFLASH_DEBUG_PRINT=${DFLASH_DEBUG_PRINT:-0}
export MODEL_PATH DRAFT_MODEL_PATH DFLASH_DEBUG_PRINT MEM_FRACTION_STATIC
[[ -n "${MAX_RUNNING_REQUESTS:-}" ]] && export MAX_RUNNING_REQUESTS
setsid bash "${ASYN_TRAIN_ROOT}/scripts/serve/serve_thresh_head.sh" \
    "${CANDIDATE_LEN_MIN}" \
    "${SPEC_TIMING}" \
    "${THRESHOLD_RATE}" \
    "${DYNAMIC_VERIFY_LEN}" \
    "${DYNAMIC_VERIFY_EMA_ALPHA}" &
SERVER_PID=$!
trap 'kill -- -${SERVER_PID} 2>/dev/null || true; lsof -ti :"${PORT}" | xargs -r kill -9 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true' EXIT INT TERM

for ((i=0; i<SERVER_STARTUP_TIMEOUT_S; i++)); do
    if curl -fsS -m 2 "${BASE_URL}/get_model_info" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if ! curl -fsS -m 2 "${BASE_URL}/get_model_info" >/dev/null 2>&1; then
    echo "Server failed to start within ${SERVER_STARTUP_TIMEOUT_S}s." >&2
    exit 1
fi

python "$BIN_DIR/benchmark.py" \
    --base-url "${BASE_URL}" \
    --model "${MODEL_PATH}" \
    --dataset "${DATASET}" \
    --max-samples "${NUM_SAMPLES}" \
    --num-prompts "${NUM_SAMPLES}" \
    --concurrency "${CONCURRENCY}" \
    "$@"
