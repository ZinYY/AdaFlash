#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

PORT=${PORT:-6784}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-9B}
TP_SIZE=${TP_SIZE:-1}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.80}

export SGLANG_DISABLE_CUDNN_CHECK=1
SPEC_NUM_STEPS=${SPEC_NUM_STEPS:-3}
SPEC_EAGLE_TOPK=${SPEC_EAGLE_TOPK:-1}
SPEC_NUM_DRAFT_TOKENS=${SPEC_NUM_DRAFT_TOKENS:-4}
SPEC_TIMING=${SPEC_TIMING:-0}

python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-9B \
    --port "${PORT}" \
    --tp-size "${TP_SIZE}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --context-length 262144 \
    --reasoning-parser qwen3 \
    --speculative-algo NEXTN \
    --speculative-num-steps "${SPEC_NUM_STEPS}" \
    --speculative-eagle-topk "${SPEC_EAGLE_TOPK}" \
    --speculative-num-draft-tokens "${SPEC_NUM_DRAFT_TOKENS}" \
    --mamba-scheduler-strategy extra_buffer \
    --host 0.0.0.0 \
    --trust-remote-code \
    --skip-server-warmup \
    $([ "${SPEC_TIMING}" = "1" ] && echo "--spec-timing") \
    "$@"
