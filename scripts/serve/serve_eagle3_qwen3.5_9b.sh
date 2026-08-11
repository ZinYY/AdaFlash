#!/usr/bin/env bash

# https://github.com/sgl-project/sglang/pull/20104

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

PORT=${PORT:-6784}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-9B}
DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH:-BLR2/Qwen3.5-9B-Eagle3-ShareGPT}
TP_SIZE=${TP_SIZE:-1}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.80}

export SGLANG_DISABLE_CUDNN_CHECK=1
SPEC_NUM_STEPS=${SPEC_NUM_STEPS:-5}
SPEC_EAGLE_TOPK=${SPEC_EAGLE_TOPK:-4}
SPEC_NUM_DRAFT_TOKENS=${SPEC_NUM_DRAFT_TOKENS:-16}
SPEC_TIMING=${SPEC_TIMING:-0}

python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path "${DRAFT_MODEL_PATH}" \
    --speculative-num-steps "${SPEC_NUM_STEPS}" \
    --speculative-eagle-topk "${SPEC_EAGLE_TOPK}" \
    --speculative-num-draft-tokens "${SPEC_NUM_DRAFT_TOKENS}" \
    --mamba-scheduler-strategy extra_buffer \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tp-size "${TP_SIZE}" \
    --dtype bfloat16 \
    --attention-backend fa3 \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --trust-remote-code \
    --skip-server-warmup \
    $([ "${SPEC_TIMING}" = "1" ] && echo "--spec-timing") \
    "$@"
