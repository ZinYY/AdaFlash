#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

PORT=${PORT:-6784}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-Coder-30B-A3B-Instruct}
DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH:-lmsys/SGLang-EAGLE3-Qwen3-Coder-30B-A3B-Instruct-SpecForge}
TP_SIZE=${TP_SIZE:-1}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.80}

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
# EAGLE-3 speculative knobs (SGLang cookbook defaults; safe with --attention-backend fa3)
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
