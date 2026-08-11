#!/usr/bin/env bash
# Serve Qwen/Qwen3-8B with SGLang EAGLE-3 speculative decoding.
#
# Default draft: Tengyunw/qwen3_8b_eagle3 (HF). Train your own via SpecForge, e.g.
#   SpecForge/examples/run_qwen3_8b_eagle3_online.sh
#
# Optional overlap scheduler (experimental):
#   export SGLANG_ENABLE_SPEC_V2=1
#
# Higher-throughput tuning (see Tengyunw/qwen3_8b_eagle3 README):
#   SPEC_NUM_STEPS=6 SPEC_EAGLE_TOPK=10 SPEC_NUM_DRAFT_TOKENS=32 bash serve_eagle3.sh

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

PORT=${PORT:-6784}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH:-Tengyunw/qwen3_8b_eagle3}
TP_SIZE=${TP_SIZE:-1}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.80}

# EAGLE-3 speculative knobs (SGLang cookbook defaults; safe with --attention-backend fa3)
SPEC_NUM_STEPS=${SPEC_NUM_STEPS:-6}
SPEC_EAGLE_TOPK=${SPEC_EAGLE_TOPK:-10}
SPEC_NUM_DRAFT_TOKENS=${SPEC_NUM_DRAFT_TOKENS:-32}
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
