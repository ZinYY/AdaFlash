#!/usr/bin/env bash
# Serve Qwen/Qwen3-8B with SGLang (standard autoregressive decoding, no speculative decoding).

PORT=${PORT:-6784}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
# MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-9B}
# MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-Coder-30B-A3B-Instruct}
TP_SIZE=${TP_SIZE:-1}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.80}

export SGLANG_DISABLE_CUDNN_CHECK=1

python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tp-size "${TP_SIZE}" \
    --dtype bfloat16 \
    --attention-backend fa3 \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --trust-remote-code \
    --skip-server-warmup \
    "$@"
