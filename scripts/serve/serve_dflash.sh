SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

PORT=${PORT:-6784}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH:-z-lab/Qwen3-8B-DFlash-b16}
# MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-9B}
# DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH:-z-lab/Qwen3.5-9B-DFlash}
# MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-Coder-30B-A3B-Instruct}
# DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH:-z-lab/Qwen3-Coder-30B-A3B-DFlash}
TP_SIZE=${TP_SIZE:-1}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.80}

QWEN35_COMPAT_ARGS=()
if [[ "${MODEL_PATH,,}" == *"qwen3.5"* ]]; then
    export SGLANG_DISABLE_CUDNN_CHECK="${SGLANG_DISABLE_CUDNN_CHECK:-1}"
    export SGLANG_ENABLE_SPEC_V2="${SGLANG_ENABLE_SPEC_V2:-1}"
    QWEN35_COMPAT_ARGS+=(--mamba-scheduler-strategy "${MAMBA_SCHEDULER_STRATEGY:-extra_buffer}")
fi

python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path "${DRAFT_MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tp-size "${TP_SIZE}" \
    --dtype bfloat16 \
    --attention-backend fa3 \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --trust-remote-code \
    --skip-server-warmup \
    "${QWEN35_COMPAT_ARGS[@]}" \
    "$@"
