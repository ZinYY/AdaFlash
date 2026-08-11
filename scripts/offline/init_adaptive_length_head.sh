#!/usr/bin/env bash
# Offline thresh-head init: default target backend is sglang (same as training_worker_rkl).
# Use torchrun --nproc_per_node=1 so init_distributed matches tp-size=1.
set -euo pipefail
_LIB_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/../lib" &>/dev/null && pwd)
# shellcheck source=common.sh
source "$_LIB_DIR/common.sh"
cd "$ASYN_TRAIN_ROOT"

TORCHRUN=(torchrun --standalone --nproc_per_node=1 --master_port="${MASTER_PORT:-29521}")

# "${TORCHRUN[@]}" "$BIN_DIR/offline_init_adaptive_length_head.py" \
#     --target-model-path Qwen/Qwen3-8B \
#     --draft-config-path configs/qwen3-8b-dflash-thresh-head.json \
#     --initial-draft-path z-lab/Qwen3-8B-DFlash-b16 \
#     --dataset-path cache/dataset/perfectblend_qwen3_8b_regen_4096.jsonl \
#     --output-dir models/Qwen3-8B-DFlash-two-model-256-perfectblend-4096 \
#     --sglang-mem-fraction-static 0.40

# "${TORCHRUN[@]}" "$BIN_DIR/offline_init_adaptive_length_head.py" \
#     --target-model-path Qwen/Qwen3-Coder-30B-A3B-Instruct \
#     --draft-config-path configs/qwen3-coder-30b-dflash-thresh-head.json \
#     --initial-draft-path z-lab/Qwen3-Coder-30B-A3B-DFlash \
#     --dataset-path cache/dataset/perfectblend_qwen3_coder_30b_a3b_regen_4096.jsonl \
#     --output-dir models/Qwen3-Coder-30B-A3B-DFlash-two-model-256-perfectblend-4096 \
#     --sglang-mem-fraction-static 0.50

"${TORCHRUN[@]}" "$BIN_DIR/offline_init_adaptive_length_head.py" \
    --target-model-path Qwen/Qwen3.5-9B \
    --draft-config-path configs/qwen3.5-9b-dflash-thresh-head.json \
    --initial-draft-path z-lab/Qwen3.5-9B-DFlash \
    --dataset-path cache/dataset/perfectblend_qwen3_5_9b_regen_4096.jsonl \
    --output-dir models/Qwen3.5-9B-DFlash-two-model-256-perfectblend-4096 \
    --sglang-mem-fraction-static 0.40
