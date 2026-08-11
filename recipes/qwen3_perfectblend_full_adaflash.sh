# Qwen3-8B, temperature = 0 (greedy decoding) AdaFlash

_LIB_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/../scripts/lib" &>/dev/null && pwd)
# shellcheck source=common.sh
source "$_LIB_DIR/common.sh"

# perfectblend full
export MODEL_PATH=Qwen/Qwen3-8B
export INITIAL_DRAFT_PATH=models/Qwen3-8B-DFlash-two-model-256-perfectblend-4096
export DRAFT_CONFIG_PATH=$ASYN_TRAIN_ROOT/configs/qwen3-8b-dflash-thresh-head.json
export DATASET_PATH=$ASYN_TRAIN_ROOT/cache/dataset/perfectblend_full.jsonl
export SWAP_DIR=$ASYN_TRAIN_ROOT/models/perfectblend_full_rkl_two_model
export LOG_DIR=$ASYN_TRAIN_ROOT/logs_perfectblend_full_rkl_two_model
export INFER_TEMPERATURE=0.0
export TRAIN_BUFFER_EPOCHS=2
export TRAIN_MAX_LENGTH=2048
export TRAIN_THRESHOLD=128
export TRAIN_BATCH_SIZE=1
export TRAIN_GRADIENT_ACCUMULATION_STEPS=2
export TRAIN_LR=3e-4
export TRAIN_MAX_STEPS=0
export MAX_DRAFT_VERSION=0
bash "$ASYN_TRAIN_ROOT/scripts/pipeline/run_async_pipeline_adaflash.sh" "2" "3,4"