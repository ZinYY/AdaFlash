# Qwen3-Coder-30B-A3B-Instruct, temperature = 0 (greedy decoding), AdaFlash
_LIB_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/../scripts/lib" &>/dev/null && pwd)
# shellcheck source=common.sh
source "$_LIB_DIR/common.sh"


# math_qa
export MODEL_PATH=Qwen/Qwen3-Coder-30B-A3B-Instruct
export INITIAL_DRAFT_PATH=models/Qwen3-Coder-30B-A3B-DFlash-two-model-256-perfectblend-4096
export DRAFT_CONFIG_PATH=$ASYN_TRAIN_ROOT/configs/qwen3-coder-30b-dflash-thresh-head.json
export DATASET_PATH=$ASYN_TRAIN_ROOT/cache/dataset/math_qa_train.jsonl
export SWAP_DIR=$ASYN_TRAIN_ROOT/temp/qwen3_coder/math_qa/with_head
export LOG_DIR=$ASYN_TRAIN_ROOT/logs_coder_math_qa_with_head
export INFER_TEMPERATURE=0.0
export MAX_DRAFT_VERSION=201
export SGLANG_MEM_FRACTION_STATIC=0.50
export TRAIN_ATTENTION_BACKEND=sdpa
bash "$ASYN_TRAIN_ROOT/scripts/pipeline/run_async_pipeline_adaflash.sh" "5" "6,7"


# gsm8k
export MODEL_PATH=Qwen/Qwen3-Coder-30B-A3B-Instruct
export INITIAL_DRAFT_PATH=models/Qwen3-Coder-30B-A3B-DFlash-two-model-256-perfectblend-4096
export DRAFT_CONFIG_PATH=$ASYN_TRAIN_ROOT/configs/qwen3-coder-30b-dflash-thresh-head.json
export DATASET_PATH=$ASYN_TRAIN_ROOT/cache/dataset/gsm8k_train.jsonl
export SWAP_DIR=$ASYN_TRAIN_ROOT/temp/qwen3_coder/gsm8k/with_head
export LOG_DIR=$ASYN_TRAIN_ROOT/logs_coder_gsm8k_with_head
export INFER_TEMPERATURE=0.0
export MAX_DRAFT_VERSION=201
export SGLANG_MEM_FRACTION_STATIC=0.50
export TRAIN_ATTENTION_BACKEND=sdpa
bash "$ASYN_TRAIN_ROOT/scripts/pipeline/run_async_pipeline_adaflash.sh" "5" "6,7"


# opencodeinstruct
export MODEL_PATH=Qwen/Qwen3-Coder-30B-A3B-Instruct
export INITIAL_DRAFT_PATH=models/Qwen3-Coder-30B-A3B-DFlash-two-model-256-perfectblend-4096
export DRAFT_CONFIG_PATH=$ASYN_TRAIN_ROOT/configs/qwen3-coder-30b-dflash-thresh-head.json
export DATASET_PATH=$ASYN_TRAIN_ROOT/cache/dataset/opencodeinstruct_train.jsonl
export SWAP_DIR=$ASYN_TRAIN_ROOT/temp/qwen3_coder/opencodeinstruct/with_head
export LOG_DIR=$ASYN_TRAIN_ROOT/logs_coder_opencodeinstruct_with_head
export INFER_TEMPERATURE=0.0
export MAX_DRAFT_VERSION=201
export SGLANG_MEM_FRACTION_STATIC=0.50
export TRAIN_ATTENTION_BACKEND=sdpa
bash "$ASYN_TRAIN_ROOT/scripts/pipeline/run_async_pipeline_adaflash.sh" "5" "6,7"


# codealpaca
export MODEL_PATH=Qwen/Qwen3-Coder-30B-A3B-Instruct
export INITIAL_DRAFT_PATH=models/Qwen3-Coder-30B-A3B-DFlash-two-model-256-perfectblend-4096
export DRAFT_CONFIG_PATH=$ASYN_TRAIN_ROOT/configs/qwen3-coder-30b-dflash-thresh-head.json
export DATASET_PATH=$ASYN_TRAIN_ROOT/cache/dataset/codealpaca-20k_train.jsonl
export SWAP_DIR=$ASYN_TRAIN_ROOT/temp/qwen3_coder/codealpaca/with_head
export LOG_DIR=$ASYN_TRAIN_ROOT/logs_coder_codealpaca_with_head
export INFER_TEMPERATURE=0.0
export MAX_DRAFT_VERSION=151
export SGLANG_MEM_FRACTION_STATIC=0.50
export TRAIN_ATTENTION_BACKEND=sdpa
bash "$ASYN_TRAIN_ROOT/scripts/pipeline/run_async_pipeline_adaflash.sh" "5" "6,7"


# sharegpt
export MODEL_PATH=Qwen/Qwen3-Coder-30B-A3B-Instruct
export INITIAL_DRAFT_PATH=models/Qwen3-Coder-30B-A3B-DFlash-two-model-256-perfectblend-4096
export DRAFT_CONFIG_PATH=$ASYN_TRAIN_ROOT/configs/qwen3-coder-30b-dflash-thresh-head.json
export DATASET_PATH=$ASYN_TRAIN_ROOT/cache/dataset/sharegpt_train.jsonl
export SWAP_DIR=$ASYN_TRAIN_ROOT/temp/qwen3_coder/sharegpt/with_head
export LOG_DIR=$ASYN_TRAIN_ROOT/logs_coder_sharegpt_with_head
export INFER_TEMPERATURE=0.0
export MAX_DRAFT_VERSION=201
export SGLANG_MEM_FRACTION_STATIC=0.50
export TRAIN_ATTENTION_BACKEND=sdpa
bash "$ASYN_TRAIN_ROOT/scripts/pipeline/run_async_pipeline_adaflash.sh" "5" "6,7"
