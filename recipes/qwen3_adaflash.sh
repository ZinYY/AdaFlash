# Qwen3-8B, temperature = 0 (greedy decoding), AdaFlash
_LIB_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/../scripts/lib" &>/dev/null && pwd)
# shellcheck source=common.sh
source "$_LIB_DIR/common.sh"


# math_qa
export MODEL_PATH=Qwen/Qwen3-8B
export INITIAL_DRAFT_PATH=models/Qwen3-8B-DFlash-two-model-256-perfectblend-4096
export DRAFT_CONFIG_PATH=$ASYN_TRAIN_ROOT/configs/qwen3-8b-dflash-thresh-head.json
export DATASET_PATH=$ASYN_TRAIN_ROOT/cache/dataset/math_qa_train.jsonl
export SWAP_DIR=$ASYN_TRAIN_ROOT/temp/qwen3/math_qa/with_head
export LOG_DIR=$ASYN_TRAIN_ROOT/logs_math_qa_with_head
export INFER_TEMPERATURE=0.0
export MAX_DRAFT_VERSION=201
bash "$ASYN_TRAIN_ROOT/scripts/pipeline/run_async_pipeline_adaflash.sh" "2" "3,4"

python plot_mean.py \
--jsonl $ASYN_TRAIN_ROOT/temp/qwen3/math_qa/with_head/inference_stats.jsonl \
--out-dir $ASYN_TRAIN_ROOT/temp/qwen3/math_qa/with_head

# gsm8k
export MODEL_PATH=Qwen/Qwen3-8B
export INITIAL_DRAFT_PATH=models/Qwen3-8B-DFlash-two-model-256-perfectblend-4096
export DRAFT_CONFIG_PATH=$ASYN_TRAIN_ROOT/configs/qwen3-8b-dflash-thresh-head.json
export DATASET_PATH=$ASYN_TRAIN_ROOT/cache/dataset/gsm8k_train.jsonl
export SWAP_DIR=$ASYN_TRAIN_ROOT/temp/qwen3/gsm8k/with_head
export LOG_DIR=$ASYN_TRAIN_ROOT/logs_gsm8k_with_head
export INFER_TEMPERATURE=0.0
export MAX_DRAFT_VERSION=61
bash "$ASYN_TRAIN_ROOT/scripts/pipeline/run_async_pipeline_adaflash.sh" "2" "3,4"

python plot_mean.py \
--jsonl $ASYN_TRAIN_ROOT/temp/qwen3/gsm8k/with_head/inference_stats.jsonl \
--out-dir $ASYN_TRAIN_ROOT/temp/qwen3/gsm8k/with_head

# opencodeinstruct
export MODEL_PATH=Qwen/Qwen3-8B
export INITIAL_DRAFT_PATH=models/Qwen3-8B-DFlash-two-model-256-perfectblend-4096
export DRAFT_CONFIG_PATH=$ASYN_TRAIN_ROOT/configs/qwen3-8b-dflash-thresh-head.json
export DATASET_PATH=$ASYN_TRAIN_ROOT/cache/dataset/opencodeinstruct_train.jsonl
export SWAP_DIR=$ASYN_TRAIN_ROOT/temp/qwen3/opencodeinstruct/with_head
export LOG_DIR=$ASYN_TRAIN_ROOT/logs_opencodeinstruct_with_head
export INFER_TEMPERATURE=0.0
export MAX_DRAFT_VERSION=201
bash "$ASYN_TRAIN_ROOT/scripts/pipeline/run_async_pipeline_adaflash.sh" "2" "3,4"

python plot_mean.py \
--jsonl $ASYN_TRAIN_ROOT/temp/qwen3/opencodeinstruct/with_head/inference_stats.jsonl \
--out-dir $ASYN_TRAIN_ROOT/temp/qwen3/opencodeinstruct/with_head

# codealpaca
export MODEL_PATH=Qwen/Qwen3-8B
export INITIAL_DRAFT_PATH=models/Qwen3-8B-DFlash-two-model-256-perfectblend-4096
export DRAFT_CONFIG_PATH=$ASYN_TRAIN_ROOT/configs/qwen3-8b-dflash-thresh-head.json
export DATASET_PATH=$ASYN_TRAIN_ROOT/cache/dataset/codealpaca-20k_train.jsonl
export SWAP_DIR=$ASYN_TRAIN_ROOT/temp/qwen3/codealpaca/with_head
export LOG_DIR=$ASYN_TRAIN_ROOT/logs_codealpaca_with_head
export INFER_TEMPERATURE=0.0
export MAX_DRAFT_VERSION=151
bash "$ASYN_TRAIN_ROOT/scripts/pipeline/run_async_pipeline_adaflash.sh" "2" "3,4"

python plot_mean.py \
--jsonl $ASYN_TRAIN_ROOT/temp/qwen3/codealpaca/with_head/inference_stats.jsonl \
--out-dir $ASYN_TRAIN_ROOT/temp/qwen3/codealpaca/with_head

# sharegpt
export MODEL_PATH=Qwen/Qwen3-8B
export INITIAL_DRAFT_PATH=models/Qwen3-8B-DFlash-two-model-256-perfectblend-4096
export DRAFT_CONFIG_PATH=$ASYN_TRAIN_ROOT/configs/qwen3-8b-dflash-thresh-head.json
export DATASET_PATH=$ASYN_TRAIN_ROOT/cache/dataset/sharegpt_train.jsonl
export SWAP_DIR=$ASYN_TRAIN_ROOT/temp/qwen3/sharegpt/with_head
export LOG_DIR=$ASYN_TRAIN_ROOT/logs_sharegpt_with_head
export INFER_TEMPERATURE=0.0
export MAX_DRAFT_VERSION=201
bash "$ASYN_TRAIN_ROOT/scripts/pipeline/run_async_pipeline_adaflash.sh" "2" "3,4"

python plot_mean.py \
--jsonl $ASYN_TRAIN_ROOT/temp/qwen3/sharegpt/with_head/inference_stats.jsonl \
--out-dir $ASYN_TRAIN_ROOT/temp/qwen3/sharegpt/with_head

# myblend
export MODEL_PATH=Qwen/Qwen3-8B
export INITIAL_DRAFT_PATH=models/Qwen3-8B-DFlash-two-model-256-perfectblend-4096
export DRAFT_CONFIG_PATH=$ASYN_TRAIN_ROOT/configs/qwen3-8b-dflash-thresh-head.json
export DATASET_PATH=$ASYN_TRAIN_ROOT/cache/dataset/myblend_train.jsonl
export SWAP_DIR=$ASYN_TRAIN_ROOT/temp/qwen3/myblend/with_head
export LOG_DIR=$ASYN_TRAIN_ROOT/logs_myblend_with_head
export INFER_TEMPERATURE=0.0
export MAX_DRAFT_VERSION=201
bash "$ASYN_TRAIN_ROOT/scripts/pipeline/run_async_pipeline_adaflash.sh" "2" "3,4"

python plot_mean.py \
--jsonl $ASYN_TRAIN_ROOT/temp/qwen3/myblend/with_head/inference_stats.jsonl \
--out-dir $ASYN_TRAIN_ROOT/temp/qwen3/myblend/with_head