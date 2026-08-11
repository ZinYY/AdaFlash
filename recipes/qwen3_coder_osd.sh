# Qwen3-Coder-30B-A3B-Instruct, temperature = 0 (greedy decoding), OSD, draft only (without adaptive length head)
_LIB_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/../scripts/lib" &>/dev/null && pwd)
# shellcheck source=common.sh
source "$_LIB_DIR/common.sh"


# math_qa
export MODEL_PATH=Qwen/Qwen3-Coder-30B-A3B-Instruct
export INITIAL_DRAFT_PATH=z-lab/Qwen3-Coder-30B-A3B-DFlash
export DRAFT_CONFIG_PATH=$ASYN_TRAIN_ROOT/configs/qwen3-coder-30b-dflash.json
export DATASET_PATH=$ASYN_TRAIN_ROOT/cache/dataset/math_qa_train.jsonl
export SWAP_DIR=$ASYN_TRAIN_ROOT/temp/qwen3_coder/math_qa/t0/soft
export LOG_DIR=$ASYN_TRAIN_ROOT/logs_math_qa_t0_soft
export INFER_TEMPERATURE=0.0
export MAX_DRAFT_VERSION=201
export SGLANG_MEM_FRACTION_STATIC=0.50
export TRAIN_ATTENTION_BACKEND=sdpa
bash "$ASYN_TRAIN_ROOT/scripts/pipeline/run_async_pipeline_osd.sh" "5" "6,7"

python plot_mean.py \
--jsonl $ASYN_TRAIN_ROOT/temp/qwen3_coder/math_qa/t0/soft/inference_stats.jsonl \
--out-dir $ASYN_TRAIN_ROOT/temp/qwen3_coder/math_qa/t0/soft

# gsm8k
export MODEL_PATH=Qwen/Qwen3-Coder-30B-A3B-Instruct
export INITIAL_DRAFT_PATH=z-lab/Qwen3-Coder-30B-A3B-DFlash
export DRAFT_CONFIG_PATH=$ASYN_TRAIN_ROOT/configs/qwen3-coder-30b-dflash.json
export DATASET_PATH=$ASYN_TRAIN_ROOT/cache/dataset/gsm8k_train.jsonl
export SWAP_DIR=$ASYN_TRAIN_ROOT/temp/qwen3_coder/gsm8k/t0/soft
export LOG_DIR=$ASYN_TRAIN_ROOT/logs_gsm8k_t0_soft
export INFER_TEMPERATURE=0.0
export MAX_DRAFT_VERSION=201
export SGLANG_MEM_FRACTION_STATIC=0.50
export TRAIN_ATTENTION_BACKEND=sdpa
bash "$ASYN_TRAIN_ROOT/scripts/pipeline/run_async_pipeline_osd.sh" "5" "6,7"

python plot_mean.py \
--jsonl $ASYN_TRAIN_ROOT/temp/qwen3_coder/gsm8k/t0/soft/inference_stats.jsonl \
--out-dir $ASYN_TRAIN_ROOT/temp/qwen3_coder/gsm8k/t0/soft

# opencodeinstruct
export MODEL_PATH=Qwen/Qwen3-Coder-30B-A3B-Instruct
export INITIAL_DRAFT_PATH=z-lab/Qwen3-Coder-30B-A3B-DFlash
export DRAFT_CONFIG_PATH=$ASYN_TRAIN_ROOT/configs/qwen3-coder-30b-dflash.json
export DATASET_PATH=$ASYN_TRAIN_ROOT/cache/dataset/opencodeinstruct_train.jsonl
export SWAP_DIR=$ASYN_TRAIN_ROOT/temp/qwen3_coder/opencodeinstruct/t0/soft
export LOG_DIR=$ASYN_TRAIN_ROOT/logs_opencodeinstruct_t0_soft
export INFER_TEMPERATURE=0.0
export MAX_DRAFT_VERSION=201
export SGLANG_MEM_FRACTION_STATIC=0.50
export TRAIN_ATTENTION_BACKEND=sdpa
bash "$ASYN_TRAIN_ROOT/scripts/pipeline/run_async_pipeline_osd.sh" "5" "6,7"

python plot_mean.py \
--jsonl $ASYN_TRAIN_ROOT/temp/qwen3_coder/opencodeinstruct/t0/soft/inference_stats.jsonl \
--out-dir $ASYN_TRAIN_ROOT/temp/qwen3_coder/opencodeinstruct/t0/soft

# codealpaca
export MODEL_PATH=Qwen/Qwen3-Coder-30B-A3B-Instruct
export INITIAL_DRAFT_PATH=z-lab/Qwen3-Coder-30B-A3B-DFlash
export DRAFT_CONFIG_PATH=$ASYN_TRAIN_ROOT/configs/qwen3-coder-30b-dflash.json
export DATASET_PATH=$ASYN_TRAIN_ROOT/cache/dataset/codealpaca-20k_train.jsonl
export SWAP_DIR=$ASYN_TRAIN_ROOT/temp/qwen3_coder/codealpaca/t0/soft
export LOG_DIR=$ASYN_TRAIN_ROOT/logs_codealpaca_t0_soft
export INFER_TEMPERATURE=0.0
export MAX_DRAFT_VERSION=151
export SGLANG_MEM_FRACTION_STATIC=0.50
export TRAIN_ATTENTION_BACKEND=sdpa
bash "$ASYN_TRAIN_ROOT/scripts/pipeline/run_async_pipeline_osd.sh" "5" "6,7"

python plot_mean.py \
--jsonl $ASYN_TRAIN_ROOT/temp/qwen3_coder/codealpaca/t0/soft/inference_stats.jsonl \
--out-dir $ASYN_TRAIN_ROOT/temp/qwen3_coder/codealpaca/t0/soft

# sharegpt
export MODEL_PATH=Qwen/Qwen3-Coder-30B-A3B-Instruct
export INITIAL_DRAFT_PATH=z-lab/Qwen3-Coder-30B-A3B-DFlash
export DRAFT_CONFIG_PATH=$ASYN_TRAIN_ROOT/configs/qwen3-coder-30b-dflash.json
export DATASET_PATH=$ASYN_TRAIN_ROOT/cache/dataset/sharegpt_train.jsonl
export SWAP_DIR=$ASYN_TRAIN_ROOT/temp/qwen3_coder/sharegpt/t0/soft
export INFER_TEMPERATURE=0.0
export MAX_DRAFT_VERSION=201
export SGLANG_MEM_FRACTION_STATIC=0.50
export TRAIN_ATTENTION_BACKEND=sdpa
bash "$ASYN_TRAIN_ROOT/scripts/pipeline/run_async_pipeline_osd.sh" "5" "6,7"

python plot_mean.py \
--jsonl $ASYN_TRAIN_ROOT/temp/qwen3_coder/sharegpt/t0/soft/inference_stats.jsonl \
--out-dir $ASYN_TRAIN_ROOT/temp/qwen3_coder/sharegpt/t0/soft

# myblend
export MODEL_PATH=Qwen/Qwen3-Coder-30B-A3B-Instruct
export INITIAL_DRAFT_PATH=z-lab/Qwen3-Coder-30B-A3B-DFlash
export DRAFT_CONFIG_PATH=$ASYN_TRAIN_ROOT/configs/qwen3-coder-30b-dflash.json
export DATASET_PATH=$ASYN_TRAIN_ROOT/cache/dataset/myblend_train.jsonl
export SWAP_DIR=$ASYN_TRAIN_ROOT/temp/qwen3_coder/myblend/t0/soft
export LOG_DIR=$ASYN_TRAIN_ROOT/logs_myblend_t0_soft
export INFER_TEMPERATURE=0.0
export MAX_DRAFT_VERSION=201
export SGLANG_MEM_FRACTION_STATIC=0.50
export TRAIN_ATTENTION_BACKEND=sdpa
bash "$ASYN_TRAIN_ROOT/scripts/pipeline/run_async_pipeline_osd.sh" "5" "6,7"

python plot_mean.py \
--jsonl $ASYN_TRAIN_ROOT/temp/qwen3_coder/myblend/t0/soft/inference_stats.jsonl \
--out-dir $ASYN_TRAIN_ROOT/temp/qwen3_coder/myblend/t0/soft