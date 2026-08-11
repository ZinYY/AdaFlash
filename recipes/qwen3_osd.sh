# temperature = 0 (greedy decoding), OSD, draft only (without adaptive length head)
_LIB_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/../scripts/lib" &>/dev/null && pwd)
# shellcheck source=common.sh
source "$_LIB_DIR/common.sh"


# math_qa
export MODEL_PATH=Qwen/Qwen3-8B
export INITIAL_DRAFT_PATH=z-lab/Qwen3-8B-DFlash-b16
export DRAFT_CONFIG_PATH=$ASYN_TRAIN_ROOT/configs/qwen3-8b-dflash.json
export DATASET_PATH=$ASYN_TRAIN_ROOT/cache/dataset/math_qa_train.jsonl 
export TEACHER_KD_DISABLE=0
export TEACHER_KD_TEMPERATURE=1.0
export TEACHER_KD_ALPHA=0
export INFER_TEMPERATURE=0.0
export MAX_DRAFT_VERSION=201
export SWAP_DIR=$ASYN_TRAIN_ROOT/temp/qwen3/math_qa/t0/soft
export LOG_DIR=$ASYN_TRAIN_ROOT/logs_math_qa_t0_soft
bash "$ASYN_TRAIN_ROOT/scripts/pipeline/run_async_pipeline_osd.sh" "2" "3,4"

python plot_mean.py \
--jsonl $ASYN_TRAIN_ROOT/temp/qwen3/math_qa/t0/soft/inference_stats.jsonl \
--out-dir $ASYN_TRAIN_ROOT/temp/qwen3/math_qa/t0/soft

# gsm8k
export MODEL_PATH=Qwen/Qwen3-8B
export INITIAL_DRAFT_PATH=z-lab/Qwen3-8B-DFlash-b16
export DRAFT_CONFIG_PATH=$ASYN_TRAIN_ROOT/configs/qwen3-8b-dflash.json
export DATASET_PATH=$ASYN_TRAIN_ROOT/cache/dataset/gsm8k_train.jsonl
export TEACHER_KD_DISABLE=0
export TEACHER_KD_TEMPERATURE=1.0
export TEACHER_KD_ALPHA=0
export INFER_TEMPERATURE=0.0
export MAX_DRAFT_VERSION=61
export SWAP_DIR=$ASYN_TRAIN_ROOT/temp/qwen3/gsm8k/t0/soft
export LOG_DIR=$ASYN_TRAIN_ROOT/logs_gsm8k_t0_soft
bash "$ASYN_TRAIN_ROOT/scripts/pipeline/run_async_pipeline_osd.sh" "2" "3,4"

python plot_mean.py \
--jsonl $ASYN_TRAIN_ROOT/temp/qwen3/gsm8k/t0/soft/inference_stats.jsonl \
--out-dir $ASYN_TRAIN_ROOT/temp/qwen3/gsm8k/t0/soft

# opencodeinstruct
export MODEL_PATH=Qwen/Qwen3-8B
export INITIAL_DRAFT_PATH=z-lab/Qwen3-8B-DFlash-b16
export DRAFT_CONFIG_PATH=$ASYN_TRAIN_ROOT/configs/qwen3-8b-dflash.json
export DATASET_PATH=$ASYN_TRAIN_ROOT/cache/dataset/opencodeinstruct_train.jsonl 
export TEACHER_KD_DISABLE=0
export TEACHER_KD_TEMPERATURE=1.0
export TEACHER_KD_ALPHA=0
export INFER_TEMPERATURE=0.0
export MAX_DRAFT_VERSION=201
export SWAP_DIR=$ASYN_TRAIN_ROOT/temp/qwen3/opencodeinstruct/t0/soft
export LOG_DIR=$ASYN_TRAIN_ROOT/logs_opencodeinstruct_t0_soft
bash "$ASYN_TRAIN_ROOT/scripts/pipeline/run_async_pipeline_osd.sh" "2" "3,4"

python plot_mean.py \
--jsonl $ASYN_TRAIN_ROOT/temp/qwen3/opencodeinstruct/t0/soft/inference_stats.jsonl \
--out-dir $ASYN_TRAIN_ROOT/temp/qwen3/opencodeinstruct/t0/soft

# codealpaca
export MODEL_PATH=Qwen/Qwen3-8B
export INITIAL_DRAFT_PATH=z-lab/Qwen3-8B-DFlash-b16
export DRAFT_CONFIG_PATH=$ASYN_TRAIN_ROOT/configs/qwen3-8b-dflash.json
export DATASET_PATH=$ASYN_TRAIN_ROOT/cache/dataset/codealpaca-20k_train.jsonl 
export TEACHER_KD_DISABLE=0
export TEACHER_KD_TEMPERATURE=1.0
export TEACHER_KD_ALPHA=0
export INFER_TEMPERATURE=0.0
export MAX_DRAFT_VERSION=151
export SWAP_DIR=$ASYN_TRAIN_ROOT/temp/qwen3/codealpaca/t0/soft
export LOG_DIR=$ASYN_TRAIN_ROOT/logs_codealpaca_t0_soft
bash "$ASYN_TRAIN_ROOT/scripts/pipeline/run_async_pipeline_osd.sh" "2" "3,4"

python plot_mean.py \
--jsonl $ASYN_TRAIN_ROOT/temp/qwen3/codealpaca/t0/soft/inference_stats.jsonl \
--out-dir $ASYN_TRAIN_ROOT/temp/qwen3/codealpaca/t0/soft

# sharegpt
export MODEL_PATH=Qwen/Qwen3-8B
export INITIAL_DRAFT_PATH=z-lab/Qwen3-8B-DFlash-b16
export DRAFT_CONFIG_PATH=$ASYN_TRAIN_ROOT/configs/qwen3-8b-dflash.json
export DATASET_PATH=$ASYN_TRAIN_ROOT/cache/dataset/sharegpt_train.jsonl 
export TEACHER_KD_DISABLE=0
export TEACHER_KD_TEMPERATURE=1.0
export TEACHER_KD_ALPHA=0
export INFER_TEMPERATURE=0.0
export MAX_DRAFT_VERSION=201
export SWAP_DIR=$ASYN_TRAIN_ROOT/temp/qwen3/sharegpt/t0/soft
export LOG_DIR=$ASYN_TRAIN_ROOT/logs_sharegpt_t0_soft
bash "$ASYN_TRAIN_ROOT/scripts/pipeline/run_async_pipeline_osd.sh" "2" "3,4"

python plot_mean.py \
--jsonl $ASYN_TRAIN_ROOT/temp/qwen3/sharegpt/t0/soft/inference_stats.jsonl \
--out-dir $ASYN_TRAIN_ROOT/temp/qwen3/sharegpt/t0/soft

# myblend
export MODEL_PATH=Qwen/Qwen3-8B
export INITIAL_DRAFT_PATH=z-lab/Qwen3-8B-DFlash-b16
export DRAFT_CONFIG_PATH=$ASYN_TRAIN_ROOT/configs/qwen3-8b-dflash.json
export DATASET_PATH=$ASYN_TRAIN_ROOT/cache/dataset/myblend_train.jsonl 
export TEACHER_KD_DISABLE=0
export TEACHER_KD_TEMPERATURE=1.0
export TEACHER_KD_ALPHA=0
export INFER_TEMPERATURE=0.0
export MAX_DRAFT_VERSION=201
export SWAP_DIR=$ASYN_TRAIN_ROOT/temp/qwen3/myblend/t0/soft
export LOG_DIR=$ASYN_TRAIN_ROOT/logs_myblend_t0_soft
bash "$ASYN_TRAIN_ROOT/scripts/pipeline/run_async_pipeline_osd.sh" "2" "3,4"

python plot_mean.py \
--jsonl $ASYN_TRAIN_ROOT/temp/qwen3/myblend/t0/soft/inference_stats.jsonl \
--out-dir $ASYN_TRAIN_ROOT/temp/qwen3/myblend/t0/soft