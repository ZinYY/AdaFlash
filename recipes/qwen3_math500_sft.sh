cd /root/dFlash_wu/asyn_train

export DATASET_PATH=cache/dataset/math500_train_qwen3_8b_regen.jsonl
export OUTPUT_DIR=temp/qwen3/math500/soft/model_weights
export INITIAL_DRAFT_PATH=models/Qwen3-8B-DFlash-two-model-256-perfectblend-4096
export DRAFT_CONFIG_PATH=configs/qwen3-8b-dflash-thresh-head.json
export MODEL_PATH=Qwen/Qwen3-8B
export TRAIN_LOG_INTERVAL=10
export TRAIN_MAX_LENGTH=8192
export TRAIN_TP_SIZE=1
export SGLANG_MEM_FRACTION_STATIC=0.30
export TRAIN_NUM_ANCHORS=512
export TRAIN_BATCH_SIZE=1
export TRAIN_BUFFER_EPOCHS=2
export TRAIN_GRADIENT_ACCUMULATION_STEPS=2
export TEACHER_KD_DISABLE=0
export TEACHER_KD_TEMPERATURE=1.0
export TEACHER_KD_ALPHA=0

bash scripts/offline/train_sft_once.sh 2,3