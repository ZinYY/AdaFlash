export INITIAL_DRAFT_PATH=models/Qwen3-8B-PerfectBlend-Full-RKL-v2
export DRAFT_CONFIG_PATH=configs/qwen3-8b-dflash-thresh-head.json
export DATASET_PATH=cache/dataset/perfectblend_qwen3_8b_regen_full.jsonl
export OUTPUT_DIR=models/Qwen3-8B-PerfectBlend-Full-RKL-v3
export TRAIN_BUFFER_EPOCHS=8
export TRAIN_MAX_LENGTH=2048
export TRAIN_TP_SIZE=1
export SGLANG_MEM_FRACTION_STATIC=0.40
export TRAIN_BATCH_SIZE=1
export TRAIN_GRADIENT_ACCUMULATION_STEPS=1
export TRAIN_LR=3e-4
export TRAIN_SAVE_INTERVAL=10000
bash scripts/offline/train_rkl_once.sh 1,2,3,4