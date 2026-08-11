# qwen3.5 rkl with adaptive length head
export MODEL_PATH=Qwen/Qwen3.5-9B
export DRAFT_CONFIG_PATH=configs/qwen3.5-9b-dflash-thresh-head.json
export INITIAL_DRAFT_PATH=temp/qwen3.5/math_qa/with_head/model_weights
export DATASET_PATH=cache/dataset/gsm8k_train_qwen3_5_9b_regen.jsonl
export OUTPUT_DIR=temp/qwen3.5/gsm8k/with_head/model_weights
export TRAIN_BUFFER_EPOCHS=5
export TRAIN_TP_SIZE=1
export TRAIN_BATCH_SIZE=1
export TRAIN_GRADIENT_ACCUMULATION_STEPS=1
export TRAIN_LOG_INTERVAL=25
export TRAIN_SAVE_INTERVAL=0
export SGLANG_MEM_FRACTION_STATIC=0.40
bash scripts/offline/train_rkl_once.sh 6,7

# qwen3.5 sft without adaptive length head
export MODEL_PATH=Qwen/Qwen3.5-9B
export DRAFT_CONFIG_PATH=configs/qwen3.5-9b-dflash.json
export INITIAL_DRAFT_PATH=temp/qwen3.5/math_qa/t0/soft/model_weights
export DATASET_PATH=cache/dataset/gsm8k_train_qwen3_5_9b_regen.jsonl
export OUTPUT_DIR=temp/qwen3.5/gsm8k/t0/soft/model_weights
export TRAIN_BUFFER_EPOCHS=5
export TRAIN_TP_SIZE=1
export TRAIN_BATCH_SIZE=1
export TRAIN_GRADIENT_ACCUMULATION_STEPS=1
export TEACHER_KD_DISABLE=0
export TEACHER_KD_TEMPERATURE=1.0
export TEACHER_KD_ALPHA=0
export TRAIN_LOG_INTERVAL=25
export TRAIN_SAVE_INTERVAL=0
export SGLANG_MEM_FRACTION_STATIC=0.40
bash scripts/offline/train_sft_once.sh 6,7

# # qwen3_coder
# export DRAFT_CONFIG_PATH=configs/qwen3-coder-30b-dflash-thresh-head.json
# export INITIAL_DRAFT_PATH=models/Qwen3-Coder-30B-A3B-DFlash-two-model-256-perfectblend-4096
# export DATASET_PATH=cache/dataset/gsm8k_train_qwen3_coder_regen.jsonl
# export OUTPUT_DIR=temp/qwen3_coder/gsm8k/offline_rkl_once/model_weights
# export TRAIN_BUFFER_EPOCHS=5
# export TRAIN_TP_SIZE=1
# export TRAIN_BATCH_SIZE=1
# export TRAIN_GRADIENT_ACCUMULATION_STEPS=1
# export TRAIN_LOG_INTERVAL=25
# export TRAIN_SAVE_INTERVAL=0
# export SGLANG_MEM_FRACTION_STATIC=0.40
# export TRAIN_ATTENTION_BACKEND=sdpa
# bash scripts/offline/train_rkl_once.sh 2,7