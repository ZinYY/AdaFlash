CUDA_VISIBLE_DEVICES=0 python bin/regenerate_dataset.py \
  --model-path Qwen/Qwen3-8B \
  --initial-draft-path z-lab/Qwen3-8B-DFlash-b16 \
  --dataset-path cache/dataset/perfectblend_train.jsonl \
  --dataset-start-line 1 \
  --num-samples 4096 \
  --output-jsonl cache/dataset/perfectblend_qwen3_8b_regen_4096.jsonl

CUDA_VISIBLE_DEVICES=1 python bin/regenerate_dataset.py \
  --model-path Qwen/Qwen3.5-9B \
  --initial-draft-path z-lab/Qwen3.5-9B-DFlash \
  --dataset-path cache/dataset/perfectblend_train.jsonl \
  --dataset-start-line 1 \
  --concurrency 10 \
  --num-samples 4096 \
  --output-jsonl cache/dataset/perfectblend_qwen3_5_9b_regen_4096.jsonl

CUDA_VISIBLE_DEVICES=2 python bin/regenerate_dataset.py \
  --model-path Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --initial-draft-path z-lab/Qwen3-Coder-30B-A3B-DFlash \
  --dataset-path cache/dataset/perfectblend_train.jsonl \
  --dataset-start-line 1 \
  --concurrency 10 \
  --num-samples 4096 \
  --output-jsonl cache/dataset/perfectblend_qwen3_coder_30b_a3b_instruct_regen_4096.jsonl

CUDA_VISIBLE_DEVICES=3 python bin/regenerate_dataset.py \
  --model-path Qwen/Qwen3-8B \
  --initial-draft-path z-lab/Qwen3-8B-DFlash-b16 \
  --dataset-path cache/dataset/math500_train.jsonl \
  --dataset-start-line 1 \
  --num-samples 4096 \
  --output-jsonl cache/dataset/math500_qwen3_8b_regen_4096.jsonl \
  --enable-thinking \
  --max-new-tokens 32768 \
  --temperature 0.6

CUDA_VISIBLE_DEVICES=2 python bin/regenerate_dataset.py \
  --model-path Qwen/Qwen3-8B \
  --initial-draft-path z-lab/Qwen3-8B-DFlash-b16 \
  --dataset-path cache/dataset/perfectblend_full.jsonl \
  --output-jsonl cache/dataset/perfectblend_qwen3_8b_regen_full.jsonl \
  --concurrency 16

CUDA_VISIBLE_DEVICES=4 python bin/regenerate_dataset.py \
  --model-path Qwen/Qwen3-8B \
  --initial-draft-path z-lab/Qwen3-8B-DFlash-b16 \
  --dataset-path cache/dataset/gsm8k_train.jsonl \
  --output-jsonl cache/dataset/gsm8k_train_qwen3_8b_regen.jsonl \
  --concurrency 16

CUDA_VISIBLE_DEVICES=2 python bin/regenerate_dataset.py \
  --model-path Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --initial-draft-path z-lab/Qwen3-Coder-30B-A3B-DFlash \
  --dataset-path cache/dataset/gsm8k_train.jsonl \
  --output-jsonl cache/dataset/gsm8k_train_qwen3_coder_regen.jsonl \
  --concurrency 16

CUDA_VISIBLE_DEVICES=6 python bin/regenerate_dataset.py \
  --model-path Qwen/Qwen3.5-9B \
  --initial-draft-path z-lab/Qwen3.5-9B-DFlash \
  --dataset-path cache/dataset/gsm8k_train.jsonl \
  --output-jsonl cache/dataset/gsm8k_train_qwen3_5_9b_regen.jsonl \
  --concurrency 16