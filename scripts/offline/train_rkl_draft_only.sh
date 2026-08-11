# RKL draft-only (no adaptive length head). INITIAL_DRAFT_PATH unset = from scratch.
export TRAIN_THRESH_HEAD=false
export DRAFT_CONFIG_PATH=configs/qwen3-8b-dflash.json
bash scripts/offline/train_rkl_once.sh 4,5,6,7