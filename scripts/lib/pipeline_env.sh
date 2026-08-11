# Shared env for async pipeline launchers.
# Requires: source scripts/lib/common.sh first.

: "${ASYN_TRAIN_ROOT:?source scripts/lib/common.sh before pipeline_env.sh}"

export TORCHINDUCTOR_CACHE_DIR="${ASYN_TRAIN_ROOT}/cache/compiled_kernels"
export SPECFORGE_DATA_NUM_PROC="${SPECFORGE_DATA_NUM_PROC:-8}"
