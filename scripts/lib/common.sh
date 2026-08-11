# Resolve and export ASYN_TRAIN_ROOT (directory containing pipeline/bootstrap.py).
# Source from any script under scripts/**:
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/common.sh"

if [[ -z "${ASYN_TRAIN_ROOT:-}" ]]; then
  _ASYN_LIB_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
  ASYN_TRAIN_ROOT=$(cd "${_ASYN_LIB_DIR}/../.." && pwd)
  export ASYN_TRAIN_ROOT
fi
REPO_ROOT="${REPO_ROOT:-$(dirname "$ASYN_TRAIN_ROOT")}"
export REPO_ROOT
BIN_DIR="${ASYN_TRAIN_ROOT}/bin"
