# ------------------------------------------------------------------------------
# Vendored from SpecForge (sgl-project), MIT License.
# Upstream: SpecForge/specforge/utils.py (subset)
# Full attribution and license: asyn_train/specforge/VENDOR_README.md
# ------------------------------------------------------------------------------

import logging

import torch.distributed as dist

logger = logging.getLogger(__name__)


def print_with_rank(message):
    if dist.is_available() and dist.is_initialized():
        logger.info("rank %s: %s", dist.get_rank(), message)
    else:
        logger.info("non-distributed: %s", message)
