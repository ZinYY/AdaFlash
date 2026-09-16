# ------------------------------------------------------------------------------
# Vendored from SpecForge (sgl-project), MIT License.
# Upstream: SpecForge/specforge/modeling/target/sglang_backend/__init__.py
# Full attribution and license: asyn_train/specforge/VENDOR_README.md
# ------------------------------------------------------------------------------

from .model_runner import SGLangRunner
from .utils import wrap_eagle3_logits_processors_in_module

__all__ = ["SGLangRunner", "wrap_eagle3_logits_processors_in_module"]
