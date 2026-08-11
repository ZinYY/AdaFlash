# ------------------------------------------------------------------------------
# Vendored from SpecForge (sgl-project), MIT License.
# Upstream: SpecForge/specforge/modeling/target/dflash_target_model.py
# Full attribution and license: asyn_train/specforge/VENDOR_README.md
# ------------------------------------------------------------------------------

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import require_mlp_sync, require_mlp_tp_gather
from transformers import AutoModelForCausalLM

from specforge.distributed import get_tp_group

from .sglang_backend import SGLangRunner

logger = logging.getLogger(__name__)


@dataclass
class DFlashTargetOutput:
    hidden_states: torch.Tensor  # [batch, seq_len, hidden_size] (draft: may be concat of layers)
    input_ids: torch.Tensor  # [batch, seq_len]
    attention_mask: torch.Tensor  # [batch, seq_len]
    loss_mask: torch.Tensor  # [batch, seq_len]
    # Last decoder layer before lm_head; required when hidden_states is multi-layer concat (teacher KD lm_head input).
    # None means teacher paths should use hidden_states (already last-layer, [B,S,H]).
    teacher_hidden_states: Optional[torch.Tensor] = None
    # Last-layer [B,S,H] for prob-head / acceptance labels when hidden_states may be concat of aux layers.
    last_hidden_states: Optional[torch.Tensor] = None


class DFlashTargetModel(ABC):
    """
    Abstract base class for DFlash target model backend.
    """

    def __init__(self):
        self.capture_layer_ids = None

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "DFlashTargetModel":
        """Initialize the target model backend."""

    @abstractmethod
    def generate_dflash_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> DFlashTargetOutput:
        """Generate context hidden states for DFlash training."""

    def set_capture_layers(self, layer_ids: List[int]) -> None:
        """Set which layers' hidden states to capture."""
        self.capture_layer_ids = layer_ids


class SGLangDFlashTargetModel(DFlashTargetModel):
    def __init__(self, model_runner: SGLangRunner):
        super().__init__()
        self.model_runner = model_runner

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "SGLangDFlashTargetModel":
        tp_size = dist.get_world_size(get_tp_group())
        server_args = ServerArgs(
            model_path=pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            dtype=torch_dtype,
            enable_return_hidden_states=True,  # Critical for DFlash
            disable_cuda_graph=True,
            tp_size=tp_size,
            pp_size=1,
            **kwargs,
        )

        tp_rank = dist.get_rank(get_tp_group())
        moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)
        model_config = ModelConfig.from_server_args(server_args)

        model_runner = SGLangRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=torch.cuda.current_device(),
            tp_rank=dist.get_rank(get_tp_group()),
            tp_size=server_args.tp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=server_args.ep_size,
            pp_rank=0,
            pp_size=1,
            server_args=server_args,
            nccl_port=None,
        )
        return cls(model_runner)

    def set_capture_layers(self, layer_ids: List[int]) -> None:
        super().set_capture_layers(layer_ids)
        model = self.model_runner.model
        # Keep Qwen3 / legacy path: prefer set_eagle3 when present (unchanged behavior).
        # Qwen3.5 (Qwen3VLForConditionalGeneration) has no set_eagle3; use set_dflash or aux
        # capture stays off -> last-layer H vs draft fc expecting K*H.
        if hasattr(model, "set_eagle3_layers_to_capture"):
            model.set_eagle3_layers_to_capture(layer_ids)
        elif hasattr(model, "set_dflash_layers_to_capture"):
            model.set_dflash_layers_to_capture(layer_ids)
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "layers_to_capture"):
            logger.debug("SGLang layers_to_capture=%s", inner.layers_to_capture)

    def _run_extend_forward(self, reqs) -> Tuple[object, List[int]]:
        """One SGLang extend forward; caller clears pools between calls if needed."""
        cache_params = CacheInitParams(
            disable=False,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            page_size=self.model_runner.server_args.page_size,
        )
        tree_cache = RadixCache(cache_params)

        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=self.model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()

        if require_mlp_sync(self.model_runner.server_args):
            Scheduler.prepare_mlp_sync_batch_raw(
                batch,
                dp_size=self.model_runner.server_args.dp_size,
                attn_tp_size=1,
                tp_group=self.model_runner.tp_group,
                get_idle_batch=None,
                disable_cuda_graph=self.model_runner.server_args.disable_cuda_graph,
                spec_algorithm=SpeculativeAlgorithm.NONE,
                speculative_num_draft_tokens=None,
                require_mlp_tp_gather=require_mlp_tp_gather(
                    self.model_runner.server_args
                ),
                disable_overlap_schedule=self.model_runner.server_args.disable_overlap_schedule,
                offload_tags=set(),
            )

        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL

        output = self.model_runner.forward(forward_batch)
        if hasattr(output, "logits_output"):
            output = output.logits_output

        input_lens = [len(req.origin_input_ids) for req in reqs]
        return output, input_lens

    @torch.no_grad()
    def _extend(self, reqs):
        output, input_lens = self._run_extend_forward(reqs)

        if (
            hasattr(output, "aux_hidden_states")
            and output.aux_hidden_states is not None
        ):
            hidden_states_list = torch.split(
                output.aux_hidden_states, input_lens, dim=0
            )
        elif hasattr(output, "hidden_states") and output.hidden_states is not None:
            hidden_states_list = torch.split(output.hidden_states, input_lens, dim=0)
        else:
            raise ValueError("SGLang output does not contain hidden states.")

        teacher_hidden_list = None
        th = getattr(output, "teacher_hidden_states", None)
        if th is not None:
            teacher_hidden_list = torch.split(th, input_lens, dim=0)

        last_hidden_list = None
        last_hidden = getattr(output, "last_hidden_states", None)
        if last_hidden is not None:
            last_hidden_list = torch.split(last_hidden, input_lens, dim=0)

        draft_h = hidden_states_list[0].shape[-1]
        H = int(self.model_runner.model_config.hidden_size)
        model = self.model_runner.model
        if (
            teacher_hidden_list is None
            and draft_h != H
            and getattr(model, "capture_aux_hidden_states", False)
        ):
            # Unpatched sglang: stored hidden is multi-layer concat only. Re-run once without
            # aux capture so logits_output.hidden_states is last-layer [T, H].
            was_cap = model.capture_aux_hidden_states
            self.model_runner.req_to_token_pool.clear()
            self.model_runner.token_to_kv_pool_allocator.clear()
            model.capture_aux_hidden_states = False
            try:
                output2, input_lens2 = self._run_extend_forward(reqs)
                if input_lens2 != input_lens:
                    raise RuntimeError("SGLang teacher fallback: input_lens mismatch")
                th2 = getattr(output2, "teacher_hidden_states", None)
                lh2 = getattr(output2, "last_hidden_states", None)
                hs2 = getattr(output2, "hidden_states", None)
                raw_t = th2 if th2 is not None else lh2 if lh2 is not None else hs2
                if raw_t is None:
                    raise RuntimeError(
                        "SGLang teacher fallback: no hidden_states in second forward"
                    )
                if raw_t.shape[-1] != H:
                    raise RuntimeError(
                        f"SGLang teacher fallback: expected hidden_size={H}, got {raw_t.shape[-1]}"
                    )
                teacher_hidden_list = torch.split(raw_t, input_lens, dim=0)
                logger.warning(
                    "[SGLangDFlashTargetModel] Ran an extra target forward without aux "
                    "capture to obtain last-layer hidden states for teacher KD. "
                    "Install patched sglang (teacher_hidden_states / last_hidden_states "
                    "on LogitsProcessorOutput) to avoid 2× forward cost."
                )
            finally:
                model.capture_aux_hidden_states = was_cap

        self.model_runner.req_to_token_pool.clear()
        self.model_runner.token_to_kv_pool_allocator.clear()

        return hidden_states_list, teacher_hidden_list, last_hidden_list

    @torch.no_grad()
    def generate_dflash_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> DFlashTargetOutput:
        sampling_params = SamplingParams(temperature=0, max_new_tokens=1)
        reqs, data_cache = [], []

        if isinstance(input_ids, torch.Tensor):
            input_ids_list = torch.split(input_ids, 1, dim=0)
            attn_mask_list = torch.split(attention_mask, 1, dim=0)
            loss_mask_list = torch.split(loss_mask, 1, dim=0)

        for idx, (curr_ids, curr_attn, curr_loss) in enumerate(
            zip(input_ids_list, attn_mask_list, loss_mask_list)
        ):
            req = Req(
                rid=str(idx),
                origin_input_text="",
                origin_input_ids=curr_ids.view(-1).tolist(),
                sampling_params=sampling_params,
            )
            req.fill_ids = req.origin_input_ids
            req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
            data_cache.append((curr_ids, curr_attn, curr_loss))
            reqs.append(req)

        hidden_states_list, teacher_hidden_list, last_hidden_list = self._extend(reqs)

        # Stack back to batch
        hidden_states = torch.cat([h.unsqueeze(0) for h in hidden_states_list], dim=0)
        input_ids = torch.cat([d[0] for d in data_cache], dim=0)
        attention_mask = torch.cat([d[1] for d in data_cache], dim=0)
        loss_mask = torch.cat([d[2] for d in data_cache], dim=0)
        last_hidden_states = None
        if last_hidden_list is not None:
            last_hidden_states = torch.cat([h.unsqueeze(0) for h in last_hidden_list], dim=0)

        teacher_hidden_states = None
        if teacher_hidden_list is not None:
            teacher_hidden_states = torch.cat(
                [h.unsqueeze(0) for h in teacher_hidden_list], dim=0
            )

        return DFlashTargetOutput(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            teacher_hidden_states=teacher_hidden_states,
            last_hidden_states=last_hidden_states,
        )


class HFDFlashTargetModel(DFlashTargetModel):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> "HFDFlashTargetModel":

        target_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            output_hidden_states=True,
            trust_remote_code=trust_remote_code,
            **kwargs,
        ).eval()

        if device:
            target_model = target_model.to(device)

        return cls(target_model)

    @torch.no_grad()
    def generate_dflash_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> DFlashTargetOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # hidden_states[0] = embedding output; hidden_states[i+1] = layer i output
        offset = 1
        last_hidden = outputs.hidden_states[-1]
        selected = []
        if self.capture_layer_ids is not None:
            for idx in self.capture_layer_ids:
                selected.append(outputs.hidden_states[idx + offset])
            hidden_states = torch.cat(selected, dim=-1)
            teacher_hidden_states = last_hidden
        else:
            hidden_states = last_hidden
            teacher_hidden_states = None

        return DFlashTargetOutput(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            teacher_hidden_states=teacher_hidden_states,
            last_hidden_states=last_hidden,
        )


def get_dflash_target_model(
    pretrained_model_name_or_path: str,
    backend: str = "sglang",
    torch_dtype: torch.dtype = None,
    device: str = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> DFlashTargetModel:
    if backend == "sglang":
        return SGLangDFlashTargetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    elif backend == "hf":
        return HFDFlashTargetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid backend: {backend}")
