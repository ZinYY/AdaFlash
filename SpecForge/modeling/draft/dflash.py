# ------------------------------------------------------------------------------
# Vendored from SpecForge (sgl-project), MIT License.
# Upstream: SpecForge/specforge/modeling/draft/dflash.py
# Full attribution and license: asyn_train/specforge/VENDOR_README.md
# ------------------------------------------------------------------------------

from typing import Callable, Optional

import torch
from torch import nn
from transformers import DynamicCache
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)
from typing_extensions import Tuple, Unpack


# ---------------------------------------------------------------------------
# Prob Head modules (acceptance probability prediction)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


class ProbHead(nn.Module):
    """Predicts per-position acceptance probability from draft hidden states."""
    def __init__(self, hidden_size: int, prob_head_model: str = "mlp",
                 num_layers: int = 2, bottleneck_dim: int = 64):
        super().__init__()
        self.prob_head_model = prob_head_model
        if prob_head_model == "resnet":
            self.model = nn.Sequential(
                *[ResBlock(hidden_size) for _ in range(num_layers)],
                nn.Linear(hidden_size, 1),
            )
        elif prob_head_model == "bottleneck_pool":
            # Bottleneck projection + mean-pool concat for cross-position aggregation
            self.down_proj = nn.Linear(hidden_size, bottleneck_dim)
            self.act = nn.SiLU()
            self.out_proj = nn.Linear(bottleneck_dim * 2, 1)
        else:  # mlp
            if bottleneck_dim > 1:
                self.model = nn.Sequential(
                    nn.Linear(hidden_size, bottleneck_dim),
                    nn.SiLU(),
                    nn.Linear(bottleneck_dim, 1),
                )
            else:
                self.model = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prob_head_model == "bottleneck_pool":
            h = self.act(self.down_proj(x))
            if h.dim() >= 2:
                global_ctx = h.mean(dim=-2, keepdim=True).expand_as(h)
            else:
                global_ctx = h
            return self.out_proj(torch.cat([h, global_ctx], dim=-1))
        return self.model(x)


# ---------------------------------------------------------------------------
# Threshold Head modules (adaptive threshold prediction)
# ---------------------------------------------------------------------------

class ThreshHeadSubsequent(nn.Module):
    """Predicts per-block threshold from draft logit statistics.
    Extracts softmax features (max_prob, entropy, top-k probs) per position,
    then aggregates over block and projects to a scalar threshold."""
    def __init__(self, vocab_size: int, bottleneck_dim: int = 256, top_k: int = 5):
        super().__init__()
        self.top_k = top_k
        # Per-position features: max_prob + entropy + top_k probs = 2 + top_k
        feat_dim = (2 + top_k)
        # Block-level: concat per-position features across block_size, then project
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, 1),
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """logits: [bs, block_size, vocab_size] -> [bs, 1]."""
        probs = torch.softmax(logits.float(), dim=-1)
        max_prob = probs.max(dim=-1).values  # [bs, block_size]
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # [bs, block_size]
        topk_probs = probs.topk(self.top_k, dim=-1).values  # [bs, block_size, top_k]
        # Concat features: [bs, block_size, 2 + top_k]
        features = torch.cat([max_prob.unsqueeze(-1), entropy.unsqueeze(-1), topk_probs], dim=-1)
        # Mean-pool across block positions
        pooled = features.mean(dim=-2)  # [bs, 2 + top_k]
        return torch.sigmoid(self.proj(pooled.to(self.proj[0].weight.dtype)))  # [bs, 1]


class ThreshHeadTwoModel(nn.Module):
    """Predicts per-block adaptive threshold from draft hidden states.
    Concatenates mean-pool and last-position features, projects through
    two-layer bottleneck with residual, outputs sigmoid."""
    def __init__(self, hidden_size: int, bottleneck_dim: int = 256):
        super().__init__()
        self.down = nn.Linear(hidden_size * 2, bottleneck_dim)
        self.res = nn.Sequential(nn.Linear(bottleneck_dim, bottleneck_dim), nn.SiLU())
        self.out = nn.Linear(bottleneck_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [bs, block_size, hidden_size] -> [bs, 1]."""
        feat = torch.cat([x.mean(dim=-2), x[:, -1]], dim=-1)  # [bs, 2*hidden_size]
        h = nn.functional.silu(self.down(feat))  # [bs, bottleneck_dim]
        h = h + self.res(h)  # residual
        return torch.sigmoid(self.out(h))  # [bs, 1]


def _prob_head_candidate_len(
    probs: torch.Tensor,
    threshold: float,
    len_min: int,
    block_size: int,
) -> int:
    """Compute candidate_len from prob_head predictions.
    Truncate at first k where 1 - prod(p_1..p_k) > threshold."""
    probs_draft = probs[1:].float()
    cum_accept = torch.cumprod(probs_draft, dim=0)
    exceed = ((1 - cum_accept) > threshold).nonzero(as_tuple=True)[0]
    candidate_len = int(exceed[0].item()) + 1 if len(exceed) else block_size
    return max(len_min, min(candidate_len, block_size))


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3DFlashAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [(num_target_layers // 2)]
    start = 1
    end = num_target_layers - 3
    span = end - start
    target_layer_ids = [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]
    return target_layer_ids


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    offset = 1
    selected_states = []
    for layer_id in layer_ids:
        selected_states.append(hidden_states[layer_id + offset])
    target_hidden = torch.cat(selected_states, dim=-1)
    return target_hidden


class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [
                Qwen3DFlashDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        dflash_config = getattr(config, "dflash_config", {}) or {}
        self.target_layer_ids = dflash_config.get(
            "target_layer_ids",
            build_target_layer_ids(config.num_target_layers, config.num_hidden_layers),
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = config.block_size
        self.mask_token_id = dflash_config.get("mask_token_id", None)

        # Prob head for acceptance probability prediction
        self.use_prob_head = dflash_config.get("use_prob_head", False)
        if self.use_prob_head:
            prob_head_model = dflash_config.get("prob_head_model", "mlp")
            prob_head_num_layers = dflash_config.get("prob_head_num_layers", 2)
            prob_head_bottleneck_dim = dflash_config.get("prob_head_bottleneck_dim", 64)
            self.prob_head = ProbHead(config.hidden_size, prob_head_model,
                                     prob_head_num_layers, prob_head_bottleneck_dim)

        # Threshold heads: predict adaptive threshold for confidence-based early stopping
        self.use_thresh_head_subsequent = dflash_config.get("use_thresh_head_subsequent", False)
        if self.use_thresh_head_subsequent:
            th_bottleneck = dflash_config.get("thresh_head_bottleneck_dim", 256)
            self.thresh_head_subsequent = ThreshHeadSubsequent(config.vocab_size, th_bottleneck)

        self.use_thresh_head_two_model = dflash_config.get("use_thresh_head_two_model", False)
        if self.use_thresh_head_two_model:
            th_bottleneck = dflash_config.get("thresh_head_bottleneck_dim", 256)
            self.thresh_head_two_model = ThreshHeadTwoModel(config.hidden_size, th_bottleneck)

        # When True: adaptive length head output is candidate_len/block_size (no max_probs at inference)
        self.thresh_head_direct_len = dflash_config.get("thresh_head_direct_len", False)

        self.post_init()

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)

        # Compute acceptance probability logits if prob head is enabled
        prob_logits = None
        if self.use_prob_head:
            prob_logits = self.prob_head(hidden_states)

        return hidden_states, prob_logits

    def thresh_head_forward(self, hidden_states, logits):
        """Run threshold head(s) on draft output. Returns threshold tensor or None."""
        if self.use_thresh_head_subsequent:
            return self.thresh_head_subsequent(logits)
        if self.use_thresh_head_two_model:
            return self.thresh_head_two_model(hidden_states)
        return None

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
        prob_head_mul_threshold: float = 0.3,
        prob_head_candidate_len_min: int = 2,
    ):
        self.eval()
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens

        block_size = self.block_size
        output_ids = torch.full(
            (1, max_length + block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=target.device,
        )
        position_ids = torch.arange(
            output_ids.shape[1], device=target.device
        ).unsqueeze(0)

        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()

        # Prefill stage
        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )

        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
            output.logits, temperature
        )
        target_hidden = extract_context_feature(
            output.hidden_states, self.target_layer_ids
        )

        # Decode stage
        acceptance_lengths = []
        start = input_ids.shape[1]
        while start < max_length:
            # Always run draft on full block first; candidate_len is derived from prob_head output
            full_block_ids = output_ids[:, start : start + block_size].clone()
            noise_embedding = target.model.embed_tokens(full_block_ids)
            draft_output = self(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[
                    :, past_key_values_draft.get_seq_length() : start + block_size
                ],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )
            draft_hidden = draft_output[0]
            draft_logits = target.lm_head(draft_hidden[:, -block_size + 1 :, :])
            past_key_values_draft.crop(start)
            full_block_ids[:, 1:] = sample(draft_logits)

            # Determine candidate_len from prob_head if available, else full block
            if self.use_prob_head and draft_output[1] is not None:
                probs = torch.sigmoid(draft_output[1][0, :, 0])
                # print(f"[prob_head] probs ({len(probs)}): {[round(p, 3) for p in probs.tolist()]}")
                candidate_len = _prob_head_candidate_len(
                    probs, prob_head_mul_threshold, prob_head_candidate_len_min, block_size
                )
                # print(f"[prob_head] candidate_len={candidate_len}")
            else:
                candidate_len = block_size

            block_output_ids = full_block_ids[:, :candidate_len]
            block_position_ids = position_ids[:, start : start + candidate_len]

            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )

            posterior = sample(output.logits, temperature)
            if candidate_len > 1:
                acceptance_length = (
                    (block_output_ids[:, 1:] == posterior[:, :-1])
                    .cumprod(dim=1)
                    .sum(dim=1)[0]
                    .item()
                )
                # if self.use_prob_head:
                #     per_pos = (block_output_ids[0, 1:candidate_len] == posterior[0, :candidate_len - 1]).int().tolist()
                #     print(f"[true_accept] candidate_len={candidate_len}, per-pos: {per_pos}")
            else:
                acceptance_length = 0
            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
                :, : acceptance_length + 1
            ]
            output_ids[:, start + acceptance_length + 1] = posterior[
                :, acceptance_length
            ]
            start += acceptance_length + 1
            past_key_values_target.crop(start)
            target_hidden = extract_context_feature(
                output.hidden_states, self.target_layer_ids
            )[:, : acceptance_length + 1, :]
            acceptance_lengths.append(acceptance_length + 1)
            if stop_token_ids is not None and any(
                stop_token_id in output_ids[:, num_input_tokens:]
                for stop_token_id in stop_token_ids
            ):
                break
        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
        if stop_token_ids is not None:
            stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_token_indices = torch.isin(
                output_ids[0][num_input_tokens:], stop_token_ids
            ).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[
                    :, : num_input_tokens + stop_token_indices[0] + 1
                ]

        return output_ids
