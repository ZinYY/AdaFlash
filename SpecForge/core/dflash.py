# coding=utf-8
"""DFlash Training Wrapper."""
# ------------------------------------------------------------------------------
# Vendored from SpecForge (sgl-project), MIT License.
# Upstream: SpecForge/specforge/core/dflash.py
# Full attribution and license: asyn_train/specforge/VENDOR_README.md
# ------------------------------------------------------------------------------


from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.modeling.draft.dflash import DFlashDraftModel

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    BlockMask = None
    create_block_mask = None


def _target_token_prob(hidden: torch.Tensor, token_ids: torch.Tensor,
                       lm_head, chunk: int = 256, log: bool = False) -> torch.Tensor:
    """Target model's softmax prob (or log-prob) for specific tokens, chunked to avoid OOM."""
    N = hidden.shape[0]
    out = torch.empty(N, dtype=torch.float32, device=hidden.device)
    fn = F.log_softmax if log else F.softmax
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        p = fn(lm_head(hidden[i:j]).float(), dim=-1)
        out[i:j] = p[torch.arange(j - i, device=out.device), token_ids[i:j]]
    return out


def _token_log_prob(logits_flat: torch.Tensor, token_ids: torch.Tensor,
                    chunk: int = 256) -> torch.Tensor:
    """Log-softmax prob for specific tokens from pre-computed logits, chunked to avoid OOM."""
    N = logits_flat.shape[0]
    out = torch.empty(N, dtype=torch.float32, device=logits_flat.device)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        lp = F.log_softmax(logits_flat[i:j].float(), dim=-1)
        out[i:j] = lp[torch.arange(j - i, device=out.device), token_ids[i:j]]
    return out


def _target_argmax(hidden: torch.Tensor, lm_head, chunk: int = 256) -> torch.Tensor:
    """Target model's greedy token ids, chunked to avoid OOM."""
    N = hidden.shape[0]
    out = torch.empty(N, dtype=torch.long, device=hidden.device)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        out[i:j] = lm_head(hidden[i:j]).float().argmax(dim=-1)
    return out


def create_dflash_sdpa_mask(anchor_positions, block_keep_mask, S, block_size, device):
    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    q_indices = torch.arange(Q_LEN, device=device).view(1, 1, -1, 1)  # (1, 1, Q_LEN, 1)
    kv_indices = torch.arange(KV_LEN, device=device).view(
        1, 1, 1, -1
    )  # (1, 1, 1, KV_LEN)

    q_block_ids = q_indices // block_size

    anchor_expanded = anchor_positions.view(B, 1, N, 1).repeat_interleave(
        block_size, dim=2
    )

    mask_context = (kv_indices < S) & (kv_indices < anchor_expanded)

    is_draft = kv_indices >= S
    kv_block_ids = (kv_indices - S) // block_size
    mask_draft = is_draft & (q_block_ids == kv_block_ids)

    valid_block = block_keep_mask.view(B, 1, N, 1).repeat_interleave(block_size, dim=2)

    final_mask = (mask_context | mask_draft) & valid_block
    return final_mask


def create_dflash_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    block_size: int,
    device: torch.device,
):
    """Construct Flex Attention BlockMask for DFlash training.

    KV: [Context (S tokens) | Block_0 | Block_1 | ... | Block_{n-1}]
    Q:  [Block_0 | Block_1 | ... | Block_{n-1}]

    Rules:
      1. Each block sees context strictly before its anchor (kv_idx < anchor_pos).
      2. Intra-block attention is bidirectional.
      3. Different blocks are invisible to each other.
      4. Invalid blocks (block_keep_mask=False) see nothing.
    """

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        safe_q_block_id = q_block_id.clamp(max=N - 1)
        anchor_pos = anchor_positions[b, safe_q_block_id]

        is_context = kv_idx < S
        # Strictly less than: matches inference where target_hidden[anchor_pos]
        # is not available as context.
        mask_context = is_context & (kv_idx < anchor_pos)

        is_draft = kv_idx >= S
        kv_block_id = (kv_idx - S) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)

        is_valid_block = block_keep_mask[b, safe_q_block_id]
        in_bounds = q_block_id < N
        return (mask_context | mask_draft) & is_valid_block & in_bounds

    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    return create_block_mask(
        dflash_mask_mod, B=B, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device
    )


class OnlineDFlashModel(nn.Module):
    """DFlash online training wrapper with block-wise CE loss.

    The logits distillation term is added and mixed in on the same target positions only when teacher_kd_temperature is passed and > 0; 
    if not enabled, no extra computation branch is introduced.
    """

    def __init__(
        self,
        draft_model: DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        prob_head_only: bool = False,
        prob_head_label_type: str = "hard_label",
        prob_head_loss_type: str = "ce",
        teacher_kd_temperature: Optional[float] = None,
        teacher_kd_alpha: float = 0.5,
        prob_head_pos_weight: float = 1.0,
    ):
        super().__init__()
        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.attention_backend = attention_backend
        self.num_anchors = num_anchors
        self.loss_decay_gamma = loss_decay_gamma
        self.prob_head_only = prob_head_only
        self.use_prob_head = getattr(draft_model, "use_prob_head", False)
        self.prob_head_label_type = prob_head_label_type
        self.prob_head_loss_type = prob_head_loss_type
        self.teacher_kd_temperature = teacher_kd_temperature
        self.teacher_kd_alpha = teacher_kd_alpha
        self._teacher_kd_enabled = (
            teacher_kd_temperature is not None and teacher_kd_temperature > 0
        )
        self.prob_head_pos_weight = prob_head_pos_weight

        self._cached_block_mask: Optional[BlockMask] = None
        self._cached_seq_len: Optional[int] = None
        self._cached_bsz: Optional[int] = None

    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample anchor positions per sample; returns (anchors, keep_mask)."""
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        max_n = min(self.num_anchors, int(valid_counts.max().item()) - 1)

        if max_n <= 0:
            raise ValueError("should preprocess the data.")

        indices = (
            torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        )
        masked_indices = torch.where(
            valid, indices, torch.tensor(seq_len + 1, device=device)
        )

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        anchors = gathered[:, :max_n].sort(dim=1).values

        keep_mask = torch.arange(max_n, device=device).unsqueeze(
            0
        ) < valid_counts.unsqueeze(1).clamp(max=max_n)
        anchors = torch.where(
            keep_mask, anchors, torch.tensor(0, dtype=torch.long, device=device)
        )

        return anchors, keep_mask

    def prepare_noise_input(
        self, input_ids: torch.Tensor, block_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Prepare noise input: first token of each block is real, rest are MASK."""
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        if block_ids is not None:
            is_block_start = torch.ones(bsz, seq_len, dtype=torch.bool, device=device)
            is_block_start[:, 1:] = block_ids[:, 1:] != block_ids[:, :-1]
        else:
            positions = torch.arange(seq_len, device=device)
            is_block_start = (positions % self.block_size) == 0
            is_block_start = is_block_start.unsqueeze(0).expand(bsz, -1)

        noise_input_ids = torch.full_like(input_ids, self.mask_token_id)
        noise_input_ids[is_block_start] = input_ids[is_block_start]
        return noise_input_ids

    def _create_position_ids(self, anchor_positions: torch.Tensor) -> torch.Tensor:
        """Create absolute position IDs for parallel draft blocks."""
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        pos_ids = anchor_positions.unsqueeze(-1) + offsets
        return pos_ids.view(bsz, -1)

    def _create_noise_embed(self, input_ids, anchor_positions, block_keep_mask):
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        noise_ids = torch.full(
            (bsz, n * bs), self.mask_token_id, dtype=torch.long, device=device
        )

        block_starts = torch.arange(n, device=device) * bs
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)

        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.mask_token_id, dtype=torch.long, device=device),
        )

        return self.embed_tokens(noise_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        teacher_hidden_states: Optional[torch.Tensor] = None,
        debug: bool = False,
        last_hidden_states: Optional[torch.Tensor] = None,
        collect_data: bool = False,
    ) -> Tuple:
        """Parallel block-wise training forward pass.

        ``hidden_states`` may be multi-layer concatenation for the draft ``fc`` input.
        When ``teacher_hidden_states`` is set (typically last-layer [B,S,H]), it is used
        for ``lm_head`` (teacher KD, soft prob labels). Otherwise ``hidden_states`` is used.
        ``last_hidden_states`` when set is preferred for prob-head target argmax paths; if
        omitted, falls back to the same tensor as teacher KD.
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        h_teacher = (
            teacher_hidden_states
            if teacher_hidden_states is not None
            else hidden_states
        )
        h_prob = last_hidden_states if last_hidden_states is not None else h_teacher

        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )

        noise_embedding = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )

        context_position_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        )
        draft_position_ids = self._create_position_ids(anchor_positions)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)

        if self.attention_backend == "flex_attention":
            dflash_attn_mask = create_dflash_block_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=self.block_size,
                device=device,
            )
        else:
            dflash_attn_mask = create_dflash_sdpa_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=self.block_size,
                device=device,
            )

        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=dflash_attn_mask,
        )
        # forward returns (hidden_states, prob_logits)
        if isinstance(output_hidden, tuple):
            output_hidden, prob_logits = output_hidden
        else:
            prob_logits = None

        logits = self.lm_head(output_hidden)

        # --- Labels: same-position prediction (position k predicts token anchor+k) ---
        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )

        # --- Weight mask: block validity * bounds * exclude anchor (pos 0) * loss_mask ---
        weight_mask = (
            block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        )
        weight_mask = weight_mask * valid_label_mask.float()

        pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()

        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        binary_eval_mask = weight_mask.view(-1)

        # --- Loss decay: exp(-(k-1)/γ) so k=1 (1st prediction) gets weight 1.0 ---
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            k = torch.arange(self.block_size, device=device).view(1, 1, -1)
            decay_weights = torch.exp(
                -(k - 1).clamp(min=0).float() / self.loss_decay_gamma
            )
            weight_mask = weight_mask * decay_weights

        # --- Cross entropy (draft model loss) ---
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = target_ids.view(-1)
        flat_weights = weight_mask.view(-1)

        if not self.prob_head_only:
            loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
            valid_token_count = flat_weights.sum() + 1e-6
            hard_loss = (loss_per_token * flat_weights).sum() / valid_token_count
            loss = hard_loss
            if self._teacher_kd_enabled:
                # Hinton KD: T^2 * KL; teacher logits from frozen hidden_states + lm_head.
                # Teacher distribution at the same next-token positions as CE (causal: h[pos-1] -> token pos).
                n = anchor_positions.size(1)
                n2 = n * self.block_size
                lin = torch.arange(flat_logits.size(0), device=device, dtype=torch.long)
                bb = lin // n2
                rem = lin % n2
                nn = rem // self.block_size
                kk = rem % self.block_size
                lab = safe_label_indices[bb, nn, kk]
                ptr = (lab - 1).clamp(min=0)
                h_sel = h_teacher[bb, ptr].detach()
                t_logits = self.lm_head(h_sel.to(flat_logits.dtype))
                T = self.teacher_kd_temperature
                log_p_s = F.log_softmax(flat_logits / T, dim=-1)
                p_t = F.softmax(t_logits / T, dim=-1)
                kd_row = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1) * (T * T)
                soft_loss = (kd_row * flat_weights).sum() / valid_token_count
                a = float(self.teacher_kd_alpha)
                loss = a * hard_loss + (1.0 - a) * soft_loss
        else:
            loss = torch.tensor(0.0, device=device)

        # --- Accuracy ---
        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            actual_token_count = binary_eval_mask.sum() + 1e-6
            accuracy = correct.sum().float() / actual_token_count

        # --- Data collection: compute all three label types and return raw data ---
        _needs_collect = self.use_prob_head or getattr(self.draft_model, "use_thresh_head_subsequent", False) or getattr(self.draft_model, "use_thresh_head_two_model", False)
        if collect_data and _needs_collect and h_prob is not None:
            with torch.no_grad():
                n = anchor_positions.shape[1]
                draft_logits_3d = logits.view(bsz, n, self.block_size, -1)
                k_off = torch.arange(self.block_size - 1, device=device)
                th_idx = (anchor_positions.unsqueeze(-1) + k_off).clamp(0, seq_len - 1)
                bi_idx = torch.arange(bsz, device=device)[:, None, None].expand_as(th_idx)
                target_h = h_prob[bi_idx, th_idx]
                target_h_flat = target_h.reshape(-1, h_prob.shape[-1])
                draft_argmax = torch.argmax(draft_logits_3d, dim=-1)
                tok_flat = draft_argmax[:, :, 1:].reshape(-1)

                # hard_label: draft argmax == target argmax
                target_toks = _target_argmax(target_h_flat, self.lm_head)
                labels_hard = torch.zeros(bsz, n, self.block_size, device=device)
                labels_hard[:, :, 1:] = (
                    draft_argmax[:, :, 1:].reshape(-1) == target_toks
                ).float().view(bsz, n, self.block_size - 1)

                # q(draft_argmax): target softmax prob at draft's greedy token
                q = _target_token_prob(target_h_flat, tok_flat, self.lm_head)
                labels_q = torch.zeros(bsz, n, self.block_size, device=device)
                labels_q[:, :, 1:] = q.view(bsz, n, self.block_size - 1)

                # q/p: SpecDec++ acceptance probability
                log_q = _target_token_prob(target_h_flat, tok_flat, self.lm_head, log=True)
                block_offsets = torch.arange(1, self.block_size, device=device)
                batch_base = torch.arange(bsz, device=device)[:, None, None] * (n * self.block_size)
                block_base = torch.arange(n, device=device)[None, :, None] * self.block_size
                flat_idx = (batch_base + block_base + block_offsets).reshape(-1)
                draft_logits_sel = logits.view(-1, logits.size(-1))[flat_idx]
                log_p = _token_log_prob(draft_logits_sel, tok_flat)
                labels_qp = torch.zeros(bsz, n, self.block_size, device=device)
                labels_qp[:, :, 1:] = (log_q - log_p).clamp(max=0).exp().view(bsz, n, self.block_size - 1)

                target_hidden_blocks = torch.zeros(
                    bsz,
                    n,
                    self.block_size,
                    h_prob.shape[-1],
                    device=device,
                    dtype=target_h.dtype,
                )
                target_hidden_blocks[:, :, 1:, :] = target_h

            return torch.tensor(0.0, device=device), accuracy, None, None, {
                'output_hidden': output_hidden.detach().reshape(bsz, n, self.block_size, -1),
                'draft_embedding': noise_embedding.detach().reshape(bsz, n, self.block_size, -1),
                'target_last_hidden': target_hidden_blocks.detach(),
                'target_token_ids': target_ids.detach(),
                'anchor_positions': anchor_positions.detach(),
                'labels_hard': labels_hard,
                'labels_q': labels_q,
                'labels_qp': labels_qp,
                'mask': binary_eval_mask.view(bsz, n, self.block_size),
                'block_keep_mask': block_keep_mask,
            }

        # --- Prob head loss ---
        if self.use_prob_head and prob_logits is not None:
            n = anchor_positions.shape[1]

            with torch.no_grad():
                assert h_prob is not None, (
                    "prob head requires last_hidden_states or teacher hidden for target lm_head logits"
                )
                k_off = torch.arange(self.block_size - 1, device=device)
                th_idx = (anchor_positions.unsqueeze(-1) + k_off).clamp(0, seq_len - 1)
                bi = torch.arange(bsz, device=device)[:, None, None].expand_as(th_idx)
                target_h = h_prob[bi, th_idx].reshape(-1, h_prob.shape[-1])

                if self.prob_head_label_type == "hard_label":
                    draft_pred_3d = torch.argmax(logits, dim=-1).view(bsz, -1, self.block_size)
                    target_toks = _target_argmax(target_h, self.lm_head)
                    accept_labels = torch.zeros(bsz, n, self.block_size, device=device)
                    accept_labels[:, :, 1:] = (
                        draft_pred_3d[:, :, 1:].reshape(-1) == target_toks
                    ).float().view(bsz, n, self.block_size - 1)
                else:
                    draft_argmax = torch.argmax(
                        logits.view(bsz, n, self.block_size, -1), dim=-1
                    )
                    tok_flat = draft_argmax[:, :, 1:].reshape(-1)
                    accept_labels = torch.zeros(bsz, n, self.block_size, device=device)
                    if self.prob_head_label_type == "q/p":
                        # SpecDec++ acceptance prob: min(1, q/p) = exp(min(0, log_q - log_p))
                        log_q = _target_token_prob(target_h, tok_flat, self.lm_head, log=True)
                        block_offsets = torch.arange(1, self.block_size, device=device)
                        batch_base = torch.arange(bsz, device=device)[:, None, None] * (n * self.block_size)
                        block_base = torch.arange(n, device=device)[None, :, None] * self.block_size
                        flat_idx = (batch_base + block_base + block_offsets).reshape(-1)
                        draft_logits_sel = logits.view(-1, logits.size(-1))[flat_idx]
                        log_p = _token_log_prob(draft_logits_sel, tok_flat)
                        accept_labels[:, :, 1:] = (log_q - log_p).clamp(max=0).exp().view(bsz, n, self.block_size - 1)
                    else:
                        q = _target_token_prob(target_h, tok_flat, self.lm_head)
                        accept_labels[:, :, 1:] = q.view(bsz, n, self.block_size - 1)

            # detach() hidden states so prob head loss does NOT backprop through draft model
            prob_logits_3d = prob_logits.view(bsz, -1, self.block_size, 1).squeeze(-1)
            if not self.prob_head_only:
                prob_logits_3d = self.draft_model.prob_head(output_hidden.detach()).view(
                    bsz, -1, self.block_size, 1
                ).squeeze(-1)

            flat_prob_logits = prob_logits_3d.view(-1)
            flat_accept_labels = accept_labels.view(-1)
            flat_prob_weights = binary_eval_mask

            if self.prob_head_loss_type == "kl":
                eps = 1e-9
                p_hat = torch.sigmoid(flat_prob_logits).clamp(eps, 1 - eps)
                r = flat_accept_labels.clamp(eps, 1 - eps)
                prob_loss_per_token = r * (r.log() - p_hat.log()) + (1 - r) * ((1 - r).log() - (1 - p_hat).log())
            else:  # ce
                pos_weight = torch.tensor(self.prob_head_pos_weight, device=flat_prob_logits.device)
                prob_loss_per_token = F.binary_cross_entropy_with_logits(
                    flat_prob_logits, flat_accept_labels, reduction="none", pos_weight=pos_weight
                )
            prob_valid_count = flat_prob_weights.sum() + 1e-6
            prob_head_loss = (prob_loss_per_token * flat_prob_weights).sum() / prob_valid_count

            loss = loss + prob_head_loss

            with torch.no_grad():
                prob_pred = torch.sigmoid(flat_prob_logits)
                mae = ((prob_pred - flat_accept_labels).abs() * flat_prob_weights).sum() / prob_valid_count
                mean_label = (flat_accept_labels * flat_prob_weights).sum() / prob_valid_count
                mean_pred = (prob_pred * flat_prob_weights).sum() / prob_valid_count
                # Binary accept/reject accuracy: pred>0.5 matches label>0.5
                bin_correct = ((prob_pred > 0.5) == (flat_accept_labels > 0.5)).float()
                bin_acc = (bin_correct * flat_prob_weights).sum() / prob_valid_count
                extra = {
                    "train/prob_head_mae": mae,
                    "train/prob_head_mean_label": mean_label,
                    "train/prob_head_mean_pred": mean_pred,
                    "train/prob_head_bin_acc": bin_acc,
                }
                if debug:
                    a_all = accept_labels[0]  # [n, block_size]
                    blk_sums = a_all[:, 1:].sum(dim=-1)
                    blk_weights = (block_keep_mask[0]).float()
                    valid_blocks = block_keep_mask[0].sum().int().item()
                    valid_idx = block_keep_mask[0].nonzero(as_tuple=True)[0]
                    sample_blks = [valid_idx[0], valid_idx[len(valid_idx)//2], valid_idx[-1]]

                    draft_pred_3d_0 = torch.argmax(logits, dim=-1).view(bsz, n, self.block_size)

                    sample_info = []
                    for bi in sample_blks:
                        bi = bi.item()
                        blk_th = h_prob[0, th_idx[0, bi]]
                        draft_tok = draft_pred_3d_0[0, bi, 1:].long()
                        tgt_tok = _target_argmax(blk_th, self.lm_head)
                        q_vals = _target_token_prob(blk_th, draft_tok, self.lm_head)
                        log_q = _target_token_prob(blk_th, draft_tok, self.lm_head, log=True)
                        draft_logits_blk = logits.view(bsz, n, self.block_size, -1)[0, bi, 1:]
                        log_p = _token_log_prob(draft_logits_blk, draft_tok)
                        qp_vals = (log_q - log_p).clamp(max=0).exp()

                        info = {
                            "blk": bi,
                            "anchor": anchor_positions[0, bi].item(),
                            "accept": a_all[bi, 1:].int().tolist(),
                            "draft": draft_pred_3d_0[0, bi, 1:].tolist(),
                            "target": tgt_tok.tolist(),
                            "q": [round(v, 3) for v in q_vals.tolist()],
                            "qp": [round(v, 3) for v in qp_vals.tolist()],
                        }
                        prob_probs = torch.sigmoid(prob_logits_3d[0, bi]).tolist()
                        info["prob_pred"] = [round(v, 3) for v in prob_probs]
                        sample_info.append(info)
                    extra["_debug"] = {
                        "valid_blocks": valid_blocks,
                        "mean_accept_per_block": (blk_sums * blk_weights).sum().item() / (blk_weights.sum().item() + 1e-6),
                        "sample_blocks": sample_info,
                        "acc": accuracy.item(),
                        "loss": loss.item(),
                        "prob_head_loss": prob_head_loss.item(),
                    }

            return loss, accuracy, prob_head_loss, mae, extra

        return loss, accuracy, None, None, None
