# coding=utf-8
"""Reverse-KL OnlineDFlashModel wrapper (subclasses vendored OnlineDFlashModel)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from specforge.core.dflash import (
    OnlineDFlashModel,
    _target_token_prob,
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
)

from pipeline.training.utils import _dist_rank

class OnlineDFlashModelReverseKL(OnlineDFlashModel):
    """``OnlineDFlashModel`` with ``loss = alpha * CE_hard(teacher argmax) + (1-alpha) * KL(p_S||p_T)``."""

    def __init__(
        self,
        *,
        rkl_temperature: float = 1.0,
        rkl_alpha: float = 0.5,
        rkl_div_clip_tau: Optional[float] = None,
        train_thresh_head: bool = False,
        thresh_head_loss_type: str = "mse",
        thresh_label_lookahead: int = 1,
        detailed_debug_print: bool = False,
        **kwargs: object,
    ) -> None:
        kwargs["teacher_kd_temperature"] = None  # keep base ``forward`` unused if ever called
        super().__init__(**kwargs)
        self.rkl_temperature = float(rkl_temperature)
        self.rkl_alpha = float(rkl_alpha)
        self.rkl_div_clip_tau: Optional[float] = (
            None if rkl_div_clip_tau is None else float(rkl_div_clip_tau)
        )
        # Populated across forwards within one ``_train_step``; reset via ``reset_rkl_clip_stats``.
        self._rkl_clip_num: float = 0.0
        self._rkl_clip_den: float = 0.0

        # adaptive length head training
        self.train_thresh_head = train_thresh_head
        self.thresh_head_loss_type = thresh_head_loss_type
        self.thresh_label_lookahead = thresh_label_lookahead
        self.detailed_debug_print = detailed_debug_print
        self._debug_step_count = 0

    def reset_rkl_clip_stats(self) -> None:
        """Clear per-training-run RKL clip counters (called at start of ``_train_step``)."""
        self._rkl_clip_num = 0.0
        self._rkl_clip_den = 0.0
        self._debug_step_count = 0

    def consume_rkl_clip_ratio(self) -> Optional[float]:
        """Fraction of (flat, weight>0) positions where raw RKL exceeded ``tau``; ``None`` if clipping off."""
        tau = self.rkl_div_clip_tau
        if tau is None or tau <= 0:
            return None
        if self._rkl_clip_den <= 0.0:
            return None
        return self._rkl_clip_num / self._rkl_clip_den

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        teacher_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple:
        """Same as ``OnlineDFlashModel.forward`` up to logits; CE on teacher argmax + reverse KL."""
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        h_teacher = (
            teacher_hidden_states
            if teacher_hidden_states is not None
            else hidden_states
        )

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
        if isinstance(output_hidden, tuple):
            output_hidden, prob_logits = output_hidden
        else:
            prob_logits = None

        logits = self.lm_head(output_hidden)

        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )

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

        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            k = torch.arange(self.block_size, device=device).view(1, 1, -1)
            decay_weights = torch.exp(
                -(k - 1).clamp(min=0).float() / self.loss_decay_gamma
            )
            weight_mask = weight_mask * decay_weights

        flat_logits = logits.view(-1, logits.size(-1))
        flat_data_targets = target_ids.view(-1)
        flat_weights = weight_mask.view(-1)

        flat_teacher_targets: Optional[torch.Tensor] = None
        if not self.prob_head_only:
            valid_token_count = flat_weights.sum() + 1e-6
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
            flat_teacher_targets = torch.argmax(t_logits, dim=-1)
            loss_per_token = F.cross_entropy(
                flat_logits, flat_teacher_targets, reduction="none"
            )
            hard_loss = (loss_per_token * flat_weights).sum() / valid_token_count
            T = self.rkl_temperature
            p_s = F.softmax(flat_logits / T, dim=-1)
            log_p_t = F.log_softmax(t_logits.float() / T, dim=-1).to(flat_logits.dtype)
            # KL(p_S || p_T): F.kl_div(input=log p_T, target=p_S) -> p_S * (log p_S - log p_T)
            # (forward KL in dflash used log_p_s with p_t; here we swap for reverse KL.)
            kd_raw = F.kl_div(log_p_t.detach(), p_s, reduction="none").sum(dim=-1) * (T * T)
            kd_raw /= valid_token_count
            tau = self.rkl_div_clip_tau
            if tau is not None and tau > 0:
                with torch.no_grad():
                    active = flat_weights > 1e-8
                    den = float(active.sum().item())
                    if den > 0.0:
                        num = float(((kd_raw.detach() > float(tau)) & active).sum().item())
                        self._rkl_clip_num += num
                        self._rkl_clip_den += den
                kd_row = torch.clamp(kd_raw, max=float(tau))
            else:
                kd_row = kd_raw
            rkl_loss = (kd_row * flat_weights).sum()
            a = float(self.rkl_alpha)
            loss = a * hard_loss + (1.0 - a) * rkl_loss
        else:
            loss = torch.tensor(0.0, device=device)

        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            acc_targets = (
                flat_teacher_targets
                if flat_teacher_targets is not None
                else flat_data_targets
            )
            correct = (pred_ids == acc_targets) & (binary_eval_mask > 0.5)
            actual_token_count = binary_eval_mask.sum() + 1e-6
            accuracy = correct.sum().float() / actual_token_count

        if self.use_prob_head and prob_logits is not None:
            n = anchor_positions.shape[1]

            with torch.no_grad():
                if self.soft_prob_label:
                    draft_argmax = torch.argmax(
                        logits.view(bsz, n, self.block_size, -1), dim=-1
                    )
                    k_off = torch.arange(self.block_size - 1, device=device)
                    th_idx = (anchor_positions.unsqueeze(-1) + k_off).clamp(0, seq_len - 1)
                    bi = torch.arange(bsz, device=device)[:, None, None].expand_as(th_idx)
                    target_h = h_teacher[bi, th_idx].reshape(-1, h_teacher.shape[-1])
                    tok_flat = draft_argmax[:, :, 1:].reshape(-1)

                    q = _target_token_prob(target_h, tok_flat, self.lm_head)

                    accept_labels = torch.zeros(bsz, n, self.block_size, device=device)
                    accept_labels[:, :, 1:] = q.view(bsz, n, self.block_size - 1)
                else:
                    draft_pred_3d = torch.argmax(logits, dim=-1).view(bsz, -1, self.block_size)
                    accept_labels = (draft_pred_3d == target_ids).float()

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
                prob_loss_per_token = r * (r.log() - p_hat.log()) + (1 - r) * (
                    (1 - r).log() - (1 - p_hat).log()
                )
            else:
                prob_loss_per_token = F.binary_cross_entropy_with_logits(
                    flat_prob_logits, flat_accept_labels, reduction="none"
                )
            prob_valid_count = flat_prob_weights.sum() + 1e-6
            prob_head_loss = (prob_loss_per_token * flat_prob_weights).sum() / prob_valid_count

            loss = loss + prob_head_loss

            with torch.no_grad():
                prob_pred = torch.sigmoid(flat_prob_logits)
                mae = ((prob_pred - flat_accept_labels).abs() * flat_prob_weights).sum() / prob_valid_count
                mean_label = (flat_accept_labels * flat_prob_weights).sum() / prob_valid_count
                mean_pred = (prob_pred * flat_prob_weights).sum() / prob_valid_count
                extra = {
                    "train/prob_head_mae": mae,
                    "train/prob_head_mean_label": mean_label,
                    "train/prob_head_mean_pred": mean_pred,
                }

            return loss, accuracy, prob_head_loss, mae, extra

        # adaptive length head training (detached from draft model loss)
        if self.train_thresh_head:
            thresh_head = None
            if hasattr(self.draft_model, "use_thresh_head_two_model") and self.draft_model.use_thresh_head_two_model:
                thresh_head = self.draft_model.thresh_head_two_model
                head_type = "two_model"
            elif hasattr(self.draft_model, "use_thresh_head_subsequent") and self.draft_model.use_thresh_head_subsequent:
                thresh_head = self.draft_model.thresh_head_subsequent
                head_type = "subsequent"

            if thresh_head is not None:
                n = anchor_positions.shape[1]
                # Compute thresh labels (same as offline training)
                with torch.no_grad():
                    draft_pred_3d = torch.argmax(logits, dim=-1).view(bsz, n, self.block_size)
                    labels_hard = (draft_pred_3d == target_ids).float()

                    # Compute expected accept length
                    accept_per_pos = labels_hard[:, :, 1:].float()  # [bsz, n, block_size-1]
                    accept_lens = torch.cumprod(accept_per_pos, dim=2).sum(dim=2)  # [bsz, n]
                    expected_accept = accept_lens.float()

                    # Direct len prediction: label = (E[accept] + lookahead) / block_size
                    direct_len = getattr(self.draft_model, "thresh_head_direct_len", False)
                    if direct_len:
                        cand_len = (expected_accept + self.thresh_label_lookahead).clamp(0, self.block_size)
                        thresh_labels = (cand_len / self.block_size).clamp(0, 1)  # [bsz, n]
                    else:
                        # Threshold-based: compute 1 - cumprod(conf) at accept position
                        logits_3d = logits.view(bsz, n, self.block_size, -1)
                        conf = torch.softmax(logits_3d.float(), dim=-1).max(dim=-1).values  # [bsz, n, block_size]
                        conf_draft = conf[:, :, 1:]  # [bsz, n, block_size-1]
                        cum_conf = torch.cumprod(conf_draft, dim=2)  # [bsz, n, block_size-1]
                        safe_idx = (accept_lens.long() + self.thresh_label_lookahead).clamp(0, self.block_size - 2)
                        thresh_at_accept = 1 - cum_conf.gather(2, safe_idx.unsqueeze(-1)).squeeze(-1)
                        # Full-block acceptance: threshold 1.0
                        thresh_labels = torch.where(
                            accept_lens >= self.block_size - 1,
                            torch.ones_like(thresh_at_accept),
                            thresh_at_accept
                        ).clamp(0, 1)  # [bsz, n]

                    block_mask = block_keep_mask.float()  # [bsz, n]

                # adaptive length head forward with DETACHED hidden states (critical!)
                output_hidden_3d = output_hidden.view(bsz, n, self.block_size, -1)
                if head_type == "subsequent":
                    # Use logits (already computed, no gradient needed from adaptive length head)
                    with torch.no_grad():
                        logits_for_thresh = logits.view(bsz, n, self.block_size, -1)
                    # Reshape to [bsz*n, block_size, vocab_size] for adaptive length head
                    logits_flat = logits_for_thresh.reshape(bsz * n, self.block_size, -1)
                    thresh_pred = thresh_head(logits_flat).squeeze(-1)  # [bsz*n, 1] -> [bsz*n]
                    thresh_pred = thresh_pred.view(bsz, n)  # [bsz, n]
                else:
                    # Use DETACHED hidden states
                    # Reshape to [bsz*n, block_size, hidden_size] for adaptive length head
                    thresh_in = output_hidden_3d.detach().reshape(bsz * n, self.block_size, -1)
                    thresh_pred = thresh_head(thresh_in).squeeze(-1)  # [bsz*n, 1] -> [bsz*n]
                    thresh_pred = thresh_pred.view(bsz, n)  # [bsz, n]

                # Compute thresh loss (MSE or MAE)
                if self.thresh_head_loss_type == "mae":
                    loss_per_block = (thresh_pred - thresh_labels).abs()
                else:
                    loss_per_block = (thresh_pred - thresh_labels) ** 2
                valid_count = block_mask.sum() + 1e-6
                thresh_loss = (loss_per_block * block_mask).sum() / valid_count

                # Debug print for first 5 iterations on rank 0
                if self.detailed_debug_print and self._debug_step_count < 5 and _dist_rank() == 0:
                    print(f"\n[DEBUG Iter {self._debug_step_count + 1}] adaptive length head Training")
                    print(f"  head_type={head_type}, loss_type={self.thresh_head_loss_type}")
                    print(f"  thresh_loss={thresh_loss.item():.6f}")
                    if head_type != 'subsequent':
                        print(f"  thresh_in.requires_grad={thresh_in.requires_grad} (expected: False)")
                    print(f"  output_hidden.requires_grad={output_hidden.requires_grad}")
                    print(f"  thresh_labels: mean={thresh_labels.mean().item():.4f}, std={thresh_labels.std().item():.4f}")
                    print(f"  thresh_pred: mean={thresh_pred.mean().item():.4f}, std={thresh_pred.std().item():.4f}")
                    print(f"  Draft model loss (before thresh): {loss.item():.6f}")
                    self._debug_step_count += 1

                # Add thresh loss to total (detached, so no gradient flows to draft model)
                loss = loss + thresh_loss

                with torch.no_grad():
                    thresh_mae = ((thresh_pred - thresh_labels).abs() * block_mask).sum() / valid_count

                return loss, accuracy, thresh_loss, thresh_mae, {"train/thresh_mae": thresh_mae}

        return loss, accuracy, None, None, None
