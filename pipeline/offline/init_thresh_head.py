from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

def _ensure_repo_pythonpath() -> None:
    """Vendored specforge under asyn_train; sglang_dflash for default SGLang target."""
    from pathlib import Path

    from pipeline.bootstrap import ensure_vendored_specforge
    from pipeline.paths import ASYN_TRAIN_ROOT, REPO_ROOT

    ensure_vendored_specforge()
    sp = str(REPO_ROOT / "sglang_dflash/python")
    if Path(sp).is_dir() and sp not in sys.path:
        sys.path.insert(0, sp)


_ensure_repo_pythonpath()

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from specforge.modeling.draft.dflash import DFlashDraftModel, extract_context_feature


class OfflineTargetAdapter(torch.nn.Module):
    """Wrap HF causal LM or ``Qwen3_5ForConditionalGeneration``-style MLLM for this script.

    Exposes ``.model.embed_tokens`` (text tower) and ``.lm_head`` like ``AutoModelForCausalLM``.
    """

    def __init__(self, base: torch.nn.Module) -> None:
        super().__init__()
        self.base = base
        inner = getattr(base, "model", base)
        self.model = getattr(inner, "language_model", inner)
        head = None
        if hasattr(base, "get_output_embeddings"):
            try:
                head = base.get_output_embeddings()
            except Exception:
                head = None
        if head is None:
            head = getattr(base, "lm_head", None)
        if head is None and hasattr(inner, "lm_head"):
            head = inner.lm_head
        if head is None and hasattr(self.model, "lm_head"):
            head = self.model.lm_head
        if head is None:
            raise RuntimeError(
                "Could not resolve lm_head on target model; unsupported architecture for offline init."
            )
        self.lm_head = head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        **kwargs: object,
    ) -> object:
        kwargs.pop("use_cache", None)
        if getattr(self.base.config, "model_type", None) == "qwen3_5" and "pixel_values" not in kwargs:
            kwargs["pixel_values"] = None
        return self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            use_cache=False,
            **kwargs,
        )


def _load_hf_target_any(
    src: str,
    device: torch.device,
    *,
    trust_remote_code: bool,
) -> torch.nn.Module:
    """``AutoModelForCausalLM`` when possible; else ``AutoModel`` (e.g. Qwen3.5 on Hub)."""
    common: dict = dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
    )
    try:
        m = AutoModelForCausalLM.from_pretrained(src, **common)
    except ValueError as err:
        msg = str(err).lower()
        if "unrecognized configuration" not in msg and "unrecognized config" not in msg:
            raise
        from transformers import AutoModel

        m = AutoModel.from_pretrained(src, **common)
    m = m.to(device).eval()
    return OfflineTargetAdapter(m)


def _pretrained_source(path_or_id: str) -> str:
    """Local HF checkpoint dir (absolute path) if ``path_or_id`` is such a directory; else Hub id unchanged.

    Avoids ``Path(...).resolve()`` on strings like ``z-lab/Qwen3-8B-DFlash-b16``, which would
    incorrectly become a filesystem path under cwd.
    """
    s = (path_or_id or "").strip()
    if not s:
        raise ValueError("pretrained path or repo id is empty")
    expanded = os.path.abspath(os.path.expanduser(s))
    if os.path.isdir(expanded) and os.path.isfile(os.path.join(expanded, "config.json")):
        return expanded
    return s


def _full_text_from_jsonl_row(example: Dict[str, Any]) -> str:
    """Concatenate pre-rendered chat strings (e.g. ``regenerate_dataset`` rows)."""
    try:
        p = example["prompt"]
        r = example["response"]
    except KeyError as e:
        raise ValueError("Each JSONL row must contain `prompt` and `response`.") from e
    if not isinstance(p, str) or not isinstance(r, str):
        raise TypeError("`prompt` and `response` must be strings.")
    return p + r


def build_hf_dataset(tokenizer, data_path: str, max_length: int) -> Any:
    raw = load_dataset("json", data_files=data_path)["train"]

    def tokenize(example: Dict[str, Any]) -> Dict[str, Any]:
        text = _full_text_from_jsonl_row(example)
        enc = tokenizer(text, max_length=max_length, truncation=True, padding=False)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    return raw.map(tokenize, remove_columns=raw.column_names, num_proc=4)


def collate_pad(batch: List[Dict[str, Any]], pad_id: int, max_length: int) -> Dict[str, torch.Tensor]:
    input_ids = [torch.tensor(x["input_ids"][:max_length], dtype=torch.long) for x in batch]
    attn_masks = [torch.tensor(x["attention_mask"][:max_length], dtype=torch.long) for x in batch]
    padded_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    padded_mask = torch.nn.utils.rnn.pad_sequence(attn_masks, batch_first=True, padding_value=0)
    return {"input_ids": padded_ids, "attention_mask": padded_mask}


def _freeze_module(m: object) -> None:
    """No-op for non-``nn.Module`` (e.g. ``SGLangDFlashTargetModel`` is not a submodule)."""
    if not isinstance(m, torch.nn.Module):
        return
    m.eval()
    for p in m.parameters():
        p.requires_grad = False


def _ensure_single_process_dist_env() -> None:
    """Allow ``init_distributed`` without ``torchrun`` (single rank 0 / world_size 1)."""
    if "RANK" in os.environ:
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29500 + (os.getpid() % 2500)))
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"


def compute_thresh_head_labels_rkl(
    logits: torch.Tensor,
    *,
    input_ids_row: torch.Tensor,
    anchor: int,
    padded_seq_len: int,
    block_size: int,
    thresh_label_lookahead: int,
    draft_direct_len: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Thresh labels identical to ``training_worker_rkl.OnlineDFlashModelReverseKL`` (thresh block).

    ``logits``: ``[1, block_size, vocab]`` from ``lm_head(draft_hidden)``.
    ``accept_lens`` = ``sum(cumprod(labels_hard[:, :, 1:]))`` where ``labels_hard`` is
    ``(argmax(logits) == target_ids)`` and ``target_ids`` comes from ``gather`` with
    ``safe_label_indices = (anchor + arange(B)).clamp(max=padded_seq_len - 1)``.

    Returns ``(thresh_labels, expected_accept)``, each shape ``[1]`` for ``torch.cat``.
    """
    device = logits.device
    offsets = torch.arange(block_size, device=device, dtype=torch.long)
    label_indices = anchor + offsets
    safe_label_indices = label_indices.clamp(max=padded_seq_len - 1)
    target_ids_flat = input_ids_row[safe_label_indices]

    draft_pred_3d = torch.argmax(logits, dim=-1).view(1, 1, block_size)
    target_3d = target_ids_flat.view(1, 1, block_size)
    labels_hard = (draft_pred_3d == target_3d).float()

    accept_per_pos = labels_hard[:, :, 1:].float()
    accept_lens = torch.cumprod(accept_per_pos, dim=2).sum(dim=2)
    expected_accept = accept_lens.float()
    la = int(thresh_label_lookahead)

    if draft_direct_len:
        cand_len = (expected_accept + la).clamp(0, block_size)
        thresh_labels = (cand_len / float(block_size)).clamp(0, 1)
    else:
        logits_3d = logits.view(1, 1, block_size, -1)
        conf = torch.softmax(logits_3d.float(), dim=-1).max(dim=-1).values
        conf_draft = conf[:, :, 1:]
        cum_conf = torch.cumprod(conf_draft, dim=2)
        safe_idx = (accept_lens.long() + la).clamp(0, block_size - 2)
        thresh_at_accept = 1 - cum_conf.gather(2, safe_idx.unsqueeze(-1)).squeeze(-1)
        thresh_labels = torch.where(
            accept_lens >= block_size - 1,
            torch.ones_like(thresh_at_accept),
            thresh_at_accept,
        ).clamp(0, 1)

    return thresh_labels.reshape(-1), expected_accept.reshape(-1)


def _collect_one_microbatch(
    *,
    target_backend: str,
    sglang_target: Optional[torch.nn.Module],
    hf_target: Optional[torch.nn.Module],
    embed_tokens: torch.nn.Module,
    draft: DFlashDraftModel,
    lm_head: torch.nn.Module,
    input_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    mask_token_id: int,
    num_anchors: int,
    block_size: int,
    label_lookahead: int,
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Returns (hidden, labels, mask, accept_lens) or None if no valid blocks."""
    target_layer_ids = draft.target_layer_ids
    bsz, seq_len = input_ids.shape
    h_list: List[torch.Tensor] = []
    y_list: List[torch.Tensor] = []
    m_list: List[torch.Tensor] = []
    acc_list: List[torch.Tensor] = []

    with torch.no_grad():
        if target_backend == "sglang":
            assert sglang_target is not None
            loss_mask = attn_mask.clone()
            to = sglang_target.generate_dflash_data(input_ids, attn_mask, loss_mask)
            ctx_all = to.hidden_states
        else:
            assert hf_target is not None
            out = hf_target(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )
            ctx_all = extract_context_feature(out.hidden_states, target_layer_ids)
        valid_lens = attn_mask.sum(dim=1)

        for b in range(bsz):
            vlen = int(valid_lens[b].item())
            min_pos = block_size
            max_pos = vlen - block_size
            if max_pos <= min_pos:
                continue
            n_try = min(num_anchors, max_pos - min_pos)
            perm = torch.randperm(max_pos - min_pos, device=device)[:n_try] + min_pos
            anchor_positions = perm.sort().values

            for anchor in anchor_positions:
                anchor = int(anchor.item())
                ctx = ctx_all[b : b + 1, :anchor, :]
                block_ids = input_ids[b, anchor : anchor + block_size].clone()
                noise_ids = torch.full((1, block_size), mask_token_id, dtype=torch.long, device=device)
                noise_ids[0, 0] = block_ids[0]
                noise_emb = embed_tokens(noise_ids)

                ctx_len = ctx.shape[1]
                ctx_pos = torch.arange(ctx_len, device=device)
                draft_pos = torch.arange(anchor, anchor + block_size, device=device)
                pos_ids = torch.cat([ctx_pos, draft_pos]).unsqueeze(0)

                draft_out = draft(
                    target_hidden=ctx,
                    noise_embedding=noise_emb,
                    position_ids=pos_ids,
                )
                draft_hidden = draft_out[0] if isinstance(draft_out, tuple) else draft_out.last_hidden_state
                logits = lm_head(draft_hidden)  # [1, block_size, V]

                thresh_labels, accept_lens = compute_thresh_head_labels_rkl(
                    logits,
                    input_ids_row=input_ids[b],
                    anchor=anchor,
                    padded_seq_len=seq_len,
                    block_size=block_size,
                    thresh_label_lookahead=label_lookahead,
                    draft_direct_len=bool(draft.thresh_head_direct_len),
                )

                h_list.append(draft_hidden.squeeze(0))
                # Keep 1D shape [1]; avoid squeeze -> 0-dim scalar (torch.cat cannot concat).
                y_list.append(thresh_labels.reshape(-1))
                m_list.append(torch.ones(1, device=device))
                acc_list.append(accept_lens.reshape(-1))

    if not h_list:
        return None
    return (
        torch.stack(h_list, dim=0).to(torch.bfloat16),
        torch.cat(y_list, dim=0).to(torch.bfloat16),
        torch.cat(m_list, dim=0).to(torch.bfloat16),
        torch.cat(acc_list, dim=0).to(torch.bfloat16),
    )


def offline_train_thresh_head_two_model(
    *,
    target_model_path: str,
    draft_config_path: str,
    initial_draft_path: str,
    dataset_path: str,
    output_dir: str,
    max_length: int = 4096,
    num_anchors: int = 16,
    epochs: int = 1,
    batch_size: int = 1,
    micro_batch_blocks: int = 8,
    learning_rate: float = 2e-4,
    thresh_label_lookahead: int = 1,
    thresh_head_loss_type: str = "mse",
    log_every: int = 20,
    log_rolling_steps: int = 50,
    log_thresh_head_threshold_rate: float = 1.0,
    log_candidate_len_min: Optional[int] = None,
    max_samples: Optional[int] = None,
    device: Optional[str] = None,
    trust_remote_code: bool = True,
    copy_remote_code: bool = True,
    target_model_backend: str = "sglang",
    tp_size: int = 1,
    dist_timeout: int = 30,
    sglang_backend_kwargs: Optional[Dict[str, Any]] = None,
) -> Path:
    """Train ``thresh_head_two_model`` only; save full draft to ``output_dir``.

    Thresh regression targets match ``training_worker_rkl`` (``accept_lens`` from hard
    draft-vs-teacher token matches, then ``direct_len`` scaling). ``initial_draft_path``:
    local DFlash dir or Hub id. ``target_model_path``: same rule.

    Returns the resolved output path.
    """
    import torch.distributed as dist

    from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type != "cuda":
        raise RuntimeError("CUDA is required for this script (target + draft size).")

    os.makedirs(output_dir, exist_ok=True)

    target_src = _pretrained_source(target_model_path)
    tokenizer = AutoTokenizer.from_pretrained(target_src, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[load] draft config {draft_config_path}")
    draft_config = AutoConfig.from_pretrained(draft_config_path, trust_remote_code=trust_remote_code)
    if not getattr(draft_config, "dflash_config", None):
        draft_config.dflash_config = {}
    dc = draft_config.dflash_config
    # Must match draft training / inference (e.g. Qwen3.5 uses 248070 in JSON). Do not overwrite
    # with tokenizer default 151669 — wrong mask embeddings collapse RKL accept streak.
    if dc.get("mask_token_id") is not None:
        mask_token_id = int(dc["mask_token_id"])
    else:
        if tokenizer.mask_token_id is None:
            tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        mask_token_id = int(tokenizer.mask_token_id or 151669)
        dc["mask_token_id"] = mask_token_id
    print(f"[load] mask_token_id={mask_token_id} (from draft JSON if set, else tokenizer)")

    dist_inited = False
    sglang_target: Optional[torch.nn.Module] = None
    hf_target: Optional[torch.nn.Module] = None
    try:
        if target_model_backend == "sglang":
            from specforge.distributed import init_distributed
            from specforge.modeling.target.dflash_target_model import get_dflash_target_model

            if not dist.is_initialized():
                _ensure_single_process_dist_env()
            init_distributed(timeout=dist_timeout, tp_size=tp_size)
            dist_inited = True
            print(f"[load] target (sglang) from {target_src!r}")
            sglang_target = get_dflash_target_model(
                pretrained_model_name_or_path=target_src,
                backend="sglang",
                torch_dtype=torch.bfloat16,
                device=None,
                trust_remote_code=trust_remote_code,
                **(sglang_backend_kwargs or {}),
            )
            _freeze_module(sglang_target)
        else:
            print(f"[load] target (hf) from {target_src!r}")
            hf_target = _load_hf_target_any(target_src, dev, trust_remote_code=trust_remote_code)
            _freeze_module(hf_target)

        print("[load] target embed + lm_head (safetensors slice, same as training_worker)")
        target_components = TargetEmbeddingsAndHead.from_pretrained(
            target_src,
            device=str(dev),
            trust_remote_code=trust_remote_code,
            dtype=torch.bfloat16,
        )
        _freeze_module(target_components)
        embed_tokens = target_components.embed_tokens
        lm_head = target_components.lm_head

        draft = DFlashDraftModel(draft_config).to(dev).to(torch.bfloat16)
        draft_src = _pretrained_source(initial_draft_path)
        print(f"[load] base draft from {draft_src!r} (local checkpoint or Hub id)")
        loaded = DFlashDraftModel.from_pretrained(
            draft_src,
            config=draft_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=trust_remote_code,
        )
        incomp = draft.load_state_dict(loaded.state_dict(), strict=False)
        del loaded
        if incomp.missing_keys:
            th_missing = [k for k in incomp.missing_keys if "thresh_head" in k]
            other = [k for k in incomp.missing_keys if "thresh_head" not in k]
            if th_missing:
                print(f"[load] {len(th_missing)} thresh-head keys missing in base (random init, then trained)")
            if other:
                preview = other[:12]
                print(
                    f"[load] WARNING: {len(other)} non-thresh keys missing in base "
                    f"(preview: {preview}{'...' if len(other) > 12 else ''})"
                )
        if incomp.unexpected_keys:
            print(f"[load] unexpected keys in base checkpoint (ignored): {incomp.unexpected_keys[:16]}")

        if not getattr(draft, "use_thresh_head_two_model", False) or draft.thresh_head_two_model is None:
            raise RuntimeError(
                "--draft-config-path must set dflash_config.use_thresh_head_two_model=true "
                "(see asyn_train/configs/qwen3-8b-dflash-thresh-head.json)."
            )

        if sglang_target is not None:
            sglang_target.set_capture_layers(draft.target_layer_ids)

        _freeze_module(draft)
        for p in draft.thresh_head_two_model.parameters():
            p.requires_grad = True
        draft.thresh_head_two_model.train()

        block_size = int(draft.block_size)
        cand_len_min = (
            int(log_candidate_len_min)
            if log_candidate_len_min is not None
            else int(getattr(draft, "prob_head_candidate_len_min", 2))
        )
        stats_deque: Deque[Tuple[float, float, float]] = deque(maxlen=max(1, int(log_rolling_steps)))

        ds = build_hf_dataset(tokenizer, dataset_path, max_length)
        n = len(ds)
        if max_samples is not None and max_samples < n:
            ds = ds.select(range(max_samples))
        print(f"[data] {dataset_path} -> {len(ds)} rows (max_length={max_length})")

        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_pad(b, int(tokenizer.pad_token_id), max_length),
        )

        th = draft.thresh_head_two_model
        th_dtype = next(th.parameters()).dtype
        opt = torch.optim.AdamW(th.parameters(), lr=learning_rate)
        rate = float(log_thresh_head_threshold_rate)

        def _after_step_log(
            pred_t: torch.Tensor,
            labels_t: torch.Tensor,
            mask_t: torch.Tensor,
            accept_t: torch.Tensor,
            *,
            step: int,
        ) -> None:
            with torch.no_grad():
                msum = mask_t.float().sum().clamp(min=1e-6)
                avg_acc = (accept_t.float() * mask_t.float()).sum() / msum
                avg_lbl_cand = (labels_t.float() * float(block_size) * mask_t.float()).sum() / msum
                verify = (pred_t.float() * float(block_size) * rate).round().clamp(
                    float(cand_len_min), float(block_size)
                )
                avg_ver = (verify * mask_t.float()).sum() / msum
            stats_deque.append((avg_acc.item(), avg_lbl_cand.item(), avg_ver.item()))
            if log_every > 0 and step % log_every == 0 and stats_deque:
                nwin = len(stats_deque)
                ma = sum(x[0] for x in stats_deque) / nwin
                ml = sum(x[1] for x in stats_deque) / nwin
                mv = sum(x[2] for x in stats_deque) / nwin
                print(
                    f"[train] step={step} rolling={nwin}/{log_rolling_steps} "
                    f"avg_accept_len={ma:.3f} avg_label_cand_len={ml:.3f} "
                    f"avg_pred_verify_len={mv:.3f} (thresh_rate={rate}, cand_min={cand_len_min}, B={block_size})",
                    flush=True,
                )

        global_step = 0
        for ep in range(epochs):
            pbar = tqdm(loader, desc=f"epoch {ep + 1}/{epochs}")
            accum_h: List[torch.Tensor] = []
            accum_y: List[torch.Tensor] = []
            accum_m: List[torch.Tensor] = []
            accum_a: List[torch.Tensor] = []

            for batch in pbar:
                input_ids = batch["input_ids"].to(dev)
                attn = batch["attention_mask"].to(dev)
                chunk = _collect_one_microbatch(
                    target_backend=target_model_backend,
                    sglang_target=sglang_target,
                    hf_target=hf_target,
                    embed_tokens=embed_tokens,
                    draft=draft,
                    lm_head=lm_head,
                    input_ids=input_ids,
                    attn_mask=attn,
                    mask_token_id=mask_token_id,
                    num_anchors=num_anchors,
                    block_size=block_size,
                    label_lookahead=thresh_label_lookahead,
                    device=dev,
                )
                if chunk is None:
                    continue
                h_b, y_b, m_b, a_b = chunk
                accum_h.append(h_b)
                accum_y.append(y_b)
                accum_m.append(m_b)
                accum_a.append(a_b)

                total_blocks = sum(x.shape[0] for x in accum_h)
                if total_blocks < micro_batch_blocks:
                    continue

                hidden = torch.cat(accum_h, dim=0)
                labels = torch.cat(accum_y, dim=0)
                mask = torch.cat(accum_m, dim=0)
                accept_lens = torch.cat(accum_a, dim=0)
                accum_h.clear()
                accum_y.clear()
                accum_m.clear()
                accum_a.clear()

                opt.zero_grad(set_to_none=True)
                pred = th(hidden.to(dtype=th_dtype)).squeeze(-1)
                if thresh_head_loss_type == "mae":
                    per = (pred - labels).abs()
                else:
                    per = (pred - labels) ** 2
                denom = mask.sum() + 1e-6
                loss = (per * mask).sum() / denom
                loss.backward()
                opt.step()
                global_step += 1
                _after_step_log(pred, labels, mask, accept_lens, step=global_step)
                pbar.set_postfix(loss=float(loss.item()), step=global_step)

            if accum_h:
                hidden = torch.cat(accum_h, dim=0)
                labels = torch.cat(accum_y, dim=0)
                mask = torch.cat(accum_m, dim=0)
                accept_lens = torch.cat(accum_a, dim=0)
                opt.zero_grad(set_to_none=True)
                pred = th(hidden.to(dtype=th_dtype)).squeeze(-1)
                if thresh_head_loss_type == "mae":
                    per = (pred - labels).abs()
                else:
                    per = (pred - labels) ** 2
                denom = mask.sum() + 1e-6
                loss = (per * mask).sum() / denom
                loss.backward()
                opt.step()
                global_step += 1
                _after_step_log(pred, labels, mask, accept_lens, step=global_step)
                print(f"[flush] loss={loss.item():.6f} step={global_step}")

        draft.thresh_head_two_model.eval()
        draft.eval()
        for p in draft.thresh_head_two_model.parameters():
            p.requires_grad = False

        out_path = Path(output_dir).resolve()
        print(f"[save] {out_path}")
        draft.save_pretrained(str(out_path))

        if copy_remote_code:
            from pipeline.paths import REPO_ROOT

            modeling_src = REPO_ROOT / "dflash" / "dflash" / "model.py"
            modeling_dst = out_path / "dflash.py"
            if modeling_src.is_file():
                shutil.copy2(modeling_src, modeling_dst)
                print(f"[save] copied remote code -> {modeling_dst}")

        return out_path
    finally:
        if dist_inited:
            from specforge.distributed import destroy_distributed

            destroy_distributed()


def main() -> None:
    from specforge.args import SGLangBackendArgs

    from pipeline.paths import ASYN_TRAIN_ROOT, REPO_ROOT

    p = argparse.ArgumentParser(description="Offline thresh_head_two_model init for async thresh-head pipeline.")
    p.add_argument(
        "--target-model-path",
        default="Qwen/Qwen3-8B",
        help="Target model id or local dir (with config.json).",
    )
    p.add_argument(
        "--target-model-backend",
        default="sglang",
        choices=("sglang", "hf"),
        help="Default sglang (same as training_worker). Use hf for HuggingFace-only target forward.",
    )
    p.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size for SGLang target (must match torchrun --nproc_per_node).",
    )
    p.add_argument(
        "--dist-timeout",
        type=int,
        default=30,
        help="NCCL timeout in minutes for init_distributed (sglang only).",
    )
    p.add_argument(
        "--draft-config-path",
        default=str(ASYN_TRAIN_ROOT / "configs" / "qwen3-8b-dflash-thresh-head.json"),
        help="Draft architecture JSON (same role as training_worker --draft-config-path).",
    )
    p.add_argument(
        "--initial-draft-path",
        required=True,
        help="Plain DFlash draft: local checkpoint directory (must contain config.json) or Hub repo id.",
    )
    p.add_argument(
        "--dataset-path",
        default=str(ASYN_TRAIN_ROOT / "cache" / "dataset" / "perfectblend_qwen3_8b_regen_4096.jsonl"),
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--num-anchors", type=int, default=64)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--micro-batch-blocks", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--thresh-label-lookahead", type=int, default=1)
    p.add_argument("--thresh-head-loss-type", choices=("mse", "mae"), default="mse")
    p.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print rolling train stats every N optimizer steps (0 disables).",
    )
    p.add_argument(
        "--log-rolling-steps",
        type=int,
        default=10,
        help="Rolling window size (last N steps) for avg_accept_len / label_cand / pred_verify stats.",
    )
    p.add_argument(
        "--log-thresh-head-threshold-rate",
        type=float,
        default=1.0,
        help="Must match inference --thresh-head-threshold-rate for avg_pred_verify_len (SGLang default 1.0).",
    )
    p.add_argument(
        "--log-candidate-len-min",
        type=int,
        default=None,
        help="Clamp min for pred verify len in logs; default follows draft prob_head_candidate_len_min.",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--no-copy-remote-code", action="store_true")
    p.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disable trust_remote_code on HF loads (not recommended for Qwen3).",
    )
    SGLangBackendArgs.add_args(p)
    args = p.parse_args()
    trust_remote = not args.no_trust_remote_code
    args.target_batch_size = args.batch_size
    sglang_kw = (
        SGLangBackendArgs.from_args(args).to_kwargs() if args.target_model_backend == "sglang" else None
    )

    offline_train_thresh_head_two_model(
        target_model_path=args.target_model_path,
        draft_config_path=args.draft_config_path,
        initial_draft_path=args.initial_draft_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        max_length=args.max_length,
        num_anchors=args.num_anchors,
        epochs=args.epochs,
        batch_size=args.batch_size,
        micro_batch_blocks=args.micro_batch_blocks,
        learning_rate=args.learning_rate,
        thresh_label_lookahead=args.thresh_label_lookahead,
        thresh_head_loss_type=args.thresh_head_loss_type,
        log_every=args.log_every,
        log_rolling_steps=args.log_rolling_steps,
        log_thresh_head_threshold_rate=args.log_thresh_head_threshold_rate,
        log_candidate_len_min=args.log_candidate_len_min,
        max_samples=args.max_samples,
        trust_remote_code=trust_remote,
        copy_remote_code=not args.no_copy_remote_code,
        target_model_backend=args.target_model_backend,
        tp_size=args.tp_size,
        dist_timeout=args.dist_timeout,
        sglang_backend_kwargs=sglang_kw,
    )
    print("Done.")


if __name__ == "__main__":
    main()
