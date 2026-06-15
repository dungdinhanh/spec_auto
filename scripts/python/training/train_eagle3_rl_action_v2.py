"""RL fine-tuning of an EAGLE-3 draft (chain decoding) — v2 with multiblock.

v2 changes vs v1: adds --multiblock_N / --multiblock_max_total / --filter_to_rejection_blocks,
mirroring train_dflash_rl_action_v5.py. Per-block work extracted into
`_compute_block_loss_eagle3`; main rl_step does common setup + dispatch.

Original v1 docstring follows.
==============================================================================
Faithful port of train_dflash_rl_action_v3.py. Same 3-component reward and
GRPO-style group-baseline policy gradient with static-reference KL anchor.
The only change is the draft forward: instead of one parallel block forward
returning gamma logits, we run a greedy chain rollout of length gamma and
collect per-step logits.

Reward composition (per K-th rollout, identical to v3):
    reward_k = w_traj  · r_traj_k       # action MSE
            + w_cons  · r_cons_k        # rule-based meta-action consistency
            + w_text  · r_text_sim_k    # token-level CoC similarity

Differences from v3:
  * Draft = Eagle3DraftModel built from `eagle3_draft.py` (1D or 3D M-RoPE).
  * `draft_chain_logits` replaces `draft_block_logits`. Returns the same shape
    (gamma, V) so K-sampling, KL, and reward are unchanged.
  * 3D-mrope drafts route through the same chain code path; position_ids are
    target's true 3D positions (computed via get_target_3d_position_ids).
  * --use_3d_mrope auto-detected from ckpt config.

The K-sampling, contamination/substitution, and KL anchor logic are identical
to v3 — same per-position machinery, since shape (gamma, V) matches.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers.cache_utils import DynamicCache

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper
from alpamayo_r1.models.dflash_draft import (
    get_qwen3vl_embed_and_head,
    get_target_3d_position_ids,
)
from alpamayo_r1.models.eagle3_draft import (
    Eagle3DraftModel, Eagle3DraftConfig, load_eagle3_ckpt,
)

TRAJ_FUTURE_START = 155681
PAD_ID = 151643
MASK_ID = 151662


def is_main(rank): return rank == 0


# ----------------------------------------------------------------------------
# Rule-based meta-action detector (Option B path) — copied from v3 unchanged.
# ----------------------------------------------------------------------------
def meta_action_label(
    action: torch.Tensor,
    horizon: int = 16,
    eps_long: float = 0.05,
    eps_lat: float = 0.10,
    dt: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    H = min(horizon, action.shape[-2])
    head = action[..., :H, :]
    mean_accel = head[..., 0].mean(dim=-1)
    long_lab = torch.zeros_like(mean_accel, dtype=torch.long)
    long_lab[mean_accel > eps_long] = 1
    long_lab[mean_accel < -eps_long] = -1
    net_heading = head[..., 1].sum(dim=-1) * dt
    lat_lab = torch.zeros_like(net_heading, dtype=torch.long)
    lat_lab[net_heading > eps_lat] = 1
    lat_lab[net_heading < -eps_lat] = -1
    return long_lab, lat_lab


@torch.no_grad()
def update_ref_model(ref_module: nn.Module, trainable_module: nn.Module,
                     mode: str = "replace", ema_alpha: float = 0.9):
    if mode == "replace":
        ref_module.load_state_dict(trainable_module.state_dict(), strict=False)
    elif mode == "ema":
        ref_state = dict(ref_module.named_parameters())
        for name, p_train in trainable_module.named_parameters():
            if name in ref_state:
                ref_state[name].data.mul_(ema_alpha).add_(
                    p_train.data.to(ref_state[name].dtype), alpha=1.0 - ema_alpha,
                )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    ref_module.eval()


# ----------------------------------------------------------------------------
# Dataset (unchanged from v3)
# ----------------------------------------------------------------------------
class TargetOutputDataset(Dataset):
    def __init__(self, output_dir, include_uuids=None, exclude_uuids=None,
                 max_samples=None):
        all_files = sorted(glob.glob(os.path.join(output_dir, "*.pt")))
        if include_uuids is not None:
            incl = set(include_uuids)
            all_files = [p for p in all_files if Path(p).stem in incl]
        if exclude_uuids is not None:
            excl = set(exclude_uuids)
            all_files = [p for p in all_files if Path(p).stem not in excl]
        if max_samples is not None:
            all_files = all_files[:max_samples]
        self.files = all_files

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        d = torch.load(self.files[idx], weights_only=False)
        out = {
            "prompt_input_ids": d["prompt_input_ids"].squeeze(0),
            "output_token_ids": d["output_token_ids"],
            "pixel_values": d["pixel_values"].to(torch.bfloat16),
            "image_grid_thw": d["image_grid_thw"],
            "clip_id": d["clip_id"],
        }
        return out


def collate_one(features):
    assert len(features) == 1, "batch_size must be 1 at the dataloader level"
    return features[0]


collate_fn = collate_one


# ----------------------------------------------------------------------------
# Target VLM + diffusion (unchanged from v3) — for action reward
# ----------------------------------------------------------------------------
@torch.no_grad()
def vlm_prefill(target_model, input_ids, pixel_values, image_grid_thw):
    vlm = target_model.vlm
    past = DynamicCache()
    out = vlm(
        input_ids=input_ids, pixel_values=pixel_values,
        image_grid_thw=image_grid_thw, past_key_values=past,
        use_cache=True, return_dict=True,
    )
    rope_deltas = vlm.model.rope_deltas
    return past, rope_deltas


@torch.no_grad()
def run_diffusion_on_cache(target_model, prompt_cache, rope_deltas,
                            traj_future_start_positions, seed):
    device = rope_deltas.device
    B = traj_future_start_positions.shape[0]
    n_diffusion_tokens = target_model.action_space.get_action_space_dims()[0]
    prefill_seq_len = prompt_cache.get_seq_length()

    position_ids = torch.arange(n_diffusion_tokens, device=device)
    position_ids = einops.repeat(position_ids, "l -> 3 b l", b=B).clone()
    offset = traj_future_start_positions + 1
    delta = rope_deltas + offset[:, None]
    position_ids += delta.to(position_ids.device)

    expert_dtype = next(target_model.expert.parameters()).dtype
    attention_mask = torch.zeros(
        (B, 1, n_diffusion_tokens, prefill_seq_len + n_diffusion_tokens),
        dtype=expert_dtype, device=device,
    )
    for i in range(B):
        attention_mask[i, :, :, offset[i]:-n_diffusion_tokens] = torch.finfo(
            attention_mask.dtype).min

    forward_kwargs = {}
    if target_model.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False
    expert_dtype = next(target_model.expert.parameters()).dtype

    def step_fn(x, t):
        b = x.shape[0]
        fte = target_model.action_in_proj(x.float(), t.float() if torch.is_tensor(t) else t)
        if fte.dim() == 2:
            fte = fte.view(b, n_diffusion_tokens, -1)
        fte = fte.to(expert_dtype)
        exp_out = target_model.expert(
            inputs_embeds=fte, position_ids=position_ids,
            past_key_values=prompt_cache, attention_mask=attention_mask,
            use_cache=True, **forward_kwargs,
        )
        prompt_cache.crop(prefill_seq_len)
        lh = exp_out.last_hidden_state[:, -n_diffusion_tokens:]
        pred = target_model.action_out_proj(lh).view(
            -1, *target_model.action_space.get_action_space_dims())
        return pred.float()

    torch.manual_seed(seed)
    sampled = target_model.diffusion.sample(
        batch_size=B, step_fn=step_fn, device=device,
        return_all_steps=False,
    )
    return sampled


# ----------------------------------------------------------------------------
# Target hidden state extraction (unchanged from v3)
# ----------------------------------------------------------------------------
@torch.no_grad()
def target_hidden_for_context(target_vlm, input_ids, pixel_values, image_grid_thw):
    past = DynamicCache()
    out = target_vlm(
        input_ids=input_ids, pixel_values=pixel_values,
        image_grid_thw=image_grid_thw, past_key_values=past,
        use_cache=True, output_hidden_states=True, return_dict=True,
    )
    return out.hidden_states


# ----------------------------------------------------------------------------
# EAGLE-3 chain rollout (replaces DFlash's block-parallel forward)
# ----------------------------------------------------------------------------
def draft_chain_logits(
    draft: Eagle3DraftModel,
    embed_tokens: nn.Module,
    lm_head: nn.Module,
    target_hidden_states: tuple,        # tuple of (B, T, H) per layer (hidden_states)
    target_layer_ids: list[int],
    prompt_ids: torch.Tensor,           # (1, P)
    gt_coc: torch.Tensor,               # (1, N)
    block_start: int,
    gamma: int,                         # = block_size - 1
    target_vlm,                         # for 3D position_id computation
    image_grid_thw: Optional[torch.Tensor],
    use_3d: bool,
    device: torch.device,
) -> torch.Tensor:
    """Greedy chain rollout of length gamma starting from gt_coc[block_start] as anchor.

    Mirrors `eagle3_spec_generate` chain decoding (see e2e_eagle3_spec_test*.py)
    but training-time:
      * context = prompt_ids + gt_coc[:block_start]   (length P+block_start, EXCLUDES anchor)
      * shifted_ids = full_committed[1:] + [anchor]    (anchor enters as last input embedding)
      * step 0 = draft prefill over context, last position's logit predicts position block_start+1
      * steps 1..gamma-1 = chain steps at virtual_pos=L-1, each step's input = previous step's argmax

    Returns (gamma, V) logits — same shape as DFlash's `draft_block_logits`,
    so all downstream sampling / KL / reward code is unchanged.
    """
    P = prompt_ids.shape[1]
    if block_start > 0:
        full_committed = torch.cat([prompt_ids, gt_coc[:, :block_start]], dim=1)
    else:
        full_committed = prompt_ids
    anchor = gt_coc[:, block_start:block_start + 1]  # (1, 1)
    shifted_ids = torch.cat([full_committed[:, 1:], anchor], dim=1)  # (1, L)
    L = shifted_ids.shape[1]  # = P + block_start

    # fc input over the first L target hidden positions (no anchor leak; anchor
    # enters via the last embedding only).
    target_h_layers = [target_hidden_states[idx + 1][:, :L, :] for idx in target_layer_ids]
    fc_input = draft.fc(torch.cat(target_h_layers, dim=-1))  # (1, L, H)

    # position ids
    if use_3d:
        attn_mask_full = torch.ones_like(full_committed, device=device)
        full_3d = get_target_3d_position_ids(
            target_vlm=target_vlm, input_ids=full_committed,
            image_grid_thw=image_grid_thw, attention_mask=attn_mask_full,
        )  # (3, 1, L)
        pos_ids = full_3d
        virtual_pos = full_3d[:, :, L - 1:L]  # (3, 1, 1)
    else:
        pos_ids = torch.arange(L, device=device, dtype=torch.long).unsqueeze(0)
        virtual_pos = torch.tensor([[L - 1]], device=device, dtype=torch.long)

    # ---- Step 0: draft prefill ----
    input_emb = embed_tokens(shifted_ids).to(fc_input.dtype)  # (1, L, H)
    attn_mask = draft._prepare_decoder_attention_mask(
        torch.ones((1, L), dtype=torch.bool, device=device),
        (1, L), input_emb.dtype, device, past_kv_len=0,
    )
    cache_hidden = [[], []]
    h_step0, cache_hidden = draft.midlayer(
        input_emb=input_emb, hidden_states=fc_input,
        cache_hidden=cache_hidden, attention_mask=attn_mask,
        position_ids=pos_ids,
    )
    last_h = h_step0[:, -1:, :]                                   # (1, 1, H)
    logit_0 = lm_head(draft.norm(last_h)).squeeze(0).squeeze(0)   # (V,)

    chain_logits = [logit_0]
    chain_hidden = last_h
    next_input = logit_0.argmax(dim=-1).view(1, 1)                # (1, 1)

    # ---- Chain steps 1..gamma-1 at virtual position L-1 ----
    chain_attn_mask = torch.zeros((1, 1, 1, L), dtype=fc_input.dtype, device=device)
    for step in range(1, gamma):
        input_emb_step = embed_tokens(next_input).to(chain_hidden.dtype)
        h_step, cache_hidden = draft.midlayer(
            input_emb=input_emb_step, hidden_states=chain_hidden,
            cache_hidden=cache_hidden, attention_mask=chain_attn_mask,
            position_ids=virtual_pos,
        )
        logit = lm_head(draft.norm(h_step)).squeeze(0).squeeze(0)  # (V,)
        chain_logits.append(logit)
        chain_hidden = h_step
        next_input = logit.argmax(dim=-1).view(1, 1)

    return torch.stack(chain_logits, dim=0)  # (gamma, V)


# ----------------------------------------------------------------------------
# Per-block forward + loss (used by both legacy and multiblock paths)
# ----------------------------------------------------------------------------
def _compute_block_loss_eagle3(
    *, target_model, draft, draft_ref, embed_tokens, lm_head,
    prompt_ids, gt_coc, pixel_values, image_grid_thw,
    target_hidden, action_gt,
    block_start, contam_pos_local, contam_mode, contam_N_legacy,
    args, device, rng, seed,
    P, N, B_size, K,
) -> Optional[dict]:
    """One EAGLE-3 block: draft chain forward, K rollouts, mixed CoC, reward, GRPO + KL.

    Args mirror DFlash v5 `_compute_block_loss`. Only the draft forward call
    differs (uses `draft_chain_logits` instead of `draft_block_logits`).
    """
    gamma = B_size - 1

    logits = draft_chain_logits(
        draft, embed_tokens, lm_head, target_hidden,
        draft.target_layer_ids, prompt_ids, gt_coc, block_start,
        gamma=gamma, target_vlm=target_model.vlm,
        image_grid_thw=image_grid_thw, use_3d=args.use_3d_mrope,
        device=device,
    )   # (gamma, V)
    with torch.no_grad():
        ref_logits = draft_chain_logits(
            draft_ref, embed_tokens, lm_head, target_hidden,
            draft_ref.target_layer_ids, prompt_ids, gt_coc, block_start,
            gamma=gamma, target_vlm=target_model.vlm,
            image_grid_thw=image_grid_thw, use_3d=args.use_3d_mrope,
            device=device,
        )

    B_minus_1 = gamma
    gt_available = min(B_minus_1, N - block_start - 1)

    greedy_pred = logits.argmax(dim=-1)
    target_greedy_block = gt_coc[0, block_start + 1:block_start + 1 + gt_available]
    greedy_matches_gt = (greedy_pred[:gt_available] == target_greedy_block)
    matched_full = torch.zeros(B_minus_1, dtype=torch.bool, device=device)
    rejected_full = torch.zeros(B_minus_1, dtype=torch.bool, device=device)
    matched_full[:gt_available] = greedy_matches_gt
    rejected_full[:gt_available] = ~greedy_matches_gt
    rejected_pos_idx_full = torch.nonzero(greedy_matches_gt == False, as_tuple=False).flatten()
    R = int(rejected_full.sum().item())

    # Decide contam_pos_local if caller didn't pre-specify (legacy fallback paths).
    if contam_pos_local is None:
        if contam_mode == "scattered_N":
            contam_pos_local = rejected_pos_idx_full[:contam_N_legacy].tolist()
        elif contam_mode == "fixed_m":
            m_eff = min(contam_N_legacy, gt_available)
            contam_pos_local = list(range(m_eff))
        else:
            raise ValueError(f"contam_mode {contam_mode} requires contam_pos_local")
    contam_pos_local = [int(p) for p in contam_pos_local if 0 <= int(p) < gt_available]

    contam_full = torch.zeros(B_minus_1, dtype=torch.bool, device=device)
    if contam_pos_local:
        contam_full[torch.tensor(contam_pos_local, dtype=torch.long, device=device)] = True
    stochastic_full = contam_full & rejected_full

    sampled = greedy_pred.unsqueeze(1).expand(-1, K).clone()
    n_stoch = int(stochastic_full.sum().item())
    if n_stoch > 0:
        probs_stoch = F.softmax(logits[stochastic_full] / args.temperature, dim=-1)
        sampled_stoch = torch.multinomial(probs_stoch, K, replacement=True)
        sampled[stochastic_full] = sampled_stoch

    log_probs_all = F.log_softmax(logits, dim=-1)
    log_probs_sampled = torch.gather(
        log_probs_all.unsqueeze(1).expand(-1, K, -1), -1,
        sampled.unsqueeze(-1),
    ).squeeze(-1)
    seq_log_probs = (log_probs_sampled * stochastic_full.unsqueeze(1).float()).sum(dim=0)

    matches_pre = (sampled[:gt_available] == target_greedy_block.unsqueeze(1))

    mixed_seqs = []
    for k in range(K):
        mc = gt_coc.clone().squeeze(0)
        for p in contam_pos_local:
            mc[block_start + 1 + p] = sampled[p, k]
        seq_k = torch.cat([prompt_ids.squeeze(0), mc, torch.tensor([TRAJ_FUTURE_START], device=device)])
        mixed_seqs.append(seq_k)
    if not mixed_seqs:
        return None
    batched_ids = torch.stack(mixed_seqs, dim=0)

    chunk_K = max(1, args.k_chunk_size)
    action_chunks = []
    for s in range(0, K, chunk_K):
        e = min(s + chunk_K, K)
        cur = e - s
        chunk_ids = batched_ids[s:e]
        chunk_pix = pixel_values.unsqueeze(0).expand(cur, *pixel_values.shape).reshape(
            cur * pixel_values.shape[0], *pixel_values.shape[1:]
        )
        chunk_grid = image_grid_thw.unsqueeze(0).expand(cur, *image_grid_thw.shape).reshape(
            cur * image_grid_thw.shape[0], *image_grid_thw.shape[1:]
        )
        cache_chunk, rd_chunk = vlm_prefill(target_model, chunk_ids, chunk_pix, chunk_grid)
        traj_pos_chunk = torch.full((cur,), P + N, device=device, dtype=torch.long)
        action_chunk = run_diffusion_on_cache(
            target_model, cache_chunk, rd_chunk, traj_pos_chunk, seed=seed + s,
        )
        action_chunks.append(action_chunk)
        del cache_chunk, rd_chunk
    action_mix = torch.cat(action_chunks, dim=0)

    diff = action_mix - action_gt
    mse_per_sample = (diff * diff).mean(dim=(-1, -2))
    r_traj = -mse_per_sample

    if contam_pos_local:
        contam_t = torch.tensor(contam_pos_local, dtype=torch.long, device=device)
        gt_at_contam = gt_coc[0, (block_start + 1) + contam_t]
        sampled_at_contam = sampled[contam_t, :]
        match_at = (sampled_at_contam == gt_at_contam.unsqueeze(1)).float()
        r_text_sim = match_at.mean(dim=0)
    else:
        r_text_sim = torch.zeros(K, device=device)

    if args.enable_r_cons or args.w_cons != 0.0:
        long_mix, lat_mix = meta_action_label(
            action_mix, horizon=args.consistency_horizon,
            eps_long=args.eps_long, eps_lat=args.eps_lat,
        )
        long_gt, lat_gt = meta_action_label(
            action_gt, horizon=args.consistency_horizon,
            eps_long=args.eps_long, eps_lat=args.eps_lat,
        )
        r_cons = ((long_mix == long_gt) & (lat_mix == lat_gt)).float()
    else:
        r_cons = torch.zeros(K, device=device)

    accepted_per_pos = matches_pre.long().cumprod(dim=0)
    accepted_length = accepted_per_pos.sum(dim=0).float()

    rewards = (
        args.w_traj * r_traj
        + args.w_cons * r_cons
        + args.w_text * r_text_sim
    )

    baseline = rewards.mean()
    advantage = rewards - baseline
    rl_loss = -(advantage.detach() * seq_log_probs).mean()

    with torch.no_grad():
        log_ref = F.log_softmax(ref_logits, dim=-1)
        ref_greedy = ref_logits.argmax(dim=-1)
    log_pi = F.log_softmax(logits, dim=-1)
    pi = log_pi.exp()
    kl_per_pos = (pi * (log_pi - log_ref)).sum(dim=-1)

    if args.anchor_source == "policy":
        anchor_weight = matched_full.float()
    elif args.anchor_source == "ref":
        ref_matches_full = torch.zeros(B_minus_1, dtype=torch.bool, device=device)
        ref_matches_full[:gt_available] = (
            ref_greedy[:gt_available] == target_greedy_block
        )
        anchor_weight = ref_matches_full.float()
    elif args.anchor_source == "weighted_all":
        ref_matches_full = torch.zeros(B_minus_1, dtype=torch.bool, device=device)
        ref_matches_full[:gt_available] = (
            ref_greedy[:gt_available] == target_greedy_block
        )
        anchor_weight = torch.full_like(kl_per_pos, args.anchor_rejected_weight)
        anchor_weight[ref_matches_full] = 1.0
        in_gt_mask = torch.zeros(B_minus_1, dtype=torch.bool, device=device)
        in_gt_mask[:gt_available] = True
        anchor_weight = anchor_weight * in_gt_mask.float()
    elif args.anchor_source == "all":
        anchor_weight = torch.zeros(B_minus_1, dtype=kl_per_pos.dtype, device=device)
        anchor_weight[:gt_available] = 1.0
    else:
        raise ValueError(f"Unknown anchor_source: {args.anchor_source}")

    weight_sum = anchor_weight.sum()
    if weight_sum > 0:
        kl_loss = (kl_per_pos * anchor_weight).sum() / weight_sum
    else:
        kl_loss = torch.zeros((), device=device)
    n_matched = int((anchor_weight > 0).sum().item())

    total_loss = rl_loss + args.kl_weight * kl_loss
    accept_rate = accepted_length / max(gt_available, 1)

    return {
        "total_loss": total_loss,
        "rl_loss": rl_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "mean_reward": rewards.mean().detach(),
        "std_reward": rewards.std().detach(),
        "min_mse": mse_per_sample.min().detach(),
        "mean_accepted": accepted_length.mean().detach(),
        "max_accepted": accepted_length.max().detach(),
        "gt_available": torch.tensor(float(gt_available), device=device),
        "mean_accept_rate": accept_rate.mean().detach(),
        "n_rejected": torch.tensor(float(R), device=device),
        "n_matched": torch.tensor(float(n_matched), device=device),
        "r_traj_mean": r_traj.mean().detach(),
        "r_cons_mean": r_cons.mean().detach(),
        "r_text_sim_mean": r_text_sim.mean().detach(),
        "n_contaminated": int(len(contam_pos_local)),
        "chosen_block_start": torch.tensor(float(block_start), device=device),
    }


# ----------------------------------------------------------------------------
# RL training step (v2: multiblock dispatch + filter_to_rejection_blocks)
# ----------------------------------------------------------------------------
def rl_step(
    target_model, draft, draft_ref, embed_tokens, lm_head,
    batch, args, device, rng
) -> Optional[dict]:
    prompt_ids = batch["prompt_input_ids"].to(device).unsqueeze(0)
    gt_coc = batch["output_token_ids"].to(device).unsqueeze(0)
    pixel_values = batch["pixel_values"].to(device).to(torch.bfloat16)
    image_grid_thw = batch["image_grid_thw"].to(device)

    P = prompt_ids.shape[1]
    N = gt_coc.shape[1]
    B_size = args.block_size
    gamma = B_size - 1
    if N < 2:
        return None

    # ---- Pass A: reference action from gt CoC ----
    traj_start = torch.tensor([[TRAJ_FUTURE_START]], device=device)
    seq_gt = torch.cat([prompt_ids, gt_coc, traj_start], dim=1)
    traj_start_pos_gt = torch.tensor([P + N], device=device)
    cache_gt, rd_gt = vlm_prefill(target_model, seq_gt, pixel_values, image_grid_thw)
    seed_master = rng.randrange(0, 2**31 - 1)
    action_gt = run_diffusion_on_cache(target_model, cache_gt, rd_gt,
                                        traj_start_pos_gt, seed=seed_master)
    del cache_gt, rd_gt

    # ---- Target hidden ONCE over full prompt + gt_coc ----
    context_input_ids_full = torch.cat([prompt_ids, gt_coc], dim=1)  # (1, P+N)
    target_hidden = target_hidden_for_context(
        target_model.vlm, context_input_ids_full, pixel_values, image_grid_thw,
    )

    lo = 0
    hi = max(N - 2, 0)
    candidates = list(range(lo, hi + 1))

    rej_list_size = -1
    n_candidates = -1
    rejection_info: list[tuple[int, int, int]] = []  # (bs, first_rej_pos, gt_avail_b)
    if args.filter_to_rejection_blocks or args.multiblock_N > 0:
        with torch.no_grad():
            for b in candidates:
                gt_avail_b = min(gamma, N - b - 1)
                if gt_avail_b <= 0:
                    continue
                logits_b = draft_chain_logits(
                    draft, embed_tokens, lm_head, target_hidden,
                    draft.target_layer_ids, prompt_ids, gt_coc, b,
                    gamma=gamma, target_vlm=target_model.vlm,
                    image_grid_thw=image_grid_thw, use_3d=args.use_3d_mrope,
                    device=device,
                )
                gp_b = logits_b.argmax(dim=-1)
                gt_b = gt_coc[0, b + 1:b + 1 + gt_avail_b]
                mismatches_b = (gp_b[:gt_avail_b] != gt_b)
                R_b = int(mismatches_b.sum().item())
                if R_b >= 1:
                    first_rej = int(torch.nonzero(mismatches_b).flatten()[0].item())
                    rejection_info.append((b, first_rej, gt_avail_b))
        rej_list_size = len(rejection_info)
        n_candidates = len(candidates)
        if not rejection_info:
            return None
        if args.multiblock_N == 0:
            chosen = rejection_info[rng.randrange(len(rejection_info))]
            block_start = chosen[0]
    else:
        decay = args.block_start_decay
        weights = [decay ** b for b in candidates]
        block_start = rng.choices(candidates, weights=weights, k=1)[0]

    # ---- Multiblock path ----
    if args.multiblock_N > 0:
        base_seed = rng.randrange(0, 2**31 - 1)
        block_outs: list[dict] = []
        cumulative = 0
        cursor = 0
        for (bs, first_rej, gt_avail_b) in rejection_info:
            if cumulative >= args.multiblock_max_total:
                break
            if bs < cursor:
                continue
            avail_in_window = max(0, min(args.multiblock_N, gt_avail_b - first_rej))
            n_take = min(avail_in_window, args.multiblock_max_total - cumulative)
            if n_take <= 0:
                continue
            contam_pos = list(range(first_rej, first_rej + n_take))
            block_seed = (base_seed + bs * 1009) % (2**31 - 1)
            bl = _compute_block_loss_eagle3(
                target_model=target_model, draft=draft, draft_ref=draft_ref,
                embed_tokens=embed_tokens, lm_head=lm_head,
                prompt_ids=prompt_ids, gt_coc=gt_coc, pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                target_hidden=target_hidden, action_gt=action_gt,
                block_start=bs, contam_pos_local=contam_pos,
                contam_mode=None, contam_N_legacy=0,
                args=args, device=device, rng=rng, seed=block_seed,
                P=P, N=N, B_size=B_size, K=args.k_samples,
            )
            if bl is None:
                continue
            block_outs.append(bl)
            cumulative += n_take
            cursor = bs + first_rej + n_take
        if not block_outs:
            return None
        total_loss = sum(b["total_loss"] for b in block_outs) / len(block_outs)
        def _avg(key: str):
            return sum(b[key] for b in block_outs) / len(block_outs)
        return {
            "total_loss": total_loss,
            "rl_loss": _avg("rl_loss"),
            "kl_loss": _avg("kl_loss"),
            "mean_reward": _avg("mean_reward"),
            "std_reward": _avg("std_reward"),
            "min_mse": _avg("min_mse"),
            "mean_accepted": _avg("mean_accepted"),
            "max_accepted": _avg("max_accepted"),
            "gt_available": _avg("gt_available"),
            "mean_accept_rate": _avg("mean_accept_rate"),
            "n_rejected": _avg("n_rejected"),
            "n_matched": _avg("n_matched"),
            "r_traj_mean": _avg("r_traj_mean"),
            "r_cons_mean": _avg("r_cons_mean"),
            "r_text_sim_mean": _avg("r_text_sim_mean"),
            "rej_list_size": torch.tensor(float(rej_list_size), device=device),
            "n_candidates": torch.tensor(float(n_candidates), device=device),
            "chosen_block_start": torch.tensor(float(block_outs[0]["chosen_block_start"]), device=device),
            "n_blocks": torch.tensor(float(len(block_outs)), device=device),
            "n_contaminated_total": torch.tensor(float(cumulative), device=device),
        }

    # ---- Legacy single-block path ----
    legacy_seed = rng.randrange(0, 2**31 - 1)
    if args.contamination_N > 0:
        contam_mode, contam_N_leg = "scattered_N", args.contamination_N
    else:
        contam_mode, contam_N_leg = "fixed_m", args.subst_m
    out_legacy = _compute_block_loss_eagle3(
        target_model=target_model, draft=draft, draft_ref=draft_ref,
        embed_tokens=embed_tokens, lm_head=lm_head,
        prompt_ids=prompt_ids, gt_coc=gt_coc, pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        target_hidden=target_hidden, action_gt=action_gt,
        block_start=block_start, contam_pos_local=None,
        contam_mode=contam_mode, contam_N_legacy=contam_N_leg,
        args=args, device=device, rng=rng, seed=legacy_seed,
        P=P, N=N, B_size=B_size, K=args.k_samples,
    )
    if out_legacy is None:
        return None
    out_legacy["rej_list_size"] = torch.tensor(float(rej_list_size), device=device)
    out_legacy["n_candidates"] = torch.tensor(float(n_candidates), device=device)
    return out_legacy


# ----------------------------------------------------------------------------
# Main — same as v3 except draft setup uses Eagle3DraftModel
# ----------------------------------------------------------------------------
def _build_eagle3_draft_from_ckpt(target_vlm, ckpt, dtype=torch.bfloat16):
    """Build Eagle3DraftModel from ckpt config. Auto-detects 1D vs 3D mrope."""
    sd = ckpt["state_dict"]
    cfg_dict = ckpt.get("config", {})
    target_layer_ids = ckpt.get("target_layer_ids") or [1, 17, 32]
    rollout_length = ckpt.get("rollout_length", 7)
    mrope_section = cfg_dict.get("mrope_section")
    mrope_interleaved = cfg_dict.get("mrope_interleaved", True)

    text_cfg = target_vlm.config.get_text_config()
    cfg = Eagle3DraftConfig(
        hidden_size=cfg_dict.get("hidden_size", text_cfg.hidden_size),
        intermediate_size=cfg_dict.get("intermediate_size", text_cfg.intermediate_size),
        num_hidden_layers=1,
        num_attention_heads=cfg_dict.get("num_attention_heads", text_cfg.num_attention_heads),
        num_key_value_heads=cfg_dict.get("num_key_value_heads", text_cfg.num_key_value_heads),
        vocab_size=cfg_dict.get("vocab_size", text_cfg.vocab_size),
        rms_norm_eps=cfg_dict.get("rms_norm_eps", text_cfg.rms_norm_eps),
        rope_theta=cfg_dict.get("rope_theta", getattr(text_cfg, "rope_theta", 10000)),
        max_position_embeddings=cfg_dict.get(
            "max_position_embeddings",
            getattr(text_cfg, "max_position_embeddings", 8192)),
        target_layer_ids=target_layer_ids,
        rollout_length=rollout_length,
        mrope_section=mrope_section,
        mrope_interleaved=mrope_interleaved,
    )
    model = Eagle3DraftModel(cfg).to(dtype=dtype)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    use_3d = mrope_section is not None
    print(f"[eagle3-load] target_layer_ids={target_layer_ids} rollout={rollout_length} "
          f"use_3d_mrope={use_3d} missing={len(missing)} unexpected={len(unexpected)}",
          flush=True)
    return model, use_3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_path", required=True)
    ap.add_argument("--init_draft_path", required=True)
    ap.add_argument("--target_outputs_dir", required=True)
    ap.add_argument("--val_uuids_file", required=True)
    ap.add_argument("--test_uuids_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--block_size", type=int, default=16,
                    help="block_size = gamma + 1 where gamma is chain length. "
                         "For EAGLE-3 this is the chain decoding depth + 1 bonus.")
    ap.add_argument("--num_epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--k_samples", type=int, default=5)
    ap.add_argument("--k_chunk_size", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--subst_m", type=int, default=6)
    ap.add_argument("--w_traj", type=float, default=1.0)
    ap.add_argument("--w_cons", type=float, default=0.0)
    ap.add_argument("--w_text", type=float, default=0.5)
    ap.add_argument("--enable_r_cons", action="store_true")
    ap.add_argument("--consistency_horizon", type=int, default=16)
    ap.add_argument("--eps_long", type=float, default=0.05)
    ap.add_argument("--eps_lat", type=float, default=0.10)
    ap.add_argument("--accept_bonus", type=float, default=0.0)
    ap.add_argument("--anchor_source", choices=["policy", "ref", "weighted_all", "all"],
                    default="ref")
    ap.add_argument("--anchor_rejected_weight", type=float, default=0.05)
    ap.add_argument("--contamination_N", type=int, default=3)
    # v2: multi-block contamination (v5 DFlash port)
    ap.add_argument("--multiblock_N", type=int, default=0,
                    help="Window size of CONSECUTIVE positions to contaminate per "
                         "rejection block, starting at first greedy rejection in that "
                         "block. 0 = disabled (legacy v1 single-block). >0 enables "
                         "multi-block iteration.")
    ap.add_argument("--multiblock_max_total", type=int, default=10,
                    help="Cumulative contaminated-position budget per training step.")
    ap.add_argument("--filter_to_rejection_blocks", action="store_true",
                    help="Probe every block_start, skip the clip if every block "
                         "fully accepts. Otherwise sample uniformly from rejection list.")
    ap.add_argument("--block_start_decay", type=float, default=0.8)
    ap.add_argument("--kl_weight", type=float, default=0.02)
    ap.add_argument("--max_clips", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--log_interval", type=int, default=5)
    ap.add_argument("--save_interval", type=int, default=500)
    ap.add_argument("--topk_save", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ref_update_interval", type=int, default=0)
    ap.add_argument("--ref_update_mode", choices=["replace", "ema"], default="replace")
    ap.add_argument("--ref_ema_alpha", type=float, default=0.9)
    ap.add_argument("--ref_update_gate", choices=["none", "train_rolling_rate", "eval_acceptance"],
                    default="none")
    ap.add_argument("--use_3d_mrope", action="store_true",
                    help="Force 3D-mrope draft path. Auto-set if ckpt has mrope_section.")
    ap.add_argument("--wandb_project", default="eagle3-rl-action")
    ap.add_argument("--wandb_run_name", default=None)
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(args.seed + rank)
    rng = random.Random(args.seed + rank)

    os.makedirs(args.output_dir, exist_ok=True)
    use_wandb = (not args.no_wandb) and is_main(rank)
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                   config=vars(args), dir=args.output_dir)

    if is_main(rank):
        print(f"loading target from {args.target_path}", flush=True)
    target = AlpamayoR1.from_pretrained(args.target_path, dtype=torch.bfloat16).to(device).eval()
    for p in target.parameters():
        p.requires_grad = False
    target.action_in_proj = target.action_in_proj.to(torch.float32)
    vlm = target.vlm
    embed_tokens, lm_head = get_qwen3vl_embed_and_head(vlm)
    for p in embed_tokens.parameters():
        p.requires_grad = False
    for p in lm_head.parameters():
        p.requires_grad = False

    if is_main(rank):
        print(f"loading EAGLE-3 draft init from {args.init_draft_path}", flush=True)
    ckpt = load_eagle3_ckpt(args.init_draft_path, map_location=device)
    draft, ckpt_use_3d = _build_eagle3_draft_from_ckpt(vlm, ckpt)
    draft = draft.to(device).train()
    if not args.use_3d_mrope and ckpt_use_3d:
        if is_main(rank):
            print("[note] ckpt has mrope_section -> setting --use_3d_mrope=True", flush=True)
        args.use_3d_mrope = True
    for p in draft.parameters():
        p.requires_grad = True

    import copy as _copy
    draft_ref = _copy.deepcopy(draft).eval()
    for p in draft_ref.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in draft.parameters() if p.requires_grad)
    print(f"  [rank {rank}] EAGLE-3 draft trainable: {n_params/1e6:.1f}M | "
          f"use_3d_mrope={args.use_3d_mrope} block_size={args.block_size} (gamma={args.block_size - 1})",
          flush=True)

    if world > 1:
        dist.barrier()
        draft = DDP(draft, device_ids=[local_rank], find_unused_parameters=True)
    draft_module = draft.module if hasattr(draft, "module") else draft

    test_ids = json.load(open(args.test_uuids_file))
    val_ids = json.load(open(args.val_uuids_file))
    train_ds = TargetOutputDataset(
        args.target_outputs_dir,
        exclude_uuids=list(set(test_ids) | set(val_ids)),
        max_samples=args.max_clips,
    )
    if is_main(rank):
        print(f"train clips: {len(train_ds)}", flush=True)

    train_sampler = DistributedSampler(
        train_ds, rank=rank, num_replicas=world, shuffle=True, seed=args.seed,
    ) if world > 1 else None
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=2, collate_fn=collate_one,
        pin_memory=True, drop_last=False,
    )

    optim = AdamW(
        [p for p in draft.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0,
    )

    global_step = 0
    t0 = time.time()
    from collections import deque
    best_ckpts = []
    roll_window_len = max(1, args.save_interval // max(args.log_interval, 1))
    rolling_acc_rate = deque(maxlen=roll_window_len)
    for epoch in range(args.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            if args.max_steps and global_step >= args.max_steps:
                break
            try:
                out = rl_step(
                    target, draft_module, draft_ref, embed_tokens, lm_head,
                    batch, args, device, rng,
                )
            except Exception as e:
                import traceback as _tb
                if is_main(rank):
                    print(f"  step err: {type(e).__name__}: {e}", flush=True)
                    _tb.print_exc()
                continue
            if out is None:
                continue

            loss = out["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in draft.parameters() if p.requires_grad],
                args.max_grad_norm,
            )
            optim.step()
            optim.zero_grad()
            global_step += 1

            if (args.ref_update_interval > 0
                    and global_step % args.ref_update_interval == 0
                    and global_step > 0):
                should_update = True
                if args.ref_update_gate == "train_rolling_rate":
                    if len(rolling_acc_rate) > 0:
                        cur_rate = sum(rolling_acc_rate) / len(rolling_acc_rate)
                        if not hasattr(rl_step, "_best_rate_for_ref"):
                            rl_step._best_rate_for_ref = -1.0
                        if cur_rate > rl_step._best_rate_for_ref:
                            rl_step._best_rate_for_ref = cur_rate
                            should_update = True
                        else:
                            should_update = False
                if should_update:
                    trainable_module = draft.module if hasattr(draft, "module") else draft
                    update_ref_model(
                        draft_ref, trainable_module,
                        mode=args.ref_update_mode, ema_alpha=args.ref_ema_alpha,
                    )
                    if is_main(rank):
                        print(f"  [ref_update] step={global_step}: ref updated.", flush=True)

            if is_main(rank) and global_step % args.log_interval == 0:
                rate = global_step / max(time.time() - t0, 1)
                print(
                    f"  epoch {epoch+1} step {global_step} | "
                    f"rl={out['rl_loss'].item():+.4f} "
                    f"kl={out['kl_loss'].item():.4f} "
                    f"r_total={out['mean_reward'].item():+.5f} "
                    f"r_traj={out['r_traj_mean'].item():+.5f} "
                    f"r_cons={out['r_cons_mean'].item():.3f} "
                    f"r_text={out['r_text_sim_mean'].item():.3f} "
                    f"min_mse={out['min_mse'].item():.5f} "
                    f"acc_mean={out['mean_accepted'].item():.2f} "
                    f"acc_max={out['max_accepted'].item():.0f} "
                    f"gt_avail={out['gt_available'].item():.0f} "
                    f"acc_rate={out['mean_accept_rate'].item():.3f} "
                    f"| {rate:.2f} steps/s",
                    flush=True,
                )
                if use_wandb:
                    import wandb
                    wandb.log({
                        "rl/rl_loss": out['rl_loss'].item(),
                        "rl/kl": out['kl_loss'].item(),
                        "rl/reward_mean": out['mean_reward'].item(),
                        "rl/reward_std": out['std_reward'].item(),
                        "rl/r_traj": out['r_traj_mean'].item(),
                        "rl/r_cons": out['r_cons_mean'].item(),
                        "rl/r_text_sim": out['r_text_sim_mean'].item(),
                        "rl/min_mse": out['min_mse'].item(),
                        "rl/mean_accepted": out['mean_accepted'].item(),
                        "rl/max_accepted": out['max_accepted'].item(),
                        "rl/gt_available": out['gt_available'].item(),
                        "rl/mean_accept_rate": out['mean_accept_rate'].item(),
                    }, step=global_step)

            if is_main(rank) and global_step % args.log_interval == 0 and "mean_accept_rate" in out:
                rolling_acc_rate.append(out["mean_accept_rate"].item())

            if global_step % args.save_interval == 0 and is_main(rank):
                avg_rate = sum(rolling_acc_rate) / max(len(rolling_acc_rate), 1)
                p = os.path.join(args.output_dir, f"draft_step_{global_step}.pt")
                # Save in EAGLE-3 ckpt format (compatible with load_eagle3_ckpt)
                cfg_dict = ckpt.get("config", {})
                torch.save({
                    "state_dict": draft_module.state_dict(),
                    "config": cfg_dict,
                    "target_layer_ids": ckpt.get("target_layer_ids", [1, 17, 32]),
                    "rollout_length": ckpt.get("rollout_length", 7),
                    "rolling_acc_rate": float(avg_rate),
                }, p)
                if args.topk_save > 0:
                    best_ckpts.append((avg_rate, global_step, p))
                    best_ckpts.sort(key=lambda x: -x[0])
                    while len(best_ckpts) > args.topk_save:
                        _, evict_step, evict_path = best_ckpts.pop()
                        if os.path.exists(evict_path):
                            os.remove(evict_path)
                    kept = [(s, f"{a:.3f}") for a, s, _ in best_ckpts]
                    print(f"    -> saved {p} (rate={avg_rate:.4f}) | top-{args.topk_save}: {kept}", flush=True)
                else:
                    print(f"    -> saved {p} (rate={avg_rate:.4f})", flush=True)

        if args.max_steps and global_step >= args.max_steps:
            break

    if is_main(rank):
        final = os.path.join(args.output_dir, "draft_final.pt")
        cfg_dict = ckpt.get("config", {})
        torch.save({
            "state_dict": draft_module.state_dict(),
            "config": cfg_dict,
            "target_layer_ids": ckpt.get("target_layer_ids", [1, 17, 32]),
            "rollout_length": ckpt.get("rollout_length", 7),
        }, final)
        print(f"Training complete. Final: {final}", flush=True)
        if use_wandb:
            import wandb; wandb.finish()

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
