"""RL fine-tuning of a DFlash draft with Alpamayo-paper-inspired 3-component reward — v5.

v5 vs v4: adds MULTI-BLOCK CONTAMINATION mode.

Motivation: v4 processes ONE block per training step. With contamination_N=3
we get only 3 contaminated positions per step → very sparse RL signal. v5 adds
a mode where each training step processes MULTIPLE consecutive blocks of the
same clip, accumulating up to --multiblock_max_total contaminated positions.

New semantics (gated by --multiblock_N > 0):
  * In each rejection block, contaminate N CONSECUTIVE positions starting at
    the first greedy rejection (positions r, r+1, ..., r+N-1).
  * After processing a block: advance the cursor to (block_start + r + N) and
    look for the NEXT rejection block. The prefix for that next block is the
    GT CoC (NOT the sampled tokens from the previous block).
  * Continue until cumulative contaminated positions reach --multiblock_max_total,
    or until no more rejection blocks remain in the CoC.
  * Per-block losses are AVERAGED into the final step loss (scale-invariant
    w.r.t. number of blocks accumulated).
  * Each block has its OWN K=K_samples rollouts and its OWN GRPO advantage
    (group baseline computed within block, not across blocks).

When --multiblock_N == 0 (default): identical to v4 behavior.

v4 vs v3: adds an online per-batch rejection-filter for block_start sampling.

Motivation: v3 sampled block_start ~ Geometric(decay=0.8) over [0, N-2]. Empirical
analysis on the v5-partial AARL run showed 57.5% of training steps had R=0 (zero
greedy rejections in the sampled block) → no policy gradient on those steps. Two
compounding causes: (a) decay=0.8 oversamples block_start near 0, where the
SFT-init draft is strongest, and (b) gt_avail<7 trivially all-accept due to short
end-of-CoC blocks.

v4 fix: when --filter_to_rejection_blocks is set, every step:
  1. Probes the draft greedily at every candidate block_start b in [0, N-2]
     (no_grad, fast, reuses a single full-CoC target forward).
  2. Builds the per-clip rejection list = { b : draft greedy at b has at least
     one rejection in the block }.
  3. Skips the clip entirely if the list is empty.
  4. Samples block_start uniformly from the list (no decay).

This is "online per-batch" — the rejection list is recomputed every step using
the CURRENT draft, not pre-cached, so it adapts as the policy moves.
block_start=0 is included only if the (current) draft has rejection there,
matching the strict-filter directive.

Original v3 docstring follows.
==============================================================================
RL fine-tuning of a DFlash draft with Alpamayo-paper-inspired 3-component reward — v3.

Reward composition (per K-th rollout):
    reward_k = w_traj  · r_traj_k       # action quality   (= -MSE, like v2)
            + w_cons  · r_cons_k        # meta-action consistency (rule-based)
            + w_text  · r_text_sim_k    # token-level CoC similarity (replaces LLM-judge r_reasoning)

Two operating modes selected via --enable_r_cons:
  * Option A  (default, --enable_r_cons=False / w_cons=0):
        reward = w_traj · r_traj + w_text · r_text_sim
        — drop consistency term entirely. Cleanest, fewest hyperparameters.
  * Option B  (--enable_r_cons=True, w_cons>0):
        + r_cons_k = 1[ meta_action(action_mix_k) == meta_action(action_gt) ]
        Rule-based detector defined locally (meta_action_label below) — the
        Alpamayo paper's detector was not released in the public code, so we
        re-implement a simple longitudinal+lateral classifier.

Why token-exact match for r_text_sim (not an LLM judge):
  We have a deterministic GT (target's greedy CoC). Per-position match is
  unambiguous, deterministic, and fast (a few tensor compares). An LLM judge
  would add ~5-10 s per training step plus drift / non-reproducibility. The
  Alpamayo paper used a judge only because their reward (reasoning quality) is
  fuzzy; ours ("did draft pick the same token target picked") is exact.

Differences from v2:
  * Reward composition expanded from {-MSE + accept_bonus*accepted_length} to
    the 3-component form above. Default Option A.
  * K bumped 4 → 5.
  * --accept_bonus is removed (subsumed by r_text_sim). accepted_length is
    still computed and logged as a diagnostic.
  * New args: --enable_r_cons, --w_traj, --w_cons, --w_text,
              --consistency_horizon, --eps_long, --eps_lat.

Inherited from v2 (unchanged):
  * Stochastic sampling restricted to greedy-rejected positions; KL only on
    matched positions; block_start ~ P(b) ∝ decay^b; periodic ref-update.
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
    build_dflash_draft_for_qwen3vl,
    get_qwen3vl_embed_and_head,
    extract_context_feature,
    load_draft_checkpoint,
)

# Alpamayo tokens
TRAJ_FUTURE_START = 155681
PAD_ID = 151643
MASK_ID = 151662


def is_main(rank): return rank == 0


# ----------------------------------------------------------------------------
# Rule-based meta-action detector (Option B path)
# ----------------------------------------------------------------------------
def meta_action_label(
    action: torch.Tensor,         # (..., T, 2)  [accel, curvature]
    horizon: int = 16,
    eps_long: float = 0.05,       # m/s^2 threshold for accel/decel/hold
    eps_lat: float = 0.10,        # rad threshold for left/right/straight
    dt: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (long_label, lat_label) ints in {-1, 0, +1}.

    long_label = sign(mean(accel[:H])) with tie-band ±eps_long → -1=decel, 0=hold, +1=accel.
    lat_label  = sign(integrate(curvature[:H] · v[t]) dt) with tie-band ±eps_lat
                 — net heading change. We assume v=1 (curvature alone), since the
                 trajectory we have is in normalized acc/curv space and we just
                 need a sign-style heuristic. Net heading proxy = sum(curv[:H]) * dt.

    Output shape matches action[..., 0, 0]: same leading dims as input.
    """
    H = min(horizon, action.shape[-2])
    head = action[..., :H, :]                            # (..., H, 2)
    mean_accel = head[..., 0].mean(dim=-1)               # (...)
    long_lab = torch.zeros_like(mean_accel, dtype=torch.long)
    long_lab[mean_accel > eps_long] = 1
    long_lab[mean_accel < -eps_long] = -1
    # net heading proxy = sum of curvatures (treating v as unit; ok for class-only)
    net_heading = head[..., 1].sum(dim=-1) * dt          # (...)
    lat_lab = torch.zeros_like(net_heading, dtype=torch.long)
    lat_lab[net_heading > eps_lat] = 1
    lat_lab[net_heading < -eps_lat] = -1
    return long_lab, lat_lab


@torch.no_grad()
def update_ref_model(ref_module: nn.Module, trainable_module: nn.Module,
                     mode: str = "replace", ema_alpha: float = 0.9):
    """Update reference draft from trainable draft.
       mode='replace': ref ← trainable (state_dict copy, no_grad).
       mode='ema':     ref ← ema_alpha * ref + (1 - ema_alpha) * trainable, per parameter."""
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
    # Ensure ref stays in eval mode
    ref_module.eval()


# ----------------------------------------------------------------------------
# Dataset: target_coc_outputs samples (reuse the same files as supervised training)
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
            "prompt_input_ids": d["prompt_input_ids"].squeeze(0),   # (prompt_len,)
            "output_token_ids": d["output_token_ids"],              # (num_gen,)
            "pixel_values": d["pixel_values"].to(torch.bfloat16),
            "image_grid_thw": d["image_grid_thw"],
            "clip_id": d["clip_id"],
        }
        return out


def collate_one(features):
    """Return a single sample (we always process one clip at a time outward)."""
    assert len(features) == 1, "batch_size must be 1 at the dataloader level"
    return features[0]


# Alias used in main() below
collate_fn = collate_one


# ----------------------------------------------------------------------------
# VLM + diffusion pipeline to produce an action tensor given full input_ids
# ----------------------------------------------------------------------------
@torch.no_grad()
def vlm_prefill(
    target_model: AlpamayoR1,
    input_ids: torch.Tensor,          # (B, T) — includes <|traj_future_start|> at the end
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
) -> tuple[DynamicCache, torch.Tensor]:
    """Prefill target VLM on the full (prompt + CoC + traj_future_start) sequence."""
    vlm = target_model.vlm
    past = DynamicCache()
    out = vlm(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        past_key_values=past,
        use_cache=True,
        return_dict=True,
    )
    # rope_deltas: tensor of shape (B,) — the offset applied during MRoPE
    rope_deltas = vlm.model.rope_deltas
    return past, rope_deltas


@torch.no_grad()
def run_diffusion_on_cache(
    target_model: AlpamayoR1,
    prompt_cache: DynamicCache,
    rope_deltas: torch.Tensor,
    traj_future_start_positions: torch.Tensor,   # (B,) — position of <|traj_future_start|>
    seed: int,
) -> torch.Tensor:
    """Run flow-matching diffusion on a pre-built VLM KV cache.
    Returns action of shape (B, 64, 2) in normalized space."""
    device = prompt_cache.key_cache[0].device if hasattr(prompt_cache, 'key_cache') \
             else rope_deltas.device
    # Fallback — just use a device we know is right:
    device = rope_deltas.device
    B = traj_future_start_positions.shape[0]
    n_diffusion_tokens = target_model.action_space.get_action_space_dims()[0]
    prefill_seq_len = prompt_cache.get_seq_length()

    # Build position_ids (mirrors sample_trajectories_from_data_with_vlm_rollout)
    position_ids = torch.arange(n_diffusion_tokens, device=device)
    position_ids = einops.repeat(position_ids, "l -> 3 b l", b=B).clone()
    offset = traj_future_start_positions + 1
    delta = rope_deltas + offset[:, None]
    position_ids += delta.to(position_ids.device)

    # Must match expert's compute dtype (bf16) for SDPA bias — HF requires
    # attention_mask.dtype == query.dtype when passing a float bias.
    expert_dtype = next(target_model.expert.parameters()).dtype
    attention_mask = torch.zeros(
        (B, 1, n_diffusion_tokens, prefill_seq_len + n_diffusion_tokens),
        dtype=expert_dtype, device=device,
    )
    for i in range(B):
        attention_mask[i, :, :, offset[i]:-n_diffusion_tokens] = torch.finfo(
            attention_mask.dtype
        ).min

    forward_kwargs = {}
    if target_model.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    # action_in_proj is fp32 (its forward does .float() internally); expert is bf16.
    expert_dtype = next(target_model.expert.parameters()).dtype

    def step_fn(x, t):
        b = x.shape[0]
        # action_in_proj takes float32 and returns float32
        fte = target_model.action_in_proj(x.float(), t.float() if torch.is_tensor(t) else t)
        if fte.dim() == 2:
            fte = fte.view(b, n_diffusion_tokens, -1)
        # Bridge to expert's dtype
        fte = fte.to(expert_dtype)
        exp_out = target_model.expert(
            inputs_embeds=fte,
            position_ids=position_ids,
            past_key_values=prompt_cache,
            attention_mask=attention_mask,
            use_cache=True,
            **forward_kwargs,
        )
        prompt_cache.crop(prefill_seq_len)   # keep cache pristine across steps
        lh = exp_out.last_hidden_state[:, -n_diffusion_tokens:]
        pred = target_model.action_out_proj(lh).view(
            -1, *target_model.action_space.get_action_space_dims()
        )
        # Cast back to float32 so the diffusion integrator math is stable
        return pred.float()

    torch.manual_seed(seed)
    sampled = target_model.diffusion.sample(
        batch_size=B, step_fn=step_fn, device=device,
        return_all_steps=False,
    )
    return sampled  # (B, 64, 2)


# ----------------------------------------------------------------------------
# Draft forward to get logits at block positions
# ----------------------------------------------------------------------------
@torch.no_grad()
def target_hidden_for_context(
    target_vlm, input_ids, pixel_values, image_grid_thw
) -> tuple:
    """Forward target VLM with output_hidden_states, return the layerwise
    hidden-state tuple."""
    past = DynamicCache()
    out = target_vlm(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        past_key_values=past,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    return out.hidden_states


def draft_block_logits(
    draft, embed_tokens, lm_head,
    target_hidden_states: tuple,     # tuple of (L, B, T, H) from target forward
    target_layer_ids: list[int],
    context_len: int,                # prompt_len + block_start (exclusive of anchor → no leak)
    anchor_token_id: int,            # gt_coc[block_start] — the anchor for the block
    block_size: int,
    mask_token_id: int,
    device: torch.device,
) -> torch.Tensor:
    """Run the draft forward on the block and return the (block_size - 1)
    logits used to predict tokens at positions anchor+1 .. anchor+block_size-1.
    Gradients flow through the draft.

    Matches training convention (train_dflash_distillation_v2.py):
      ctx_hidden = target_hidden[:, :start, :]   (EXCLUDES anchor — no leak)
      noise_emb  = embed([anchor, MASK, MASK, ..., MASK])  length = block_size
      pos_ids    = arange(ctx_len + block_size)
      logits = lm_head(draft_hidden[:, -(block_size - 1):, :])
    """
    ctx_all = extract_context_feature(target_hidden_states, target_layer_ids)
    ctx_hidden = ctx_all[:, :context_len, :]

    # noise tokens: [anchor, MASK, MASK, ..., MASK]
    noise_tokens = torch.full(
        (1, block_size), mask_token_id, dtype=torch.long, device=device,
    )
    noise_tokens[0, 0] = anchor_token_id
    noise_emb = embed_tokens(noise_tokens)

    pos_ids = torch.arange(context_len + block_size, device=device).unsqueeze(0)
    draft_hidden = draft(
        target_hidden=ctx_hidden,
        noise_embedding=noise_emb,
        position_ids=pos_ids,
    )
    # Take the LAST (block_size - 1) outputs — these correspond to positions
    # anchor+1..anchor+block_size-1 (offsets 1..block_size-1 from anchor).
    block_logits = lm_head(draft_hidden[:, -(block_size - 1):, :])
    return block_logits.squeeze(0)   # (block_size - 1, V)


# ----------------------------------------------------------------------------
# Per-block forward + loss (used by both legacy and multiblock paths)
# ----------------------------------------------------------------------------
def _compute_block_loss(
    *, target_model, draft, draft_ref, embed_tokens, lm_head,
    prompt_ids, gt_coc, pixel_values, image_grid_thw,
    target_hidden, action_gt,
    block_start, contam_pos_local, contam_mode, contam_N_legacy,
    args, device, rng, seed,
    P, N, B_size, K,
) -> Optional[dict]:
    """Compute one block's RL forward + reward + loss.

    Args:
      contam_pos_local: list[int] of LOCAL positions in [0, B-2] to contaminate,
        OR None to determine internally from contam_mode.
      contam_mode: when contam_pos_local is None — one of:
        "scattered_N" : first contam_N_legacy greedy-rejected positions
        "fixed_m"     : positions [0, m_eff) (legacy v2 path)
      contam_N_legacy: N for scattered_N, or m for fixed_m.
      seed: diffusion noise seed for action_mix (use distinct seeds across
        blocks so per-block r_traj variance is independent).
    """
    anchor_token = int(gt_coc[0, block_start].item())
    context_len = P + block_start                   # EXCLUDES anchor (no leak)

    # ---- Draft forward (with grad) ----
    logits = draft_block_logits(
        draft, embed_tokens, lm_head, target_hidden,
        draft.target_layer_ids, context_len, anchor_token, B_size, MASK_ID, device,
    )   # (B-1, V)
    with torch.no_grad():
        ref_logits = draft_block_logits(
            draft_ref, embed_tokens, lm_head, target_hidden,
            draft_ref.target_layer_ids, context_len, anchor_token, B_size, MASK_ID, device,
        )

    B_minus_1 = B_size - 1
    gt_available = min(B_minus_1, N - block_start - 1)

    greedy_pred = logits.argmax(dim=-1)                              # (B-1,)
    target_greedy_block = gt_coc[0, block_start + 1:block_start + 1 + gt_available]
    greedy_matches_gt = (greedy_pred[:gt_available] == target_greedy_block)
    matched_full = torch.zeros(B_minus_1, dtype=torch.bool, device=device)
    rejected_full = torch.zeros(B_minus_1, dtype=torch.bool, device=device)
    matched_full[:gt_available] = greedy_matches_gt
    rejected_full[:gt_available] = ~greedy_matches_gt

    rejected_pos_idx_full = torch.nonzero(greedy_matches_gt == False, as_tuple=False).flatten()  # (R_in,)
    R = int(rejected_full.sum().item())

    # ---- Decide contam_pos_local if caller didn't provide one ----
    if contam_pos_local is None:
        if contam_mode == "scattered_N":
            contam_pos_local = rejected_pos_idx_full[:contam_N_legacy].tolist()
        elif contam_mode == "fixed_m":
            m_eff = min(contam_N_legacy, gt_available)
            contam_pos_local = list(range(m_eff))
        else:
            raise ValueError(f"contam_mode {contam_mode} requires contam_pos_local")
    # Always clip to valid range.
    contam_pos_local = [int(p) for p in contam_pos_local if 0 <= int(p) < gt_available]

    # ---- Sampling mask: stochastic at contam positions that are also rejected ----
    contam_full = torch.zeros(B_minus_1, dtype=torch.bool, device=device)
    if contam_pos_local:
        contam_full[torch.tensor(contam_pos_local, dtype=torch.long, device=device)] = True
    stochastic_full = contam_full & rejected_full

    sampled = greedy_pred.unsqueeze(1).expand(-1, K).clone()         # (B-1, K)
    n_stoch = int(stochastic_full.sum().item())
    if n_stoch > 0:
        probs_stoch = F.softmax(logits[stochastic_full] / args.temperature, dim=-1)  # (n_stoch, V)
        sampled_stoch = torch.multinomial(probs_stoch, K, replacement=True)          # (n_stoch, K)
        sampled[stochastic_full] = sampled_stoch

    log_probs_all = F.log_softmax(logits, dim=-1)                    # (B-1, V)
    log_probs_sampled = torch.gather(
        log_probs_all.unsqueeze(1).expand(-1, K, -1), -1,
        sampled.unsqueeze(-1),
    ).squeeze(-1)
    # Policy gradient contribution from STOCHASTIC contam positions only.
    seq_log_probs = (log_probs_sampled * stochastic_full.unsqueeze(1).float()).sum(dim=0)  # (K,)

    matches_pre = (sampled[:gt_available] == target_greedy_block.unsqueeze(1))  # (gt_available, K)

    # ---- Build K mixed CoC by substituting at contam_pos_local ----
    mixed_seqs = []
    for k in range(K):
        mc = gt_coc.clone().squeeze(0)
        for p in contam_pos_local:
            mc[block_start + 1 + p] = sampled[p, k]
        seq_k = torch.cat([prompt_ids.squeeze(0), mc, torch.tensor([TRAJ_FUTURE_START], device=device)])
        mixed_seqs.append(seq_k)
    if not mixed_seqs:
        return None
    batched_ids = torch.stack(mixed_seqs, dim=0)   # (K, P+N+1)

    # ---- Target VLM + diffusion in chunks ----
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
    action_mix = torch.cat(action_chunks, dim=0)  # (K, 64, 2)

    # ---- Rewards ----
    diff = action_mix - action_gt                                   # (K, 64, 2)
    mse_per_sample = (diff * diff).mean(dim=(-1, -2))               # (K,)
    r_traj = -mse_per_sample                                        # (K,)

    # r_text on contam_pos_local (whichever positions we actually substituted)
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

    # ---- GRPO advantage + policy loss ----
    baseline = rewards.mean()
    advantage = rewards - baseline
    rl_loss = -(advantage.detach() * seq_log_probs).mean()

    # ---- KL anchor ----
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
# RL training step
# ----------------------------------------------------------------------------
def rl_step(
    target_model, draft, draft_ref, embed_tokens, lm_head,
    batch, args, device, rng
) -> Optional[dict]:
    """Run one RL step on one clip. Returns dict of loss terms or None if skipped."""
    prompt_ids = batch["prompt_input_ids"].to(device).unsqueeze(0)     # (1, P)
    gt_coc = batch["output_token_ids"].to(device).unsqueeze(0)         # (1, N)
    pixel_values = batch["pixel_values"].to(device).to(torch.bfloat16)
    image_grid_thw = batch["image_grid_thw"].to(device)

    P = prompt_ids.shape[1]
    N = gt_coc.shape[1]
    B_size = args.block_size
    # Need at least 1 anchor token + 1 GT token after it to produce any training signal.
    # Short clips are OK now — we'll just run a partial-tail block.
    if N < 2:
        return None

    # ---- Pass A: reference action from gt CoC ----
    traj_start = torch.tensor([[TRAJ_FUTURE_START]], device=device)
    seq_gt = torch.cat([prompt_ids, gt_coc, traj_start], dim=1)   # (1, P+N+1)
    traj_start_pos_gt = torch.tensor([P + N], device=device)       # index of <|traj_future_start|>
    cache_gt, rd_gt = vlm_prefill(target_model, seq_gt, pixel_values, image_grid_thw)
    seed = rng.randrange(0, 2**31 - 1)
    action_gt = run_diffusion_on_cache(target_model, cache_gt, rd_gt,
                                        traj_start_pos_gt, seed=seed)   # (1, 64, 2)

    # ---- Choose block ----
    # v4: when --filter_to_rejection_blocks is set, probe every candidate
    # block_start with the CURRENT draft (no_grad), build a list of
    # block_starts that have at least one greedy rejection, and sample uniformly
    # from that list. Skip the clip if no block_start has any rejection.
    # Otherwise (legacy v3 path), sample with P(b) ∝ decay^b.
    lo = 0
    hi = max(N - 2, 0)                              # inclusive; ensures >= 1 GT token
    candidates = list(range(lo, hi + 1))

    # Compute target hidden states ONCE on prompt + gt_coc (full CoC) — needed
    # both by the probe pass (slicing per block_start) and by the with-grad
    # draft forward at the chosen block_start. This replaces the v3
    # per-block-start `target_hidden_for_context(prompt + gt_coc[:b+1])` call;
    # the longer forward gives identical hiddens at every position 0..P+N-1
    # by causal masking, and we slice as needed.
    context_input_ids_full = torch.cat([prompt_ids, gt_coc], dim=1)   # (1, P+N)
    target_hidden = target_hidden_for_context(
        target_model.vlm, context_input_ids_full, pixel_values, image_grid_thw
    )

    rej_list_size = -1   # sentinel: "filter not used"; positive = list size
    n_candidates = -1
    # rejection_info: list of (block_start, first_rejection_pos_in_block, gt_avail_b)
    # — first_rejection_pos is the LOCAL position (0..B-2) of the first greedy
    # rejection in that block. Used by multiblock mode to pick the contam window.
    rejection_info: list[tuple[int, int, int]] = []
    if args.filter_to_rejection_blocks or args.multiblock_N > 0:
        # === Probe pass: find block_starts with R >= 1 under current draft ===
        with torch.no_grad():
            for b in candidates:
                ctx_len_b = P + b
                anchor_b = int(gt_coc[0, b].item())
                gt_avail_b = min(B_size - 1, N - b - 1)
                if gt_avail_b <= 0:
                    continue
                logits_b = draft_block_logits(
                    draft, embed_tokens, lm_head, target_hidden,
                    draft.target_layer_ids, ctx_len_b, anchor_b, B_size,
                    MASK_ID, device,
                )                                                       # (B-1, V)
                gp_b = logits_b.argmax(dim=-1)                          # (B-1,)
                gt_b = gt_coc[0, b + 1:b + 1 + gt_avail_b]
                mismatches_b = (gp_b[:gt_avail_b] != gt_b)              # (gt_avail_b,) bool
                R_b = int(mismatches_b.sum().item())
                if R_b >= 1:
                    first_rej = int(torch.nonzero(mismatches_b).flatten()[0].item())
                    rejection_info.append((b, first_rej, gt_avail_b))
        rej_list_size = len(rejection_info)
        n_candidates = len(candidates)
        if not rejection_info:
            return None
        # For legacy single-block (multiblock_N == 0): sample uniformly.
        if args.multiblock_N == 0:
            chosen = rejection_info[rng.randrange(len(rejection_info))]
            block_start = chosen[0]
    else:
        decay = args.block_start_decay
        weights = [decay ** b for b in candidates]
        block_start = rng.choices(candidates, weights=weights, k=1)[0]

    # ---- Multiblock contamination path (v5) ----
    if args.multiblock_N > 0:
        # rejection_info is sorted ascending by block_start (candidates is range).
        base_seed = rng.randrange(0, 2**31 - 1)
        block_outs: list[dict] = []
        cumulative = 0
        cursor = 0   # block_start must be >= cursor (skips overlap with previous contam region)
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
            bl = _compute_block_loss(
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
        # Average all per-block losses → one scalar.
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

    # ---- Legacy single-block path (v4-identical behavior) ----
    legacy_seed = rng.randrange(0, 2**31 - 1)
    if args.contamination_N > 0:
        contam_mode, contam_N_leg = "scattered_N", args.contamination_N
    else:
        contam_mode, contam_N_leg = "fixed_m", args.subst_m
    out_legacy = _compute_block_loss(
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


# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_path", required=True)
    ap.add_argument("--init_draft_path", required=True)
    ap.add_argument("--target_outputs_dir", required=True)
    ap.add_argument("--val_uuids_file", required=True)
    ap.add_argument("--test_uuids_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_draft_layers", type=int, default=4)
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--num_target_features", type=int, default=5,
                    help="Number of target hidden-state layers concatenated as cross-attn "
                         "input to the draft. v6/v7 default 5; legacy v2 used num_draft_layers.")
    ap.add_argument("--num_epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--k_samples", type=int, default=5,
                    help="K rollouts per training step (group size for advantage baseline). v3 default 5.")
    ap.add_argument("--k_chunk_size", type=int, default=5,
                    help="Chunk size along the K dimension when running target VLM + diffusion. "
                         "Memory peaks at chunk_K samples through the target. v3 default 5.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--subst_m", type=int, default=6,
                    help="Number of block positions to substitute with draft samples (action-MSE signal). "
                         "Remaining positions stay as gt_coc. Must be <= block_size - 1.")
    # v3 reward composition
    ap.add_argument("--w_traj", type=float, default=1.0,
                    help="Weight on r_traj = -MSE(action_mix, action_gt). Default 1.0.")
    ap.add_argument("--w_cons", type=float, default=0.0,
                    help="Weight on r_cons (rule-based meta-action consistency). Set to 0 for Option A. "
                         "Try 0.1 for Option B.")
    ap.add_argument("--w_text", type=float, default=0.5,
                    help="Weight on r_text_sim (token-level CoC overlap on substituted positions). Default 0.5.")
    ap.add_argument("--enable_r_cons", action="store_true",
                    help="Enable rule-based meta-action consistency reward (Option B). "
                         "Equivalent to setting w_cons>0; either flag triggers the computation.")
    ap.add_argument("--consistency_horizon", type=int, default=16,
                    help="Number of action steps used by the meta-action detector (1.6 s @ 10 Hz).")
    ap.add_argument("--eps_long", type=float, default=0.05,
                    help="Tie-band for longitudinal classification (m/s^2). |mean accel| <= eps_long → 'hold'.")
    ap.add_argument("--eps_lat", type=float, default=0.10,
                    help="Tie-band for lateral classification (rad). |sum curvature * dt| <= eps_lat → 'straight'.")
    # v2 holdover (kept for back-compat; default 0 makes it inert in v3)
    ap.add_argument("--accept_bonus", type=float, default=0.0,
                    help="DEPRECATED in v3 — was used in v2 reward. Kept for arg compatibility; default 0 (inert).")
    # v3 KL anchor source
    ap.add_argument("--anchor_source", choices=["policy", "ref", "weighted_all"],
                    default="policy",
                    help="Where to compute the KL anchor mask. "
                         "'policy' (default, v2 behavior): anchor where CURRENT draft's argmax "
                         "matches target greedy. Anchor set shifts as policy drifts — pathological. "
                         "'ref': anchor where the FROZEN ref draft's argmax matches target greedy. "
                         "Anchor set is static, doesn't shift. "
                         "'weighted_all': KL on every gt_available position with reduced weight "
                         "(--anchor_rejected_weight) where ref does NOT match target greedy.")
    ap.add_argument("--anchor_rejected_weight", type=float, default=0.05,
                    help="With --anchor_source weighted_all: KL weight at positions where ref doesn't "
                         "match target greedy. Default 0.05.")
    ap.add_argument("--contamination_N", type=int, default=0,
                    help="Experiment 2: substitute draft's first N rejected-token predictions "
                         "into GT CoC. N=0 uses legacy fixed-m substitution. N>=1 enables "
                         "contamination reward. SCATTERED-MISMATCH semantics (legacy).")
    # v5: multi-block contamination
    ap.add_argument("--multiblock_N", type=int, default=0,
                    help="v5: window size of CONSECUTIVE positions to contaminate per "
                         "rejection block, starting at the first greedy rejection in that "
                         "block. 0 = disabled (v4 single-block behavior). >0 enables "
                         "multi-block iteration.")
    ap.add_argument("--multiblock_max_total", type=int, default=10,
                    help="v5: cumulative contaminated-position budget per training step. "
                         "Once total contaminated positions across processed blocks reach "
                         "this cap, stop iterating.")
    ap.add_argument("--block_start_decay", type=float, default=0.8,
                    help="Geometric decay for block_start sampling weights. "
                         "P(block_start=b) ∝ decay^b. decay=0.8 → block_start=0 sampled ~5x more than block_start=8. "
                         "Lower = stronger early-position preference. "
                         "Ignored when --filter_to_rejection_blocks is set.")
    ap.add_argument("--filter_to_rejection_blocks", action="store_true",
                    help="v4: per-batch online rejection filter. Probe every block_start "
                         "in [0, N-2] with the current draft (no_grad), keep only those "
                         "with at least one greedy rejection, sample uniformly from that list. "
                         "Skip the clip if every block fully accepts. Replaces decay-weighted sampling.")
    ap.add_argument("--kl_weight", type=float, default=0.02)
    ap.add_argument("--max_clips", type=int, default=None,
                    help="Optional cap for smoke tests.")
    ap.add_argument("--max_steps", type=int, default=None,
                    help="Optional cap on total optimizer steps (smoke test).")
    ap.add_argument("--log_interval", type=int, default=5)
    ap.add_argument("--save_interval", type=int, default=500)
    ap.add_argument("--topk_save", type=int, default=0,
                    help="If >0, keep only the top-K checkpoints by rolling-average acceptance_rate "
                         "(saves at save_interval, but deletes all but the K best so far).")
    ap.add_argument("--seed", type=int, default=42)
    # ---- EMA / periodic ref-update args (Option B defaults) ----
    ap.add_argument("--ref_update_interval", type=int, default=0,
                    help="Update reference draft every N training steps. 0 = static ref (default, like v2). "
                         "Recommended: 1000 for periodic updates.")
    ap.add_argument("--ref_update_mode", choices=["replace", "ema"], default="replace",
                    help="How to update ref. 'replace' copies trainable→ref; 'ema' blends.")
    ap.add_argument("--ref_ema_alpha", type=float, default=0.9,
                    help="EMA decay if mode=ema. Higher = slower update. Half-life ~7 updates at α=0.9.")
    ap.add_argument("--ref_update_gate", choices=["none", "train_rolling_rate", "eval_acceptance"],
                    default="none",
                    help="Gate the update on a quality signal. 'none' = always update. "
                         "'train_rolling_rate' = only if rolling acc_rate > best so far (free, biased). "
                         "'eval_acceptance' = run a small held-out eval and only update if mean acc_length improved. "
                         "(not yet implemented)")
    ap.add_argument("--wandb_project", default="dflash-rl-action")
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
        print(f"loading target from {args.target_path}")
    target = AlpamayoR1.from_pretrained(
        args.target_path, dtype=torch.bfloat16,
    ).to(device).eval()
    for p in target.parameters():
        p.requires_grad = False
    # action_in_proj has internal `x.float()` / `timesteps.float()` casts that
    # clash with bf16 weights. Keep it fp32 end-to-end — cheap, small module.
    target.action_in_proj = target.action_in_proj.to(torch.float32)
    vlm = target.vlm
    embed_tokens, lm_head = get_qwen3vl_embed_and_head(vlm)
    for p in embed_tokens.parameters():
        p.requires_grad = False
    for p in lm_head.parameters():
        p.requires_grad = False

    # Build draft (policy) and load checkpoint
    if is_main(rank):
        print(f"loading draft init from {args.init_draft_path}")
    ckpt = load_draft_checkpoint(args.init_draft_path, map_location=device)
    num_layers = ckpt["num_draft_layers"] or args.num_draft_layers
    bsz = ckpt["block_size"] or args.block_size
    mask_id = ckpt["mask_token_id"] or MASK_ID
    if bsz != args.block_size:
        if is_main(rank):
            print(f"  NOTE: ckpt block_size={bsz} overrides arg block_size={args.block_size}")
    args.block_size = bsz

    # Decide if draft uses M-RoPE (warm/bs16 runs do)
    # This is our standard path for bs16 drafts:
    from alpamayo_r1.models.dflash_draft_mrope import build_dflash_draft_mrope_for_qwen3vl
    # v6/v7 ckpts use num_target_features=5 (decoupled from num_draft_layers).
    # Build target_layer_ids accordingly so fc.weight shape matches the ckpt.
    from alpamayo_r1.models.dflash_draft import build_target_layer_ids as _btli
    n_target_layers = vlm.config.get_text_config().num_hidden_layers
    n_feat = args.num_target_features if args.num_target_features is not None else num_layers
    target_layer_ids = _btli(n_target_layers, n_feat)
    if is_main(rank):
        print(f"  num_target_features={n_feat} -> target_layer_ids={target_layer_ids}")
    def build():
        return build_dflash_draft_mrope_for_qwen3vl(
            vlm, num_draft_layers=num_layers,
            block_size=bsz, mask_token_id=mask_id,
            target_layer_ids=target_layer_ids,
        ).to(torch.bfloat16).to(device)

    draft = build().train()
    draft.load_state_dict(ckpt["state_dict"], strict=False)
    # Ensure every draft parameter is trainable (some ranks have seen 0
    # trainable params after prior `for p in target.parameters(): p.requires_grad=False`
    # if any were mistakenly shared).
    for p in draft.parameters():
        p.requires_grad = True

    # Build π_ref by deep-copying draft AFTER it's on-device and loaded, so
    # parameter topology is guaranteed identical.
    import copy as _copy
    draft_ref = _copy.deepcopy(draft).eval()
    for p in draft_ref.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in draft.parameters() if p.requires_grad)
    n_param_tensors = sum(1 for p in draft.parameters() if p.requires_grad)
    print(f"  [rank {rank}] Draft trainable: {n_params/1e6:.1f}M in "
          f"{n_param_tensors} tensors | num_layers={num_layers} "
          f"block_size={bsz} mask_id={mask_id}", flush=True)

    if world > 1:
        dist.barrier()
        draft = DDP(draft, device_ids=[local_rank], find_unused_parameters=True)
    draft_module = draft.module if hasattr(draft, "module") else draft

    # Data
    test_ids = json.load(open(args.test_uuids_file))
    val_ids = json.load(open(args.val_uuids_file))
    train_ds = TargetOutputDataset(
        args.target_outputs_dir,
        exclude_uuids=list(set(test_ids) | set(val_ids)),
        max_samples=args.max_clips,
    )
    if is_main(rank):
        print(f"train clips: {len(train_ds)}")

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
    # Top-K checkpoint tracking
    from collections import deque
    best_ckpts = []    # list of (avg_acc_rate, step, path) — kept sorted desc
    roll_window_len = max(1, args.save_interval // max(args.log_interval, 1))
    rolling_acc_rate = deque(maxlen=roll_window_len)
    # v4: counters for the rejection-filter path. Reset at every log_interval.
    skipped_no_rejection = 0       # clips skipped due to empty rejection list
    seen_clips = 0                 # clips actually consumed (skip + train)
    rej_list_sizes = []            # rejection_list size per non-skipped clip
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
                    print(f"  step err: {type(e).__name__}: {e}")
                    _tb.print_exc()
                continue
            seen_clips += 1
            is_skip = (out is None)

            # DDP correctness: every rank must call backward each iteration so
            # the all-reduce completes. On skip, we issue a zero-gradient
            # backward so the rank still participates. The averaged gradient on
            # this iteration is (sum of valid grads) / world_size — i.e. a
            # diluted step rather than no step. Models stay synchronized
            # because optim.step() applies the same averaged gradient on every
            # rank. global_step is incremented uniformly so save/log intervals
            # fire consistently.
            if is_skip:
                skipped_no_rejection += 1
                zero_loss = sum(
                    p.sum() for p in draft.parameters() if p.requires_grad
                ) * 0.0
                zero_loss.backward()
            else:
                if "rej_list_size" in out:
                    rls = out["rej_list_size"].item()
                    if rls >= 0:
                        rej_list_sizes.append(rls)
                loss = out["total_loss"]
                loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [p for p in draft.parameters() if p.requires_grad],
                args.max_grad_norm,
            )
            optim.step()
            optim.zero_grad()
            global_step += 1

            if is_skip:
                # Don't log a step we couldn't measure.
                continue

            # ---- Periodic ref-update (EMA / replace) ----
            if (args.ref_update_interval > 0
                    and global_step % args.ref_update_interval == 0
                    and global_step > 0):
                # Compute gate decision (currently only `none` and `train_rolling_rate` supported)
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
                # 'eval_acceptance' gate is not yet implemented — fall through to update
                if should_update:
                    # In DDP mode, all ranks have synced trainable weights.
                    # Update ref locally on each rank (deterministic since trainable is synced).
                    trainable_module = draft.module if hasattr(draft, "module") else draft
                    update_ref_model(
                        draft_ref, trainable_module,
                        mode=args.ref_update_mode, ema_alpha=args.ref_ema_alpha,
                    )
                    if is_main(rank):
                        gate_label = args.ref_update_gate
                        print(f"  [ref_update] step={global_step} mode={args.ref_update_mode} "
                              f"alpha={args.ref_ema_alpha} gate={gate_label}: ref updated.",
                              flush=True)
                else:
                    if is_main(rank):
                        print(f"  [ref_update] step={global_step} gate=train_rolling_rate "
                              f"GATE FAILED (cur_rate did not improve): ref kept.", flush=True)

            if is_main(rank) and global_step % args.log_interval == 0:
                rate = global_step / max(time.time() - t0, 1)
                # v4: rejection-filter telemetry
                if args.filter_to_rejection_blocks and seen_clips > 0:
                    skip_frac = skipped_no_rejection / max(seen_clips, 1)
                    mean_rls = (sum(rej_list_sizes) / max(len(rej_list_sizes), 1)
                                if rej_list_sizes else 0.0)
                    filter_str = (f"skip%={100*skip_frac:.1f} "
                                  f"rej_n={mean_rls:.1f}/{out['n_candidates'].item():.0f} "
                                  f"bs={out['chosen_block_start'].item():.0f} ")
                else:
                    filter_str = ""
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
                    f"{filter_str}| {rate:.2f} steps/s",
                    flush=True,
                )
                # Reset interval counters so each log line is over the window since
                # last log, not cumulative.
                skipped_no_rejection = 0
                seen_clips = 0
                rej_list_sizes = []
                if use_wandb:
                    import wandb
                    wandb_dict = {
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
                    }
                    if args.filter_to_rejection_blocks:
                        wandb_dict["rl/skip_frac"] = skip_frac
                        wandb_dict["rl/rej_list_size_mean"] = mean_rls
                        wandb_dict["rl/n_candidates"] = out['n_candidates'].item()
                        wandb_dict["rl/chosen_block_start"] = out['chosen_block_start'].item()
                    wandb.log(wandb_dict, step=global_step)

            # Push acc_rate into rolling window (only on log steps where we have data)
            if is_main(rank) and global_step % args.log_interval == 0 and "mean_accept_rate" in out:
                rolling_acc_rate.append(out["mean_accept_rate"].item())

            if global_step % args.save_interval == 0 and is_main(rank):
                avg_rate = sum(rolling_acc_rate) / max(len(rolling_acc_rate), 1)
                p = os.path.join(args.output_dir, f"draft_step_{global_step}.pt")
                torch.save({
                    "state_dict": draft_module.state_dict(),
                    "mask_token_id": mask_id,
                    "num_draft_layers": num_layers,
                    "block_size": bsz,
                    "rolling_acc_rate": float(avg_rate),
                }, p)
                if args.topk_save > 0:
                    # Insert and prune to top-K
                    best_ckpts.append((avg_rate, global_step, p))
                    best_ckpts.sort(key=lambda x: -x[0])
                    while len(best_ckpts) > args.topk_save:
                        _, evict_step, evict_path = best_ckpts.pop()
                        if os.path.exists(evict_path):
                            os.remove(evict_path)
                    kept = [(s, f"{a:.3f}") for a, s, _ in best_ckpts]
                    print(f"    -> saved {p} (rate={avg_rate:.4f}) | top-{args.topk_save}: {kept}")
                else:
                    print(f"    -> saved {p} (rate={avg_rate:.4f})")

        if args.max_steps and global_step >= args.max_steps:
            break

    if is_main(rank):
        final = os.path.join(args.output_dir, "draft_final.pt")
        torch.save({
            "state_dict": draft_module.state_dict(),
            "mask_token_id": mask_id,
            "num_draft_layers": num_layers,
            "block_size": bsz,
        }, final)
        print(f"Training complete. Final: {final}")
        if use_wandb:
            import wandb; wandb.finish()

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
