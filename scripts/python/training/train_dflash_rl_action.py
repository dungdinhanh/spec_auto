"""RL fine-tuning of a DFlash draft with trajectory-action-MSE as reward.

Design (six decisions pinned 2026-04-21):
  1. Reward  = − MSE(action_gt, action_mix) in normalized (accel, curvature) space.
  2. Block   = random `block_start ∼ Uniform(0, N − block_size)` per step.
  3. Init    = dflash_L4_lr1e-4_ep15_bs16_sharon1/draft_final.pt (block_size=16).
  4. KL ref  = frozen copy of init, kl_weight = 0.02.
  5. Sample T= 1.0 during RL; argmax at inference.
  6. K       = 4 samples per step, batched through target VLM + diffusion.

Per training step:
  Pass A (once, no grad):
     VLM forward(prompt + gt_coc + <|traj_future_start|>)  → kv cache A
     manual_seed(s); action_gt = diffusion.sample(cache A)
  Draft forward (once, with grad):
     build context (prompt + gt_coc[:block_start+1]) → target hidden → draft
     → logits at 15 block positions
  K = 4 samples from draft logits (with grad through log_prob):
     sample_k, log_prob_k, mixed_coc_k
  Pass B (batched K=4 via target+diffusion, no grad):
     manual_seed(s); action_mix_k = diffusion.sample(cache B_k)
     reward_k = − MSE(action_gt, action_mix_k)
  Loss:
     advantage = reward − group_mean(reward)
     rl_loss = − Σ advantage · log_prob
     kl_loss = 0.02 · KL(π_current || π_ref)  over the 15 block positions
     total.backward(); optim.step()
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
    # SAMPLING weight (not loss weight): probability of picking each block_start
    # position. We pick ONE block_start per step with P(block_start=b) ∝ decay^b,
    # so early positions are sampled more often. Loss and gradients are unchanged;
    # we just visit early positions more frequently during training.
    #
    # - INCLUDES block_start=0 (the first-block case e2e evaluates most heavily).
    # - Block may overshoot N (partial tail), handled by gt_available below.
    # - With decay=0.8: block_start=0 sampled ~5x more often than block_start=8.
    lo = 0
    hi = max(N - 2, 0)                              # inclusive; ensures >= 1 GT token
    candidates = list(range(lo, hi + 1))
    decay = args.block_start_decay
    weights = [decay ** b for b in candidates]
    block_start = rng.choices(candidates, weights=weights, k=1)[0]
    anchor_token = int(gt_coc[0, block_start].item())
    context_len = P + block_start                   # EXCLUDES anchor (no leak)

    # ---- Draft forward (with grad) ----
    # Need target hidden states over (prompt + gt_coc[:block_start + 1]) — the
    # full prompt + tokens 0..block_start inclusive of the anchor. We'll slice
    # the hidden states to `context_len` positions (= P + block_start) inside
    # `draft_block_logits` so the anchor's hidden state is not leaked.
    context_input_ids = torch.cat(
        [prompt_ids, gt_coc[:, :block_start + 1]], dim=1
    )
    target_hidden = target_hidden_for_context(
        target_model.vlm, context_input_ids, pixel_values, image_grid_thw
    )
    logits = draft_block_logits(
        draft, embed_tokens, lm_head, target_hidden,
        draft.target_layer_ids, context_len, anchor_token, B_size, MASK_ID, device,
    )   # (block_size - 1, V)
    with torch.no_grad():
        ref_logits = draft_block_logits(
            draft_ref, embed_tokens, lm_head, target_hidden,
            draft_ref.target_layer_ids, context_len, anchor_token, B_size, MASK_ID, device,
        )

    # ---- Sample K sequences ----
    K = args.k_samples
    probs = F.softmax(logits / args.temperature, dim=-1)            # (15, V)
    # torch.multinomial samples rowwise; use K draws per row
    # shape of sampled: (15, K)
    sampled = torch.multinomial(probs, K, replacement=True)
    log_probs_all = F.log_softmax(logits, dim=-1)                   # (15, V)
    # Pick the log_prob at each sampled token: shape (15, K)
    log_probs_sampled = torch.gather(
        log_probs_all.unsqueeze(1).expand(-1, K, -1), -1,
        sampled.unsqueeze(-1),
    ).squeeze(-1)
    seq_log_probs = log_probs_sampled.sum(dim=0)                    # (K,)

    # ---- Build K mixed CoC, batched VLM + diffusion ----
    block_end = block_start + B_size
    gt_available = min(B_size - 1, N - block_start - 1)   # GT positions in this block

    # Pre-compute which positions in each sample's draft differ from GT (needed
    # for contamination reward and also for acceptance_length below).
    target_greedy_block = gt_coc[0, block_start + 1:block_start + 1 + gt_available]
    matches_pre = (sampled[:gt_available] == target_greedy_block.unsqueeze(1))  # (gt_available, K) bool

    mixed_seqs = []
    if args.contamination_N > 0:
        # === Contamination reward ===
        # For each K sample, find the first `contamination_N` positions where the
        # draft's prediction differs from target greedy. Substitute ONLY at those
        # positions; all other positions stay at target greedy. The K samples
        # differ only in their rejected-token substitutions, giving per-sample
        # action-MSE variance that directly reflects how badly each draft's first
        # N mistakes would perturb the target's action.
        N_contam = args.contamination_N
        for k in range(K):
            mc = gt_coc.clone().squeeze(0)
            placed = 0
            for p in range(gt_available):
                if not matches_pre[p, k].item():
                    mc[block_start + 1 + p] = sampled[p, k]
                    placed += 1
                    if placed >= N_contam:
                        break
            seq_k = torch.cat([prompt_ids.squeeze(0), mc, torch.tensor([TRAJ_FUTURE_START], device=device)])
            mixed_seqs.append(seq_k)
    else:
        # === Legacy fixed-m substitution ===
        m_eff = min(args.subst_m, gt_available)
        for k in range(K):
            mc = gt_coc.clone().squeeze(0)
            if m_eff > 0:
                mc[block_start + 1:block_start + 1 + m_eff] = sampled[:m_eff, k]
            seq_k = torch.cat([prompt_ids.squeeze(0), mc, torch.tensor([TRAJ_FUTURE_START], device=device)])
            mixed_seqs.append(seq_k)
    batched_ids = torch.stack(mixed_seqs, dim=0)   # (K, P+N+1)
    batched_pix = pixel_values.unsqueeze(0).expand(K, *pixel_values.shape).reshape(
        K * pixel_values.shape[0], *pixel_values.shape[1:]
    )
    batched_grid = image_grid_thw.unsqueeze(0).expand(K, *image_grid_thw.shape).reshape(
        K * image_grid_thw.shape[0], *image_grid_thw.shape[1:]
    )
    cache_mix, rd_mix = vlm_prefill(target_model, batched_ids, batched_pix, batched_grid)
    traj_pos_mix = torch.full((K,), P + N, device=device, dtype=torch.long)
    action_mix = run_diffusion_on_cache(target_model, cache_mix, rd_mix,
                                         traj_pos_mix, seed=seed)   # (K, 64, 2)

    # ---- Rewards ----
    # action_gt: (1, 64, 2). action_mix: (K, 64, 2). Broadcast.
    diff = action_mix - action_gt                                  # (K, 64, 2)
    mse_per_sample = (diff * diff).mean(dim=(-1, -2))               # (K,)

    # Acceptance length from the pre-computed matches matrix.
    matches = matches_pre                                                              # (gt_available, K)
    accepted_per_pos = matches.long().cumprod(dim=0)                                   # (gt_available, K)
    accepted_length = accepted_per_pos.sum(dim=0).float()                              # (K,) in [0, gt_available]

    # Hybrid reward: local action-MSE signal on m spliced positions + global
    # acceptance-length signal across the whole block.
    rewards = -mse_per_sample + args.accept_bonus * accepted_length  # (K,)

    # ---- Advantage (group baseline) ----
    baseline = rewards.mean()
    advantage = rewards - baseline                                  # (K,)

    # ---- Losses ----
    # Policy gradient: maximize E[advantage * log_prob].
    # seq_log_probs still has grad; advantage is detached.
    rl_loss = -(advantage.detach() * seq_log_probs).mean()

    # KL(pi_current || pi_ref) averaged over 15 positions
    with torch.no_grad():
        log_ref = F.log_softmax(ref_logits, dim=-1)                 # (15, V)
    log_pi = F.log_softmax(logits, dim=-1)                          # (15, V)
    pi = log_pi.exp()
    kl_per_pos = (pi * (log_pi - log_ref)).sum(dim=-1)              # (15,)
    kl_loss = kl_per_pos.mean()

    total_loss = rl_loss + args.kl_weight * kl_loss

    # Acceptance rate is fraction of GT positions accepted (denominator-
    # adjusted so it's comparable across block_start values with different
    # gt_available).
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
    }


# ----------------------------------------------------------------------------
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
    ap.add_argument("--num_epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--k_samples", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--subst_m", type=int, default=6,
                    help="Number of block positions to substitute with draft samples (action-MSE signal). "
                         "Remaining positions stay as gt_coc. Must be <= block_size - 1.")
    ap.add_argument("--accept_bonus", type=float, default=0.15,
                    help="Reward bonus per accepted token: reward_k = -mse + accept_bonus * accepted_length_k.")
    ap.add_argument("--contamination_N", type=int, default=0,
                    help="Experiment 2: substitute draft's first N rejected-token predictions "
                         "into GT CoC. N=0 uses legacy fixed-m substitution. N>=1 enables "
                         "contamination reward.")
    ap.add_argument("--block_start_decay", type=float, default=0.8,
                    help="Geometric decay for block_start sampling weights. "
                         "P(block_start=b) ∝ decay^b. decay=0.8 → block_start=0 sampled ~5x more than block_start=8. "
                         "Lower = stronger early-position preference.")
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
    def build():
        return build_dflash_draft_mrope_for_qwen3vl(
            vlm, num_draft_layers=num_layers,
            block_size=bsz, mask_token_id=mask_id,
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

            if is_main(rank) and global_step % args.log_interval == 0:
                rate = global_step / max(time.time() - t0, 1)
                print(
                    f"  epoch {epoch+1} step {global_step} | "
                    f"rl={out['rl_loss'].item():+.4f} "
                    f"kl={out['kl_loss'].item():.4f} "
                    f"reward_mean={out['mean_reward'].item():+.5f} "
                    f"reward_std={out['std_reward'].item():.5f} "
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
                        "rl/min_mse": out['min_mse'].item(),
                        "rl/mean_accepted": out['mean_accepted'].item(),
                        "rl/max_accepted": out['max_accepted'].item(),
                        "rl/gt_available": out['gt_available'].item(),
                        "rl/mean_accept_rate": out['mean_accept_rate'].item(),
                    }, step=global_step)

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
