"""DFlash block diffusion draft training for Alpamayo-R1 (DDP, multi-GPU).

Training follows the DFlash paper (arXiv 2602.06036):
  - Block diffusion: draft predicts block_size-1 masked tokens in parallel
  - Position-weighted CE: w_k = exp(-(k-1)/block_size), emphasizing early positions
  - Target hidden states injected as cross-attention context into every draft layer
  - Draft shares frozen embed_tokens and lm_head with target

Trains on a mixture of:
  - UltraChat (text-only instruction data, general capability)
  - Alpamayo CoC clips (multimodal driving data, domain specialization)

Usage (single GPU):
    python scripts/train_dflash_distillation.py --target_path ... --ultrachat_dir ...

Usage (multi-GPU via torchrun):
    torchrun --nproc_per_node=4 scripts/train_dflash_distillation.py --target_path ...
"""
from __future__ import annotations

import os
import sys
import argparse
import glob
import json
import math
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper
from alpamayo_r1.models.dflash_draft import (
    build_dflash_draft_for_qwen3vl,
    get_qwen3vl_embed_and_head,
    extract_context_feature,
)

import shutil
import tempfile


def _wrap_draft_ckpt(state_dict, args):
    """Wrap state_dict with training metadata so eval can read mask_token_id etc.
    from the checkpoint directly."""
    return {
        "state_dict": state_dict,
        "mask_token_id": args.mask_token_id,
        "num_draft_layers": args.num_draft_layers,
        "block_size": args.block_size,
    }


def _safe_save(state_dict, path, use_tmp=False):
    """Save checkpoint, optionally via local /tmp to work around NFS issues.

    Args:
        use_tmp: If True, save to /tmp first then copy to destination.
                 Use when output_dir is on virtiofs/NFS that can't handle
                 direct torch.save (no lock files, atomic rename fails).
    """
    dst = Path(path)
    try:
        if not use_tmp:
            torch.save(state_dict, dst)
        else:
            with tempfile.NamedTemporaryFile(dir="/tmp", suffix=".pt", delete=False) as tmp:
                tmp_path = tmp.name
            torch.save(state_dict, tmp_path)
            tmp_size = os.path.getsize(tmp_path)
            if dst.exists():
                ts = int(time.time())
                dst = dst.with_suffix(f".{ts}.pt")
            shutil.copyfile(tmp_path, dst)
            os.unlink(tmp_path)
        print(f"    -> saved {dst} ({os.path.getsize(dst) / 1e6:.1f}MB)")
    except Exception as e:
        print(f"    !! SAVE FAILED for {dst}: {type(e).__name__}: {e}")
        # Keep the /tmp copy if NFS write failed
        if use_tmp and os.path.exists(tmp_path):
            print(f"    !! Checkpoint preserved at {tmp_path} ({tmp_size / 1e6:.1f}MB)")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class CoCClipDataset(Dataset):
    """Lazy-loading dataset over pre-cached Alpamayo .pt clip files."""

    def __init__(self, clips_dir: str, max_clips: int | None = None,
                 offset: int = 0):
        all_files = sorted(glob.glob(os.path.join(clips_dir, "*.pt")))
        all_files = all_files[offset:]
        if max_clips is not None:
            all_files = all_files[:max_clips]
        self.clip_files = all_files

    def __len__(self):
        return len(self.clip_files)

    def __getitem__(self, idx):
        clip = torch.load(self.clip_files[idx], weights_only=False)
        clip["source"] = "coc"
        return clip


class TargetOutputDataset(Dataset):
    """Dataset of pre-generated target model outputs for self-distillation.

    Each .pt file contains:
        - prompt_input_ids: (1, prompt_len) — tokenized prompt with image tokens
        - output_token_ids: (num_generated,) — target's greedy CoC tokens
        - output_logits: (num_generated, vocab_size) — target's logits (fp16)
        - pixel_values: (N, D) — preprocessed image features
        - image_grid_thw: (num_images, 3) — image grid dimensions
        - prompt_len, num_generated, clip_id
    """

    def __init__(self, output_dir: str, max_samples: int | None = None,
                 offset: int = 0,
                 include_uuids: list | None = None,
                 exclude_uuids: list | None = None):
        all_files = sorted(glob.glob(os.path.join(output_dir, "*.pt")))
        if include_uuids is not None:
            incl = set(include_uuids)
            all_files = [p for p in all_files if Path(p).stem in incl]
        if exclude_uuids is not None:
            excl = set(exclude_uuids)
            all_files = [p for p in all_files if Path(p).stem not in excl]
        all_files = all_files[offset:]
        if max_samples is not None:
            all_files = all_files[:max_samples]
        self.files = all_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            d = torch.load(self.files[idx], weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.files[idx]}: {e}") from e
        # Concatenate prompt + output as the full sequence for training
        prompt_ids = d["prompt_input_ids"].squeeze(0)  # (prompt_len,)
        output_ids = d["output_token_ids"]              # (num_generated,)
        full_ids = torch.cat([prompt_ids, output_ids], dim=0).unsqueeze(0)  # (1, total_len)

        result = {
            "input_ids": full_ids,
            "output_logits": d["output_logits"],  # (num_generated, V)
            "prompt_len": d["prompt_len"],
            "num_generated": d["num_generated"],
            "source": "target_output",
        }
        # Visual inputs
        if "pixel_values" in d:
            result["pixel_values"] = d["pixel_values"].to(torch.bfloat16)
        if "image_grid_thw" in d:
            result["image_grid_thw"] = d["image_grid_thw"]
        if "pixel_values_videos" in d:
            result["pixel_values_videos"] = d["pixel_values_videos"].to(torch.bfloat16)
        if "video_grid_thw" in d:
            result["video_grid_thw"] = d["video_grid_thw"]
        return result


class UltraChatDataset(Dataset):
    """Text-only instruction dataset from UltraChat parquet files."""

    def __init__(self, ultrachat_dir: str, split: str = "train_sft", max_samples: int | None = None):
        import pandas as pd
        parquet_files = sorted(glob.glob(os.path.join(ultrachat_dir, f"{split}*.parquet")))
        if not parquet_files:
            parquet_files = sorted(glob.glob(os.path.join(ultrachat_dir, "**", f"{split}*.parquet"), recursive=True))
        dfs = [pd.read_parquet(f) for f in parquet_files]
        self.data = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        if max_samples is not None and len(self.data) > max_samples:
            self.data = self.data.sample(n=max_samples, random_state=42).reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        messages = row["messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)
        return {"messages": messages, "source": "ultrachat"}


# ---------------------------------------------------------------------------
# Block diffusion loss (DFlash paper Section 3.3)
# ---------------------------------------------------------------------------

def block_diffusion_loss(
    draft_logits: torch.Tensor,
    labels: torch.Tensor,
    block_size: int,
    ignore_index: int = -100,
) -> dict[str, torch.Tensor]:
    """Position-weighted CE loss following DFlash paper.

    Within each block of `block_size` predictions, earlier positions get
    exponentially higher weight because errors at early positions invalidate
    all subsequent tokens during speculative verification.

    Weight: w_k = exp(-(k-1) / block_size)  for position k=1..block_size-1

    Args:
        draft_logits:  (B, L, V) — draft predictions for each position
        labels:        (B, L)   — ground truth token IDs
        block_size:    block size used in DFlash
        ignore_index:  token ID to ignore
    """
    B, L, V = draft_logits.shape

    # Build position weights: repeating pattern of [w_1, w_2, ..., w_{bs-1}]
    # across the sequence length
    block_weights = torch.tensor(
        [math.exp(-(k) / block_size) for k in range(block_size)],
        device=draft_logits.device, dtype=draft_logits.dtype,
    )
    # Tile to cover sequence length
    num_repeats = (L + block_size - 1) // block_size
    position_weights = block_weights.repeat(num_repeats)[:L]  # (L,)

    # Compute per-token CE loss
    per_token_ce = F.cross_entropy(
        draft_logits.view(-1, V),
        labels.view(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view(B, L)  # (B, L)

    # Mask ignored positions
    valid_mask = (labels != ignore_index).float()  # (B, L)

    # Apply position weights
    weighted_ce = per_token_ce * position_weights.unsqueeze(0) * valid_mask
    loss = weighted_ce.sum() / valid_mask.sum().clamp(min=1)

    # Also compute unweighted CE for logging
    unweighted_ce = (per_token_ce * valid_mask).sum() / valid_mask.sum().clamp(min=1)

    return {
        "loss": loss,
        "weighted_ce": loss,
        "unweighted_ce": unweighted_ce,
    }


# ---------------------------------------------------------------------------
# Training step (block diffusion)
# ---------------------------------------------------------------------------

def tokenize_messages(messages, tokenizer):
    """Convert messages to input_ids for text-only samples."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    return tokens


def train_step(
    target_vlm: nn.Module,
    draft: nn.Module,
    embed_tokens: nn.Embedding,
    lm_head: nn.Linear,
    batch: dict,
    processor,
    tokenizer,
    draft_target_layer_ids: list[int],
    block_size: int,
    mask_token_id: int,
    device: torch.device,
    overlapping_blocks: bool = False,
    random_mask: bool = False,
) -> dict:
    """One block-diffusion training step.

    1. Run target model → get hidden states (context features)
    2. Build masked blocks: [anchor, MASK, MASK, ..., MASK] for each position
    3. Run draft on masked blocks with target context → predict masked tokens
    4. Position-weighted CE loss on predictions vs ground truth
    """
    source = batch.get("source", "unknown")

    # --- Tokenize / prepare inputs ---
    if source == "target_output":
        # Pre-tokenized target output with saved visual features
        input_ids = batch["input_ids"].to(device)
        inputs = {"input_ids": input_ids}
        for vkey in ["pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"]:
            if vkey in batch:
                inputs[vkey] = batch[vkey].to(device)
    elif source == "coc":
        inputs = processor.apply_chat_template(
            batch["messages"],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    else:
        inputs = tokenize_messages(batch["messages"], tokenizer)
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    input_ids = inputs["input_ids"]        # (1, L)
    seq_len = input_ids.shape[1]
    if seq_len < block_size + 1:
        return None  # sequence too short

    # --- Target forward (frozen) → hidden states ---
    #
    # use_cache=True + fresh DynamicCache. This routes through Qwen3-VL's
    # prefill path which correctly applies M-RoPE (the model's native 3D
    # multimodal position encoding). use_cache=False misroutes for multimodal
    # inputs and produces broken hidden states (verified: model's own argmax
    # becomes <i#> image tokens under use_cache=False + auto position_ids).
    #
    # Verified bit-identical to vlm_spec_generate's prefill hidden states at
    # inference, so train/infer distributions match.
    from transformers.cache_utils import DynamicCache
    with torch.no_grad():
        target_kwargs = dict(
            input_ids=input_ids,
            attention_mask=inputs.get("attention_mask"),
            past_key_values=DynamicCache(),
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        # Pass any visual inputs (images or videos)
        for vkey in ["pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"]:
            if inputs.get(vkey) is not None:
                target_kwargs[vkey] = inputs[vkey]

        # Log visual input status on first call
        if not hasattr(train_step, "_logged_visual"):
            visual_keys = [k for k in target_kwargs if "pixel" in k or "grid" in k]
            if visual_keys:
                shapes = {k: tuple(target_kwargs[k].shape) for k in visual_keys}
                print(f"  [VISUAL] Target forward WITH images: {shapes}")
            else:
                print(f"  [VISUAL] WARNING: Target forward WITHOUT images!")
            train_step._logged_visual = True

        target_out = target_vlm(**target_kwargs)

    target_hidden = extract_context_feature(
        target_out.hidden_states, draft_target_layer_ids
    )  # (1, L, H*num_target_layers)

    # --- Build masked blocks and run draft ---
    # Only train on blocks in the output region (after prompt_len).
    prompt_len = batch.get("prompt_len", 0)
    if isinstance(prompt_len, torch.Tensor):
        prompt_len = prompt_len.item()

    # Output region: start from <|cot_start|> as anchor (not \n before it).
    # <|cot_start|> is always the first output token and deterministic,
    # so use it as anchor rather than wasting a prediction slot on it.
    output_start = prompt_len

    # Build list of block starting positions
    if overlapping_blocks:
        # Stride-1 overlapping: every position in output region is a block start
        all_starts = list(range(output_start, seq_len - block_size))
        if not all_starts:
            return None
        # Subsample to limit compute (max ~16 blocks per sample)
        max_blocks = 16
        if len(all_starts) > max_blocks:
            import random as _rng
            all_starts = sorted(_rng.sample(all_starts, max_blocks))
    else:
        # Non-overlapping blocks (original behavior)
        first_block_start = prompt_len
        num_blocks = (seq_len - first_block_start - 1) // block_size
        if num_blocks < 1:
            return None
        all_starts = []
        for b_idx in range(num_blocks):
            start = first_block_start + b_idx * block_size
            if start + block_size >= seq_len:
                break
            all_starts.append(start)
        if not all_starts:
            return None

    all_draft_logits = []
    all_labels = []
    all_block_starts = []

    for start in all_starts:
        end = start + block_size

        block_ids = input_ids[:, start:end].clone()

        if random_mask:
            # Random mask schedule: mask a random number of positions 1..block_size-1.
            # Anchor (position 0) is always unmasked.
            # num_masked sampled uniformly from [1, block_size-1].
            num_maskable = block_size - 1
            num_masked = torch.randint(1, num_maskable + 1, (1,)).item()
            mask_positions = torch.randperm(num_maskable)[:num_masked] + 1  # +1 to skip anchor
            block_ids[:, mask_positions] = mask_token_id
        else:
            # Full mask (original behavior): mask all positions except anchor
            mask_positions = torch.arange(1, block_size)
            block_ids[:, 1:] = mask_token_id

        noise_embedding = embed_tokens(block_ids)  # (1, block_size, H)

        # Context: target hidden states of the committed prefix ONLY (positions
        # 0..start-1). Slicing :start instead of :end prevents leakage of GT
        # block tokens into the draft's cross-attention context, matching what
        # `vlm_spec_generate` provides at inference time.
        ctx_hidden = target_hidden[:, :start, :]
        ctx_len = ctx_hidden.shape[1]
        pos_ids = torch.arange(ctx_len + block_size, device=device).unsqueeze(0)

        # Draft forward
        draft_hidden = draft(
            target_hidden=ctx_hidden,
            noise_embedding=noise_embedding,
            position_ids=pos_ids,
        )
        # Draft predicts positions after the anchor (last block_size-1 positions)
        block_logits = lm_head(draft_hidden[:, -(block_size - 1):, :])  # (1, block_size-1, V)

        # Ground truth: tokens at positions start+1 to start+block_size-1
        block_labels = input_ids[:, start + 1:end].clone()  # (1, block_size-1)

        if random_mask:
            # Only compute loss on masked positions
            mask_set = set(mask_positions.tolist())
            for k in range(block_size - 1):
                if (k + 1) not in mask_set:
                    block_labels[:, k] = -100

        all_draft_logits.append(block_logits)
        all_labels.append(block_labels)
        all_block_starts.append(start)

    if not all_draft_logits:
        return None

    # Concatenate all blocks
    draft_logits_cat = torch.cat(all_draft_logits, dim=1)
    labels_cat = torch.cat(all_labels, dim=1)

    losses = block_diffusion_loss(
        draft_logits=draft_logits_cat,
        labels=labels_cat,
        block_size=block_size,
    )

    # KL divergence loss against target logits (for self-distillation)
    kl_weight = batch.get("_kl_weight", 0.0)
    target_logits = batch.get("output_logits")
    if kl_weight > 0 and target_logits is not None and prompt_len > 0:
        target_logits = target_logits.to(device).float()
        kl_loss = torch.tensor(0.0, device=device)
        kl_count = 0
        for bi, start in enumerate(all_block_starts):
            for k in range(block_size - 1):
                global_pos = start + k + 1
                tgt_idx = global_pos - prompt_len
                if 0 <= tgt_idx < target_logits.shape[0]:
                    draft_lp = F.log_softmax(all_draft_logits[bi][0, k, :].float(), dim=-1)
                    target_p = F.softmax(target_logits[tgt_idx, :], dim=-1)
                    kl_loss += F.kl_div(draft_lp, target_p, reduction='sum')
                    kl_count += 1
        if kl_count > 0:
            losses["kl"] = kl_loss / kl_count

    return losses


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def is_main_process(rank):
    return rank == 0


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_validation(draft, draft_module, vlm, embed_tokens, lm_head,
                    val_loader, args, device, epoch, global_step,
                    rank, use_wandb, max_batches=None,
                    processor=None, tokenizer=None):
    """Run validation loop using the SAME loss-computation path as training
    (via `train_step`), under torch.no_grad(). This guarantees val shares the
    non-leaky ctx_hidden fix and the same block layout / mask schedule as train.

    If max_batches is set, only run that many batches.
    """
    draft.eval()
    val_wce, val_uce, val_count = 0.0, 0.0, 0
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            if max_batches and i >= max_batches:
                break
            try:
                losses = train_step(
                    target_vlm=vlm,
                    draft=draft_module,
                    embed_tokens=embed_tokens,
                    lm_head=lm_head,
                    batch=sample,
                    processor=processor,
                    tokenizer=tokenizer,
                    draft_target_layer_ids=draft_module.target_layer_ids,
                    block_size=args.block_size,
                    mask_token_id=args.mask_token_id,
                    device=device,
                    overlapping_blocks=args.overlapping_blocks,
                    random_mask=args.random_mask,
                )
                if losses is None:
                    continue
                val_wce += float(losses["weighted_ce"].item())
                val_uce += float(losses["unweighted_ce"].item())
                val_count += 1
            except Exception as e:
                continue
    draft.train()
    if is_main_process(rank) and val_count > 0:
        v_wce = val_wce / val_count
        v_uce = val_uce / val_count
        label = "Val" if max_batches is None else f"Val({val_count})"
        print(f"  {label} @step {global_step} epoch {epoch+1}: wce={v_wce:.4f} ce={v_uce:.4f}")
        if use_wandb:
            import wandb
            prefix = "val" if max_batches is None else "val_quick"
            wandb.log({f"{prefix}/weighted_ce": v_wce, f"{prefix}/ce": v_uce}, step=global_step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_path", type=str, required=True)
    parser.add_argument("--clips_dir", type=str, default=None)
    parser.add_argument("--target_outputs_dir", type=str, default=None,
                        help="Dir of pre-generated target outputs (.pt) for self-distillation")
    parser.add_argument("--ultrachat_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_clips", type=int, default=9000)
    parser.add_argument("--val_clips", type=int, default=1000)
    parser.add_argument("--max_ultrachat", type=int, default=50000)
    parser.add_argument("--log_interval", type=int, default=50,
                        help="Log train loss to stdout/wandb every N steps.")
    parser.add_argument("--val_interval", type=int, default=50,
                        help="Run quick validation every N steps (uses --val_batches clips).")
    parser.add_argument("--val_uuids_file", type=str, default=None,
                        help="JSON list of validation clip UUIDs. When set, training excludes "
                             "these and validation uses exactly these (ignores --val_clips / offset).")
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--num_draft_layers", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=4)
    parser.add_argument("--mask_token_id", type=int, default=151662,
                        help="Qwen3-VL token id used as mask. Default 151662 = <|fim_pad|>, a "
                             "semantically inert placeholder (Alpamayo never generates FIM tokens). "
                             "Earlier runs used 151643 = <|endoftext|>; pass explicitly to reproduce.")
    parser.add_argument("--overlapping_blocks", action="store_true",
                        help="Use stride-1 overlapping blocks instead of non-overlapping")
    parser.add_argument("--random_mask", action="store_true",
                        help="Random mask schedule (vary num masked positions) instead of full mask")
    parser.add_argument("--pretrained_draft", type=str, default=None,
                        help="Path to pretrained DFlash draft weights (safetensors or .pt) to initialize from")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint dir to resume from (loads draft weights, optimizer, scheduler, step)")
    parser.add_argument("--kl_weight", type=float, default=0.0,
                        help="Weight for KL divergence loss against target logits (0.0 = disabled)")

    parser.add_argument("--seed", type=int, default=42)

    # Wandb
    parser.add_argument("--wandb_project", type=str, default="dflash-distillation")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--save_via_tmp", action="store_true",
                        help="Save checkpoints via /tmp to work around virtiofs/NFS issues")
    parser.add_argument("--val_batches", type=int, default=50,
                        help="Number of val batches per epoch (0=skip mid-training val, full val runs at end)")

    args = parser.parse_args()

    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if is_main_process(rank):
        os.makedirs(args.output_dir, exist_ok=True)
        config_path = os.path.join(args.output_dir, "config.json")
        if not os.path.exists(config_path):
            config_str = json.dumps(vars(args), indent=2)
            if args.save_via_tmp:
                tmp_cfg = "/tmp/_dflash_config.json"
                with open(tmp_cfg, "w") as f:
                    f.write(config_str)
                shutil.copyfile(tmp_cfg, config_path)
                os.unlink(tmp_cfg)
            else:
                with open(config_path, "w") as f:
                    f.write(config_str)

    # --- Wandb init (main process only) ---
    use_wandb = not args.no_wandb and is_main_process(rank)
    if use_wandb:
        try:
            import wandb
            run_name = args.wandb_run_name or f"L{args.num_draft_layers}_bs{args.block_size}_lr{args.lr}"
            if args.save_via_tmp:
                wandb_dir = "/tmp/wandb_runs"
                os.makedirs(wandb_dir, exist_ok=True)
            else:
                wandb_dir = args.output_dir
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=vars(args),
                dir=wandb_dir,
            )
            print(f"Wandb initialized: {wandb.run.url}")
        except Exception as e:
            print(f"Wandb init failed: {e}, continuing without wandb")
            use_wandb = False

    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    # --- Load target model (frozen) ---
    if is_main_process(rank):
        print(f"Loading target model from {args.target_path}...")
    target = AlpamayoR1.from_pretrained(
        args.target_path, dtype=torch.bfloat16,
    ).to(device).eval()
    for p in target.parameters():
        p.requires_grad = False

    vlm = target.vlm
    embed_tokens, lm_head = get_qwen3vl_embed_and_head(vlm)
    tokenizer = target.tokenizer
    processor = helper.get_processor(tokenizer)

    # --- Build draft model (trainable) ---
    if is_main_process(rank):
        print("Building DFlash draft model...")
    draft = build_dflash_draft_for_qwen3vl(
        vlm,
        num_draft_layers=args.num_draft_layers,
        block_size=args.block_size,
        mask_token_id=args.mask_token_id,
    ).to(device).train()

    # Load pretrained DFlash draft weights if provided
    if args.pretrained_draft:
        if is_main_process(rank):
            print(f"Loading pretrained draft from {args.pretrained_draft}")
        from alpamayo_r1.models.dflash_draft import load_draft_checkpoint
        ckpt = load_draft_checkpoint(args.pretrained_draft, map_location=device)
        if is_main_process(rank) and ckpt["mask_token_id"] is not None and ckpt["mask_token_id"] != args.mask_token_id:
            print(f"  WARNING: pretrained draft was trained with mask_token_id={ckpt['mask_token_id']} "
                  f"but current training uses {args.mask_token_id}. "
                  f"The draft's learned mapping from mask-embedding to predictions will be invalidated.")
        state_dict = ckpt["state_dict"]
        msg = draft.load_state_dict(state_dict, strict=False)
        if is_main_process(rank):
            print(f"  Loaded: {len(state_dict)} keys, missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")

    if world_size > 1:
        draft = DDP(draft, device_ids=[local_rank])
    draft_module = draft.module if isinstance(draft, DDP) else draft

    if is_main_process(rank):
        num_params = sum(p.numel() for p in draft_module.parameters() if p.requires_grad)
        print(f"Draft model: {num_params / 1e6:.1f}M trainable parameters")
        print(f"Block size: {args.block_size}")
        print(f"Target layers used: {draft_module.target_layer_ids}")

    # --- Datasets ---
    train_datasets, val_datasets = [], []

    # Self-distillation: pre-generated target outputs (preferred)
    if args.target_outputs_dir and os.path.isdir(args.target_outputs_dir):
        val_uuids = None
        if args.val_uuids_file:
            import json as _json
            with open(args.val_uuids_file) as _f:
                val_uuids = _json.load(_f)
            if is_main_process(rank):
                print(f"Loaded {len(val_uuids)} validation UUIDs from {args.val_uuids_file}")
            # Train: everything except val UUIDs (respecting max_clips if set)
            train_tgt = TargetOutputDataset(args.target_outputs_dir,
                                            max_samples=args.max_clips,
                                            exclude_uuids=val_uuids)
            train_datasets.append(train_tgt)
            val_tgt = TargetOutputDataset(args.target_outputs_dir,
                                          include_uuids=val_uuids)
            val_datasets.append(val_tgt)
        else:
            train_tgt = TargetOutputDataset(args.target_outputs_dir, max_samples=args.max_clips)
            train_datasets.append(train_tgt)
            if args.val_clips > 0:
                val_tgt = TargetOutputDataset(args.target_outputs_dir, max_samples=args.val_clips,
                                              offset=args.max_clips)
                val_datasets.append(val_tgt)
        if is_main_process(rank):
            print(f"Target outputs: {len(train_tgt)} train, {len(val_tgt) if val_datasets else 0} val")

    # Raw CoC clips (legacy — images may not pass through correctly)
    elif args.clips_dir and os.path.isdir(args.clips_dir):
        train_coc = CoCClipDataset(args.clips_dir, max_clips=args.max_clips)
        train_datasets.append(train_coc)
        if args.val_clips > 0:
            val_coc = CoCClipDataset(args.clips_dir, max_clips=args.val_clips,
                                     offset=args.max_clips)
            val_datasets.append(val_coc)
        if is_main_process(rank):
            print(f"CoC clips: {len(train_coc)} train, {len(val_coc) if val_datasets else 0} val")

    if args.ultrachat_dir and os.path.isdir(args.ultrachat_dir):
        uc_ds = UltraChatDataset(args.ultrachat_dir, max_samples=args.max_ultrachat)
        train_datasets.append(uc_ds)
        if is_main_process(rank):
            print(f"UltraChat: {len(uc_ds)} samples")

    if not train_datasets:
        raise ValueError("Provide at least one of --clips_dir or --ultrachat_dir")

    dataset = ConcatDataset(train_datasets)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        dataset, batch_size=1, sampler=sampler,
        shuffle=(sampler is None), num_workers=0,
        collate_fn=lambda x: x[0],
    )

    val_loader = None
    if val_datasets:
        val_dataset = ConcatDataset(val_datasets)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
        val_loader = DataLoader(
            val_dataset, batch_size=1, sampler=val_sampler,
            shuffle=False, num_workers=0,
            collate_fn=lambda x: x[0],
        )

    if is_main_process(rank):
        print(f"Total training samples: {len(dataset)} (across {world_size} GPUs)")
        if val_loader:
            print(f"Validation samples: {len(val_dataset)}")

    # --- Optimizer ---
    optimizer = AdamW(draft.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.num_epochs * len(loader) // args.grad_accum_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

    # --- Resume from checkpoint ---
    start_epoch = 0
    start_step = 0
    global_step = 0
    last_val_step = -1  # prevents re-running val 4x while global_step sticks (grad_accum)
    if args.resume:
        resume_path = os.path.join(args.resume, "resume.pt") if os.path.isdir(args.resume) else args.resume
        if os.path.exists(resume_path):
            if is_main_process(rank):
                print(f"Resuming from {resume_path}")
            ckpt = torch.load(resume_path, map_location=device)
            draft_module.load_state_dict(ckpt["draft"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            global_step = ckpt["global_step"]
            start_epoch = ckpt["epoch"]
            start_step = ckpt.get("step_in_epoch", 0) + 1
            if is_main_process(rank):
                print(f"  Resumed: epoch={start_epoch+1}, step={start_step}, global_step={global_step}")
        else:
            if is_main_process(rank):
                print(f"  WARNING: resume path {resume_path} not found, starting fresh")

    # --- Training loop ---
    for epoch in range(start_epoch, args.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_wce = 0.0
        epoch_uce = 0.0
        epoch_count = 0
        t0 = time.time()

        optimizer.zero_grad()
        skip_to = start_step if epoch == start_epoch else 0
        for i, batch in enumerate(loader):
            if i < skip_to:
                continue
            try:
                batch["_kl_weight"] = args.kl_weight
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    losses = train_step(
                        target_vlm=vlm,
                        draft=draft_module,
                        embed_tokens=embed_tokens,
                        lm_head=lm_head,
                        batch=batch,
                        processor=processor,
                        tokenizer=tokenizer,
                        draft_target_layer_ids=draft_module.target_layer_ids,
                        block_size=args.block_size,
                        mask_token_id=args.mask_token_id,
                        device=device,
                        overlapping_blocks=args.overlapping_blocks,
                        random_mask=args.random_mask,
                    )

                if losses is None:
                    continue

                loss = losses["loss"] / args.grad_accum_steps
                if "kl" in losses and args.kl_weight > 0:
                    loss = loss + args.kl_weight * losses["kl"] / args.grad_accum_steps
                loss.backward()

                step_wce = losses["weighted_ce"].item()
                step_uce = losses["unweighted_ce"].item()
                step_kl = losses["kl"].item() if "kl" in losses else 0.0
                epoch_wce += step_wce
                epoch_uce += step_uce
                epoch_count += 1

                if use_wandb and (i + 1) % args.grad_accum_steps == 0:
                    log_dict = {
                        "step/weighted_ce": step_wce,
                        "step/ce": step_uce,
                    }
                    if step_kl > 0:
                        log_dict["step/kl"] = step_kl
                    wandb.log(log_dict, step=global_step)

            except Exception as e:
                if is_main_process(rank):
                    print(f"  skip sample {i}: {type(e).__name__}: {e}")
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                continue

            if (i + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(draft.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if is_main_process(rank) and (i + 1) % args.log_interval == 0 and epoch_count > 0:
                avg_wce = epoch_wce / epoch_count
                avg_uce = epoch_uce / epoch_count
                elapsed = time.time() - t0
                samples_per_sec = epoch_count / elapsed
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"  epoch {epoch+1} step {i+1}/{len(loader)} | "
                    f"wce={avg_wce:.4f} ce={avg_uce:.4f} | "
                    f"lr={lr_now:.2e} | {samples_per_sec:.1f} samples/s"
                )
                if use_wandb:
                    wandb.log({
                        "train/weighted_ce": avg_wce,
                        "train/ce": avg_uce,
                        "train/loss": avg_wce,
                        "train/lr": lr_now,
                        "train/samples_per_sec": samples_per_sec,
                        "train/epoch": epoch + 1,
                        "train/global_step": global_step,
                    }, step=global_step)

            # --- Mid-epoch quick validation every args.val_interval steps ---
            if (val_loader is not None and args.val_interval > 0
                    and global_step > 0 and global_step % args.val_interval == 0):
                _run_validation(draft, draft_module, vlm, embed_tokens, lm_head,
                                val_loader, args, device, epoch, global_step,
                                rank, use_wandb, max_batches=args.val_batches,
                                processor=processor, tokenizer=tokenizer)

            if is_main_process(rank) and global_step > 0 and global_step % args.save_interval == 0:
                ckpt_path = os.path.join(args.output_dir, f"draft_step_{global_step}.pt")
                _safe_save(_wrap_draft_ckpt(draft_module.state_dict(), args), ckpt_path, use_tmp=args.save_via_tmp)
                resume_ckpt = {
                    "draft": draft_module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                    "step_in_epoch": i,
                }
                _safe_save(resume_ckpt, os.path.join(args.output_dir, "resume.pt"), use_tmp=args.save_via_tmp)
                print(f"  saved: {ckpt_path} + resume.pt")

        if is_main_process(rank):
            avg_wce = epoch_wce / max(epoch_count, 1)
            avg_uce = epoch_uce / max(epoch_count, 1)
            print(f"Epoch {epoch+1}/{args.num_epochs} | wce={avg_wce:.4f} ce={avg_uce:.4f} | samples={epoch_count}")
            ckpt_path = os.path.join(args.output_dir, f"draft_epoch_{epoch+1}.pt")
            _safe_save(_wrap_draft_ckpt(draft_module.state_dict(), args), ckpt_path)
            print(f"  saved: {ckpt_path}")
            if use_wandb:
                wandb.log({
                    "epoch/weighted_ce": avg_wce,
                    "epoch/ce": avg_uce,
                    "epoch/num": epoch + 1,
                    "epoch/samples": epoch_count,
                }, step=global_step)

        # --- Validation (limited batches at end of epoch) ---
        if val_loader is not None and args.val_batches > 0:
            _run_validation(draft, draft_module, vlm, embed_tokens, lm_head,
                            val_loader, args, device, epoch, global_step,
                            rank, use_wandb, max_batches=args.val_batches,
                            processor=processor, tokenizer=tokenizer)

    # --- Full validation at end of training ---
    if val_loader is not None:
        if is_main_process(rank):
            print("Running full validation...")
        _run_validation(draft, draft_module, vlm, embed_tokens, lm_head,
                        val_loader, args, device, args.num_epochs - 1, global_step,
                        rank, use_wandb, max_batches=None,
                        processor=processor, tokenizer=tokenizer)

    if is_main_process(rank):
        final_path = os.path.join(args.output_dir, "draft_final.pt")
        _safe_save(_wrap_draft_ckpt(draft_module.state_dict(), args), final_path, use_tmp=args.save_via_tmp)
        print(f"Training complete. Final model: {final_path}")
        if use_wandb:
            wandb.save(final_path)
            wandb.finish()
            # Copy wandb logs from /tmp to output_dir when using save_via_tmp
            if args.save_via_tmp and os.path.isdir("/tmp/wandb_runs/wandb"):
                dst = os.path.join(args.output_dir, "wandb")
                try:
                    shutil.copytree("/tmp/wandb_runs/wandb", dst, copy_function=shutil.copyfile)
                    print(f"  Copied wandb logs to {dst}")
                except Exception as e:
                    print(f"  Warning: could not copy wandb logs: {e}")

    cleanup()


if __name__ == "__main__":
    main()
