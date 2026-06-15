"""Train the EAGLE-3-style AR draft for Qwen3-VL.

Per-sample step:
  1. Target VLM forward once with the visual inputs → full hidden states.
  2. Pick a single target layer's hidden (= `context_hidden`, B×T×H).
  3. Draft forward(embed_tokens(input_ids), context_hidden) → draft hidden.
  4. Logits = lm_head(draft_hidden). CE loss at OUTPUT positions only, shifted
     by 1 for next-token prediction.

Inference (outside this script): draft chains its own output hidden for new
proposed positions; target's hidden is used for committed positions.
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
from alpamayo_r1.models.dflash_draft import get_qwen3vl_embed_and_head
from alpamayo_r1.models.autoregressive_draft import (
    build_ar_draft_for_qwen3vl,
    warm_start_ar_draft_from_target,
    _wrap_ar_ckpt,
)

PAD_ID = 151643


def is_main(rank): return rank == 0


class TargetOutputDataset(Dataset):
    def __init__(self, output_dir, include_uuids=None, exclude_uuids=None,
                 max_samples=None, offset=0):
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

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        d = torch.load(self.files[idx], weights_only=False)
        prompt_ids = d["prompt_input_ids"].squeeze(0)
        output_ids = d["output_token_ids"]
        full_ids = torch.cat([prompt_ids, output_ids], dim=0).unsqueeze(0)
        r = {
            "input_ids": full_ids,
            "prompt_len": int(d["prompt_len"]),
            "num_generated": int(d["num_generated"]),
        }
        if "pixel_values" in d:
            r["pixel_values"] = d["pixel_values"].to(torch.bfloat16)
        if "image_grid_thw" in d:
            r["image_grid_thw"] = d["image_grid_thw"]
        return r


def collate(features):
    max_len = max(f["input_ids"].shape[1] for f in features)
    B = len(features)
    input_ids = torch.full((B, max_len), PAD_ID, dtype=torch.long)
    attn = torch.zeros((B, max_len), dtype=torch.long)
    prompt_lens = torch.zeros((B,), dtype=torch.long)
    num_gen = torch.zeros((B,), dtype=torch.long)
    pix_list, grid_list = [], []
    for i, f in enumerate(features):
        L = f["input_ids"].shape[1]
        input_ids[i, :L] = f["input_ids"][0]
        attn[i, :L] = 1
        prompt_lens[i] = f["prompt_len"]
        num_gen[i] = f["num_generated"]
        if "pixel_values" in f: pix_list.append(f["pixel_values"])
        if "image_grid_thw" in f: grid_list.append(f["image_grid_thw"])
    out = {"input_ids": input_ids, "attention_mask": attn,
           "prompt_len": prompt_lens, "num_generated": num_gen}
    if pix_list: out["pixel_values"] = torch.cat(pix_list, dim=0)
    if grid_list: out["image_grid_thw"] = torch.cat(grid_list, dim=0)
    return out


def mrope_pos_ids(T, B, device):
    p = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    return p.unsqueeze(0).expand(3, -1, -1).contiguous()


def train_step(draft_module, vlm, embed_tokens, lm_head, target_layer_ids,
               batch, device):
    input_ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    prompt_lens = batch["prompt_len"].to(device)
    num_gen = batch["num_generated"].to(device)

    B, T = input_ids.shape
    if T < 2:
        return None

    # --- Target forward (frozen) → hidden states ---
    target_kwargs = dict(input_ids=input_ids, attention_mask=attn,
                         use_cache=True, output_hidden_states=True,
                         past_key_values=DynamicCache(), return_dict=True)
    if "pixel_values" in batch:
        target_kwargs["pixel_values"] = batch["pixel_values"].to(device).to(torch.bfloat16)
    if "image_grid_thw" in batch:
        target_kwargs["image_grid_thw"] = batch["image_grid_thw"].to(device)

    with torch.no_grad():
        tout = vlm(**target_kwargs)
    # hidden_states is tuple length (num_layers + 1): [embedding, layer_0, layer_1, ...]
    # EAGLE-3: pick multiple layers' outputs = hidden_states[idx + 1] for each idx.
    context_hiddens = [tout.hidden_states[idx + 1] for idx in target_layer_ids]

    # --- Draft forward ---
    input_embeds = embed_tokens(input_ids)  # (B, T, H)
    pos_ids = mrope_pos_ids(T, B, device)
    draft_hidden = draft_module(
        input_embeds=input_embeds,
        context_hidden=context_hiddens,
        position_ids=pos_ids,
        use_cache=False,
    )
    logits = lm_head(draft_hidden)  # (B, T, V)

    # Next-token CE loss on output positions
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    pos = torch.arange(T - 1, device=device).unsqueeze(0).expand(B, -1)
    valid = ((pos + 1) >= prompt_lens.unsqueeze(1)) & \
            ((pos + 1) < (prompt_lens + num_gen).unsqueeze(1)) & \
            (attn[:, 1:] > 0)
    labels = shift_labels.clone()
    labels[~valid] = -100
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                           labels.view(-1),
                           ignore_index=-100,
                           reduction="mean")
    if torch.isnan(loss) or torch.isinf(loss):
        return None
    return {"loss": loss, "ce": loss.detach()}


@torch.inference_mode()
def run_val(draft_module, vlm, embed_tokens, lm_head, target_layer_ids,
            val_loader, device, max_batches=None):
    draft_module.eval()
    total, count = 0.0, 0
    for i, batch in enumerate(val_loader):
        if max_batches and i >= max_batches: break
        try:
            out = train_step(draft_module, vlm, embed_tokens, lm_head,
                              target_layer_ids, batch, device)
            if out is None: continue
            total += float(out["ce"].detach().cpu().item())
            count += 1
        except Exception as e:
            continue
        finally:
            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()
    draft_module.train()
    return total / max(count, 1), count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_path", required=True)
    ap.add_argument("--target_outputs_dir", required=True)
    ap.add_argument("--val_uuids_file", required=True)
    ap.add_argument("--test_uuids_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_draft_layers", type=int, default=1,
                    help="Paper-faithful EAGLE-3 uses 1 transformer block.")
    ap.add_argument("--target_layer_ids", type=str, default=None,
                    help="Comma-sep list of target layer indices to fuse, e.g. '12,24,35'. "
                         "Default = single last layer (legacy 1-feature variant).")
    ap.add_argument("--num_epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--val_interval", type=int, default=200)
    ap.add_argument("--save_interval", type=int, default=500)
    ap.add_argument("--val_batches", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warm_start", action="store_true")
    ap.add_argument("--wandb_project", default="dflash-distillation")
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
    random.seed(args.seed + rank)

    os.makedirs(args.output_dir, exist_ok=True)
    use_wandb = (not args.no_wandb) and is_main(rank)

    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                   config=vars(args), dir=args.output_dir)

    if is_main(rank):
        print(f"loading target from {args.target_path}")
    target = AlpamayoR1.from_pretrained(args.target_path, dtype=torch.bfloat16
                                         ).to(device).eval()
    for p in target.parameters():
        p.requires_grad = False
    vlm = target.vlm
    embed_tokens, lm_head = get_qwen3vl_embed_and_head(vlm)

    n_tgt = vlm.config.get_text_config().num_hidden_layers
    if args.target_layer_ids:
        target_layer_ids = [int(x) for x in args.target_layer_ids.split(",")]
        target_layer_ids = [i if i >= 0 else n_tgt + i for i in target_layer_ids]
    else:
        target_layer_ids = [n_tgt - 1]

    if is_main(rank):
        print(f"building AR draft EAGLE-3 (L={args.num_draft_layers}, target_layer_ids={target_layer_ids})")
    draft = build_ar_draft_for_qwen3vl(
        target, num_draft_layers=args.num_draft_layers,
        target_layer_ids=target_layer_ids,
    ).to(device).train()
    if args.warm_start:
        tgt_layers = vlm.language_model.layers
        layer_ids = [int((i + 1) * n_tgt / (args.num_draft_layers + 1))
                     for i in range(args.num_draft_layers)]
        if is_main(rank):
            print(f"[warm_start] AR draft layers <- target layers {layer_ids}")
        warm_start_ar_draft_from_target(draft, tgt_layers, layer_ids,
                                         verbose=is_main(rank))

    n_params = sum(p.numel() for p in draft.parameters() if p.requires_grad)
    if is_main(rank):
        print(f"Draft params (trainable): {n_params/1e6:.1f}M")

    if world > 1:
        draft = DDP(draft, device_ids=[local_rank], find_unused_parameters=False)
    draft_module = draft.module if hasattr(draft, "module") else draft

    test_ids = json.load(open(args.test_uuids_file))
    val_ids = json.load(open(args.val_uuids_file))
    train_ds = TargetOutputDataset(args.target_outputs_dir,
                                    exclude_uuids=list(set(test_ids) | set(val_ids)))
    val_ds = TargetOutputDataset(args.target_outputs_dir, include_uuids=val_ids)
    if is_main(rank):
        print(f"train={len(train_ds)} val={len(val_ds)}")

    train_sampler = DistributedSampler(train_ds, rank=rank, num_replicas=world,
                                        shuffle=True, seed=args.seed) if world > 1 else None
    val_sampler = DistributedSampler(val_ds, rank=rank, num_replicas=world,
                                      shuffle=False) if world > 1 else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=args.num_workers,
                              collate_fn=collate, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            sampler=val_sampler, num_workers=2,
                            collate_fn=collate, pin_memory=True)

    optim = AdamW([p for p in draft.parameters() if p.requires_grad],
                  lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    total_steps = max(1, args.num_epochs * len(train_loader) // args.grad_accum_steps)
    sched = CosineAnnealingLR(optim, T_max=total_steps)

    global_step = 0
    last_save = last_val = -1
    t0 = time.time()
    for epoch in range(args.num_epochs):
        if train_sampler is not None: train_sampler.set_epoch(epoch)
        draft.train()
        ep_ce, ep_count = 0.0, 0
        for i, batch in enumerate(train_loader):
            out = train_step(draft_module, vlm, embed_tokens, lm_head,
                              target_layer_ids, batch, device)
            if out is None: continue
            loss = out["loss"] / args.grad_accum_steps
            loss.backward()
            if (i + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in draft.parameters() if p.requires_grad],
                    args.max_grad_norm)
                optim.step(); sched.step(); optim.zero_grad()
                global_step += 1
                if is_main(rank) and global_step % args.log_interval == 0:
                    rate = global_step * args.batch_size * world / max(time.time() - t0, 1)
                    print(f"  epoch {epoch+1} step {global_step} | ce={float(out['ce']):.4f} | "
                          f"lr={sched.get_last_lr()[0]:.2e} | {rate:.1f} samples/s", flush=True)
                    if use_wandb:
                        import wandb
                        wandb.log({"train/ce": float(out['ce']),
                                   "lr": sched.get_last_lr()[0]}, step=global_step)
                if (global_step % args.val_interval == 0) and (global_step != last_val):
                    last_val = global_step
                    v_ce, v_count = run_val(draft_module, vlm, embed_tokens, lm_head,
                                              target_layer_ids, val_loader, device,
                                              max_batches=args.val_batches)
                    if is_main(rank):
                        print(f"  Val({v_count}) @step {global_step} epoch {epoch+1}: ce={v_ce:.4f}")
                        if use_wandb:
                            import wandb
                            wandb.log({"val_quick/ce": v_ce}, step=global_step)
                if (global_step % args.save_interval == 0) and (global_step != last_save):
                    last_save = global_step
                    if is_main(rank):
                        ckpt = os.path.join(args.output_dir, f"draft_step_{global_step}.pt")
                        torch.save(_wrap_ar_ckpt(draft_module.state_dict(),
                                                  args.num_draft_layers,
                                                  target_layer_ids), ckpt)
                        print(f"    -> saved {ckpt}")
            ep_ce += float(out['ce']); ep_count += 1
        if is_main(rank):
            avg = ep_ce / max(ep_count, 1)
            print(f"Epoch {epoch+1}/{args.num_epochs} | ce={avg:.4f} | samples={ep_count}")
            ckpt = os.path.join(args.output_dir, f"draft_epoch_{epoch+1}.pt")
            torch.save(_wrap_ar_ckpt(draft_module.state_dict(),
                                       args.num_draft_layers,
                                       target_layer_ids), ckpt)
            print(f"    -> saved {ckpt}")

    if is_main(rank):
        print("Running full validation...")
    v_ce, v_count = run_val(draft_module, vlm, embed_tokens, lm_head,
                             target_layer_ids, val_loader, device, max_batches=None)
    if is_main(rank):
        print(f"  Val @step {global_step} epoch {args.num_epochs}: ce={v_ce:.4f}")
        final = os.path.join(args.output_dir, "draft_final.pt")
        torch.save(_wrap_ar_ckpt(draft_module.state_dict(),
                                   args.num_draft_layers,
                                   target_layer_ids), final)
        print(f"Training complete. Final: {final}")
        if use_wandb:
            import wandb
            wandb.log({"val/ce": v_ce}, step=global_step)
            wandb.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
