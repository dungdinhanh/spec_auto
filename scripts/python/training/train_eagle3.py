"""EAGLE-3 paper-faithful trainer for Qwen3-VL targets.

Mirrors `EAGLE/eagle/traineagle3/main.py`:
- Multi-step rollout (length=7).
- Soft-KL distillation against target's softmax distribution at each step.
- Loss aggregation: `sum_k 0.8^k * ploss[k]`.
- Per-step accuracy logging (argmax match rate).
- Full target vocab (no t2d/d2t subset).
- Default target_layer_ids = [1, 17, 32] (EAGLE-3's idx-2/idx-half/idx-(len-3)
  rule for a 36-layer Qwen3-VL backbone).

Differences from EAGLE-3 official:
- We use torchrun + AdamW, not DeepSpeed (matches the rest of our pipeline).
- Target is AlpamayoR1 (Qwen3-VL-8B + flow-matching action head); only its
  VLM is run during training.
- We use cached `target_coc_outputs/*.pt` (prompt + output + pixel_values).
- 1D RoPE in the draft (matches EAGLE's Llama draft); target's M-RoPE 3D
  is internal and we just consume `output_hidden_states`.
"""
from __future__ import annotations

import argparse
import glob
import json
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
from alpamayo_r1.models.dflash_draft import (
    get_qwen3vl_embed_and_head, get_target_3d_position_ids,
)
from alpamayo_r1.models.eagle3_draft import (
    build_eagle3_draft_for_qwen3vl,
    save_eagle3_ckpt,
)

PAD_ID = 151643


def is_main(rank): return rank == 0


class TargetOutputDataset(Dataset):
    """Reuses our cached `target_coc_outputs/*.pt`. Returns prompt + output
    concatenated, plus pixel_values / image_grid_thw for target's VLM forward."""

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


def build_loss_mask(prompt_lens, num_gen, attn_mask):
    """1 at positions p where (p+1) is in the output range AND attention is
    valid. This is the EAGLE-3-compatible "step-0" loss mask: at step 0 we
    train on positions whose `next-token target` is an output token."""
    B, T = attn_mask.shape
    device = attn_mask.device
    pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    valid = ((pos + 1) >= prompt_lens.unsqueeze(1).to(device)) & \
            ((pos + 1) < (prompt_lens + num_gen).unsqueeze(1).to(device)) & \
            (attn_mask > 0)
    return valid.to(torch.long)


def train_step(draft_module, vlm, embed_tokens, lm_head, target_layer_ids,
               batch, device, use_mrope_3d: bool = False):
    input_ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    prompt_lens = batch["prompt_len"].to(device)
    num_gen = batch["num_generated"].to(device)
    B, T = input_ids.shape
    if T < 2:
        return None

    # 1. Target VLM forward (frozen) with full hidden states for soft KL.
    target_kwargs = dict(input_ids=input_ids, attention_mask=attn,
                         use_cache=True, output_hidden_states=True,
                         past_key_values=DynamicCache(), return_dict=True)
    if "pixel_values" in batch:
        target_kwargs["pixel_values"] = batch["pixel_values"].to(device).to(torch.bfloat16)
    if "image_grid_thw" in batch:
        target_kwargs["image_grid_thw"] = batch["image_grid_thw"].to(device)

    with torch.no_grad():
        tout = vlm(**target_kwargs)

    # Target hiddens at the configured layers (low/mid/high). hidden_states[k]
    # is the output of layer (k-1) in HF convention; we use [layer_idx + 1].
    target_hiddens = [tout.hidden_states[idx + 1].detach() for idx in target_layer_ids]
    target_logits = tout.logits.detach()  # (B, T, V) full logits
    # Free the other 33 layers' hiddens + the target's KV cache.
    del tout
    if "past_key_values" in target_kwargs:
        del target_kwargs["past_key_values"]
    torch.cuda.empty_cache()

    # 2. Loss mask: step-0 mask. Multi-step shift happens inside the model.
    loss_mask = build_loss_mask(prompt_lens, num_gen, attn)

    # 3. (Optional) compute target's true M-RoPE 3D position_ids for the draft.
    pos_ids = None
    if use_mrope_3d:
        igt = batch.get("image_grid_thw")
        if igt is not None:
            igt = igt.to(device)
        pos_ids = get_target_3d_position_ids(
            target_vlm=vlm, input_ids=input_ids, image_grid_thw=igt, attention_mask=attn,
        )  # (3, B, T)

    # 4. Multi-step rollout inside the model.
    plosses, acces = draft_module(
        target_hiddens=target_hiddens,
        input_ids=input_ids,
        embed_tokens=embed_tokens,
        lm_head=lm_head,
        target_logits=target_logits,
        loss_mask=loss_mask,
        attention_mask=attn,
        position_ids=pos_ids,
    )
    if any(torch.isnan(p) or torch.isinf(p) for p in plosses):
        return None

    # 4. Aggregate per-step losses with EAGLE-3's 0.8^k decay.
    weights = [0.8 ** i for i in range(len(plosses))]
    total = sum(w * p for w, p in zip(weights, plosses))
    return {"loss": total, "plosses": [p.detach() for p in plosses], "acces": acces}


@torch.inference_mode()
def run_val(draft_module, vlm, embed_tokens, lm_head, target_layer_ids,
            val_loader, device, max_batches=None, use_mrope_3d: bool = False):
    draft_module.eval()
    sums = None
    count = 0
    for i, batch in enumerate(val_loader):
        if max_batches and i >= max_batches: break
        try:
            out = train_step(draft_module, vlm, embed_tokens, lm_head,
                              target_layer_ids, batch, device,
                              use_mrope_3d=use_mrope_3d)
            if out is None: continue
            if sums is None:
                sums = [0.0 for _ in out["plosses"]]
            for j, p in enumerate(out["plosses"]):
                sums[j] += float(p.cpu().item())
            count += 1
        except Exception:
            continue
        finally:
            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()
    draft_module.train()
    if sums is None or count == 0:
        return [0.0], 0
    return [s / count for s in sums], count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_path", required=True)
    ap.add_argument("--target_outputs_dir", required=True)
    ap.add_argument("--val_uuids_file", required=True)
    ap.add_argument("--test_uuids_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--target_layer_ids", type=str, default="1,17,32",
                    help="Comma-sep layer indices fused at draft input. "
                         "EAGLE-3 default for 36-layer Qwen3-VL = 1,17,32.")
    ap.add_argument("--rollout_length", type=int, default=7,
                    help="Multi-step rollout length (EAGLE-3 paper: 7).")
    ap.add_argument("--use_mrope3d_draft", action="store_true",
                    help="Use M-RoPE 3D rotary in the draft (the natural EAGLE-3 "
                         "baseline for multimodal targets). Default OFF: 1D arange "
                         "rotary, which is OUR PAPER CONTRIBUTION (frees draft "
                         "attention from vision-token clustering — see "
                         "project_eagle3_1d_vs_3d_claim.md).")
    ap.add_argument("--num_epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--val_interval", type=int, default=500)
    ap.add_argument("--save_interval", type=int, default=1000)
    ap.add_argument("--val_batches", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
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

    target_layer_ids = [int(x) for x in args.target_layer_ids.split(",")]

    if is_main(rank):
        print(f"loading target from {args.target_path}")
    target = AlpamayoR1.from_pretrained(args.target_path, dtype=torch.bfloat16
                                         ).to(device).eval()
    for p in target.parameters():
        p.requires_grad = False
    vlm = target.vlm
    embed_tokens, lm_head = get_qwen3vl_embed_and_head(vlm)

    if is_main(rank):
        print(f"building EAGLE-3 draft (target_layer_ids={target_layer_ids}, "
              f"rollout_length={args.rollout_length}, "
              f"use_mrope3d_draft={args.use_mrope3d_draft})")
    draft = build_eagle3_draft_for_qwen3vl(
        target, target_layer_ids=target_layer_ids,
        rollout_length=args.rollout_length,
        use_mrope_3d=args.use_mrope3d_draft,
    ).to(device)
    if is_main(rank):
        n_params = sum(p.numel() for p in draft.parameters())
        print(f"draft params: {n_params / 1e6:.1f}M")

    if world > 1:
        draft_ddp = DDP(draft, device_ids=[local_rank], find_unused_parameters=False)
    else:
        draft_ddp = draft

    # Splits.
    with open(args.val_uuids_file) as f:
        val_uuids = json.load(f)
    with open(args.test_uuids_file) as f:
        test_uuids = json.load(f)
    excl = set(val_uuids) | set(test_uuids)

    train_ds = TargetOutputDataset(args.target_outputs_dir, exclude_uuids=excl)
    val_ds = TargetOutputDataset(args.target_outputs_dir, include_uuids=val_uuids)

    if is_main(rank):
        print(f"train clips: {len(train_ds)}, val clips: {len(val_ds)}")

    train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True) \
        if world > 1 else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              shuffle=(train_sampler is None), num_workers=args.num_workers,
                              collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=None,
                            shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate, pin_memory=True)

    optim = AdamW(draft.parameters(), lr=args.lr)
    total_steps = max(1, len(train_loader) * args.num_epochs // args.grad_accum_steps)
    sched = CosineAnnealingLR(optim, T_max=total_steps)

    global_step = 0
    for epoch in range(args.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        draft.train()
        for batch_idx, batch in enumerate(train_loader):
            t0 = time.time()
            out = train_step(draft_ddp, vlm, embed_tokens, lm_head,
                              target_layer_ids, batch, device,
                              use_mrope_3d=args.use_mrope3d_draft)
            if out is None:
                continue
            loss = out["loss"]
            (loss / args.grad_accum_steps).backward()
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(draft.parameters(), args.max_grad_norm)
                optim.step()
                sched.step()
                optim.zero_grad()
                global_step += 1

            if is_main(rank) and global_step % args.log_interval == 0:
                logd = {"train/lr": optim.param_groups[0]["lr"],
                        "train/loss": float(loss.detach().cpu().item()),
                        "train/step_time": time.time() - t0,
                        "epoch": epoch, "global_step": global_step}
                for k, p in enumerate(out["plosses"]):
                    logd[f"train/ploss_{k}"] = float(p.cpu().item())
                for k, a in enumerate(out["acces"]):
                    logd[f"train/acc_{k}"] = a
                if use_wandb:
                    import wandb
                    wandb.log(logd, step=global_step)
                print(f"[ep{epoch} step{global_step}] "
                      f"loss={logd['train/loss']:.3f} "
                      f"ploss0={logd['train/ploss_0']:.3f} "
                      f"acc0={logd['train/acc_0']:.3f} "
                      f"acc6={logd.get('train/acc_6', 0):.3f}")

            if is_main(rank) and global_step > 0 and global_step % args.save_interval == 0:
                save_eagle3_ckpt(draft, os.path.join(args.output_dir, f"draft_step_{global_step}.pt"))

            if is_main(rank) and global_step > 0 and global_step % args.val_interval == 0:
                val_plosses, n = run_val(draft_ddp, vlm, embed_tokens, lm_head,
                                          target_layer_ids, val_loader, device,
                                          max_batches=args.val_batches,
                                          use_mrope_3d=args.use_mrope3d_draft)
                logd = {"val/clips_used": n, "global_step": global_step}
                for k, p in enumerate(val_plosses):
                    logd[f"val/ploss_{k}"] = p
                if use_wandb:
                    import wandb
                    wandb.log(logd, step=global_step)
                print(f"[ep{epoch} VAL step{global_step}] " +
                      " ".join([f"p{k}={p:.3f}" for k, p in enumerate(val_plosses)]))

        if is_main(rank):
            save_eagle3_ckpt(draft, os.path.join(args.output_dir, f"draft_epoch_{epoch+1}.pt"))

    if is_main(rank):
        save_eagle3_ckpt(draft, os.path.join(args.output_dir, "draft_final.pt"))


if __name__ == "__main__":
    main()
