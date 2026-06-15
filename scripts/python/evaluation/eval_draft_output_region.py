"""Evaluate draft accuracy on output region only (matching training setup).

Uses target_coc_outputs format (pre-generated with pixel_values).
Reports position accuracy and acceptance length on the output tokens.
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper
from alpamayo_r1.models.dflash_draft import (
    build_dflash_draft_for_qwen3vl,
    get_qwen3vl_embed_and_head,
    extract_context_feature,
)


@torch.no_grad()
def evaluate(target, draft, embed_tokens, lm_head, tokenizer,
             files, block_size, device, num_clips=50):
    total_gt_match = 0
    total_positions = 0
    total_accept = 0.0
    total_blocks = 0
    total_clips = 0
    pos_accs = [0] * (block_size - 1)
    pos_counts = [0] * (block_size - 1)

    for fi in range(min(num_clips, len(files))):
        data = torch.load(files[fi], weights_only=False)
        prompt_ids = data["prompt_input_ids"].squeeze(0)
        output_ids = data["output_token_ids"]
        full_ids = torch.cat([prompt_ids, output_ids], dim=0).unsqueeze(0).to(device)
        prompt_len = len(prompt_ids)

        tkw = dict(input_ids=full_ids, use_cache=False, output_hidden_states=True, return_dict=True)
        pv = data.get("pixel_values")
        igt = data.get("image_grid_thw")
        if pv is not None:
            tkw["pixel_values"] = pv.to(torch.bfloat16).to(device)
        if igt is not None:
            tkw["image_grid_thw"] = igt.to(device)

        tout = target(**tkw)
        th = extract_context_feature(tout.hidden_states, draft.target_layer_ids)

        seq_len = full_ids.shape[1]
        # Non-overlapping blocks in output region (matches inference)
        first_start = max(0, prompt_len - 1)
        clip_blocks = 0
        clip_accept = 0

        for start in range(first_start, seq_len - block_size, block_size):
            end = start + block_size
            if end > seq_len:
                break

            block_ids = full_ids[:, start:end].clone()
            block_ids[:, 1:] = draft.mask_token_id  # full mask (inference mode)
            noise = embed_tokens(block_ids)
            ctx = th[:, :end, :]
            pos = torch.arange(ctx.shape[1] + block_size, device=device).unsqueeze(0)

            dh = draft(target_hidden=ctx, noise_embedding=noise, position_ids=pos,
                       past_key_values=None, use_cache=False)
            dl = lm_head(dh[:, -(block_size - 1):, :])
            dp = dl.argmax(dim=-1)
            gt = full_ids[0, start + 1:end]

            # Per-position accuracy
            for k in range(block_size - 1):
                if start + k + 1 >= seq_len:
                    break
                pos_counts[k] += 1
                if dp[0, k].item() == gt[k].item():
                    total_gt_match += 1
                    pos_accs[k] += 1
                total_positions += 1

            # Acceptance length
            acc = 0
            for k in range(block_size - 1):
                if start + k + 1 >= seq_len:
                    break
                if dp[0, k].item() == gt[k].item():
                    acc += 1
                else:
                    break
            clip_accept += acc
            clip_blocks += 1

        if clip_blocks > 0:
            total_accept += clip_accept / clip_blocks
            total_blocks += clip_blocks
            total_clips += 1

        if (fi + 1) % 10 == 0:
            acc_so_far = total_gt_match / max(total_positions, 1)
            avg_accept = total_accept / max(total_clips, 1)
            print(f"  [{fi+1}/{num_clips}] pos_acc={acc_so_far:.1%} avg_accept={avg_accept:.2f}")

    # Summary
    if total_positions == 0:
        print("No valid positions evaluated.")
        return

    overall_acc = total_gt_match / total_positions
    avg_accept = total_accept / max(total_clips, 1)
    print(f"\n--- RESULTS ({total_clips} clips, block_size={block_size}) ---")
    print(f"  Overall position accuracy: {total_gt_match}/{total_positions} ({overall_acc:.1%})")
    print(f"  Avg acceptance length: {avg_accept:.2f} / {block_size - 1}")
    print(f"  Per-position accuracy:")
    for k in range(block_size - 1):
        if pos_counts[k] > 0:
            print(f"    pos {k+1}: {pos_accs[k]}/{pos_counts[k]} ({pos_accs[k]/pos_counts[k]:.1%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_path", type=str, required=True)
    parser.add_argument("--draft_path", type=str, required=True)
    parser.add_argument("--target_outputs_dir", type=str, required=True)
    parser.add_argument("--num_draft_layers", type=int, default=2)
    parser.add_argument("--block_size", type=int, default=8)
    parser.add_argument("--mask_token_id", type=int, default=151662,
                        help="Fallback mask token id (fim_pad). Overridden by checkpoint's "
                             "mask_token_id if the checkpoint is wrapped with metadata. "
                             "For legacy plain state_dicts (no metadata), pass --mask_token_id "
                             "explicitly to match how the draft was trained.")
    parser.add_argument("--num_clips", type=int, default=50)
    args = parser.parse_args()

    device = "cuda"

    print(f"Loading target from {args.target_path}...")
    model = AlpamayoR1.from_pretrained(args.target_path, dtype=torch.bfloat16)
    target = model.vlm.to(device).eval()
    embed_tokens, lm_head = get_qwen3vl_embed_and_head(target)
    tokenizer = model.tokenizer

    from alpamayo_r1.models.dflash_draft import load_draft_checkpoint
    print(f"Loading draft checkpoint from {args.draft_path}...")
    ckpt = load_draft_checkpoint(args.draft_path, map_location=device)
    if ckpt["mask_token_id"] is None:
        print(f"  WARNING: legacy plain state_dict (no metadata). Using CLI --mask_token_id={args.mask_token_id}.")
        print(f"           If this draft was trained with a different mask token, results will be garbage.")
    mask_id = ckpt["mask_token_id"] if ckpt["mask_token_id"] is not None else args.mask_token_id
    num_layers = ckpt["num_draft_layers"] if ckpt["num_draft_layers"] is not None else args.num_draft_layers
    bsz = ckpt["block_size"] if ckpt["block_size"] is not None else args.block_size
    print(f"  From checkpoint: mask_token_id={mask_id}, num_draft_layers={num_layers}, block_size={bsz}")
    args.block_size = bsz

    print(f"Building draft (layers={num_layers}, bs={bsz}, mask_id={mask_id})...")
    draft = build_dflash_draft_for_qwen3vl(
        target, num_draft_layers=num_layers,
        block_size=bsz, mask_token_id=mask_id,
    ).to(torch.bfloat16).to(device).eval()
    msg = draft.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"  Loaded: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")

    files = sorted(glob.glob(os.path.join(args.target_outputs_dir, "*.pt")))
    print(f"Valid target output files: {len(files)}")

    evaluate(target, draft, embed_tokens, lm_head, tokenizer,
             files, args.block_size, device, args.num_clips)


if __name__ == "__main__":
    main()
