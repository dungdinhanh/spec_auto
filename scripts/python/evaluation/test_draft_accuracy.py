"""Test draft prediction accuracy on training/validation clips WITH full multimodal input.

Processes clips exactly like training: images go through Qwen3-VL visual encoder,
then tests if the draft can predict the target's next tokens.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/test_draft_accuracy.py \
        --target_path /path/to/Alpamayo-R1-10B \
        --draft_path /path/to/draft_epoch_1.pt \
        --clips_dir /path/to/alpamayo_clips \
        --num_train 20 --num_val 20
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper
from alpamayo_r1.models.dflash_draft import (
    build_dflash_draft_for_qwen3vl,
    get_qwen3vl_embed_and_head,
    extract_context_feature,
)


def process_clip(clip, processor, device):
    """Process a clip exactly like training — returns input_ids, pixel_values_videos, etc."""
    inputs = processor.apply_chat_template(
        clip["messages"],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}


@torch.no_grad()
def test_on_clip(target, draft, embed_tokens, lm_head, inputs, block_size, device):
    """Test draft accuracy on a single clip with full multimodal input."""
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    if seq_len < block_size + 1:
        return None

    # Target forward WITH images
    target_kwargs = dict(
        input_ids=input_ids,
        attention_mask=inputs.get("attention_mask"),
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    if inputs.get("pixel_values_videos") is not None:
        target_kwargs["pixel_values_videos"] = inputs["pixel_values_videos"]
    if inputs.get("video_grid_thw") is not None:
        target_kwargs["video_grid_thw"] = inputs["video_grid_thw"]

    target_out = target(**target_kwargs)
    target_hidden = extract_context_feature(target_out.hidden_states, draft.target_layer_ids)
    target_preds = target_out.logits.argmax(dim=-1)

    # Test draft on multiple blocks
    num_blocks = min((seq_len - 1) // block_size, 10)

    # Two sets of metrics: vs ground truth (dataset) and vs target model
    gt_pos_accs = {k: 0 for k in range(1, block_size)}
    tgt_pos_accs = {k: 0 for k in range(1, block_size)}
    gt_accepted_total = 0
    tgt_accepted_total = 0
    total_blocks = 0

    for b_idx in range(num_blocks):
        start = b_idx * block_size
        end = start + block_size
        if end >= seq_len:
            break

        block_ids = input_ids[:, start:end].clone()
        block_ids[:, 1:] = draft.mask_token_id
        noise_embedding = embed_tokens(block_ids)

        ctx_hidden = target_hidden[:, :end, :]
        ctx_len = ctx_hidden.shape[1]
        pos_ids = torch.arange(ctx_len + block_size, device=device).unsqueeze(0)

        draft_hidden = draft(
            target_hidden=ctx_hidden,
            noise_embedding=noise_embedding,
            position_ids=pos_ids,
            past_key_values=None,
            use_cache=False,
        )
        draft_logits = lm_head(draft_hidden[:, -(block_size - 1):, :])
        draft_preds = draft_logits.argmax(dim=-1)

        # Ground truth: actual next tokens from the dataset
        gt_next = input_ids[:, start + 1:end]

        # Target model's greedy predictions
        target_next = target_preds[:, start:end - 1]

        for k in range(block_size - 1):
            if draft_preds[0, k] == gt_next[0, k]:
                gt_pos_accs[k + 1] += 1
            if draft_preds[0, k] == target_next[0, k]:
                tgt_pos_accs[k + 1] += 1

        # Acceptance length vs ground truth
        gt_acc = 0
        for k in range(block_size - 1):
            if draft_preds[0, k] == gt_next[0, k]:
                gt_acc += 1
            else:
                break
        gt_accepted_total += gt_acc

        # Acceptance length vs target model
        tgt_acc = 0
        for k in range(block_size - 1):
            if draft_preds[0, k] == target_next[0, k]:
                tgt_acc += 1
            else:
                break
        tgt_accepted_total += tgt_acc

        total_blocks += 1

    if total_blocks == 0:
        return None

    return {
        "total_blocks": total_blocks,
        "gt_avg_accepted": gt_accepted_total / total_blocks,
        "tgt_avg_accepted": tgt_accepted_total / total_blocks,
        "gt_pos_accs": {k: v / total_blocks for k, v in gt_pos_accs.items()},
        "tgt_pos_accs": {k: v / total_blocks for k, v in tgt_pos_accs.items()},
        "seq_len": seq_len,
    }


def run_test(target, draft, embed_tokens, lm_head, processor, clip_files,
             block_size, device, label=""):
    """Run accuracy test on a list of clip files."""
    results = []
    for i, cf in enumerate(clip_files):
        try:
            clip = torch.load(cf, weights_only=False)
            inputs = process_clip(clip, processor, device)
            result = test_on_clip(target, draft, embed_tokens, lm_head, inputs,
                                  block_size, device)
            if result is None:
                continue
            results.append(result)
            gt_str = " ".join(
                f"p{k}={result['gt_pos_accs'].get(k, 0):.0%}"
                for k in range(1, min(5, block_size))
            )
            print(f"  [{i+1}/{len(clip_files)}] seq={result['seq_len']} "
                  f"blocks={result['total_blocks']} "
                  f"gt_accept={result['gt_avg_accepted']:.2f} "
                  f"tgt_accept={result['tgt_avg_accepted']:.2f} "
                  f"gt:[{gt_str}]")
        except Exception as e:
            print(f"  [{i+1}/{len(clip_files)}] skip: {type(e).__name__}: {str(e)[:80]}")
            torch.cuda.empty_cache()
            continue

    if not results:
        print(f"  No valid results for {label}")
        return {}

    # Aggregate
    summary = {}
    for k in range(1, block_size):
        summary[f"gt_pos_{k}_acc"] = sum(r["gt_pos_accs"].get(k, 0) for r in results) / len(results)
        summary[f"tgt_pos_{k}_acc"] = sum(r["tgt_pos_accs"].get(k, 0) for r in results) / len(results)
    summary["gt_avg_accepted"] = sum(r["gt_avg_accepted"] for r in results) / len(results)
    summary["tgt_avg_accepted"] = sum(r["tgt_avg_accepted"] for r in results) / len(results)
    summary["num_clips"] = len(results)
    summary["avg_seq_len"] = sum(r["seq_len"] for r in results) / len(results)

    print(f"\n  {label} Summary ({len(results)} clips):")
    print(f"  {'Pos':<5} {'vs GT (dataset)':<18} {'vs Target (model)':<18}")
    print(f"  {'-'*40}")
    for k in range(1, min(block_size, 9)):
        print(f"  {k:<5} {summary[f'gt_pos_{k}_acc']:<18.1%} {summary[f'tgt_pos_{k}_acc']:<18.1%}")
    print(f"  {'Avg acceptance:':<20} GT={summary['gt_avg_accepted']:.2f}  Target={summary['tgt_avg_accepted']:.2f}  (max={block_size - 1})")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_path", type=str, required=True)
    parser.add_argument("--draft_path", type=str, required=True)
    parser.add_argument("--clips_dir", type=str, required=True)
    parser.add_argument("--num_draft_layers", type=int, default=5)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--mask_token_id", type=int, default=151662,
                        help="Fallback mask token id (fim_pad). Overridden by checkpoint's "
                             "mask_token_id if wrapped. For legacy plain state_dicts, pass explicitly.")
    parser.add_argument("--num_train", type=int, default=20)
    parser.add_argument("--num_val", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = "cuda"

    print(f"Loading target from {args.target_path}...")
    model = AlpamayoR1.from_pretrained(args.target_path, dtype=torch.bfloat16)
    target = model.vlm.to(device).eval()
    embed_tokens, lm_head = get_qwen3vl_embed_and_head(target)
    tokenizer = model.tokenizer
    processor = helper.get_processor(tokenizer)

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

    # Split clips: first max_clips for train, next val_clips for val
    all_clips = sorted(glob.glob(os.path.join(args.clips_dir, "*.pt")))
    train_clips = all_clips[:args.num_train]
    # Val clips from offset 9000 (matching training config)
    val_clips = all_clips[9000:9000 + args.num_val]

    all_results = {}

    print(f"\n{'='*60}")
    print(f"TEST ON TRAINING DATA ({len(train_clips)} clips)")
    print(f"{'='*60}")
    all_results["train"] = run_test(
        target, draft, embed_tokens, lm_head, processor,
        train_clips, args.block_size, device, label="Train"
    )

    print(f"\n{'='*60}")
    print(f"TEST ON VALIDATION DATA ({len(val_clips)} clips)")
    print(f"{'='*60}")
    all_results["val"] = run_test(
        target, draft, embed_tokens, lm_head, processor,
        val_clips, args.block_size, device, label="Val"
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
