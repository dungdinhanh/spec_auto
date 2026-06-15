"""Regenerate specific corrupted target outputs as {clip_id}_regen.pt.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/regen_corrupted_outputs.py \
        --target_path /mnt/resv-harry-6f72s/dungda/models/Alpamayo-R1-10B \
        --clips_dir /mnt/resv-harry-6f72s/dungda/data/alpamayo_clips \
        --output_dir /mnt/resv-harry-6f72s/dungda/runs/target_coc_outputs \
        --clip_ids 1cb171a9-3600-4a7c-b03b-6a00d2260bd3 883c103f-b201-4d7d-b994-97a837b6ae9c
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper
from generate_target_outputs import generate_with_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_path", type=str, required=True)
    parser.add_argument("--clips_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--clip_ids", type=str, nargs="+", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    device = "cuda"

    print(f"Loading target from {args.target_path}...")
    model = AlpamayoR1.from_pretrained(args.target_path, dtype=torch.bfloat16)
    target = model.vlm.to(device).eval()
    tokenizer = model.tokenizer
    processor = helper.get_processor(tokenizer)

    for clip_id in args.clip_ids:
        clip_path = os.path.join(args.clips_dir, f"{clip_id}.pt")
        out_path = os.path.join(args.output_dir, f"{clip_id}_r.pt")

        if not os.path.exists(clip_path):
            print(f"SKIP {clip_id}: clip file not found at {clip_path}")
            continue

        if os.path.exists(out_path):
            print(f"SKIP {clip_id}: _r file already exists")
            continue

        print(f"Generating {clip_id}...")
        t0 = time.time()

        clip = torch.load(clip_path, weights_only=False)

        # Build prompt (system + user only)
        prompt_messages = []
        for m in clip["messages"]:
            if m["role"] == "assistant":
                break
            prompt_messages.append(m)

        inputs = processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        result = generate_with_logits(
            target, inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
        )

        save_data = {
            "clip_id": clip_id,
            "prompt_input_ids": inputs["input_ids"].cpu(),
            "prompt_len": prompt_len,
            "output_token_ids": result["output_token_ids"],
            "output_logits": result["output_logits"].to(torch.float16),
            "num_generated": result["num_generated"],
            "temperature": 0.0,
        }
        for vkey in ["pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"]:
            if inputs.get(vkey) is not None:
                val = inputs[vkey].cpu()
                if val.is_floating_point():
                    val = val.to(torch.float16)
                save_data[vkey] = val

        # Save to /tmp first, then copy to NFS (virtiofs can't set timestamps)
        tmp_path = f"/tmp/{clip_id}_r.pt"
        torch.save(save_data, tmp_path)
        # Use copyfile (content only, no metadata/timestamps) for virtiofs compat
        shutil.copyfile(tmp_path, out_path)
        os.remove(tmp_path)

        elapsed = time.time() - t0
        print(f"  Done: {result['num_generated']} tokens, {elapsed:.1f}s -> {out_path}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
