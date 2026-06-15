"""Generate target model outputs with logits for DFlash self-distillation.

For each clip, runs the Alpamayo-R1 target model autoregressively and saves:
  - input_ids: the full prompt (system + user with images)
  - output_ids: generated tokens
  - logits: per-token logits at each generation step (for KL distillation)
  - pixel_values_videos, video_grid_thw: visual inputs (for draft training)

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_target_outputs.py \
        --target_path /path/to/Alpamayo-R1-10B \
        --clips_dir /path/to/alpamayo_clips \
        --output_dir /path/to/target_outputs \
        --max_clips 9000 --max_new_tokens 512

    # Parallel on 4 GPUs:
    for i in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$i python scripts/generate_target_outputs.py \
            --target_path ... --clips_dir ... --output_dir ... \
            --shard $i --num_shards 4 &
    done
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper


@torch.no_grad()
def generate_with_logits(model, inputs, max_new_tokens=512, temperature=0.0,
                         save_logits=True):
    """Autoregressive generation that (optionally) saves logits at each step.

    Returns dict with input_ids, output_token_ids, and output_logits.
    When `save_logits=False`, logits are neither accumulated (saves host RAM)
    nor returned (`output_logits` is None) — used for the token-IDs-only export.
    """
    input_ids = inputs["input_ids"]
    device = input_ids.device
    prompt_len = input_ids.shape[1]

    # Prefill
    prefill_kwargs = dict(
        input_ids=input_ids,
        attention_mask=inputs.get("attention_mask"),
        use_cache=True,
        return_dict=True,
    )
    # Handle both image and video inputs
    for key in ["pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"]:
        if inputs.get(key) is not None:
            prefill_kwargs[key] = inputs[key]

    output = model(**prefill_kwargs)
    past_key_values = output.past_key_values

    # First token
    last_logits = output.logits[:, -1:, :]  # (1, 1, V)
    if temperature > 0:
        probs = torch.softmax(last_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs.squeeze(1), 1)
    else:
        next_token = last_logits.argmax(dim=-1)

    all_tokens = [next_token.squeeze()]  # list of scalar tensors
    all_logits = [last_logits.squeeze(0).cpu()] if save_logits else None  # (1, V) on CPU

    # Decode loop
    # Stop at <|traj_future_start|> (155681) — after this, diffusion takes over
    # Also stop at <|im_end|> (151645) and <|cot_end|> (155678)
    stop_ids = {155681, 151645, 155678}
    for step in range(max_new_tokens - 1):
        cache_position = torch.tensor([prompt_len + step + 1], device=device)
        output = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = output.past_key_values

        last_logits = output.logits[:, -1:, :]
        if temperature > 0:
            probs = torch.softmax(last_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs.squeeze(1), 1)
        else:
            next_token = last_logits.argmax(dim=-1)

        all_tokens.append(next_token.squeeze())
        if save_logits:
            all_logits.append(last_logits.squeeze(0).cpu())

        if next_token.item() in stop_ids:
            break

    output_token_ids = torch.stack(all_tokens)  # (num_generated,)
    output_logits = torch.cat(all_logits, dim=0) if save_logits else None  # (num_generated, V)

    return {
        "output_token_ids": output_token_ids.cpu(),
        "output_logits": output_logits,  # None when save_logits=False
        "num_generated": len(all_tokens),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_path", type=str, required=True)
    parser.add_argument("--clips_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_clips", type=int, default=9000)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0 for greedy, >0 for sampling")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--include_uuids_file", type=str, default=None,
                        help="JSON list of clip UUIDs to process (all others are skipped). "
                             "Useful for generating only missing/new clips while holding out test UUIDs.")
    parser.add_argument("--no_logits", action="store_true",
                        help="Do not accumulate or save per-token logits. Output keeps "
                             "token IDs + pixel_values only (~35MB/file vs ~42MB). KL "
                             "distillation is unavailable for these outputs.")
    args = parser.parse_args()

    device = "cuda"
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading target from {args.target_path}...")
    model = AlpamayoR1.from_pretrained(args.target_path, dtype=torch.bfloat16)
    target = model.vlm.to(device).eval()
    tokenizer = model.tokenizer
    processor = helper.get_processor(tokenizer)

    # Get clips for this shard
    all_clips = sorted(glob.glob(os.path.join(args.clips_dir, "*.pt")))
    if args.include_uuids_file:
        import json
        with open(args.include_uuids_file) as f:
            include = set(json.load(f))
        all_clips = [p for p in all_clips if Path(p).stem in include]
        print(f"Include list restricts to {len(all_clips)} clips")
    all_clips = all_clips[:args.max_clips]
    shard_size = len(all_clips) // args.num_shards
    start = args.shard * shard_size
    end = start + shard_size if args.shard < args.num_shards - 1 else len(all_clips)
    clip_files = all_clips[start:end]

    print(f"Shard {args.shard}/{args.num_shards}: clips {start}-{end} ({len(clip_files)} clips)")
    print(f"Max new tokens: {args.max_new_tokens}, temperature: {args.temperature}")

    t0 = time.time()
    generated = 0
    skipped = 0

    for i, cf in enumerate(clip_files):
        clip_id = Path(cf).stem
        out_path = os.path.join(args.output_dir, f"{clip_id}.pt")

        # Skip if already generated
        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            clip = torch.load(cf, weights_only=False)

            # Build prompt: system + user only (no assistant — let model generate)
            prompt_messages = []
            for m in clip["messages"]:
                if m["role"] == "assistant":
                    break
                prompt_messages.append(m)

            # Process with images
            inputs = processor.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

            prompt_len = inputs["input_ids"].shape[1]

            # Generate (optionally without logits)
            result = generate_with_logits(
                target, inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                save_logits=not args.no_logits,
            )

            # Save: prompt input_ids + generated tokens (+ logits unless --no_logits)
            save_data = {
                "clip_id": clip_id,
                "prompt_input_ids": inputs["input_ids"].cpu(),
                "prompt_len": prompt_len,
                "output_token_ids": result["output_token_ids"],
                "num_generated": result["num_generated"],
                "temperature": args.temperature,
            }
            if not args.no_logits:
                save_data["output_logits"] = result["output_logits"].to(torch.float16)  # save space
            # Also save visual inputs for draft training
            for vkey in ["pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"]:
                if inputs.get(vkey) is not None:
                    val = inputs[vkey].cpu()
                    if val.is_floating_point():
                        val = val.to(torch.float16)
                    save_data[vkey] = val

            torch.save(save_data, out_path)
            generated += 1

            elapsed = time.time() - t0
            rate = generated / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(clip_files)}] {clip_id}: {result['num_generated']} tokens, "
                  f"{rate:.2f} clips/s, {generated} done, {skipped} skipped")

        except Exception as e:
            print(f"  [{i+1}/{len(clip_files)}] SKIP {clip_id}: {type(e).__name__}: {str(e)[:100]}")
            torch.cuda.empty_cache()
            continue

    elapsed = time.time() - t0
    print(f"\nDone: {generated} generated, {skipped} skipped, {elapsed:.0f}s total")


if __name__ == "__main__":
    main()
