"""Benchmark speculative decoding vs autoregressive baseline on Alpamayo-R1.

Measures wall-clock time, tokens/sec, and acceptance rate for:
  1. Autoregressive (target-only) generation
  2. Speculative decoding with DFlash draft

Usage:
    # Text-only (no images):
    python scripts/benchmark_spec_decoding.py \
        --target_path /path/to/Alpamayo-R1-10B \
        --draft_path /path/to/draft_final.pt \
        --num_samples 20

    # With precomputed clip data (.pt files from cache_alpamayo_clips.py):
    python scripts/benchmark_spec_decoding.py \
        --target_path /path/to/Alpamayo-R1-10B \
        --draft_path /path/to/draft_final.pt \
        --clips_dir /path/to/alpamayo_clips \
        --num_samples 10

    # Sweep block sizes:
    python scripts/benchmark_spec_decoding.py \
        --target_path /path/to/Alpamayo-R1-10B \
        --draft_path /path/to/draft_final.pt \
        --block_sizes 2 4 8 16
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper
from alpamayo_r1.models.dflash_draft import (
    build_dflash_draft_for_qwen3vl,
    get_qwen3vl_embed_and_head,
    vlm_spec_generate,
)


def load_target(target_path: str, device: str = "cuda"):
    """Load Alpamayo-R1 target model."""
    model = AlpamayoR1.from_pretrained(target_path, dtype=torch.bfloat16)
    return model.vlm.to(device).eval()


def load_draft(target, draft_path: str, num_draft_layers: int, block_size: int, device: str = "cuda"):
    """Build and load DFlash draft model."""
    draft = build_dflash_draft_for_qwen3vl(
        target, num_draft_layers=num_draft_layers, block_size=block_size,
        mask_token_id=151643,
    ).to(torch.bfloat16).to(device).eval()

    if draft_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(draft_path, device=device)
    else:
        state_dict = torch.load(draft_path, map_location=device, weights_only=False)
    draft.load_state_dict(state_dict, strict=False)
    return draft


@torch.no_grad()
def baseline_generate(target, input_ids, max_new_tokens=256, temperature=0.0, stop_token_ids=None):
    """Autoregressive generation (target only, no draft)."""
    device = input_ids.device
    num_input_tokens = input_ids.shape[1]
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        out = target(input_ids=generated, use_cache=False, return_dict=True)
        logits = out.logits[:, -1:, :]
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs.squeeze(1), 1)
        else:
            next_token = logits.argmax(dim=-1)
        generated = torch.cat([generated, next_token], dim=1)

        if stop_token_ids and next_token.item() in stop_token_ids:
            break

    return {
        "output_ids": generated,
        "num_input_tokens": num_input_tokens,
        "num_output_tokens": generated.shape[1] - num_input_tokens,
    }


@torch.no_grad()
def baseline_generate_kv(target, input_ids, max_new_tokens=256, temperature=0.0, stop_token_ids=None):
    """Autoregressive generation with KV cache (target only)."""
    from transformers.cache_utils import DynamicCache

    device = input_ids.device
    num_input_tokens = input_ids.shape[1]
    output_ids = input_ids.clone()
    past_key_values = DynamicCache()

    # Prefill
    out = target(input_ids=input_ids, past_key_values=past_key_values,
                 use_cache=True, return_dict=True)
    if temperature > 0:
        probs = torch.softmax(out.logits[:, -1:, :] / temperature, dim=-1)
        next_token = torch.multinomial(probs.squeeze(1), 1)
    else:
        next_token = out.logits[:, -1:, :].argmax(dim=-1)
    output_ids = torch.cat([output_ids, next_token], dim=1)

    # Decode
    for step in range(max_new_tokens - 1):
        cache_pos = torch.tensor([num_input_tokens + step + 1], device=device)
        out = target(input_ids=next_token, past_key_values=past_key_values,
                     cache_position=cache_pos, use_cache=True, return_dict=True)
        if temperature > 0:
            probs = torch.softmax(out.logits[:, -1:, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs.squeeze(1), 1)
        else:
            next_token = out.logits[:, -1:, :].argmax(dim=-1)
        output_ids = torch.cat([output_ids, next_token], dim=1)

        if stop_token_ids and next_token.item() in stop_token_ids:
            break

    return {
        "output_ids": output_ids,
        "num_input_tokens": num_input_tokens,
        "num_output_tokens": output_ids.shape[1] - num_input_tokens,
    }


def build_text_prompts(num_samples: int):
    """Create simple text prompts for benchmarking."""
    prompts = [
        "Explain the concept of autonomous driving in simple terms.",
        "What are the main challenges in self-driving car technology?",
        "Describe how a neural network processes camera images for driving.",
        "What is the difference between L3 and L5 autonomous driving?",
        "How does a vision-language model help with driving decisions?",
        "What sensors do self-driving cars typically use?",
        "Explain how trajectory prediction works in autonomous vehicles.",
        "What is chain-of-thought reasoning in the context of driving AI?",
        "Describe the role of LiDAR in autonomous driving systems.",
        "How do autonomous vehicles handle unexpected obstacles?",
        "What is the importance of map data for self-driving cars?",
        "Explain the concept of end-to-end learning for autonomous driving.",
        "How do self-driving cars make lane change decisions?",
        "What are the safety considerations for deploying autonomous vehicles?",
        "Describe the training process for a vision-based driving model.",
        "How does reinforcement learning apply to autonomous driving?",
        "What is the role of simulation in testing self-driving systems?",
        "Explain how object detection works in driving scenarios.",
        "What are the ethical considerations of autonomous vehicles?",
        "How do autonomous vehicles communicate with traffic infrastructure?",
    ]
    return prompts[:num_samples]


def run_benchmark(
    target,
    draft,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 256,
    block_sizes: list[int] | None = None,
    device: str = "cuda",
):
    """Run the full benchmark suite."""
    if block_sizes is None:
        block_sizes = [draft.block_size] if draft is not None else []

    results = {}

    # --- Baseline: autoregressive with KV cache ---
    print("\n=== Baseline: Autoregressive (KV cache) ===")
    ar_times = []
    ar_tokens = []
    for i, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Warmup on first sample
        if i == 0:
            _ = baseline_generate_kv(target, input_ids, max_new_tokens=10)
            torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = baseline_generate_kv(target, input_ids, max_new_tokens=max_new_tokens)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed = t1 - t0
        n_tokens = out["num_output_tokens"]
        ar_times.append(elapsed)
        ar_tokens.append(n_tokens)
        print(f"  [{i+1}/{len(prompts)}] {n_tokens} tokens in {elapsed:.2f}s ({n_tokens/elapsed:.1f} tok/s)")

    avg_ar_time = sum(ar_times) / len(ar_times)
    avg_ar_tokens = sum(ar_tokens) / len(ar_tokens)
    avg_ar_tps = sum(t/e for t, e in zip(ar_tokens, ar_times)) / len(ar_times)
    results["autoregressive"] = {
        "avg_time": avg_ar_time,
        "avg_tokens": avg_ar_tokens,
        "avg_tokens_per_sec": avg_ar_tps,
    }
    print(f"  Average: {avg_ar_tokens:.0f} tokens, {avg_ar_time:.2f}s, {avg_ar_tps:.1f} tok/s")

    # --- Speculative decoding with different block sizes ---
    if draft is not None:
        for bs in block_sizes:
            print(f"\n=== Speculative Decoding (block_size={bs}) ===")
            spec_times = []
            spec_tokens = []
            all_acceptance = []

            for i, prompt in enumerate(prompts):
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

                # Warmup
                if i == 0:
                    _ = vlm_spec_generate(
                        target, draft, input_ids,
                        max_new_tokens=10, block_size=bs,
                    )
                    torch.cuda.synchronize()

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = vlm_spec_generate(
                    target, draft, input_ids,
                    max_new_tokens=max_new_tokens, block_size=bs,
                )
                torch.cuda.synchronize()
                t1 = time.perf_counter()

                elapsed = t1 - t0
                n_tokens = out["num_output_tokens"]
                acc_lens = out["acceptance_lengths"]
                avg_acc = sum(acc_lens) / max(len(acc_lens), 1)

                spec_times.append(elapsed)
                spec_tokens.append(n_tokens)
                all_acceptance.extend(acc_lens)

                print(f"  [{i+1}/{len(prompts)}] {n_tokens} tokens in {elapsed:.2f}s "
                      f"({n_tokens/elapsed:.1f} tok/s, avg_accept={avg_acc:.2f})")

            avg_spec_time = sum(spec_times) / len(spec_times)
            avg_spec_tokens = sum(spec_tokens) / len(spec_tokens)
            avg_spec_tps = sum(t/e for t, e in zip(spec_tokens, spec_times)) / len(spec_times)
            avg_acceptance = sum(all_acceptance) / max(len(all_acceptance), 1)
            speedup = avg_ar_tps / avg_spec_tps if avg_spec_tps > 0 else 0

            results[f"spec_bs{bs}"] = {
                "avg_time": avg_spec_time,
                "avg_tokens": avg_spec_tokens,
                "avg_tokens_per_sec": avg_spec_tps,
                "avg_acceptance_length": avg_acceptance,
                "speedup_vs_ar": 1.0 / speedup if speedup > 0 else 0,
            }
            print(f"  Average: {avg_spec_tokens:.0f} tokens, {avg_spec_time:.2f}s, "
                  f"{avg_spec_tps:.1f} tok/s, accept={avg_acceptance:.2f}, "
                  f"speedup={1.0/speedup if speedup > 0 else 0:.2f}x")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<30} {'tok/s':>8} {'speedup':>8}")
    print("-" * 60)
    ar_tps = results["autoregressive"]["avg_tokens_per_sec"]
    print(f"{'Autoregressive (KV cache)':<30} {ar_tps:>8.1f} {'1.00x':>8}")
    for bs in block_sizes:
        key = f"spec_bs{bs}"
        if key in results:
            tps = results[key]["avg_tokens_per_sec"]
            speedup = results[key]["speedup_vs_ar"]
            acc = results[key]["avg_acceptance_length"]
            print(f"{'Spec bs=' + str(bs):<30} {tps:>8.1f} {speedup:>7.2f}x  (accept={acc:.2f})")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark speculative decoding efficiency")
    parser.add_argument("--target_path", type=str, required=True, help="Path to Alpamayo-R1 or Qwen3-VL model")
    parser.add_argument("--draft_path", type=str, default=None, help="Path to draft checkpoint (.pt)")
    parser.add_argument("--num_draft_layers", type=int, default=1)
    parser.add_argument("--block_sizes", type=int, nargs="+", default=[4], help="Block sizes to benchmark")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--clips_dir", type=str, default=None, help="Optional: Alpamayo clip .pt files for multimodal test")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load target
    print(f"\nLoading target from {args.target_path}...")
    target = load_target(args.target_path, device)
    print(f"Target: {sum(p.numel() for p in target.parameters()) / 1e6:.0f}M params")

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.environ.get("PROCESSOR_PATH", args.target_path)
    )

    # Load draft (if provided)
    draft = None
    if args.draft_path:
        print(f"Loading draft from {args.draft_path}...")
        max_bs = max(args.block_sizes)
        draft = load_draft(target, args.draft_path, args.num_draft_layers, max_bs, device)
        print(f"Draft: {sum(p.numel() for p in draft.parameters()) / 1e6:.1f}M params")

    # Build prompts
    prompts = build_text_prompts(args.num_samples)
    print(f"\nBenchmarking with {len(prompts)} prompts, max_new_tokens={args.max_new_tokens}")

    # Run benchmark
    results = run_benchmark(
        target, draft, tokenizer, prompts,
        max_new_tokens=args.max_new_tokens,
        block_sizes=args.block_sizes if draft else None,
        device=device,
    )

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
