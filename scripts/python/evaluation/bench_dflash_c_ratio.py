"""Microbenchmark DFlash draft cost ratio c = T_draft_block / T_target_step.

Theoretical speedup S = L / (1 + c) — we already have L from e2e_spec_test.py;
this script supplies c so the S column on the table is grounded in measurement
instead of a back-of-envelope estimate.

Measures on a fixed Alpamayo-style prompt context: vision + text prefill of
~3000 tokens, then in a steady-state continuation regime.
  - T_target_step : single target forward at q_len=1 with cache_position
  - T_draft_block : single draft forward producing block_size-1 predictions
                    with target_hidden as cross-attention K/V context

Usage:
  python bench_dflash_c_ratio.py \
      --target_path /home/ubuntu/local_data/models/Alpamayo-R1-10B \
      --draft_path  /home/ubuntu/local_data/runs/.../draft_final.pt \
      --clip_path   /home/ubuntu/local_data/runs/target_coc_outputs/SOME-UUID.pt \
      --num_draft_layers 2 --block_size 16 --num_target_features 5 \
      --warmup 5 --iters 50
"""
from __future__ import annotations
import argparse, sys, time, json
from pathlib import Path
import torch

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.models.dflash_draft import (
    build_dflash_draft_for_qwen3vl,
    build_target_layer_ids,
    extract_context_feature,
    load_draft_checkpoint,
    get_qwen3vl_embed_and_head,
)
from transformers.cache_utils import DynamicCache


def bench(target, draft, embed_tokens, lm_head, prompt_ids, pixel_values, image_grid_thw,
          block_size, mask_id, warmup, iters, device):
    # Prefill target on prompt to warm KV cache + target_hidden context.
    past = DynamicCache()
    pkw = dict(input_ids=prompt_ids, past_key_values=past, use_cache=True,
               output_hidden_states=True, return_dict=True)
    if pixel_values is not None: pkw["pixel_values"] = pixel_values
    if image_grid_thw is not None: pkw["image_grid_thw"] = image_grid_thw
    out_t = target(**pkw)
    target_hidden = extract_context_feature(out_t.hidden_states, draft.target_layer_ids)
    P = prompt_ids.shape[1]

    # --- T_target_step: single forward at q_len=1 with cache_position ---
    next_tok = torch.randint(1000, 50000, (1, 1), device=device)
    cache_pos = torch.tensor([P], device=device, dtype=torch.long)
    for _ in range(warmup):
        _ = target(input_ids=next_tok, past_key_values=past, cache_position=cache_pos,
                   use_cache=True, return_dict=True)
        past.crop(P)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = target(input_ids=next_tok, past_key_values=past, cache_position=cache_pos,
                   use_cache=True, return_dict=True)
        past.crop(P)
    torch.cuda.synchronize()
    T_target_step = (time.perf_counter() - t0) / iters

    # --- T_draft_block: single draft forward producing B-1 predictions ---
    ctx_hidden = target_hidden[:, :P, :]
    anchor_tok = int(prompt_ids[0, -1].item())
    noise_tokens = torch.full((1, block_size), mask_id, dtype=torch.long, device=device)
    noise_tokens[0, 0] = anchor_tok
    noise_emb = embed_tokens(noise_tokens)
    pos_ids = torch.arange(P + block_size, device=device).unsqueeze(0)
    for _ in range(warmup):
        dh = draft(target_hidden=ctx_hidden, noise_embedding=noise_emb, position_ids=pos_ids)
        _ = lm_head(dh[:, -(block_size - 1):, :])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        dh = draft(target_hidden=ctx_hidden, noise_embedding=noise_emb, position_ids=pos_ids)
        _ = lm_head(dh[:, -(block_size - 1):, :])
    torch.cuda.synchronize()
    T_draft_block = (time.perf_counter() - t0) / iters

    return T_target_step, T_draft_block


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_path", required=True)
    ap.add_argument("--draft_path", required=True)
    ap.add_argument("--clip_path", required=True,
                    help="Path to a target_coc_outputs/*.pt — used to get a real prompt.")
    ap.add_argument("--num_draft_layers", type=int, required=True)
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--num_target_features", type=int, default=5)
    ap.add_argument("--mask_token_id", type=int, default=151662)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--output_json", type=str, default=None)
    args = ap.parse_args()

    device = "cuda"
    dt = torch.bfloat16
    print(f"loading target ...", flush=True)
    model = AlpamayoR1.from_pretrained(args.target_path, dtype=dt)
    target = model.vlm.to(device).eval()
    embed_tokens, lm_head = get_qwen3vl_embed_and_head(target)

    print(f"loading draft from {args.draft_path}", flush=True)
    ckpt = load_draft_checkpoint(args.draft_path, map_location=device)
    mask_id = ckpt["mask_token_id"] or args.mask_token_id
    num_layers = ckpt["num_draft_layers"] or args.num_draft_layers
    bsz = ckpt["block_size"] or args.block_size

    _n_target = target.config.get_text_config().num_hidden_layers
    tlids = build_target_layer_ids(_n_target, args.num_target_features)
    print(f"  num_draft_layers={num_layers}  block_size={bsz}  "
          f"target_layer_ids={tlids}", flush=True)

    draft = build_dflash_draft_for_qwen3vl(
        target, num_draft_layers=num_layers, block_size=bsz,
        mask_token_id=mask_id, target_layer_ids=tlids,
    ).to(dt).to(device).eval()
    draft.load_state_dict(ckpt["state_dict"], strict=False)

    print(f"loading clip from {args.clip_path}", flush=True)
    clip = torch.load(args.clip_path, weights_only=False)
    prompt_ids = clip["prompt_input_ids"].to(device)
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    pixel_values = clip["pixel_values"].to(dt).to(device) if "pixel_values" in clip else None
    image_grid_thw = clip["image_grid_thw"].to(device) if "image_grid_thw" in clip else None

    T_target_step, T_draft_block = bench(
        target, draft, embed_tokens, lm_head,
        prompt_ids, pixel_values, image_grid_thw,
        bsz, mask_id, args.warmup, args.iters, device,
    )
    c = T_draft_block / T_target_step
    print(f"\n=== RESULTS ===")
    print(f"  T_target_step : {T_target_step*1000:.3f} ms")
    print(f"  T_draft_block : {T_draft_block*1000:.3f} ms")
    print(f"  c = T_draft_block / T_target_step = {c:.4f}")
    print(f"  S(L)        = L / (1 + c)  with c={c:.4f}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({
                "draft_path": args.draft_path,
                "num_draft_layers": num_layers,
                "block_size": bsz,
                "num_target_features": args.num_target_features,
                "T_target_step_ms": T_target_step * 1000,
                "T_draft_block_ms": T_draft_block * 1000,
                "c": c,
                "iters": args.iters,
            }, f, indent=2)
        print(f"  written: {args.output_json}")


if __name__ == "__main__":
    main()
