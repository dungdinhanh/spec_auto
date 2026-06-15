"""1-GPU smoke test — verify training pipeline imports, model loads, and a
single draft forward pass works on Katana flora before kicking off full train.

Specifically checks:
  - AlpamayoR1.from_pretrained loads with flash_attention_2 config (must
    fall back to sdpa if flash-attn not installed)
  - target_coc_outputs can be read and feeds into the draft
  - M-RoPE draft forward pass runs without error
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper
from alpamayo_r1.models.dflash_draft import (
    get_qwen3vl_embed_and_head,
    extract_context_feature,
)
from alpamayo_r1.models.dflash_draft_mrope import build_dflash_draft_mrope_for_qwen3vl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_path", required=True)
    ap.add_argument("--target_outputs_dir", required=True)
    ap.add_argument("--val_uuids_file", required=True)
    ap.add_argument("--num_draft_layers", type=int, default=3)
    ap.add_argument("--block_size", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda")

    t0 = time.time()
    print(f"[smoke] loading AlpamayoR1 from {args.target_path} ...", flush=True)
    try:
        target = AlpamayoR1.from_pretrained(
            args.target_path, dtype=torch.bfloat16,
        ).to(device).eval()
        attn_used = getattr(target.vlm.config, "_attn_implementation",
                            getattr(target.vlm.config, "attn_implementation", "?"))
        print(f"[smoke] loaded with attn_implementation={attn_used}", flush=True)
    except Exception as e:
        print(f"[smoke] initial load FAILED: {type(e).__name__}: {e}", flush=True)
        print("[smoke] retrying with attn_implementation=sdpa ...", flush=True)
        target = AlpamayoR1.from_pretrained(
            args.target_path, dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to(device).eval()
        print("[smoke] sdpa load OK", flush=True)

    for p in target.parameters():
        p.requires_grad = False
    print(f"[smoke] target loaded in {time.time()-t0:.1f}s", flush=True)

    # --- Load one sample from target_coc_outputs ---
    with open(args.val_uuids_file) as f:
        val_uuids = json.load(f)
    sample_uuid = val_uuids[0]
    sample_path = os.path.join(args.target_outputs_dir, f"{sample_uuid}.pt")
    print(f"[smoke] loading sample {sample_uuid} ...", flush=True)
    sample = torch.load(sample_path, map_location="cpu", weights_only=False)
    print(f"[smoke] sample keys: {list(sample.keys())}", flush=True)
    for k in ("prompt_input_ids", "output_token_ids", "pixel_values", "image_grid_thw"):
        v = sample.get(k)
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}", flush=True)

    # --- Target forward (get hidden states) ---
    prompt_ids = sample["prompt_input_ids"].to(device)
    output_ids = sample["output_token_ids"].to(device)
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    if output_ids.ndim == 1:
        output_ids = output_ids.unsqueeze(0)

    full_ids = torch.cat([prompt_ids, output_ids], dim=1)
    pixel_values = sample["pixel_values"].to(device).to(torch.bfloat16)
    image_grid_thw = sample["image_grid_thw"].to(device)
    print(f"[smoke] running target forward (prompt+output)={full_ids.shape[1]} tokens ...", flush=True)

    vlm = target.vlm
    with torch.no_grad():
        out = vlm(
            input_ids=full_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden_states = out.hidden_states
    print(f"[smoke] target forward OK — {len(hidden_states)} hidden layers, last shape={tuple(hidden_states[-1].shape)}", flush=True)

    # --- Build draft ---
    embed_tokens, lm_head = get_qwen3vl_embed_and_head(vlm)
    print(f"[smoke] building DFlash M-RoPE draft (L={args.num_draft_layers}) ...", flush=True)
    draft = build_dflash_draft_mrope_for_qwen3vl(
        vlm,
        num_draft_layers=args.num_draft_layers,
        block_size=args.block_size,
        mask_token_id=151662,
    ).to(device).to(torch.bfloat16)
    target_layer_ids = draft.target_layer_ids
    print(f"[smoke] draft built. target_layer_ids={target_layer_ids}", flush=True)
    print(f"[smoke] draft params (trainable): {sum(p.numel() for p in draft.parameters() if p.requires_grad)/1e6:.1f}M", flush=True)

    # --- Single draft forward (mirror training block 0) ---
    ctx_all = extract_context_feature(hidden_states, target_layer_ids)
    start = prompt_ids.shape[1]
    block_size = args.block_size
    ctx_len = start + 1
    end = start + block_size
    ctx_hidden = ctx_all[:, :ctx_len, :]

    mask_token_id = 151662
    noise_tokens = torch.full((1, block_size - 1), mask_token_id,
                              dtype=torch.long, device=device)
    noise_embedding = embed_tokens(noise_tokens)
    pos_ids = torch.arange(ctx_len + block_size - 1, device=device).unsqueeze(0)

    with torch.no_grad():
        draft_hidden = draft(
            target_hidden=ctx_hidden,
            noise_embedding=noise_embedding,
            position_ids=pos_ids,
        )
    block_logits = lm_head(draft_hidden[:, -(block_size - 1):, :])
    print(f"[smoke] draft forward OK — block_logits shape={tuple(block_logits.shape)}", flush=True)

    print(f"[smoke] ALL CHECKS PASSED in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
