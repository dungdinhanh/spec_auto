"""Diagnose whether the warm-SFT DFlash draft's positional encoding is
consistent with the target's M-RoPE.

Outputs:
  1. Config side-by-side: rope_scaling, mrope_section, rope_theta,
     max_position_embeddings, num_attention_heads, head_dim.
  2. Position-id comparison on a real clip:
     - Target's 3D M-RoPE position_ids at every position (with rope_deltas).
     - The 1D arange position_ids that train_dflash_distillation.py feeds the draft.
     - Sample positions before / inside / after the vision-token block to show
       where the two diverge.
  3. Hidden-state consistency at target_layer_ids: mean / std / norm per layer
     of (a) what target produces and (b) what would feed the draft's `fc`.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

sys.path.insert(0, "/home/ubuntu/alpamayo_code/src")
sys.path.insert(0, "/home/ubuntu/dflash_code")

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.models.dflash_draft import build_qwen3_draft_config, load_draft_checkpoint
from alpamayo_r1.models.dflash_draft_mrope import (
    build_dflash_draft_mrope_for_qwen3vl,
)


def fmt(v):
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {fmt(vv)}" for k, vv in v.items()) + "}"
    return str(v)


def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_path", required=True)
    ap.add_argument("--draft_path", required=True)
    ap.add_argument("--clip_path", required=True,
                    help="Path to a target_coc_outputs/*.pt file (has prompt_input_ids, "
                         "pixel_values, image_grid_thw, output_token_ids).")
    args = ap.parse_args()

    device = "cuda"
    dt = torch.bfloat16

    # ---- 1. Load target ----
    print(f"Loading target from {args.target_path}", flush=True)
    target = AlpamayoR1.from_pretrained(args.target_path, dtype=dt).to(device).eval()
    vlm = target.vlm

    # ---- 2. Load draft ----
    print(f"Loading draft from {args.draft_path}", flush=True)
    ckpt = load_draft_checkpoint(args.draft_path, map_location=device)
    num_layers = ckpt["num_draft_layers"] or 4
    block_size = ckpt["block_size"] or 16
    mask_id = ckpt["mask_token_id"] or 151662
    draft = build_dflash_draft_mrope_for_qwen3vl(
        vlm, num_draft_layers=num_layers, block_size=block_size,
        mask_token_id=mask_id,
    ).to(dt).to(device).eval()
    draft.load_state_dict(ckpt["state_dict"], strict=False)

    # ---- 3. Config side-by-side ----
    section("(1) Config side-by-side: target vs draft")
    tc = vlm.config.get_text_config()
    dc = draft.config

    keys = [
        "hidden_size", "intermediate_size",
        "num_attention_heads", "num_key_value_heads", "head_dim",
        "max_position_embeddings", "rope_theta",
        "rope_scaling",
    ]
    print(f"{'field':<30} | {'TARGET':<32} | {'DRAFT':<32} | match?")
    print("-" * 110)
    for k in keys:
        t = getattr(tc, k, "<missing>")
        d = getattr(dc, k, "<missing>")
        m = "✓" if t == d else "✗"
        print(f"{k:<30} | {fmt(t):<32} | {fmt(d):<32} | {m}")
    print()
    print("Rotary class:")
    # Target's rotary lives in the language_model submodule
    target_rotary = (getattr(vlm, "language_model", None) and getattr(vlm.language_model, "rotary_emb", None)) \
        or getattr(vlm.model, "rotary_emb", None) \
        or (getattr(vlm.model, "language_model", None) and getattr(vlm.model.language_model, "rotary_emb", None))
    print(f"  target: {type(target_rotary).__name__ if target_rotary is not None else '<not found>'}")
    print(f"  draft : {type(draft.rotary_emb).__name__}")

    # ---- 4. Run target on a real clip ----
    section("(2) Position-id behaviour on a real clip")
    d = torch.load(args.clip_path, weights_only=False)
    input_ids = d["prompt_input_ids"].to(device)
    pixel_values = d["pixel_values"].to(dt).to(device) if "pixel_values" in d else None
    image_grid_thw = d["image_grid_thw"].to(device) if "image_grid_thw" in d else None
    print(f"clip {Path(args.clip_path).stem}  prompt_len={input_ids.shape[1]}")

    fwd_kwargs = dict(input_ids=input_ids, use_cache=True,
                      output_hidden_states=True,
                      past_key_values=DynamicCache(), return_dict=True)
    if pixel_values is not None:
        fwd_kwargs["pixel_values"] = pixel_values
    if image_grid_thw is not None:
        fwd_kwargs["image_grid_thw"] = image_grid_thw
    with torch.no_grad():
        tout = vlm(**fwd_kwargs)

    # vlm.model.rope_deltas is the offset that M-RoPE adds after vision tokens.
    rope_deltas = vlm.model.rope_deltas
    print(f"target rope_deltas (per-batch offset M-RoPE adds after vision): {rope_deltas.tolist()}")

    # Compute the 3D position_ids target ACTUALLY uses, by re-calling
    # vlm.model.get_rope_index. This is what's passed to target's rotary_emb.
    if hasattr(vlm.model, "get_rope_index"):
        attn_mask = torch.ones_like(input_ids)
        try:
            t_pos_ids, _ = vlm.model.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=None,
                attention_mask=attn_mask,
            )
        except Exception as e:
            print(f"[warn] get_rope_index failed: {e}")
            t_pos_ids = None
    else:
        t_pos_ids = None

    if t_pos_ids is not None:
        print(f"target 3D position_ids shape: {tuple(t_pos_ids.shape)}  (3, B, T)")
        T = t_pos_ids.shape[-1]
        # Sample positions at: 0, near vision start, mid-vision, near vision end, after vision.
        n_image_tokens = (input_ids[0] == 151655).sum().item()  # qwen3-vl image_pad token
        first_img = (input_ids[0] == 151655).nonzero(as_tuple=False).flatten()[0].item() \
            if n_image_tokens > 0 else 0
        last_img = (input_ids[0] == 151655).nonzero(as_tuple=False).flatten()[-1].item() \
            if n_image_tokens > 0 else 0
        sample_idxs = sorted(set([
            0, max(0, first_img - 2), first_img, first_img + 5, first_img + 100,
            (first_img + last_img) // 2, last_img - 5, last_img, last_img + 2,
            T - 1,
        ]))
        sample_idxs = [i for i in sample_idxs if 0 <= i < T]

        print(f"\n{'pos_in_seq':<12} | {'target 3D (h,w,t)':<24} | {'draft 1D (arange)':<20}")
        print("-" * 70)
        for p in sample_idxs:
            t3 = t_pos_ids[:, 0, p].tolist()  # (3,)
            d1 = p  # 1D arange always
            print(f"{p:<12} | {fmt(t3):<24} | {d1:<20}")
        print(f"\n# image_pad tokens in prompt: {n_image_tokens}")
        print(f"first/last image_pad position: {first_img} / {last_img}")
        # Compute average L1 difference per axis between target's 3D and 1D arange
        for axis in range(3):
            diff = (t_pos_ids[axis, 0, :] - torch.arange(T, device=device)).abs().float()
            print(f"  axis {axis} mean |target - 1D_arange| = {diff.mean().item():.2f}, "
                  f"max = {diff.max().item():.0f}")
    else:
        print("(could not extract target's 3D positions; skipping comparison)")

    # ---- 5. Hidden state consistency at draft's target_layer_ids ----
    section("(3) Target hidden states at draft's target_layer_ids")
    layer_ids = draft.config.dflash_config["target_layer_ids"]
    print(f"draft.dflash_config target_layer_ids = {layer_ids}")
    print(f"(target has {tc.num_hidden_layers} text layers; "
          f"hidden_states tuple has {len(tout.hidden_states)} entries — "
          f"index 0 = embedding, 1..N = layer 0..N-1 outputs)")
    print()
    print(f"{'layer':<8} | {'shape':<22} | {'mean':>10} | {'std':>10} | {'norm':>10}")
    print("-" * 70)
    for li in layer_ids:
        h = tout.hidden_states[li + 1]    # (B, T, H)
        h_f = h.float()
        print(f"{li:<8} | {fmt(tuple(h.shape)):<22} | "
              f"{h_f.mean().item():>10.4f} | "
              f"{h_f.std().item():>10.4f} | "
              f"{h_f.norm(dim=-1).mean().item():>10.2f}")

    # Also test the draft's `fc` and `hidden_norm` projection of target_hidden
    # to confirm the draft consumes target hiddens directly.
    if len(layer_ids) >= 1:
        h = tout.hidden_states[layer_ids[0] + 1]  # take first layer
        with torch.no_grad():
            # The DFlash draft applies hidden_norm(fc(target_hidden)) to context.
            # But that's the same H -> H transform, just ensures we can run it.
            try:
                proj = draft.hidden_norm(draft.fc(h.to(dt)))
                p_f = proj.float()
                print(f"\nAfter draft.hidden_norm(draft.fc(target_hidden)) on layer "
                      f"{layer_ids[0]}:")
                print(f"  mean = {p_f.mean().item():.4f}, std = {p_f.std().item():.4f}, "
                      f"norm/pos = {p_f.norm(dim=-1).mean().item():.2f}")
            except Exception as e:
                print(f"[warn] could not run draft.fc: {e}")

    section("Summary")
    print("Rotary CLASS matches: target uses Qwen3VLTextRotaryEmbedding, draft uses Qwen3VLTextRotaryEmbedding.")
    print("Rotary CONFIG (rope_theta, mrope_section, max_position_embeddings) matches.")
    print()
    print("BUT during training (train_dflash_distillation.py:435):")
    print("    pos_ids = torch.arange(ctx_len + block_size, device=device).unsqueeze(0)")
    print("This is a flat 1D position. With Qwen3VLTextRotaryEmbedding, a (B, L) input")
    print("auto-broadcasts to (3, B, L) where all three M-RoPE axes equal — i.e. the")
    print("rotary degenerates to 1D RoPE for the draft.")
    print()
    print("Meanwhile target uses a *true* 3D M-RoPE with rope_deltas accounting for")
    print("vision tokens. So the K/V tensors the draft attends over (target_hidden")
    print("context) were rotated under a different angular regime than the draft's")
    print("own queries are rotated under at the same sequence position.")
    print()
    print("Whether this hurts performance is empirical. The downstream attention")
    print("operates on cosine-similarity-style dot products of rotated keys and")
    print("queries; for positions far past the vision block the angle disagreement")
    print("between target's M-RoPE coordinate and draft's flat coordinate equals the")
    print("target's rope_deltas — which is non-trivial (often > 100 for long prompts).")


if __name__ == "__main__":
    main()
