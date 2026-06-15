"""Compare cross-attention statistics between the 1D-baseline and 3D-fix DFlash drafts.

Hypothesis: at output (CoC) positions, the 3D-trained draft puts MORE attention
mass on the vision-token block (because spatially-adjacent vision tokens
cluster in 3D angle-space and look similar -> attention "leaks" into them),
and within the vision block the attention has HIGHER entropy (more spread)
because individual tokens are less distinguishable. Conversely text-only
context positions (system + user instruction + already-generated CoC tokens)
should retain similar or higher attention focus in the 1D regime.

For each draft we:
  1. Build target hidden states for one cached clip
  2. Run a single block at the first CoC position (anchor at <|cot_start|>)
  3. Hook layer 0's Qwen3DFlashAttention to capture q,k just before softmax
  4. Compute attention weights = softmax(q @ k^T / sqrt(d)) for output positions
  5. Split context into [vision_mask, text_mask] using input_ids
  6. Report:
     - mean attention mass on vision vs text  (per output pos, averaged across heads)
     - entropy WITHIN vision attention      (Shannon, over vision positions)
     - entropy WITHIN text attention        (Shannon, over text positions)

Run on one GPU (4-7 free on sharon1). Output: stdout table + JSON.
"""
import argparse, json, math, sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/ubuntu/katana_transfer/code/src")

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.models.dflash_draft import (
    build_dflash_draft_for_qwen3vl,
    extract_context_feature,
    get_qwen3vl_embed_and_head,
    get_target_3d_position_ids,
    load_draft_checkpoint,
)
from alpamayo_r1.models.dflash_draft_mrope import (
    build_dflash_draft_mrope_for_qwen3vl,
    build_dflash_draft_mrope3d_for_qwen3vl,
)
from transformers.cache_utils import DynamicCache

VISION_PAD = 151655   # <|image_pad|>
COT_START = 155677


def shannon(p, dim=-1, eps=1e-12):
    """Shannon entropy along `dim`. p must sum to 1 along dim."""
    return -(p * (p.clamp_min(eps).log())).sum(dim=dim)


@torch.no_grad()
def capture_one_block(target, vlm, draft, embed_tokens, lm_head,
                       full_ids, image_grid_thw, pixel_values,
                       use_3d_mrope, layer_to_hook=0):
    """Run target prefill -> draft one-block forward at first CoC position.
    Returns dict with attn weights shape (n_heads, q_len, k_len) for that layer.
    """
    device = full_ids.device
    block_size = draft.config.block_size
    layer_ids = draft.target_layer_ids
    mask_id = draft.mask_token_id

    # Target prefill
    pkw = dict(input_ids=full_ids, use_cache=True, output_hidden_states=True,
               past_key_values=DynamicCache(), return_dict=True)
    if pixel_values is not None: pkw["pixel_values"] = pixel_values
    if image_grid_thw is not None: pkw["image_grid_thw"] = image_grid_thw
    tout = vlm(**pkw)
    target_hidden = extract_context_feature(tout.hidden_states, layer_ids)  # (1, T, H*4)
    T = target_hidden.shape[1]

    # Find anchor position = first <|cot_start|>.
    anchor_idx = (full_ids[0] == COT_START).nonzero(as_tuple=False).flatten()
    if anchor_idx.numel() == 0:
        raise RuntimeError("no <|cot_start|> in input_ids")
    start = int(anchor_idx[0].item())
    end = start + block_size
    if end > T:
        end = T
        block_size_eff = end - start
    else:
        block_size_eff = block_size

    # Build masked block: anchor (cot_start) + (B-1) MASK tokens.
    block_ids = full_ids[:, start:end].clone()
    block_ids[:, 1:] = mask_id
    noise = embed_tokens(block_ids)               # (1, B, H)
    ctx = target_hidden[:, :start, :]             # (1, ctx_len, H*4)

    # Position ids
    if use_3d_mrope:
        full_3d = get_target_3d_position_ids(target, full_ids,
                                              image_grid_thw=image_grid_thw)
        # extend by block_size for the next-block window
        last = full_3d[:, :, -1:]
        extra = torch.arange(1, block_size + 1, device=device).view(1, 1, -1)
        full_ext = torch.cat([full_3d, last + extra], dim=-1)
        pos = full_ext[:, :, :start + block_size_eff]
    else:
        pos = torch.arange(start + block_size_eff, device=device).unsqueeze(0)

    # Hook layer_to_hook's self_attn to capture q, k pre-softmax.
    captured = {}
    layer = draft.layers[layer_to_hook].self_attn
    orig_forward = layer.forward

    def wrapped(self, hidden_states, target_hidden, position_embeddings,
                attention_mask, **kw):
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len_local = target_hidden.shape[1]
        q = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)        # (B, n_heads, q_len, head_dim)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len_local + q_len, -1, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len_local + q_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)        # (B, n_kv_heads, ctx+q, head_dim)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        from dflash.model import apply_rotary_pos_emb
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        # Manual attention compute (don't pass through SDPA so we get the weights).
        # Repeat KV for GQA.
        if k_rot.shape[1] != q_rot.shape[1]:
            n_rep = q_rot.shape[1] // k_rot.shape[1]
            k_rep = k_rot.repeat_interleave(n_rep, dim=1)
            v_rep = v.repeat_interleave(n_rep, dim=1)
        else:
            k_rep = k_rot; v_rep = v
        scores = torch.matmul(q_rot, k_rep.transpose(-1, -2)) * self.scaling
        if attention_mask is not None:
            scores = scores + attention_mask
        attn = F.softmax(scores.float(), dim=-1)
        captured["attn"] = attn.detach().cpu()    # (B, n_heads, q_len, ctx+q_len)
        captured["ctx_len"] = ctx_len_local
        captured["q_len"] = q_len
        out = torch.matmul(attn.to(v_rep.dtype), v_rep)
        out = out.transpose(1, 2).reshape(bsz, q_len, -1)
        return self.o_proj(out), None

    # Bind wrapped to instance
    import types
    layer.forward = types.MethodType(wrapped, layer)
    try:
        _ = draft(target_hidden=ctx, noise_embedding=noise, position_ids=pos)
    finally:
        layer.forward = orig_forward
    return captured, start, block_size_eff


def analyze(captured, full_ids, ctx_start_pos, vision_pad_id=VISION_PAD):
    """Bin context positions into vision/text and compute attention statistics.

    captured["attn"]: (1, n_heads, q_len, ctx_len + q_len)
    """
    attn = captured["attn"][0]                    # (n_heads, q_len, ctx+q)
    ctx_len = captured["ctx_len"]
    q_len = captured["q_len"]
    n_heads, _, total_kv = attn.shape

    # Context positions are full_ids[0, :ctx_start_pos] (since draft attends
    # over only the committed prefix).
    ctx_ids = full_ids[0, :ctx_len].cpu()
    vision_mask_ctx = (ctx_ids == vision_pad_id)        # (ctx_len,) on CPU
    text_mask_ctx = ~vision_mask_ctx
    n_vis = int(vision_mask_ctx.sum().item())
    n_txt = int(text_mask_ctx.sum().item())

    # Slice the context portion of attention.
    attn_ctx = attn[:, :, :ctx_len]               # (n_heads, q_len, ctx_len)
    # Sum attention mass on vision vs text per (head, q_pos).
    vmass = attn_ctx[..., vision_mask_ctx].sum(dim=-1)   # (n_heads, q_len)
    tmass = attn_ctx[..., text_mask_ctx].sum(dim=-1)
    # (Sanity: vmass + tmass + attention on noise positions ≈ 1.)
    nmass = attn[:, :, ctx_len:].sum(dim=-1)              # noise self-attention

    # Entropy WITHIN vision: renormalize over vision context only.
    eps = 1e-12
    vis_attn = attn_ctx[..., vision_mask_ctx]
    vis_norm = vis_attn / vis_attn.sum(dim=-1, keepdim=True).clamp_min(eps)
    H_vis = shannon(vis_norm, dim=-1)             # (n_heads, q_len)
    H_vis_max = math.log(n_vis) if n_vis > 0 else 1.0

    txt_attn = attn_ctx[..., text_mask_ctx]
    txt_norm = txt_attn / txt_attn.sum(dim=-1, keepdim=True).clamp_min(eps)
    H_txt = shannon(txt_norm, dim=-1)             # (n_heads, q_len)
    H_txt_max = math.log(n_txt) if n_txt > 0 else 1.0

    # Average across heads and across output positions (we evaluate at all
    # masked positions in the block — they all predict CoC tokens).
    return {
        "n_heads": int(n_heads),
        "n_vision_ctx": n_vis,
        "n_text_ctx": n_txt,
        "vision_mass_mean":   float(vmass.mean().item()),
        "text_mass_mean":     float(tmass.mean().item()),
        "noise_mass_mean":    float(nmass.mean().item()),
        "H_vision_mean":      float(H_vis.mean().item()),
        "H_vision_norm":      float((H_vis / H_vis_max).mean().item()),  # in [0,1]
        "H_text_mean":        float(H_txt.mean().item()),
        "H_text_norm":        float((H_txt / H_txt_max).mean().item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_path", required=True)
    ap.add_argument("--draft_1d_path", required=True,
                    help="Warm SFT trained with 1D arange (legacy DFlashDraftMRoPEModel)")
    ap.add_argument("--draft_3d_path", required=True,
                    help="Warm SFT v4 trained with --use_mrope3d_draft (DFlashDraftMRoPE3DModel)")
    ap.add_argument("--clip_path", required=True)
    ap.add_argument("--out_json", default="/tmp/attn_entropy_compare.json")
    args = ap.parse_args()

    device = "cuda"
    dt = torch.bfloat16

    print("Loading target...", flush=True)
    target = AlpamayoR1.from_pretrained(args.target_path, dtype=dt).to(device).eval()
    vlm = target.vlm
    embed_tokens, lm_head = get_qwen3vl_embed_and_head(vlm)

    # Load clip
    d = torch.load(args.clip_path, weights_only=False)
    prompt_ids = d["prompt_input_ids"].to(device)
    output_ids = d["output_token_ids"].to(device)
    full_ids = torch.cat([prompt_ids[0], output_ids], dim=0).unsqueeze(0)
    pixel_values = d.get("pixel_values").to(dt).to(device) if d.get("pixel_values") is not None else None
    image_grid_thw = d.get("image_grid_thw").to(device) if d.get("image_grid_thw") is not None else None
    print(f"clip {Path(args.clip_path).stem}  full_len={full_ids.shape[1]}")

    results = {}

    for label, ckpt_path, use_3d in [
        ("1D-baseline", args.draft_1d_path, False),
        ("3D-v4",       args.draft_3d_path, True),
    ]:
        print(f"\n=== {label} ===", flush=True)
        print(f"Loading from {ckpt_path}")
        ckpt = load_draft_checkpoint(ckpt_path, map_location=device)
        nL = ckpt["num_draft_layers"] or 4
        bsz = ckpt["block_size"] or 16
        mask_id = ckpt["mask_token_id"] or 151662
        if use_3d:
            draft = build_dflash_draft_mrope3d_for_qwen3vl(
                vlm, num_draft_layers=nL, block_size=bsz, mask_token_id=mask_id,
            ).to(dt).to(device).eval()
        else:
            draft = build_dflash_draft_mrope_for_qwen3vl(
                vlm, num_draft_layers=nL, block_size=bsz, mask_token_id=mask_id,
            ).to(dt).to(device).eval()
        draft.load_state_dict(ckpt["state_dict"], strict=False)

        captured, start, bsz_eff = capture_one_block(
            target, vlm, draft, embed_tokens, lm_head,
            full_ids, image_grid_thw, pixel_values,
            use_3d_mrope=use_3d, layer_to_hook=0,
        )
        stats = analyze(captured, full_ids, start)
        results[label] = stats
        print(f"  n_heads={stats['n_heads']} ctx_split: vision={stats['n_vision_ctx']} text={stats['n_text_ctx']}")
        print(f"  attention mass: vision={stats['vision_mass_mean']:.4f} "
              f"text={stats['text_mass_mean']:.4f} noise={stats['noise_mass_mean']:.4f}")
        print(f"  entropy WITHIN vision: H={stats['H_vision_mean']:.3f} "
              f"(normalised by log(n_vis)={stats['H_vision_norm']:.3f})")
        print(f"  entropy WITHIN text:   H={stats['H_text_mean']:.3f} "
              f"(normalised by log(n_txt)={stats['H_text_norm']:.3f})")
        del draft
        torch.cuda.empty_cache()

    print("\n=== Summary (1D-baseline -> 3D-v4) ===")
    a = results["1D-baseline"]; b = results["3D-v4"]
    print(f"  vision attention mass: {a['vision_mass_mean']:.4f} -> {b['vision_mass_mean']:.4f}  "
          f"(Δ {b['vision_mass_mean']-a['vision_mass_mean']:+.4f})")
    print(f"  text   attention mass: {a['text_mass_mean']:.4f} -> {b['text_mass_mean']:.4f}  "
          f"(Δ {b['text_mass_mean']-a['text_mass_mean']:+.4f})")
    print(f"  H WITHIN vision (norm): {a['H_vision_norm']:.3f} -> {b['H_vision_norm']:.3f}  "
          f"(Δ {b['H_vision_norm']-a['H_vision_norm']:+.3f})")
    print(f"  H WITHIN text   (norm): {a['H_text_norm']:.3f} -> {b['H_text_norm']:.3f}  "
          f"(Δ {b['H_text_norm']-a['H_text_norm']:+.3f})")
    print()
    print("Hypothesis predictions for 3D-v4 vs 1D-baseline:")
    print("  - HIGHER vision attention mass (vision tokens cluster in angle-space, "
          "spilling attention)")
    print("  - HIGHER entropy WITHIN vision (within-cluster tokens look similar, "
          "spread softmax)")
    print("  - LOWER attention mass / lower entropy on text (less budget left)")

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON to {args.out_json}")


if __name__ == "__main__":
    main()
