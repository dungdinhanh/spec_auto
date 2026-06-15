"""Two analyses on baseline spec-decode (v6-RM L=2 ep19, FIXED stop-check):

A. Per-position rejection rate.
   For each block position p in 1..B-1, compute P(reject at p | reached p) over
   all iterations. Tells us which positions are structurally hard.

B. Iter-index vs acceptance.
   For each iteration index i within a clip's generation (i=1,2,3,…),
   compute mean accepted_length. Tells us whether later iters within the same
   clip are systematically easier or harder than earlier iters.

Writes a markdown report to claude_report/.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import torch
from transformers import DynamicCache

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper
from alpamayo_r1.models.dflash_draft import (
    build_dflash_draft_for_qwen3vl,
    load_draft_checkpoint,
    build_target_layer_ids,
    get_qwen3vl_embed_and_head,
    extract_context_feature,
)
from dflash.model import sample

sys.path.insert(0, "/home/ubuntu/katana_transfer/code/claude_mod")
from e2e_spec_test import build_inputs  # noqa: E402


@torch.inference_mode()
def run_one_clip(target, draft, inputs, block_size, max_new_tokens, stop_token_ids):
    """Replica of vlm_spec_generate's FIXED loop. Yields per-iter info."""
    device = inputs["input_ids"].device
    input_ids = inputs["input_ids"]
    mask_token_id = draft.mask_token_id

    embed_tokens, lm_head = get_qwen3vl_embed_and_head(target)

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids = torch.full(
        (1, max_length + block_size), mask_token_id, dtype=torch.long, device=device,
    )
    output_ids[:, :num_input_tokens] = input_ids

    past_target = DynamicCache()
    past_draft = DynamicCache()

    prefill_kwargs = dict(
        input_ids=input_ids,
        attention_mask=inputs.get("attention_mask"),
        past_key_values=past_target,
        use_cache=True,
        output_hidden_states=True,
        logits_to_keep=1,
        return_dict=True,
    )
    for k in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
        v = inputs.get(k)
        if v is not None:
            prefill_kwargs[k] = v
    out = target(**prefill_kwargs)
    first_token = sample(out.logits, temperature=0.0)
    output_ids[:, num_input_tokens : num_input_tokens + 1] = first_token
    target_hidden = extract_context_feature(out.hidden_states, draft.target_layer_ids)

    draft_position_ids = torch.arange(
        max_length + block_size, device=device, dtype=torch.long
    ).unsqueeze(0)

    start = num_input_tokens
    iter_idx = 0
    while start < max_length:
        iter_idx += 1
        block_output_ids = output_ids[:, start : start + block_size].clone()
        noise_embedding = embed_tokens(block_output_ids)
        draft_hidden = draft(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=draft_position_ids[
                :, past_draft.get_seq_length() : start + block_size
            ],
            past_key_values=past_draft,
            use_cache=True,
        )
        draft_logits = lm_head(draft_hidden[:, -block_size + 1 :, :])
        past_draft.crop(start)
        block_output_ids[:, 1:] = sample(draft_logits, temperature=0.0)

        cache_position = torch.arange(start, start + block_size, device=device, dtype=torch.long)
        verify_out = target(
            input_ids=block_output_ids,
            past_key_values=past_target,
            cache_position=cache_position,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        posterior = sample(verify_out.logits, temperature=0.0)
        accepted = (block_output_ids[0, 1:] == posterior[0, : block_size - 1]).cumprod(dim=0).sum().item()

        yield {"iter_idx": iter_idx, "accepted_length": int(accepted)}

        output_ids[:, start : start + accepted + 1] = block_output_ids[:, : accepted + 1]
        output_ids[:, start + accepted + 1] = posterior[:, accepted]
        start += accepted + 1
        past_target.crop(start)
        target_hidden = extract_context_feature(verify_out.hidden_states, draft.target_layer_ids)[:, : accepted + 1, :]

        # FIXED stop check: includes the bonus at position `start`.
        if stop_token_ids is not None and any(
            sid in output_ids[0, num_input_tokens : start + 1].tolist() for sid in stop_token_ids
        ):
            break


def main():
    DEFAULT_STOP_TOKENS = [155681, 151645]
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_path", required=True)
    ap.add_argument("--draft_path", required=True)
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--uuids_file", type=str, default=None)
    ap.add_argument("--max_clips", type=int, default=100)
    ap.add_argument("--num_draft_layers", type=int, default=2)
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--mask_token_id", type=int, default=151662)
    ap.add_argument("--num_target_features", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--stop_token_ids", type=int, nargs="+", default=DEFAULT_STOP_TOKENS)
    ap.add_argument("--output_md", type=str, required=True)
    args = ap.parse_args()

    device = "cuda"
    print("Loading target …", flush=True)
    model = AlpamayoR1.from_pretrained(args.target_path, dtype=torch.bfloat16)
    target = model.vlm.to(device).eval()
    tokenizer = model.tokenizer
    processor = helper.get_processor(tokenizer)

    print("Loading draft …", flush=True)
    ckpt = load_draft_checkpoint(args.draft_path, map_location=device)
    mask_id = ckpt["mask_token_id"] if ckpt["mask_token_id"] is not None else args.mask_token_id
    num_layers = ckpt["num_draft_layers"] if ckpt["num_draft_layers"] is not None else args.num_draft_layers
    bsz = ckpt["block_size"] if ckpt["block_size"] is not None else args.block_size

    tlids = None
    if args.num_target_features is not None:
        n_text = target.config.get_text_config().num_hidden_layers
        tlids = build_target_layer_ids(n_text, args.num_target_features)
    draft = build_dflash_draft_for_qwen3vl(
        target, num_draft_layers=num_layers,
        block_size=bsz, mask_token_id=mask_id,
        target_layer_ids=tlids,
    ).to(torch.bfloat16).to(device).eval()
    draft.load_state_dict(ckpt["state_dict"], strict=False)

    all_clips = sorted(glob.glob(os.path.join(args.clips_dir, "*.pt")))
    if args.uuids_file:
        with open(args.uuids_file) as f:
            wanted = set(json.load(f))
        clips = [p for p in all_clips if Path(p).stem in wanted]
    else:
        clips = all_clips
    clips = clips[: args.max_clips]
    print(f"walking {len(clips)} clips", flush=True)

    # Per-iter records: (iter_idx_within_clip, accepted_length).
    records = []
    iters_per_clip = []
    for ci, cf in enumerate(clips):
        try:
            inputs = build_inputs(cf, processor, device)
        except Exception as e:
            print(f"  [{ci+1}] SKIP {Path(cf).stem}: {type(e).__name__}: {e}", flush=True)
            continue
        clip_iters = 0
        for ev in run_one_clip(target, draft, inputs,
                               block_size=bsz, max_new_tokens=args.max_new_tokens,
                               stop_token_ids=args.stop_token_ids):
            records.append((ev["iter_idx"], ev["accepted_length"]))
            clip_iters += 1
        iters_per_clip.append(clip_iters)
        if (ci + 1) % 20 == 0:
            print(f"  [{ci+1}/{len(clips)}] total_iters={len(records)}", flush=True)

    print(f"\nCollected {len(records)} iters across {len(iters_per_clip)} clips", flush=True)

    # ---------- Analysis A: per-position rejection ----------
    # For each position p in 1..B-1, count:
    #   reached_p = number of iters with accepted_length >= p-1  (i.e., draft proposed pos p)
    #               which is: iters with accepted_length >= 0 (always true for p=1)
    #               and accepted_length >= 1 for p=2 (i.e., draft was correct at pos 1)
    #               ...
    #               accepted_length >= p-1 for pos p.
    #   rejected_at_p = number of iters with accepted_length == p-1.
    # P(reject at p | reached p) = rejected_at_p / reached_p.
    B = bsz
    reached = [0] * (B + 1)
    rejected_at = [0] * (B + 1)
    for _, acc in records:
        # Iter "reaches" position p if accepted_length >= p-1.
        for p in range(1, B):
            if acc >= p - 1:
                reached[p] += 1
        # Iter "is rejected at" position p if accepted_length == p-1.
        if 0 <= acc < B - 1:
            rejected_at[acc + 1] += 1
        # If acc == B-1: full accept, no rejection.

    pos_table = []
    for p in range(1, B):
        rch = reached[p]
        rej = rejected_at[p]
        rate = rej / rch if rch > 0 else 0.0
        pos_table.append((p, rch, rej, rate))

    # ---------- Analysis B: iter-index vs accepted_length ----------
    by_iter_idx = {}
    for ii, acc in records:
        by_iter_idx.setdefault(ii, []).append(acc)
    iter_table = []
    for ii in sorted(by_iter_idx):
        vals = by_iter_idx[ii]
        mean_acc = sum(vals) / len(vals)
        full_accept_rate = sum(1 for v in vals if v == B - 1) / len(vals)
        rej_at_pos1 = sum(1 for v in vals if v == 0) / len(vals)
        iter_table.append((ii, len(vals), mean_acc, full_accept_rate, rej_at_pos1))

    # ---------- Write report ----------
    lines = []
    lines.append("# Block-position rejection + iter-index analysis (BASELINE, fixed stop-check)\n")
    lines.append(f"- Draft: `{args.draft_path}`")
    lines.append(f"- Block size B = {bsz} (anchor + {bsz-1} predicted positions)")
    lines.append(f"- Clips analysed: {len(iters_per_clip)}  |  Total iters: {len(records)}")
    lines.append(f"- Iters per clip: mean={sum(iters_per_clip)/max(len(iters_per_clip),1):.2f}, "
                 f"min={min(iters_per_clip) if iters_per_clip else 0}, "
                 f"max={max(iters_per_clip) if iters_per_clip else 0}\n")

    lines.append("## A. Per-position rejection rate")
    lines.append("`P(reject at pos p | reached pos p)` — how likely the draft was wrong AT position p, "
                 "given the iteration reached it (= all previous positions accepted).\n")
    lines.append("| pos | reached | rejected | rate |")
    lines.append("|---:|---:|---:|---:|")
    for p, rch, rej, rate in pos_table:
        lines.append(f"| {p} | {rch} | {rej} | {rate:.4f} |")
    full_accepts = sum(1 for _, acc in records if acc == B - 1)
    lines.append(f"\n**Full-accept rate (no rejection in block)**: {full_accepts}/{len(records)} "
                 f"= {full_accepts/max(len(records),1):.4f}\n")

    lines.append("## B. Iter-index within clip vs acceptance")
    lines.append("For each iter index `i` (= the i-th spec iteration within a single clip's generation), "
                 "we report how many clips had at least that many iters, the mean accepted_length at "
                 "that iter, the full-accept rate, and the pos-1 rejection rate.\n")
    lines.append("| iter_idx | n_clips_reached | mean_accepted | full_accept_rate | pos1_reject_rate |")
    lines.append("|---:|---:|---:|---:|---:|")
    for ii, n, ma, fa, rj1 in iter_table:
        lines.append(f"| {ii} | {n} | {ma:.3f} | {fa:.4f} | {rj1:.4f} |")

    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text("\n".join(lines))
    print(f"Wrote report to {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
