"""Acceptance RATE analysis: normalize each iter's accepted_length by the
max possible given the remaining CoC budget for that clip.

For each iter i within a clip:
  tokens_emitted_i = accepted_i + 1  (k draft accepts + 1 target bonus)
  remaining_before_i = total_emitted_for_clip - sum(tokens_emitted_j for j<i)
  possible_i = min(B-1, remaining_before_i - 1)
      (the iter must emit accepted+1 ≤ remaining, so accepted ≤ remaining-1)
  rate_i = accepted_i / possible_i   (undefined if possible_i == 0)

Aggregations:
  A. By iter_idx FROM START  (1, 2, 3, ...): does later iters become harder?
  B. By iter_idx FROM END   (-1, -2, -3, ...): the last iter is intrinsically
     the "stop emission" iter and might be uniquely hard.
  C. Per-position rejection (unchanged from earlier analysis, for reference).

Writes a markdown report.
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
        use_cache=True, output_hidden_states=True,
        logits_to_keep=1, return_dict=True,
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
            target_hidden=target_hidden, noise_embedding=noise_embedding,
            position_ids=draft_position_ids[:, past_draft.get_seq_length() : start + block_size],
            past_key_values=past_draft, use_cache=True,
        )
        draft_logits = lm_head(draft_hidden[:, -block_size + 1 :, :])
        past_draft.crop(start)
        block_output_ids[:, 1:] = sample(draft_logits, temperature=0.0)
        cache_position = torch.arange(start, start + block_size, device=device, dtype=torch.long)
        verify_out = target(
            input_ids=block_output_ids, past_key_values=past_target,
            cache_position=cache_position, use_cache=True,
            output_hidden_states=True, return_dict=True,
        )
        posterior = sample(verify_out.logits, temperature=0.0)
        accepted = (block_output_ids[0, 1:] == posterior[0, : block_size - 1]).cumprod(dim=0).sum().item()
        yield {"iter_idx": iter_idx, "accepted_length": int(accepted)}
        output_ids[:, start : start + accepted + 1] = block_output_ids[:, : accepted + 1]
        output_ids[:, start + accepted + 1] = posterior[:, accepted]
        start += accepted + 1
        past_target.crop(start)
        target_hidden = extract_context_feature(verify_out.hidden_states, draft.target_layer_ids)[:, : accepted + 1, :]
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
    ap.add_argument("--max_clips", type=int, default=150)
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

    # Per-clip list of iter records: each iter has (iter_idx, accepted_length).
    per_clip: list[list[dict]] = []
    for ci, cf in enumerate(clips):
        try:
            inputs = build_inputs(cf, processor, device)
        except Exception as e:
            print(f"  [{ci+1}] SKIP {Path(cf).stem}: {type(e).__name__}: {e}", flush=True)
            continue
        clip_records = list(run_one_clip(
            target, draft, inputs,
            block_size=bsz, max_new_tokens=args.max_new_tokens,
            stop_token_ids=args.stop_token_ids,
        ))
        per_clip.append(clip_records)
        if (ci + 1) % 25 == 0:
            print(f"  [{ci+1}/{len(clips)}] clips processed", flush=True)

    B = bsz
    total_iters = sum(len(c) for c in per_clip)
    print(f"\nCollected {total_iters} iters across {len(per_clip)} clips", flush=True)

    # ---------- enrich each iter with possible_i, remaining_before_i, tokens_emitted ----------
    enriched = []  # list of dicts across all iters
    for cidx, recs in enumerate(per_clip):
        n_iters = len(recs)
        total_emitted = sum(r["accepted_length"] + 1 for r in recs)
        cum = 0
        for i, r in enumerate(recs):
            acc = r["accepted_length"]
            emitted = acc + 1
            remaining_before = total_emitted - cum
            possible = min(B - 1, remaining_before - 1)  # accepted ≤ remaining - 1
            possible = max(0, possible)
            enriched.append({
                "clip_idx": cidx,
                "iter_idx_fwd": i + 1,                # 1, 2, 3, ...
                "iter_idx_rev": n_iters - i,         # for last iter = 1, second-last = 2, ...
                "accepted": acc,
                "emitted": emitted,
                "remaining_before": remaining_before,
                "possible": possible,
                "n_iters_this_clip": n_iters,
            })
            cum += emitted

    # ---------- Aggregation A: by iter_idx FROM START ----------
    by_fwd: dict[int, list] = {}
    for e in enriched:
        by_fwd.setdefault(e["iter_idx_fwd"], []).append(e)
    fwd_table = []
    for ii in sorted(by_fwd):
        evs = by_fwd[ii]
        sum_acc = sum(e["accepted"] for e in evs)
        sum_pos = sum(e["possible"] for e in evs)
        mean_acc = sum_acc / len(evs)
        mean_pos = sum_pos / len(evs)
        rate = sum_acc / sum_pos if sum_pos > 0 else 0.0
        n_skipped = sum(1 for e in evs if e["possible"] == 0)
        fwd_table.append((ii, len(evs), mean_acc, mean_pos, rate, n_skipped))

    # ---------- Aggregation B: by iter_idx FROM END ----------
    by_rev: dict[int, list] = {}
    for e in enriched:
        by_rev.setdefault(e["iter_idx_rev"], []).append(e)
    rev_table = []
    for ii in sorted(by_rev):
        evs = by_rev[ii]
        sum_acc = sum(e["accepted"] for e in evs)
        sum_pos = sum(e["possible"] for e in evs)
        mean_acc = sum_acc / len(evs)
        mean_pos = sum_pos / len(evs)
        rate = sum_acc / sum_pos if sum_pos > 0 else 0.0
        n_skipped = sum(1 for e in evs if e["possible"] == 0)
        rev_table.append((ii, len(evs), mean_acc, mean_pos, rate, n_skipped))

    # ---------- Per-position rejection (rate-style: only count iters that
    # COULD have reached pos p, i.e., possible >= p) ----------
    pos_reached = [0] * (B + 1)
    pos_rejected = [0] * (B + 1)
    for e in enriched:
        for p in range(1, B):
            # Iter could reach pos p only if it had budget for at least p positions.
            if e["possible"] < p:
                continue
            # Iter reached pos p if accepted >= p - 1 (i.e., draft was right at all p-1 prior positions).
            if e["accepted"] >= p - 1:
                pos_reached[p] += 1
                if e["accepted"] == p - 1:
                    pos_rejected[p] += 1

    pos_table = []
    for p in range(1, B):
        rch, rej = pos_reached[p], pos_rejected[p]
        rate = rej / rch if rch > 0 else 0.0
        pos_table.append((p, rch, rej, rate))

    # ---------- Write report ----------
    lines = []
    lines.append("# Block acceptance-rate analysis (BASELINE, fixed stop-check)\n")
    lines.append(f"- Draft: `{args.draft_path}`")
    lines.append(f"- Block size B = {bsz}; max draft positions per iter = {bsz-1}")
    lines.append(f"- Clips: {len(per_clip)}  |  total iters: {total_iters}\n")

    lines.append("Each iter is normalised by `possible = min(B-1, remaining_CoC_tokens - 1)`. "
                 "The previous (un-normalised) analysis was biased by clips running out of CoC budget; "
                 "this version reports `accepted / possible` so a draft that perfectly emits the last "
                 "3 tokens of CoC counts the same as a draft that perfectly emits 15.\n")

    lines.append("## A. Iter-index FROM START (forward)")
    lines.append("`mean_accepted` and `mean_possible` are per-iter averages; `rate = Σaccepted / Σpossible` "
                 "is the bias-corrected acceptance rate. `n_skipped` iters had possible=0 (only the bonus left).\n")
    lines.append("| iter_idx | n_iters | mean_accepted | mean_possible | **rate** | n_skipped |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for ii, n, ma, mp, rate, sk in fwd_table:
        lines.append(f"| {ii} | {n} | {ma:.3f} | {mp:.3f} | **{rate:.4f}** | {sk} |")

    lines.append("\n## B. Iter-index FROM END (reverse)")
    lines.append("`iter_idx_rev = 1` means the LAST iter of the clip's generation, `2` the second-to-last, etc.\n")
    lines.append("| iter_from_end | n_iters | mean_accepted | mean_possible | **rate** | n_skipped |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for ii, n, ma, mp, rate, sk in rev_table:
        lines.append(f"| {ii} | {n} | {ma:.3f} | {mp:.3f} | **{rate:.4f}** | {sk} |")

    lines.append("\n## C. Per-position rejection (only iters with budget for that position)")
    lines.append("`P(reject at pos p | reached pos p AND possible >= p)` — only count iters that COULD "
                 "have reached pos p given their remaining-CoC budget. Removes the 'iter near end of clip "
                 "couldn't have reached pos 10 anyway' bias.\n")
    lines.append("| pos | reached | rejected | rate |")
    lines.append("|---:|---:|---:|---:|")
    for p, rch, rej, rate in pos_table:
        lines.append(f"| {p} | {rch} | {rej} | {rate:.4f} |")

    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text("\n".join(lines))
    print(f"Wrote report to {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
