"""For each clip, capture the LAST spec-decode iter in full detail and write
a markdown report. Per-clip we record:
  - iter_idx (= n_iters in this clip's generation)
  - anchor token (id + decoded)
  - draft's full B-1 token block (ids + decoded)
  - target's argmax B-1 token block (ids + decoded) computed at verify time
  - accepted_length, possible (= min(B-1, remaining-1)), rate
  - per-position match marks
The report is sorted by rate descending (rate=1.0 cases first) so it's easy
to scan how the rate=1.0 cases actually look.
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
    """Same FIXED loop as analyse_block_acceptance_rate; yields per-iter detail
    including the decoded blocks for each iter.
    """
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
        anchor_id = block_output_ids[0, 0].item()
        noise_embedding = embed_tokens(block_output_ids)
        draft_hidden = draft(
            target_hidden=target_hidden, noise_embedding=noise_embedding,
            position_ids=draft_position_ids[:, past_draft.get_seq_length() : start + block_size],
            past_key_values=past_draft, use_cache=True,
        )
        draft_logits = lm_head(draft_hidden[:, -block_size + 1 :, :])
        past_draft.crop(start)
        draft_block_ids = sample(draft_logits, temperature=0.0)[0]
        block_output_ids[:, 1:] = draft_block_ids.unsqueeze(0)

        cache_position = torch.arange(start, start + block_size, device=device, dtype=torch.long)
        verify_out = target(
            input_ids=block_output_ids, past_key_values=past_target,
            cache_position=cache_position, use_cache=True,
            output_hidden_states=True, return_dict=True,
        )
        posterior = sample(verify_out.logits, temperature=0.0)[0]
        target_block_ids = posterior[: block_size - 1]
        accepted = (block_output_ids[0, 1:] == posterior[: block_size - 1]).cumprod(dim=0).sum().item()

        yield {
            "iter_idx": iter_idx,
            "start_pos": start,
            "anchor_id": anchor_id,
            "draft_block_ids": draft_block_ids.detach().cpu().tolist(),
            "target_block_ids": target_block_ids.detach().cpu().tolist(),
            "accepted_length": int(accepted),
            "bonus_id": int(posterior[accepted].item()),
        }
        output_ids[:, start : start + accepted + 1] = block_output_ids[:, : accepted + 1]
        output_ids[:, start + accepted + 1] = posterior[accepted]
        start += accepted + 1
        past_target.crop(start)
        target_hidden = extract_context_feature(verify_out.hidden_states, draft.target_layer_ids)[:, : accepted + 1, :]
        if stop_token_ids is not None and any(
            sid in output_ids[0, num_input_tokens : start + 1].tolist() for sid in stop_token_ids
        ):
            break


def decode_each(tokenizer, ids):
    return [tokenizer.decode([t]) if t >= 0 else "<oob>" for t in ids]


def decode_run(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=False)


def main():
    DEFAULT_STOP_TOKENS = [155681, 151645]
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_path", required=True)
    ap.add_argument("--draft_path", required=True)
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--uuids_file", type=str, default=None)
    ap.add_argument("--max_clips", type=int, default=200)
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

    B = bsz
    last_iters = []
    for ci, cf in enumerate(clips):
        clip_id = Path(cf).stem
        try:
            inputs = build_inputs(cf, processor, device)
        except Exception as e:
            print(f"  [{ci+1}] SKIP {clip_id}: {type(e).__name__}: {e}", flush=True)
            continue
        recs = list(run_one_clip(
            target, draft, inputs,
            block_size=bsz, max_new_tokens=args.max_new_tokens,
            stop_token_ids=args.stop_token_ids,
        ))
        if not recs:
            continue
        n_iters = len(recs)
        total_emitted = sum(r["accepted_length"] + 1 for r in recs)
        cum = sum(r["accepted_length"] + 1 for r in recs[:-1])
        last = recs[-1]
        remaining = total_emitted - cum
        possible = max(0, min(B - 1, remaining - 1))
        rate = last["accepted_length"] / possible if possible > 0 else float("nan")
        last_iters.append({
            "clip": clip_id,
            "n_iters": n_iters,
            "remaining": remaining,
            "possible": possible,
            "accepted": last["accepted_length"],
            "rate": rate,
            "anchor_id": last["anchor_id"],
            "draft_block_ids": last["draft_block_ids"],
            "target_block_ids": last["target_block_ids"],
            "bonus_id": last["bonus_id"],
        })
        if (ci + 1) % 25 == 0:
            print(f"  [{ci+1}/{len(clips)}] last_iters collected={len(last_iters)}", flush=True)

    # Sort by rate desc (NaN goes last), then by accepted desc.
    def sort_key(c):
        r = c["rate"]
        r_key = -1e9 if r != r else r  # NaN treated as very low so it sorts last (with reverse=True)
        return (r_key, c["accepted"])
    last_iters.sort(key=sort_key, reverse=True)

    # Summary stats
    n_total = len(last_iters)
    n_skipped = sum(1 for c in last_iters if c["possible"] == 0)
    n_rate1 = sum(1 for c in last_iters if c["possible"] > 0 and c["rate"] == 1.0)
    n_rate0 = sum(1 for c in last_iters if c["possible"] > 0 and c["rate"] == 0.0)
    n_other = n_total - n_skipped - n_rate1 - n_rate0
    rates = [c["rate"] for c in last_iters if c["possible"] > 0]
    mean_rate = sum(rates) / max(len(rates), 1)

    lines = []
    lines.append("# Last-iter detail report (BASELINE, fixed stop-check, v6-RM L=2 ep19)\n")
    lines.append(f"- Block size B = {bsz} (anchor + {bsz-1} draftable positions)")
    lines.append(f"- Clips analysed: {n_total}\n")
    lines.append("## Summary\n")
    lines.append(f"- Total clips with at least one iter: **{n_total}**")
    lines.append(f"- Last-iter `possible == 0` (only the bonus left, no draft positions to evaluate): **{n_skipped}**  ({100*n_skipped/n_total:.1f}%)")
    lines.append(f"- Among the {n_total - n_skipped} non-skipped last iters:")
    lines.append(f"  - **rate = 1.0** (draft perfectly nailed all `possible` positions): **{n_rate1}**  ({100*n_rate1/max(n_total - n_skipped,1):.1f}%)")
    lines.append(f"  - **rate = 0.0** (draft missed even at pos 1): **{n_rate0}**  ({100*n_rate0/max(n_total - n_skipped,1):.1f}%)")
    lines.append(f"  - in-between: **{n_other}**")
    lines.append(f"- Mean rate over non-skipped last iters: **{mean_rate:.3f}**\n")

    lines.append("## Per-clip detail (sorted by rate desc)\n")
    lines.append("Convention:")
    lines.append("- `possible` = `min(B-1, remaining_CoC_tokens - 1)`; if 0, the iter only had room for the bonus token.")
    lines.append("- `accepted` = number of draft tokens that matched target's argmax.")
    lines.append("- `rate` = `accepted / possible`.")
    lines.append("- The draft block is what the draft proposed; the target block is what target's argmax was at the SAME positions during the verify forward.\n")

    for idx, c in enumerate(last_iters, 1):
        anchor = tokenizer.decode([c["anchor_id"]]) if c["anchor_id"] >= 0 else "<oob>"
        bonus = tokenizer.decode([c["bonus_id"]]) if c["bonus_id"] >= 0 else "<oob>"
        rate_str = "N/A (possible=0)" if c["possible"] == 0 else f"{c['rate']:.3f}"
        draft_run = decode_run(tokenizer, c["draft_block_ids"])
        target_run = decode_run(tokenizer, c["target_block_ids"])
        lines.append(f"### #{idx}  clip `{c['clip']}`  (n_iters={c['n_iters']}, last iter)")
        lines.append(f"- remaining = {c['remaining']},  possible = {c['possible']},  accepted = {c['accepted']},  rate = {rate_str}")
        lines.append(f"- anchor (id={c['anchor_id']}): `{anchor!r}`    bonus (id={c['bonus_id']}): `{bonus!r}`")
        lines.append(f"- draft block (run):  `{draft_run!r}`")
        lines.append(f"- target block (run): `{target_run!r}`")
        # Per-position table only for the FIRST few positions (= possible + 1) to keep the report scannable
        n_show = max(c["possible"], 3) if c["possible"] > 0 else min(5, len(c["draft_block_ids"]))
        n_show = min(n_show, len(c["draft_block_ids"]))
        de = decode_each(tokenizer, c["draft_block_ids"][:n_show])
        te = decode_each(tokenizer, c["target_block_ids"][:n_show])
        lines.append("")
        lines.append("| pos | draft id | draft tok | target id | target tok | match |")
        lines.append("|---:|---:|---|---:|---|:-:|")
        for p in range(n_show):
            di = c["draft_block_ids"][p]
            ti = c["target_block_ids"][p]
            m = "✓" if di == ti else "✗"
            lines.append(f"| {p+1} | {di} | `{de[p]!r}` | {ti} | `{te[p]!r}` | {m} |")
        lines.append("")

    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text("\n".join(lines))
    print(f"Wrote report to {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
