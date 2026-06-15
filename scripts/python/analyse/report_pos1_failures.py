"""Capture cases where the BASELINE spec-decode rejects at position 1.

For each such iteration we record:
  - The anchor token (last verified token, conditioning the rejected proposal)
  - The draft's full proposed block (positions 1..B-1, decoded)
  - The target's argmax block (positions 1..B-1, decoded) under that anchor
  - Which positions matched (cumulative, until first mismatch)

Run target=baseline T=1 vlm_spec_generate semantics (no refinement, no first_ar)
on a small number of clips, stop after collecting N cases, write a markdown
report to claude_report/.

Usage example:
  python scripts/python/analyse/report_pos1_failures.py \
    --target_path /home/ubuntu/local_data/models/Alpamayo-R1-10B \
    --draft_path  /home/ubuntu/local_data/runs/dflash_L2_lr1e-4_ep50_bs16_warm_v6_randomMask_sharon1/draft_epoch_19.pt \
    --clips_dir /home/ubuntu/local_data/runs/target_coc_outputs \
    --uuids_file /home/ubuntu/katana_transfer/splits/val_uuids_v3.json \
    --num_draft_layers 2 --block_size 16 --num_target_features 5 \
    --max_cases 30 --max_clips 12 \
    --output_md /home/ubuntu/local_data/runs/_pos1_failures.md
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

# We import indirectly to avoid path issues — assume PYTHONPATH is set as in e2e_spec_test.
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper
from alpamayo_r1.models.dflash_draft import (
    build_dflash_draft_for_qwen3vl,
    load_dflash_weights,
    load_draft_checkpoint,
    build_target_layer_ids,
    get_qwen3vl_embed_and_head,
    extract_context_feature,
)
from dflash.model import sample

# Reuse build_inputs from e2e_spec_test to keep prompt construction identical.
sys.path.insert(0, "/home/ubuntu/katana_transfer/code/claude_mod")
from e2e_spec_test import build_inputs  # noqa: E402


@torch.inference_mode()
def run_one_clip(target, draft, inputs, processor, tokenizer, block_size, max_new_tokens, stop_token_ids):
    """Replica of vlm_spec_generate's loop with per-iter intercepts.

    Yields dicts {anchor, draft_block, target_block, accepted_length} for every
    iteration. The CALLER filters for accepted_length == 0 (first-token failures).
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
    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        anchor_id = block_output_ids[0, 0].item()
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
        draft_block_ids = sample(draft_logits, temperature=0.0)[0]  # (B-1,)
        block_output_ids[:, 1:] = draft_block_ids.unsqueeze(0)

        cache_position = torch.arange(start, start + block_size, device=device, dtype=torch.long)
        verify_out = target(
            input_ids=block_output_ids,
            past_key_values=past_target,
            cache_position=cache_position,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        posterior = sample(verify_out.logits, temperature=0.0)[0]  # (B,)
        # posterior[i] = target argmax predicting position start+i+1 given prefix up to start+i.
        target_block_ids = posterior[: block_size - 1]  # (B-1,)  predictions for positions 1..B-1

        # Acceptance length: longest matching prefix between draft and target argmax.
        accepted = (block_output_ids[0, 1:] == posterior[: block_size - 1]).cumprod(dim=0).sum().item()

        yield {
            "anchor_id": anchor_id,
            "draft_block_ids": draft_block_ids.detach().cpu().tolist(),
            "target_block_ids": target_block_ids.detach().cpu().tolist(),
            "accepted_length": int(accepted),
            "start_pos": start,
        }

        output_ids[:, start : start + accepted + 1] = block_output_ids[:, : accepted + 1]
        output_ids[:, start + accepted + 1] = posterior[accepted]
        start += accepted + 1

        past_target.crop(start)
        target_hidden = extract_context_feature(
            verify_out.hidden_states, draft.target_layer_ids
        )[:, : accepted + 1, :]

        if stop_token_ids is not None and any(
            sid in output_ids[0, num_input_tokens:start + 1].tolist() for sid in stop_token_ids
        ):
            break


def decode_each(tokenizer, ids: list[int]) -> list[str]:
    """Decode each id individually so we can lay them out side-by-side."""
    return [tokenizer.decode([t]) if t >= 0 else "<oob>" for t in ids]


def decode_run(tokenizer, ids: list[int]) -> str:
    """Decode a sequence of ids as a continuous string."""
    return tokenizer.decode(ids, skip_special_tokens=False)


def main():
    DEFAULT_STOP_TOKENS = [155681, 151645]  # <|traj_future_start|>, <|im_end|>
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_path", required=True)
    ap.add_argument("--draft_path", required=True)
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--uuids_file", type=str, default=None)
    ap.add_argument("--offset", type=int, default=9000)
    ap.add_argument("--num_clips", type=int, default=50)
    ap.add_argument("--max_clips", type=int, default=10, help="walk through at most this many clips")
    ap.add_argument("--max_cases", type=int, default=30, help="stop after collecting this many first-token failures")
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
        clips = all_clips[args.offset : args.offset + args.num_clips]
    clips = clips[: args.max_clips]
    print(f"walking {len(clips)} clips, collecting up to {args.max_cases} first-token failures", flush=True)

    cases = []
    for ci, cf in enumerate(clips):
        if len(cases) >= args.max_cases:
            break
        try:
            inputs = build_inputs(cf, processor, device)
        except Exception as e:
            print(f"  [{ci+1}] SKIP {Path(cf).stem}: {type(e).__name__}: {e}", flush=True)
            continue
        clip_id = Path(cf).stem
        iter_idx = 0
        for ev in run_one_clip(
            target, draft, inputs, processor, tokenizer,
            block_size=bsz, max_new_tokens=args.max_new_tokens,
            stop_token_ids=args.stop_token_ids,
        ):
            iter_idx += 1
            if ev["accepted_length"] == 0:
                cases.append({
                    "clip": clip_id,
                    "iter": iter_idx,
                    "start_pos": ev["start_pos"],
                    "anchor_id": ev["anchor_id"],
                    "draft_block_ids": ev["draft_block_ids"],
                    "target_block_ids": ev["target_block_ids"],
                })
                if len(cases) >= args.max_cases:
                    break
        print(f"  [{ci+1}/{len(clips)}] {clip_id}: total iters={iter_idx} cumulative_cases={len(cases)}", flush=True)

    print(f"\nCollected {len(cases)} cases. Writing report to {args.output_md}", flush=True)

    lines = []
    lines.append("# Spec-decode first-token failures (baseline, v6-RM L=2 ep19)\n")
    lines.append(f"- Draft ckpt: `{args.draft_path}`")
    lines.append(f"- Block size: {bsz} (anchor + {bsz-1} predicted positions)")
    lines.append(f"- Clips walked: {len(clips)}, cases collected: {len(cases)}\n")
    lines.append("**Each case shows the anchor (last verified token), the draft's full proposed block "
                 "(positions 1..B-1) and the target's argmax block under the same anchor. The first "
                 "draft position MISMATCHED the target — so the spec iteration ended here, "
                 "emitting only the bonus = target[1].**\n")
    for ci, c in enumerate(cases, 1):
        anchor_txt = tokenizer.decode([c["anchor_id"]]) if c["anchor_id"] >= 0 else "<oob>"
        draft_each = decode_each(tokenizer, c["draft_block_ids"])
        target_each = decode_each(tokenizer, c["target_block_ids"])
        draft_run = decode_run(tokenizer, c["draft_block_ids"])
        target_run = decode_run(tokenizer, c["target_block_ids"])
        match_marks = ["✓" if d == t else "✗" for d, t in zip(c["draft_block_ids"], c["target_block_ids"])]
        lines.append(f"## Case {ci} — clip `{c['clip']}` iter {c['iter']} (start_pos={c['start_pos']})\n")
        lines.append(f"- Anchor token (id={c['anchor_id']}): `{anchor_txt!r}`")
        lines.append(f"- Draft block (decoded as run): `{draft_run!r}`")
        lines.append(f"- Target block (decoded as run): `{target_run!r}`\n")
        lines.append("| pos | draft id | draft tok | target id | target tok | match |")
        lines.append("|---:|---:|---|---:|---|:-:|")
        for p, (di, ti, dt, tt, m) in enumerate(zip(c["draft_block_ids"], c["target_block_ids"], draft_each, target_each, match_marks), 1):
            lines.append(f"| {p} | {di} | `{dt!r}` | {ti} | `{tt!r}` | {m} |")
        lines.append("")

    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text("\n".join(lines))
    print(f"Wrote {len(lines)} lines to {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
