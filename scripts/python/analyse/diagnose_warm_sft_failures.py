"""Failure-case analysis for warm SFT draft on training clips.

For each (clip, block_start), compute greedy predictions and check:
  - SUCCESS = greedy matches GT at every gt_available position (acc_rate=1.0)
  - FAILURE = greedy fails at some position; record first-failure position

Aggregate:
  1. Success rate by gt_avail bucket → answers "does it succeed on short blocks?"
  2. Success rate by block_start → answers "does it fail more often at b > 0?"
  3. First-failure-position distribution among failing blocks
  4. Joint (block_start, gt_avail) → success rate matrix
"""
import argparse, glob, json, os, sys
from pathlib import Path
import torch

sys.path.insert(0, "/home/ubuntu/alpamayo_code/src")
sys.path.insert(0, "/home/ubuntu/dflash_code")
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.models.dflash_draft import (
    build_dflash_draft_for_qwen3vl, get_qwen3vl_embed_and_head,
    extract_context_feature, load_draft_checkpoint,
)
from transformers.cache_utils import DynamicCache


@torch.no_grad()
def measure(target, draft, embed_tokens, lm_head, files, block_size, device):
    mask_id = draft.mask_token_id
    B = block_size
    layer_ids = draft.target_layer_ids

    # Per-(clip, block_start) records
    rows = []   # list of (block_start, gt_avail, success, first_fail_pos)

    for fi, f in enumerate(files):
        d = torch.load(f, weights_only=False)
        prompt_ids = d["prompt_input_ids"].to(device)
        output_ids = d["output_token_ids"].to(device)
        pv = d.get("pixel_values")
        if pv is not None:
            pv = pv.to(next(target.parameters()).dtype).to(device)
        igt = d.get("image_grid_thw").to(device) if d.get("image_grid_thw") is not None else None
        P = prompt_ids.shape[1]
        N = output_ids.shape[0]
        if N < 2: continue
        full_ids = torch.cat([prompt_ids[0], output_ids], dim=0).unsqueeze(0)

        # Prefill target
        cache_T = DynamicCache()
        kw = dict(input_ids=full_ids, past_key_values=cache_T,
                  use_cache=True, output_hidden_states=True, return_dict=True)
        if pv is not None: kw["pixel_values"] = pv
        if igt is not None: kw["image_grid_thw"] = igt
        po = target(**kw)
        target_hidden = extract_context_feature(po.hidden_states, layer_ids)

        for block_start in range(0, N - 1):
            gt_av = min(B - 1, N - block_start - 1)
            if gt_av <= 0: break
            anchor = full_ids[0, P + block_start].item()
            ctx_len = P + block_start
            ctx_hidden = target_hidden[:, :ctx_len, :]

            noise_ids = torch.full((1, B), mask_id, dtype=full_ids.dtype, device=device)
            noise_ids[:, 0] = anchor
            noise_emb = embed_tokens(noise_ids)
            pos_ids = torch.arange(ctx_len + B, device=device).unsqueeze(0)
            dh = draft(target_hidden=ctx_hidden, noise_embedding=noise_emb,
                       position_ids=pos_ids)
            dl = lm_head(dh[:, -(B - 1):, :])
            argmax = dl[0, :gt_av].argmax(dim=-1)
            gt = full_ids[0, P + block_start + 1:P + block_start + 1 + gt_av]

            # find first failure
            first_fail = -1
            for p in range(gt_av):
                if argmax[p].item() != gt[p].item():
                    first_fail = p
                    break
            success = (first_fail == -1)
            rows.append((block_start, gt_av, success, first_fail if not success else gt_av))

        if (fi + 1) % 20 == 0:
            print(f"  {fi+1}/{len(files)} clips processed", flush=True)

    return rows


def report(rows):
    n = len(rows)
    print(f"\n=== Total (clip, block_start) records: {n} ===")
    n_success = sum(1 for _, _, s, _ in rows if s)
    n_fail = n - n_success
    print(f"  Success (matched all gt_avail): {n_success} ({100*n_success/n:.2f}%)")
    print(f"  Failure (at least one mismatch): {n_fail} ({100*n_fail/n:.2f}%)")

    # 1. By gt_avail bucket
    buckets_gt = [(1, 3), (4, 7), (8, 11), (12, 15)]
    print(f"\n=== Success rate by gt_avail bucket ===")
    print(f"  {'bucket':<14} | {'n_records':>10} | {'success':>10} | {'success rate':>14}")
    for lo, hi in buckets_gt:
        sub = [r for r in rows if lo <= r[1] <= hi]
        if sub:
            ns = sum(1 for r in sub if r[2])
            print(f"  gt_avail [{lo:>2}, {hi:>2}] | {len(sub):>10} | {ns:>10} | {100*ns/len(sub):>13.2f}%")

    # 2. By block_start bucket
    buckets_b = [(0, 0), (1, 2), (3, 5), (6, 9), (10, 99)]
    print(f"\n=== Success rate by block_start ===")
    print(f"  {'bucket':<16} | {'n_records':>10} | {'success':>10} | {'success rate':>14}")
    for lo, hi in buckets_b:
        sub = [r for r in rows if lo <= r[0] <= hi]
        if sub:
            ns = sum(1 for r in sub if r[2])
            label = f"b={lo}" if lo == hi else f"b∈[{lo},{hi if hi<99 else '∞'}]"
            print(f"  {label:<16} | {len(sub):>10} | {ns:>10} | {100*ns/len(sub):>13.2f}%")

    # 3. First-failure-position distribution among failures only
    print(f"\n=== First failure position (among {n_fail} failing records) ===")
    print(f"  {'fail position p':<18} | {'count':>8} | {'pct':>7}")
    fail_pos_counts = [0] * 16
    for _, gt_av, s, fp in rows:
        if not s:
            fail_pos_counts[fp] += 1
    cum = 0
    for p in range(16):
        if fail_pos_counts[p] > 0:
            cum += fail_pos_counts[p]
            print(f"  fail at p = {p:>3}     | {fail_pos_counts[p]:>8} | {100*fail_pos_counts[p]/n_fail:>6.2f}% (cumul {100*cum/n_fail:.2f}%)")

    # 4. Joint (block_start, gt_avail) success rate
    print(f"\n=== Joint (block_start, gt_avail) success rate ===")
    print(f"  {'gt_avail':<12} ", end="")
    bs_buckets = [(0, 0), (1, 2), (3, 5), (6, 9), (10, 99)]
    for lo, hi in bs_buckets:
        label = f"b={lo}" if lo == hi else f"b∈[{lo},{hi if hi<99 else '∞'}]"
        print(f"| {label:<10} ", end="")
    print()
    for gt_lo, gt_hi in buckets_gt:
        print(f"  gt∈[{gt_lo:>2},{gt_hi:>2}]   ", end="")
        for bs_lo, bs_hi in bs_buckets:
            sub = [r for r in rows if gt_lo <= r[1] <= gt_hi and bs_lo <= r[0] <= bs_hi]
            if sub:
                ns = sum(1 for r in sub if r[2])
                cell = f"{100*ns/len(sub):.0f}% (n={len(sub)})"
                print(f"| {cell:<10} ", end="")
            else:
                print(f"| {'—':<10} ", end="")
        print()

    # 5. Highlight the user's specific questions
    print(f"\n=== Direct answers ===")
    short_blocks = [r for r in rows if r[1] <= 5]
    long_blocks = [r for r in rows if r[1] >= 10]
    if short_blocks:
        print(f"  Short blocks (gt_avail ≤ 5):  {len(short_blocks)} records, success rate = {100*sum(1 for r in short_blocks if r[2])/len(short_blocks):.2f}%")
    if long_blocks:
        print(f"  Long blocks (gt_avail ≥ 10):  {len(long_blocks)} records, success rate = {100*sum(1 for r in long_blocks if r[2])/len(long_blocks):.2f}%")
    b0 = [r for r in rows if r[0] == 0]
    bgt0 = [r for r in rows if r[0] > 0]
    print(f"  block_start = 0:    {len(b0)} records, success rate = {100*sum(1 for r in b0 if r[2])/len(b0):.2f}%")
    print(f"  block_start > 0:    {len(bgt0)} records, success rate = {100*sum(1 for r in bgt0 if r[2])/len(bgt0):.2f}%")
    bgt0_long = [r for r in rows if r[0] > 0 and r[1] >= 10]
    if bgt0_long:
        print(f"  block_start > 0 AND gt_avail >= 10:  {len(bgt0_long)} records, success rate = {100*sum(1 for r in bgt0_long if r[2])/len(bgt0_long):.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_path", required=True)
    ap.add_argument("--draft_path", required=True)
    ap.add_argument("--target_outputs_dir", required=True)
    ap.add_argument("--uuids_file", required=True)
    ap.add_argument("--num_draft_layers", type=int, default=4)
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--mask_token_id", type=int, default=151662)
    args = ap.parse_args()

    device = "cuda"
    dt = torch.bfloat16
    print(f"loading target bf16...", flush=True)
    model = AlpamayoR1.from_pretrained(args.target_path, dtype=dt)
    target = model.vlm.to(device).eval()
    embed_tokens, lm_head = get_qwen3vl_embed_and_head(target)

    print(f"loading draft from {args.draft_path}", flush=True)
    ckpt = load_draft_checkpoint(args.draft_path, map_location=device)
    mask_id = ckpt["mask_token_id"] or args.mask_token_id
    nL = ckpt["num_draft_layers"] or args.num_draft_layers
    bsz = ckpt["block_size"] or args.block_size
    draft = build_dflash_draft_for_qwen3vl(
        target, num_draft_layers=nL, block_size=bsz, mask_token_id=mask_id,
    ).to(dt).to(device).eval()
    draft.load_state_dict(ckpt["state_dict"], strict=False)

    all_files = sorted(glob.glob(os.path.join(args.target_outputs_dir, "*.pt")))
    uuids = set(json.load(open(args.uuids_file)))
    files = [p for p in all_files if Path(p).stem in uuids]
    print(f"processing {len(files)} clips", flush=True)

    rows = measure(target, draft, embed_tokens, lm_head, files, bsz, device)
    report(rows)


if __name__ == "__main__":
    main()
