"""For each (clip, block_start, within-block-position), compute the rank of the
GT (target greedy) token in the draft's distribution. Aggregate top-K hit rates
by (a) within-block-position range, (b) block_start index.

Output:
- Overall top-K hit rate (K=20, 25, 30) and rank percentiles
- Top-K hit rate by within-block-position range: [0,6), [6,10), [10,16)
- Top-K hit rate by block_start index (per-bs)
"""
import argparse, glob, json, os, sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/ubuntu/alpamayo_code/src")
sys.path.insert(0, "/home/ubuntu/dflash_code")
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.models.dflash_draft import (
    build_dflash_draft_for_qwen3vl, get_qwen3vl_embed_and_head,
    extract_context_feature, load_draft_checkpoint,
)
from transformers.cache_utils import DynamicCache


@torch.no_grad()
def measure(target, draft, embed_tokens, lm_head, files, block_size, device, label):
    mask_id = draft.mask_token_id
    B = block_size
    layer_ids = draft.target_layer_ids

    # Aggregators
    K_THRESHOLDS = [20, 25, 30]
    # Per range: (range_idx, K_threshold) → cumulative count and total positions
    RANGE_BOUNDS = [(0, 6), (6, 10), (10, 16)]   # within-block prediction index
    range_counts = [[0]*len(K_THRESHOLDS) for _ in RANGE_BOUNDS]
    range_total = [0 for _ in RANGE_BOUNDS]
    overall_counts = [0]*len(K_THRESHOLDS)
    overall_total = 0
    # Per block_start (cap at 100): { bs: {K: count, total: int} }
    bs_counts = [[0]*len(K_THRESHOLDS) for _ in range(100)]
    bs_total = [0 for _ in range(100)]
    # Rank distribution
    rank_buckets = {1:0, 5:0, 10:0, 20:0, 30:0, 50:0, 100:0, 1000:0, int(1e9):0}

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
        seq_len = full_ids.shape[1]

        # Prefill target on full sequence
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
            dl = lm_head(dh[:, -(B - 1):, :])     # (1, B-1, V)
            logits = dl[0]                         # (B-1, V)

            gt = full_ids[0, P + block_start + 1:P + block_start + 1 + gt_av]   # (gt_av,)

            # For each within-block prediction position p in [0, gt_av), compute
            # the rank of gt[p] in logits[p].
            for p in range(gt_av):
                # Compute rank of GT token: count tokens with higher logit
                gt_logit = logits[p, gt[p]].item()
                rank = (logits[p] > gt_logit).sum().item() + 1   # 1-indexed
                # Top-K hit?
                for ki, K in enumerate(K_THRESHOLDS):
                    if rank <= K:
                        overall_counts[ki] += 1
                        if block_start < 100:
                            bs_counts[block_start][ki] += 1
                        for ri, (lo, hi) in enumerate(RANGE_BOUNDS):
                            if lo <= p < hi:
                                range_counts[ri][ki] += 1
                                break
                overall_total += 1
                if block_start < 100:
                    bs_total[block_start] += 1
                for ri, (lo, hi) in enumerate(RANGE_BOUNDS):
                    if lo <= p < hi:
                        range_total[ri] += 1
                        break
                # Rank histogram
                for thresh in [1, 5, 10, 20, 30, 50, 100, 1000, int(1e9)]:
                    if rank <= thresh:
                        rank_buckets[thresh] += 1
                        break

        if (fi + 1) % 20 == 0:
            print(f"  [{label}] {fi+1}/{len(files)} clips processed", flush=True)

    print(f"\n=== {label} on {len(files)} clips, total positions: {overall_total} ===")
    print(f"\nRank distribution (cumulative %):")
    cum = 0
    for thresh in [1, 5, 10, 20, 30, 50, 100, 1000, int(1e9)]:
        cum += rank_buckets[thresh]
        thresh_lab = "all" if thresh > 100000 else f"≤{thresh}"
        print(f"  rank {thresh_lab:<6}: {rank_buckets[thresh]:>6} positions ({100*cum/overall_total:>5.2f}% cumulative)")

    print(f"\nOverall top-K hit rates:")
    for ki, K in enumerate(K_THRESHOLDS):
        print(f"  top-{K:>2}: {overall_counts[ki]:>6} / {overall_total} = {100*overall_counts[ki]/overall_total:.3f}%")

    print(f"\nTop-K hit rates by WITHIN-BLOCK position range:")
    print(f"  range          | n_positions | top-20  | top-25  | top-30")
    for ri, (lo, hi) in enumerate(RANGE_BOUNDS):
        rng_lab = f"[{lo}, {hi})"
        n = range_total[ri]
        if n > 0:
            row = " | ".join(f"{100*range_counts[ri][ki]/n:>6.2f}%" for ki in range(len(K_THRESHOLDS)))
            print(f"  {rng_lab:<14} | {n:>11} | {row}")

    print(f"\nTop-K hit rates by BLOCK_START index (b) — only b's with ≥30 positions:")
    print(f"  block_start | n_positions | top-20  | top-25  | top-30")
    for b in range(100):
        if bs_total[b] >= 30:
            n = bs_total[b]
            row = " | ".join(f"{100*bs_counts[b][ki]/n:>6.2f}%" for ki in range(len(K_THRESHOLDS)))
            print(f"  {b:>11} | {n:>11} | {row}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_path", required=True)
    ap.add_argument("--draft_path", required=True)
    ap.add_argument("--target_outputs_dir", required=True)
    ap.add_argument("--uuids_file", required=True)
    ap.add_argument("--num_draft_layers", type=int, default=4)
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--mask_token_id", type=int, default=151662)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    device = "cuda"
    dt = torch.bfloat16
    print(f"[{args.label}] loading target bf16...", flush=True)
    model = AlpamayoR1.from_pretrained(args.target_path, dtype=dt)
    target = model.vlm.to(device).eval()
    embed_tokens, lm_head = get_qwen3vl_embed_and_head(target)

    print(f"[{args.label}] loading draft from {args.draft_path}", flush=True)
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
    print(f"[{args.label}] processing {len(files)} clips", flush=True)

    measure(target, draft, embed_tokens, lm_head, files, bsz, device, args.label)


if __name__ == "__main__":
    main()
