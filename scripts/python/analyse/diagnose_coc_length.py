"""Compute CoC length distribution over target_coc_outputs/*.pt files.
For each clip: N = output_token_ids.shape[0]. Aggregate histogram + percentiles.

Default: sample 2000 random clips for speed. Use --all to process every file
in the directory.
"""
import argparse, glob, os, random
from pathlib import Path
import torch
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_outputs_dir", required=True)
    ap.add_argument("--uuids_file", default=None,
                    help="Optional: only count clips whose stems are in this JSON list.")
    ap.add_argument("--n_sample", type=int, default=2000,
                    help="Sample at most this many clips. 0 = all.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    all_files = sorted(glob.glob(os.path.join(args.target_outputs_dir, "*.pt")))
    if args.uuids_file:
        uuids = set(json.load(open(args.uuids_file)))
        all_files = [p for p in all_files if Path(p).stem in uuids]
    print(f"Total clips available: {len(all_files)}")

    if args.n_sample > 0 and args.n_sample < len(all_files):
        rng = random.Random(args.seed)
        files = rng.sample(all_files, args.n_sample)
        print(f"Sampling {args.n_sample} clips with seed {args.seed}")
    else:
        files = all_files
        print(f"Processing all {len(files)} clips")

    lengths = []
    bad = 0
    for i, f in enumerate(files):
        try:
            d = torch.load(f, weights_only=False, map_location="cpu")
            N = int(d["output_token_ids"].shape[0])
            lengths.append(N)
            del d
        except Exception as e:
            bad += 1
            if bad <= 5:
                print(f"  bad file {f}: {e}")
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(files)} processed ({bad} bad)", flush=True)

    if not lengths:
        print("No valid clips!")
        return

    lengths = sorted(lengths)
    n = len(lengths)
    print(f"\n=== Stats over {n} clips ({bad} bad files skipped) ===")
    print(f"  min    = {lengths[0]}")
    print(f"  max    = {lengths[-1]}")
    print(f"  mean   = {sum(lengths)/n:.2f}")
    print(f"  median = {lengths[n//2]}")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9]:
        idx = min(n - 1, int(n * p / 100))
        print(f"  p{p:>4}: N = {lengths[idx]}")

    print(f"\n=== Histogram of CoC length ===")
    print(f"  N           | count | pct    | cumul (≤N)")
    bins = list(range(2, 25)) + [30, 40, 60, 100, 200]
    cumul = 0
    bin_idx = 0
    for v in lengths:
        cumul += 1
    cumul = 0
    last_idx = 0
    for thresh in bins:
        cnt = sum(1 for v in lengths if v <= thresh)
        # Count in this bucket = cnt - last cumul
        if thresh == bins[0]:
            bucket = cnt
        else:
            bucket = cnt - last_idx
        pct = 100 * bucket / n
        cum_pct = 100 * cnt / n
        print(f"  N ≤ {thresh:>4}  | {bucket:>5} | {pct:>5.2f}% | {cum_pct:>5.2f}%")
        last_idx = cnt

    # CoC length thresholds the user cares about
    print(f"\n=== Key thresholds ===")
    for thresh in [10, 12, 14, 15, 16, 18, 20, 22, 25, 30]:
        ge = sum(1 for v in lengths if v >= thresh)
        lt = sum(1 for v in lengths if v < thresh)
        print(f"  N >= {thresh:>2}: {ge:>5} clips ({100*ge/n:>5.2f}%)   |   N < {thresh:>2}: {lt:>5} clips ({100*lt/n:>5.2f}%)")

    # Block_start coverage: for each b, how many clips have N >= b+2 (so b is a valid block_start)
    print(f"\n=== Block_start coverage (clips with valid b = clips with N >= b+2) ===")
    print(f"  block_start | n_clips_reach | %_clips_reach | gt_avail at b (mean)")
    for b in range(0, 22):
        reach = [v for v in lengths if v >= b + 2]
        if not reach:
            continue
        # gt_available = min(15, N - b - 1)
        gt_avs = [min(15, v - b - 1) for v in reach]
        mean_gt_av = sum(gt_avs) / len(gt_avs)
        print(f"  {b:>11} | {len(reach):>13} | {100*len(reach)/n:>12.2f}% | {mean_gt_av:>5.2f}")


if __name__ == "__main__":
    main()
