"""Diff weight tensors of three drafts:
  - warm SFT init
  - v2 RL step 1500 (apparent peak rolling acc_rate)
  - v2 RL step 3500 (post-collapse)

Report per-tensor max |Δ|, mean |Δ|, L2 ratio, and overall summary.
"""
import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, "/home/ubuntu/katana_transfer/code/src")
from alpamayo_r1.models.dflash_draft import load_draft_checkpoint


def diff_state_dicts(sd_a, sd_b, label):
    print(f"\n=== diff: {label} ===")
    print(f"  state_dict keys A: {len(sd_a)}, B: {len(sd_b)}")
    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    if only_a:
        print(f"  keys only in A: {len(only_a)}, sample: {list(only_a)[:5]}")
    if only_b:
        print(f"  keys only in B: {len(only_b)}, sample: {list(only_b)[:5]}")

    common = sorted(keys_a & keys_b)
    total_l2_diff_sq = 0.0
    total_l2_a_sq = 0.0
    total_max_abs = 0.0
    total_mean_abs_weighted_sum = 0.0
    total_numel = 0
    n_identical = 0
    n_different = 0
    per_tensor_summary = []

    for k in common:
        a = sd_a[k].float()
        b = sd_b[k].float()
        if a.shape != b.shape:
            print(f"  shape mismatch on {k}: {a.shape} vs {b.shape}")
            continue
        d = (a - b).float()
        max_abs = d.abs().max().item()
        mean_abs = d.abs().mean().item()
        l2_diff = d.norm().item()
        l2_a = a.norm().item()
        rel = l2_diff / max(l2_a, 1e-12)

        if max_abs == 0.0:
            n_identical += 1
        else:
            n_different += 1

        total_l2_diff_sq += l2_diff ** 2
        total_l2_a_sq += l2_a ** 2
        total_max_abs = max(total_max_abs, max_abs)
        total_mean_abs_weighted_sum += mean_abs * d.numel()
        total_numel += d.numel()
        per_tensor_summary.append((k, max_abs, mean_abs, rel, d.numel()))

    print(f"  common keys: {len(common)}")
    print(f"  identical tensors: {n_identical}")
    print(f"  different tensors: {n_different}")
    print(f"  total params compared: {total_numel:,}")
    print(f"  global max |Δ|:       {total_max_abs:.6e}")
    print(f"  global mean |Δ|:      {total_mean_abs_weighted_sum / max(total_numel,1):.6e}")
    print(f"  global ||Δ||_2 / ||A||_2: {(total_l2_diff_sq**0.5) / max(total_l2_a_sq**0.5, 1e-12):.6e}")

    # Top changed tensors by max |Δ|
    print(f"\n  Top-10 tensors by max |Δ|:")
    per_tensor_summary.sort(key=lambda x: -x[1])
    print(f"  {'tensor name':<60} | max|Δ|       | mean|Δ|      | ||Δ||/||A||  | numel")
    for name, mx, mn, rel, n in per_tensor_summary[:10]:
        print(f"  {name[:60]:<60} | {mx:.4e} | {mn:.4e} | {rel:.4e} | {n:,}")

    return total_max_abs, total_mean_abs_weighted_sum / max(total_numel,1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--ckpt1", required=True)
    ap.add_argument("--ckpt1_label", default="ckpt1")
    ap.add_argument("--ckpt2", required=True)
    ap.add_argument("--ckpt2_label", default="ckpt2")
    args = ap.parse_args()

    print(f"loading {args.init}")
    init = load_draft_checkpoint(args.init, map_location="cpu")["state_dict"]
    print(f"loading {args.ckpt1}")
    c1 = load_draft_checkpoint(args.ckpt1, map_location="cpu")["state_dict"]
    print(f"loading {args.ckpt2}")
    c2 = load_draft_checkpoint(args.ckpt2, map_location="cpu")["state_dict"]

    diff_state_dicts(init, c1, f"init vs {args.ckpt1_label}")
    diff_state_dicts(init, c2, f"init vs {args.ckpt2_label}")
    diff_state_dicts(c1, c2, f"{args.ckpt1_label} vs {args.ckpt2_label}")


if __name__ == "__main__":
    main()
