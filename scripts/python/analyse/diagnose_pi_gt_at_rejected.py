"""For each (clip, block_start) pair, identify greedy-rejected positions
(where argmax != GT). Measure pi(GT) — the draft's probability mass on the
GT token — at those positions, under bf16 softmax with T=1.0 (matches training).

Comparing two ckpts: if RL ckpt has higher pi(GT) at rejected positions
(while argmax is unchanged), it explains why training-time K=4 stochastic
acc_rate climbed even though greedy acc_rate didn't.
"""
import argparse, glob, json, os, sys, random
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/ubuntu/katana_transfer/code/src")
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.models.dflash_draft import (
    build_dflash_draft_for_qwen3vl, get_qwen3vl_embed_and_head,
    extract_context_feature, load_draft_checkpoint,
)
from transformers.cache_utils import DynamicCache


@torch.no_grad()
def measure_pi_gt(target, draft, embed_tokens, lm_head, files, block_size, device, label):
    mask_id = draft.mask_token_id
    B = block_size
    layer_ids = draft.target_layer_ids

    # Aggregators
    sum_pi_gt_at_rejected = 0.0
    n_rejected = 0
    sum_pi_gt_at_matched = 0.0  # control: should be huge (greedy IS GT here)
    n_matched = 0
    # Histogram of pi(GT) at rejected positions
    bins = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    hist = [0] * (len(bins) - 1)
    # Top-K rank of GT among rejected positions
    rank_buckets = {1:0, 2:0, 3:0, 5:0, 10:0, 50:0, 1e9:0}

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
            dl = lm_head(dh[:, -(B - 1):, :])      # (1, B-1, V)
            probs = F.softmax(dl[0].float(), dim=-1)  # (B-1, V) in fp32 for safety

            gt = full_ids[0, P + block_start + 1:P + block_start + 1 + gt_av]   # (gt_av,)
            argmax = dl[0, :gt_av].argmax(dim=-1)
            pi_at_gt = probs[:gt_av].gather(-1, gt.unsqueeze(-1)).squeeze(-1)   # (gt_av,)

            for p in range(gt_av):
                if argmax[p].item() == gt[p].item():
                    sum_pi_gt_at_matched += pi_at_gt[p].item()
                    n_matched += 1
                else:
                    val = pi_at_gt[p].item()
                    sum_pi_gt_at_rejected += val
                    n_rejected += 1
                    # histogram bin
                    for j in range(len(bins) - 1):
                        if bins[j] <= val < bins[j+1]:
                            hist[j] += 1
                            break
                    # rank of GT in this rejected position's distribution
                    sorted_probs, sorted_idx = probs[p].sort(descending=True)
                    rank = (sorted_idx == gt[p].item()).nonzero(as_tuple=True)[0].item() + 1
                    for thresh in [1, 2, 3, 5, 10, 50, 1e9]:
                        if rank <= thresh:
                            rank_buckets[thresh] += 1
                            break

        if (fi + 1) % 25 == 0:
            print(f"  [{label}] {fi+1}/{len(files)} clips processed", flush=True)

    if n_rejected == 0:
        print(f"\n[{label}] no rejected positions found")
        return

    print(f"\n=== {label} ===")
    print(f"total positions: {n_rejected + n_matched}")
    print(f"  matched (greedy=GT): {n_matched}  ({100*n_matched/(n_rejected+n_matched):.1f}%)")
    print(f"  rejected (greedy!=GT): {n_rejected}  ({100*n_rejected/(n_rejected+n_matched):.1f}%)")
    print(f"\nMean pi(GT) at MATCHED positions: {sum_pi_gt_at_matched/max(n_matched,1):.4f}")
    print(f"Mean pi(GT) at REJECTED positions: {sum_pi_gt_at_rejected/max(n_rejected,1):.4f}")
    print(f"\nHistogram of pi(GT) at rejected positions ({n_rejected} positions):")
    print(f"  {'range':<14} | count | %")
    for j in range(len(bins) - 1):
        rng = f"[{bins[j]:.3f}, {bins[j+1]:.3f})"
        print(f"  {rng:<14} | {hist[j]:>5} | {100*hist[j]/n_rejected:>5.2f}")
    print(f"\nRank of GT at rejected positions:")
    cum = 0
    for thresh in [1, 2, 3, 5, 10, 50, 1e9]:
        cum += rank_buckets[thresh]
        label_t = "all" if thresh > 100 else f"top-{int(thresh)}"
        print(f"  {label_t:<8}: {rank_buckets[thresh]:>5} ({100*cum/n_rejected:>5.2f}% cumulative)")

    # Report key statistic for the experiment
    mean_pi_rej = sum_pi_gt_at_rejected / max(n_rejected, 1)
    return mean_pi_rej


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

    measure_pi_gt(target, draft, embed_tokens, lm_head, files, bsz, device, args.label)


if __name__ == "__main__":
    main()
