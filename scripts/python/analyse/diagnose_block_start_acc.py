"""Diagnose: why does training-time mean_accept_rate (~0.83) not match eval L (~4.5)?

For 100 train clips, measure:
  - acc_length at block_start=0 (greedy, fixed start)
  - acc_length sampled with decay=0.8 (training distribution)
  - acc_length uniform over [0, N-2]
  - eval-style: sequential acc_lengths starting from 0
  - per-block_start acc_rate curve (averaged across clips)

If training-time 0.83 matches "block_start=0 only" or "decay=0.8 weighted",
that proves the gap is the block_start sampling bias inside the training loop.
"""
import argparse, glob, json, os, sys, random
from pathlib import Path
import torch

sys.path.insert(0, "/home/ubuntu/katana_transfer/code/src")
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.models.dflash_draft import (
    build_dflash_draft_for_qwen3vl, get_qwen3vl_embed_and_head,
    extract_context_feature, load_draft_checkpoint,
)
from transformers.cache_utils import DynamicCache


@torch.no_grad()
def per_block_start_acc(target, draft, embed_tokens, lm_head, files, block_size, device):
    """For each clip, for each block_start in [0, N-2], measure acc_length under
    greedy decoding. Return per-block_start arrays + summary stats."""
    mask_id = draft.mask_token_id
    B = block_size
    layer_ids = draft.target_layer_ids

    # Per-block_start accumulators (block_start max = 100 to keep simple; CoCs are short)
    per_bs_acc_sum = [0.0] * 100
    per_bs_count = [0] * 100

    # Per-clip stats:
    bs0_accs = []          # acc_length at block_start=0 only
    decay08_accs = []      # acc_length sampled with decay=0.8 (per clip, 1 sample)
    uniform_accs = []      # acc_length sampled uniformly over [0, N-2]
    sequential_iter_tokens = []   # eval-style: iter_tokens for each iteration sequentially
    sequential_iters_per_clip = []
    coc_lens = []

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
        coc_lens.append(N)

        full_ids = torch.cat([prompt_ids[0], output_ids], dim=0).unsqueeze(0)
        seq_len = full_ids.shape[1]

        # Prefill target on full sequence (no incremental — we just need hidden states)
        cache_T = DynamicCache()
        kw = dict(input_ids=full_ids, past_key_values=cache_T,
                  use_cache=True, output_hidden_states=True, return_dict=True)
        if pv is not None: kw["pixel_values"] = pv
        if igt is not None: kw["image_grid_thw"] = igt
        po = target(**kw)
        target_hidden = extract_context_feature(po.hidden_states, layer_ids)

        # Compute acc_length at every block_start from 0 to N-2
        # context_len = P + block_start, gt_avail = min(B-1, N - block_start - 1)
        accs_for_clip = []  # acc_length per block_start
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
            dp = dl.argmax(dim=-1)        # (1, B-1)

            gt = full_ids[0, P + block_start + 1:P + block_start + 1 + gt_av]
            acc = 0
            for k in range(gt.shape[0]):
                if dp[0, k].item() == gt[k].item(): acc += 1
                else: break

            accs_for_clip.append(acc)
            if block_start < 100:
                per_bs_acc_sum[block_start] += acc / max(gt_av, 1)
                per_bs_count[block_start] += 1

        # Stats for this clip:
        if accs_for_clip:
            bs0_accs.append(accs_for_clip[0])
            # Decay=0.8 weighted single sample
            cands = list(range(len(accs_for_clip)))
            weights = [0.8 ** b for b in cands]
            random.seed(fi)
            chosen = random.choices(cands, weights=weights, k=1)[0]
            decay08_accs.append(accs_for_clip[chosen])
            # Uniform single sample
            chosen_u = random.choice(cands)
            uniform_accs.append(accs_for_clip[chosen_u])

            # Sequential eval-style: start at 0, advance by acc+1
            start = 0
            iters_this_clip = 0
            while start < len(accs_for_clip):
                acc_here = accs_for_clip[start]
                iter_tokens = acc_here + 1
                sequential_iter_tokens.append(iter_tokens)
                start += iter_tokens
                iters_this_clip += 1
            sequential_iters_per_clip.append(iters_this_clip)

        if (fi + 1) % 25 == 0:
            print(f"  [{fi+1}/{len(files)}]", flush=True)

    # Summaries
    n_clips = len(coc_lens)
    print(f"\n=== {n_clips} clips, mean CoC length={sum(coc_lens)/n_clips:.1f} ===")

    # Each per-clip-acc converted to acc_rate: acc/min(B-1, gt_av)
    # For block_start=0, gt_av = min(B-1, N-1). For most clips with N>=16, gt_av=B-1=15.
    # We'll just report acc/gt_av using the actual gt_av.

    def mean_rate(accs, gt_avs):
        rates = [a/g for a, g in zip(accs, gt_avs)]
        return sum(rates) / max(len(rates), 1)

    # Reconstruct gt_av for block_start=0
    gt_av_bs0 = [min(B - 1, n - 1) for n in coc_lens]

    print(f"\nblock_start=0 only:")
    print(f"  mean acc = {sum(bs0_accs)/n_clips:.2f}")
    print(f"  mean acc_rate = {mean_rate(bs0_accs, gt_av_bs0):.4f}")
    print(f"  implied L = {sum(bs0_accs)/n_clips + 1:.3f}")

    # decay=0.8 weighted (training-style)
    # Need gt_av per clip for the chosen block_start... not tracking, approximate with B-1=15
    print(f"\ndecay=0.8 weighted block_start (training-style sampling):")
    print(f"  mean acc = {sum(decay08_accs)/n_clips:.2f}")
    print(f"  mean acc_rate (assuming gt_av={B-1}) = {sum(decay08_accs)/n_clips/(B-1):.4f}")

    print(f"\nuniform block_start:")
    print(f"  mean acc = {sum(uniform_accs)/n_clips:.2f}")
    print(f"  mean acc_rate (assuming gt_av={B-1}) = {sum(uniform_accs)/n_clips/(B-1):.4f}")

    print(f"\neval-style (sequential, start=0, advance by acc+1):")
    n_iters = len(sequential_iter_tokens)
    L_eval = sum(sequential_iter_tokens) / n_iters
    print(f"  total iters across {n_clips} clips: {n_iters}")
    print(f"  mean iters per clip: {n_iters/n_clips:.2f}")
    print(f"  L = mean iter_tokens = {L_eval:.3f}")
    print(f"  acc_rate (L-1)/{B-1} = {(L_eval-1)/(B-1):.4f}")

    print(f"\n--- per-block_start acc_rate curve (first 16 positions) ---")
    print(f"  block_start | n_clips | mean_acc_rate")
    for bs in range(min(16, max(per_bs_count))):
        if per_bs_count[bs] > 0:
            rate = per_bs_acc_sum[bs] / per_bs_count[bs]
            print(f"  {bs:>11d} | {per_bs_count[bs]:>7d} | {rate:.4f}")


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
    print(f"[{args.label}] evaluating on {len(files)} clips", flush=True)

    per_block_start_acc(target, draft, embed_tokens, lm_head, files, bsz, device)


if __name__ == "__main__":
    main()
