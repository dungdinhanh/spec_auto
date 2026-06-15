"""Aggregate per-run eval JSONs across the v3 sweep into
claude_report/v3_sweep_results.md.

Reads $FLORA/runs/<RUN_NAME>/eval/eval_results.json for each run and builds a
markdown report with:
  - Training/val/test cached-logits accuracy + acceptance
  - E2E on-shelf acceptance rate + speedup
  - E2E off-shelf acceptance rate + speedup
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path


def _fmt_pct(x, digits=1):
    if x is None: return "—"
    return f"{x*100:.{digits}f}%"

def _fmt_f(x, digits=2):
    if x is None: return "—"
    return f"{x:.{digits}f}"


def extract_cached(cached):
    if not cached: return {}
    splits = cached.get("splits", {})
    out = {}
    for k in ("train", "val", "test"):
        r = splits.get(k) or {}
        out[k] = {
            "accuracy": r.get("accuracy"),
            "acceptance": r.get("acceptance"),
            "clips": r.get("clips"),
        }
    return out


def extract_e2e(e2e):
    """Read e2e_spec_test.py output JSON.

    avg_iter_tokens = avg tokens emitted per spec iteration (= accepted + 1 bonus).
    acceptance length = avg_iter_tokens (max = block_size)
    acceptance rate  = (avg_iter_tokens - 1) / (block_size - 1)
    """
    if not e2e: return {}
    avg_iter = e2e.get("avg_iter_tokens")
    block_size = e2e.get("block_size")
    speedup = e2e.get("speedup")
    if avg_iter is not None and block_size and block_size > 1:
        accept_rate = (avg_iter - 1.0) / (block_size - 1)
    else:
        accept_rate = None
    return {
        "accept_len": avg_iter,
        "accept_rate": accept_rate,
        "block_size": block_size,
        "speedup": speedup,
        "num_clips": e2e.get("num_clips_evaluated"),
        "ar_tps": e2e.get("ar_tokens_per_sec"),
        "sp_tps": e2e.get("sp_tokens_per_sec"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="/srv/scratch/flora/dungda/runs")
    ap.add_argument("--pattern", default="dflash_L*_lr*_bs4_mrope_v3")
    ap.add_argument("--output", default="claude_report/v3_sweep_results.md")
    args = ap.parse_args()

    run_dirs = sorted(glob.glob(os.path.join(args.runs_root, args.pattern)))
    rows = []
    for rd in run_dirs:
        name = Path(rd).name
        eval_json = Path(rd) / "eval" / "eval_results.json"
        if not eval_json.exists():
            rows.append({"run": name, "status": "no eval_results.json"})
            continue
        er = json.load(open(eval_json))
        cached = extract_cached(er.get("cached"))
        onshelf = extract_e2e(er.get("e2e_onshelf"))
        offshelf = extract_e2e(er.get("e2e_offshelf"))
        # Parse L, lr from name: dflash_L<N>_lr<LR>_...
        L = name.split("_L", 1)[1].split("_", 1)[0] if "_L" in name else "?"
        lr = name.split("_lr", 1)[1].split("_", 1)[0] if "_lr" in name else "?"
        rows.append({
            "run": name, "L": L, "lr": lr,
            "cached": cached, "onshelf": onshelf, "offshelf": offshelf,
            "status": "ok",
        })

    # Build markdown
    lines = []
    lines.append("# DFlash v3 Sweep Results")
    lines.append("")
    lines.append("Sweep: `num_draft_layers ∈ {1,2,3,4}` × `lr ∈ {5e-5, 1e-4, 2e-4}`, "
                 "block=8, bs=4, grad_accum=1, 4× H200, 6 epochs, M-RoPE + overlapping + random_mask + KL=1.0.")
    lines.append("")

    # Cached-logits section
    lines.append("## Cached-logits accuracy (target_coc_outputs)")
    lines.append("")
    lines.append("| L | lr | train acc | train acc/blk | val acc | val acc/blk | test acc | test acc/blk |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("status") != "ok":
            lines.append(f"| — | — | {r['run']}: {r['status']} | | | | | |")
            continue
        c = r["cached"]
        tr, va, te = c.get("train", {}), c.get("val", {}), c.get("test", {})
        lines.append(
            f"| {r['L']} | {r['lr']} | "
            f"{_fmt_pct(tr.get('accuracy'))} | {_fmt_f(tr.get('acceptance'))} | "
            f"{_fmt_pct(va.get('accuracy'))} | {_fmt_f(va.get('acceptance'))} | "
            f"{_fmt_pct(te.get('accuracy'))} | {_fmt_f(te.get('acceptance'))} |"
        )

    lines.append("")
    lines.append("## E2E speculative decoding")
    lines.append("")
    lines.append("Acceptance length = avg tokens emitted per spec iteration (max = block_size=8). "
                 "Acceptance rate = (length − 1) / (block_size − 1).")
    lines.append("")
    lines.append("| L | lr | on-shelf len | on-shelf rate | on-shelf speedup | off-shelf len | off-shelf rate | off-shelf speedup |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("status") != "ok": continue
        on, off = r["onshelf"], r["offshelf"]
        lines.append(
            f"| {r['L']} | {r['lr']} | "
            f"{_fmt_f(on.get('accept_len'))} | {_fmt_pct(on.get('accept_rate'))} | {_fmt_f(on.get('speedup'))}× | "
            f"{_fmt_f(off.get('accept_len'))} | {_fmt_pct(off.get('accept_rate'))} | {_fmt_f(off.get('speedup'))}× |"
        )

    lines.append("")
    lines.append("## Missing / failed runs")
    for r in rows:
        if r.get("status") != "ok":
            lines.append(f"- `{r['run']}`: {r['status']}")

    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_path} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
