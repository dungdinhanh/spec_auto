"""Aggregate v6 ep50 sweep results — find best epoch per (run, split)."""
import json, glob, re
from pathlib import Path

def parse(run_dir, c):
    epoch_re = re.compile(r"epoch_(\d+)_(val|test|offshelf)\.json$")
    rows = {}  # epoch -> {split: L}
    for f in sorted(glob.glob(f"{run_dir}/_eval_sweep/*.json")):
        m = epoch_re.search(f)
        if not m: continue
        ep = int(m.group(1)); split = m.group(2)
        try:
            j = json.loads(open(f).read())
            L = j.get("avg_iter_tokens")
            if L is None: continue
            rows.setdefault(ep, {})[split] = L
        except Exception as e:
            print(f"skip {f}: {e}")
    return rows, c

def fmt(L, c): return f"{L:.2f} / {L/(1+c):.2f}x"

def report(name, run_dir, c):
    print(f"\n===== {name} (c={c}) =====")
    rows, _ = parse(run_dir, c)
    print(f"{'epoch':>5} | {'val (L / S)':>12} | {'test (L / S)':>12} | {'off-shelf (L / S)':>17}")
    for ep in sorted(rows):
        r = rows[ep]
        v = fmt(r.get('val', 0), c) if 'val' in r else '—'
        t = fmt(r.get('test', 0), c) if 'test' in r else '—'
        o = fmt(r.get('offshelf', 0), c) if 'offshelf' in r else '—'
        print(f"{ep:>5} | {v:>12} | {t:>12} | {o:>17}")
    # find best per split
    print("\nBest epoch per split (by L):")
    for sp in ('val', 'test', 'offshelf'):
        best_ep = max(rows, key=lambda e: rows[e].get(sp, -1))
        bl = rows[best_ep].get(sp, 0)
        print(f"  {sp:>9}: ep {best_ep:>2}  L={bl:.3f}  S={bl/(1+c):.3f}x")

if __name__ == "__main__":
    L4 = "/home/ubuntu/local_data/runs/dflash_L4_lr1e-4_ep50_bs16_warm_v6_sharon1"
    L2 = "/home/ubuntu/local_data/runs/dflash_L2_lr1e-4_ep50_bs16_warm_v6_sharon1"
    report("v6 L=4 ep50 sweep", L4, c=0.213)
    report("v6 L=2 ep50 sweep", L2, c=0.165)
