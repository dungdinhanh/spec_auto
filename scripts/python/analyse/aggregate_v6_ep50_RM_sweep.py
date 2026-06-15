"""Aggregate v6 ep50 + random_mask sweep results."""
import json, glob, re

def parse(run_dir):
    epoch_re = re.compile(r"epoch_(\d+)_(val|test|offshelf)\.json$")
    rows = {}
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
            pass
    return rows

def report(name, run_dir, c):
    print(f"\n===== {name} (c={c}) =====")
    rows = parse(run_dir)
    print(f"{'ep':>3} | {'val':>5} {'S':>5} | {'test':>5} {'S':>5} | {'off':>5} {'S':>5}")
    for ep in sorted(rows):
        r = rows[ep]
        v = r.get('val', 0); t = r.get('test', 0); o = r.get('offshelf', 0)
        print(f"{ep:>3} | {v:>5.2f} {v/(1+c):>5.2f} | {t:>5.2f} {t/(1+c):>5.2f} | {o:>5.2f} {o/(1+c):>5.2f}")
    print("Best epoch per split:")
    for sp in ('val', 'test', 'offshelf'):
        best_ep = max(rows, key=lambda e: rows[e].get(sp, -1))
        bl = rows[best_ep][sp]
        print(f"  {sp:>9}: ep {best_ep:>2}  L={bl:.3f}  S={bl/(1+c):.3f}")

if __name__ == "__main__":
    L4 = "/home/ubuntu/local_data/runs/dflash_L4_lr1e-4_ep50_bs16_warm_v6_randomMask_sharon1"
    L2 = "/home/ubuntu/local_data/runs/dflash_L2_lr1e-4_ep50_bs16_warm_v6_randomMask_sharon1"
    report("v6 L=4 ep50 + RM sweep", L4, c=0.213)
    report("v6 L=2 ep50 + RM sweep", L2, c=0.165)
