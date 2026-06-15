"""Aggregate iterative-refinement grid results.

For each (model, T, split): read avg_iter_tokens (real L) and compute
S_eff = L / (1 + T_eff * c) using c_L4=0.213, c_L2=0.165 and the
effective_steps from the cosine schedule.
"""
import json, glob, re, math

OUT = "/home/ubuntu/local_data/runs/_refine_eval"
C = {"L4": 0.213, "L2": 0.165}

def cos_sched(M, T):
    if T <= 1 or M <= 1: return [M]
    masked_after = [M] + [int(round(M*math.cos(math.pi/2*t/T))) for t in range(1,T)] + [0]
    reveals = [masked_after[t-1]-masked_after[t] for t in range(1,T+1)]
    reveals = [max(0,r) for r in reveals]
    diff = M - sum(reveals)
    if diff != 0: reveals[-1] += diff
    return [r for r in reveals if r > 0]

def main():
    rows = {}
    for f in sorted(glob.glob(f"{OUT}/*.json")):
        m = re.search(r"(L[24])_T(\d+)_(val|test|offshelf)\.json$", f)
        if not m: continue
        name, T, split = m.group(1), int(m.group(2)), m.group(3)
        d = json.loads(open(f).read())
        L = d["avg_iter_tokens"]
        T_eff = len(cos_sched(15, T))
        c = C[name]
        S = L / (1 + T_eff * c)
        rows[(name, T, split)] = (L, T_eff, S, d.get("speedup", 0.0))

    for name in ("L4", "L2"):
        print(f"\n===== {name}  c={C[name]}  =====")
        print(f"{'T':>2} {'T_eff':>5} | {'val_L':>6} {'val_S':>6} | {'test_L':>6} {'test_S':>6} | {'off_L':>6} {'off_S':>6} | wall_speedup(off)")
        for T in (1, 2, 3, 5):
            sched = cos_sched(15, T)
            r_val = rows.get((name, T, "val"), (0, 0, 0, 0))
            r_test = rows.get((name, T, "test"), (0, 0, 0, 0))
            r_off = rows.get((name, T, "offshelf"), (0, 0, 0, 0))
            T_eff = r_val[1] or r_off[1] or r_test[1] or len(sched)
            print(f"{T:>2} {T_eff:>5} | {r_val[0]:>6.3f} {r_val[2]:>6.3f} | "
                  f"{r_test[0]:>6.3f} {r_test[2]:>6.3f} | "
                  f"{r_off[0]:>6.3f} {r_off[2]:>6.3f} | "
                  f"{r_off[3]:>5.3f}   sched={sched}")

if __name__ == "__main__":
    main()
