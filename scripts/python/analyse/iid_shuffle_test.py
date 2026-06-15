"""IID-shuffle test: extract per-step acc_rate values from v2 RL training log,
then compare the actual rolling-acc_rate trajectory to N shuffled trajectories
where the same per-step values are randomly reshuffled (no model in the loop).

If shuffled trajectories show similar shape characteristics (range, peak height,
trough depth, autocorrelation), the real trajectory is consistent with noise
plus 10-sample autocorrelation, no model trend needed.
"""
import re
import sys
import numpy as np

LOG_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/rl_e2v2_v2_N5.log"
SAVE_INTERVAL = 250
LOG_INTERVAL = 25
WINDOW = SAVE_INTERVAL // LOG_INTERVAL   # 10
N_SHUFFLES = 5000

# 1. Extract per-step acc_rate values from log
data = []
with open(LOG_PATH) as f:
    for line in f:
        m = re.search(r"step (\d+) \|.*acc_rate=([\d.]+)", line)
        if m:
            data.append((int(m.group(1)), float(m.group(2))))

steps = np.array([s for s, _ in data], dtype=int)
accs = np.array([a for _, a in data], dtype=float)
print(f"Extracted {len(accs)} per-step values, step range: {steps.min()}..{steps.max()}")
print(f"Mean: {accs.mean():.4f}, std: {accs.std():.4f}")
print(f"Window size (deque maxlen): {WINDOW}")
print()


def rolling_means_at_saves(per_step_values, save_step_idxs):
    """For each save_step idx in save_step_idxs, compute mean of last `WINDOW`
    values from per_step_values[:save_step_idx+1]."""
    means = []
    for idx in save_step_idxs:
        end = idx + 1
        start = max(0, end - WINDOW)
        means.append(per_step_values[start:end].mean())
    return np.array(means)


# 2. Identify save points: step % save_interval == 0 and step > 0
# In our extracted data (one entry per log_interval), saves happen every (save_interval/log_interval) = 10 entries.
# But the actual save_step is the index in the deque where save fires. Since deque is appended at log_interval
# and save fires at save_interval, save fires at log_interval idx 9, 19, 29, ... (0-indexed, every 10 logged entries).
n_logged = len(accs)
save_idxs = list(range(WINDOW - 1, n_logged, WINDOW))   # 9, 19, 29, ...
save_steps = steps[save_idxs]
print(f"Save points: {len(save_idxs)}, at logged-step idx: {save_idxs[:5]}... corresponding to training steps: {save_steps[:5].tolist()}...")
print()

# 3. Actual rolling-mean trajectory
actual_traj = rolling_means_at_saves(accs, save_idxs)
print(f"=== ACTUAL trajectory ({len(actual_traj)} save points) ===")
print(f"  range: [{actual_traj.min():.4f}, {actual_traj.max():.4f}], spread = {actual_traj.max() - actual_traj.min():.4f}")
print(f"  mean: {actual_traj.mean():.4f}, std: {actual_traj.std():.4f}")
print(f"  peak (max - mean): {actual_traj.max() - actual_traj.mean():+.4f}")
print(f"  trough (min - mean): {actual_traj.min() - actual_traj.mean():+.4f}")
# autocorrelation lag-1
if len(actual_traj) > 1:
    ac = np.corrcoef(actual_traj[:-1], actual_traj[1:])[0, 1]
    print(f"  lag-1 autocorrelation: {ac:.4f}")
print()
print("  trajectory:")
for i, (s, m) in enumerate(zip(save_steps, actual_traj)):
    print(f"    save{i+1:>3} (step {s:>5}): {m:.4f}")
print()

# 4. N IID shuffles: shuffle the per-step values, compute the same rolling-mean trajectory
rng = np.random.default_rng(42)
shuffled_ranges = np.zeros(N_SHUFFLES)
shuffled_max = np.zeros(N_SHUFFLES)
shuffled_min = np.zeros(N_SHUFFLES)
shuffled_lag1_ac = np.zeros(N_SHUFFLES)
shuffled_max_consec_rise = np.zeros(N_SHUFFLES)   # largest contiguous rise in saves
shuffled_max_consec_fall = np.zeros(N_SHUFFLES)
for k in range(N_SHUFFLES):
    perm = rng.permutation(accs)
    traj = rolling_means_at_saves(perm, save_idxs)
    shuffled_ranges[k] = traj.max() - traj.min()
    shuffled_max[k] = traj.max()
    shuffled_min[k] = traj.min()
    if len(traj) > 1:
        shuffled_lag1_ac[k] = np.corrcoef(traj[:-1], traj[1:])[0, 1]
    # Largest rise in any contiguous subwindow
    max_rise = max_fall = 0.0
    for i in range(len(traj)):
        for j in range(i+1, len(traj)):
            d = traj[j] - traj[i]
            if d > max_rise: max_rise = d
            if -d > max_fall: max_fall = -d
    shuffled_max_consec_rise[k] = max_rise
    shuffled_max_consec_fall[k] = max_fall

# 5. Compare actual to shuffle distribution
def percentile_rank(x, dist):
    """Return percentile of x within dist (0-100)."""
    return float((dist <= x).mean() * 100)

actual_range = actual_traj.max() - actual_traj.min()
actual_max_rise = max(actual_traj[j] - actual_traj[i] for i in range(len(actual_traj)) for j in range(i+1, len(actual_traj)))
actual_max_fall = max(actual_traj[i] - actual_traj[j] for i in range(len(actual_traj)) for j in range(i+1, len(actual_traj)))
actual_lag1_ac = np.corrcoef(actual_traj[:-1], actual_traj[1:])[0, 1]

print(f"=== {N_SHUFFLES} IID shuffles, comparing actual to shuffled distribution ===")
print()
print(f"  Range (max - min):")
print(f"    actual:    {actual_range:.4f}")
print(f"    shuffled:  mean={shuffled_ranges.mean():.4f}, std={shuffled_ranges.std():.4f}, median={np.median(shuffled_ranges):.4f}")
print(f"    p5={np.percentile(shuffled_ranges, 5):.4f}, p95={np.percentile(shuffled_ranges, 95):.4f}")
print(f"    actual percentile within shuffled dist: {percentile_rank(actual_range, shuffled_ranges):.1f}%")
print()
print(f"  Max:")
print(f"    actual: {actual_traj.max():.4f}")
print(f"    shuffled: median={np.median(shuffled_max):.4f}, p5={np.percentile(shuffled_max, 5):.4f}, p95={np.percentile(shuffled_max, 95):.4f}")
print(f"    actual percentile: {percentile_rank(actual_traj.max(), shuffled_max):.1f}%")
print()
print(f"  Min:")
print(f"    actual: {actual_traj.min():.4f}")
print(f"    shuffled: median={np.median(shuffled_min):.4f}, p5={np.percentile(shuffled_min, 5):.4f}, p95={np.percentile(shuffled_min, 95):.4f}")
print(f"    actual percentile (lower=more extreme): {percentile_rank(actual_traj.min(), shuffled_min):.1f}%")
print()
print(f"  Largest contiguous rise (any save i to save j>i):")
print(f"    actual: {actual_max_rise:.4f}")
print(f"    shuffled: mean={shuffled_max_consec_rise.mean():.4f}, p95={np.percentile(shuffled_max_consec_rise, 95):.4f}")
print(f"    actual percentile: {percentile_rank(actual_max_rise, shuffled_max_consec_rise):.1f}%")
print()
print(f"  Largest contiguous fall:")
print(f"    actual: {actual_max_fall:.4f}")
print(f"    shuffled: mean={shuffled_max_consec_fall.mean():.4f}, p95={np.percentile(shuffled_max_consec_fall, 95):.4f}")
print(f"    actual percentile: {percentile_rank(actual_max_fall, shuffled_max_consec_fall):.1f}%")
print()
print(f"  lag-1 autocorrelation between consecutive saves:")
print(f"    actual: {actual_lag1_ac:.4f}")
print(f"    shuffled: mean={shuffled_lag1_ac.mean():.4f}, std={shuffled_lag1_ac.std():.4f}")
print(f"    actual percentile: {percentile_rank(actual_lag1_ac, shuffled_lag1_ac):.1f}%")
print()
# 6. Sample 5 example shuffled trajectories for visual comparison
print(f"=== Sample 5 IID shuffled trajectories ===")
for trial in range(5):
    perm = rng.permutation(accs)
    traj = rolling_means_at_saves(perm, save_idxs)
    print(f"  shuffle {trial+1}: ", end="")
    print(" ".join(f"{m:.3f}" for m in traj))
print()
print(f"=== ACTUAL trajectory: ", end="")
print(" ".join(f"{m:.3f}" for m in actual_traj))
