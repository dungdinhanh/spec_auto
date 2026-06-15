"""Trajectory-history disturbance experiment.

For a single source clip, run target VLM+diffusion under several variants:
  - no_fuse  : prompt with placeholder traj_history tokens (matches existing animations)
  - src_hist : fuse with the source clip's own ego_history (canonical / sanity)
  - donor_*  : fuse with a donor clip's ego_history (history is wrong relative to images/CoC)

Compare action[0..5] across variants. If actions barely change, the model isn't
conditioning on history. If they shift meaningfully, history is a real input.
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
import torch
import einops

sys.path.insert(0, "/home/ubuntu/alpamayo_code/src")
sys.path.insert(0, "/home/ubuntu/dflash_code")

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from transformers import AutoProcessor
from transformers.cache_utils import DynamicCache

TRAJ_FUTURE_START = 155681
TRAJ_HISTORY_PLACEHOLDER = 155684

# Donors picked from a 60-clip scan to span ego_history regimes.
# label, donor_uuid, descriptive note
DONORS = [
    ("no_fuse",          None,                                       "(placeholder tokens, no history fused)"),
    ("src_hist",         "SRC",                                       "(source clip's own ego_history)"),
    ("donor_stationary", "00755b25-51ab-49b4-ae9d-29f80f554db0",     "stationary, v=0 m/s, travel=0 m"),
    ("donor_highway",    "00af351a-7b12-44c8-9dba-e4b8a70cd653",     "highway straight, v=35.7 m/s, travel=53.5 m"),
    ("donor_high_speed", "010d70d3-4efa-4d82-b43c-37d420a8cbbe",     "high speed, v=26.4 m/s, travel=40.2 m"),
    ("donor_turning",    "00f2e502-9fba-43a1-9eb3-4bed06862570",     "stationary turning, v=0 m/s, yaw=9.5 rad"),
    ("donor_slow_turn",  "0014feab-2f1a-4cbf-bac7-d1ea793f7d93",     "slow turning, v=3.5 m/s, yaw=0.4 rad"),
]


@torch.no_grad()
def vlm_prefill(target_model, input_ids, pixel_values, image_grid_thw):
    vlm = target_model.vlm
    past = DynamicCache()
    vlm(
        input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw,
        past_key_values=past, use_cache=True, return_dict=True,
    )
    return past, vlm.model.rope_deltas


@torch.no_grad()
def run_diffusion(target_model, prompt_cache, rope_deltas, traj_future_start_pos, seed):
    device = rope_deltas.device
    B = traj_future_start_pos.shape[0]
    n_dt = target_model.action_space.get_action_space_dims()[0]
    prefill_seq_len = prompt_cache.get_seq_length()

    position_ids = torch.arange(n_dt, device=device)
    position_ids = einops.repeat(position_ids, "l -> 3 b l", b=B).clone()
    offset = traj_future_start_pos + 1
    position_ids += (rope_deltas + offset[:, None]).to(position_ids.device)

    expert_dtype = next(target_model.expert.parameters()).dtype
    attention_mask = torch.zeros(
        (B, 1, n_dt, prefill_seq_len + n_dt), dtype=expert_dtype, device=device,
    )
    for i in range(B):
        attention_mask[i, :, :, offset[i]:-n_dt] = torch.finfo(attention_mask.dtype).min

    forward_kwargs = {}
    if target_model.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    in_dtype = next(target_model.action_in_proj.parameters()).dtype
    out_dtype = next(target_model.action_out_proj.parameters()).dtype
    initial_seq_len = prompt_cache.get_seq_length()

    torch.manual_seed(seed)

    def step_fn(x, t):
        if prompt_cache.get_seq_length() > initial_seq_len:
            prompt_cache.crop(initial_seq_len)
        fte = target_model.action_in_proj(
            x.to(in_dtype), t.to(in_dtype) if torch.is_tensor(t) else t,
        )
        n_dtt = fte.shape[1]
        exp_out = target_model.expert(
            inputs_embeds=fte.to(expert_dtype), past_key_values=prompt_cache,
            attention_mask=attention_mask, position_ids=position_ids,
            use_cache=False, **forward_kwargs,
        )
        lh = exp_out.last_hidden_state[:, -n_dtt:].to(out_dtype)
        return target_model.action_out_proj(lh).to(x.dtype)

    return target_model.diffusion.sample(
        batch_size=B, step_fn=step_fn, device=device, return_all_steps=False,
    )


def integrate(action, dt=0.1, v0=10.0):
    n = action.shape[0]
    accel = action[:, 0]; curv = action[:, 1]
    x = np.zeros(n + 1); y = np.zeros(n + 1); theta = np.zeros(n + 1)
    v = v0
    for t in range(n):
        theta[t + 1] = theta[t] + v * curv[t] * dt
        x[t + 1] = x[t] + v * np.cos(theta[t]) * dt
        y[t + 1] = y[t] + v * np.sin(theta[t]) * dt
        v = max(0.0, v + accel[t] * dt)
    return x, y


def history_speed_summary(hist_xyz):
    # hist_xyz: (T, 3) numpy
    if hist_xyz.shape[0] < 2:
        return 0.0, 0.0
    v_recent = float(np.linalg.norm(hist_xyz[-1, :2] - hist_xyz[-2, :2]) / 0.1)
    travel = float(np.sum(np.linalg.norm(np.diff(hist_xyz[:, :2], axis=0), axis=1)))
    return v_recent, travel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_path", required=True)
    ap.add_argument("--processor_path", default=None)
    ap.add_argument("--target_outputs_dir", required=True)
    ap.add_argument("--alpamayo_clips_dir", required=True)
    ap.add_argument("--clip_uuid", required=True, help="Source clip UUID")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = "cuda"
    dt_torch = torch.bfloat16

    print(f"Loading target...", flush=True)
    target_model = AlpamayoR1.from_pretrained(args.target_path, dtype=dt_torch).to(device).eval()
    target_model.action_in_proj.to(torch.float32)
    target_model.action_out_proj.to(torch.float32)

    proc_path = args.processor_path or args.target_path
    processor = AutoProcessor.from_pretrained(proc_path, trust_remote_code=True)
    tokenizer = processor.tokenizer

    coc_path = os.path.join(args.target_outputs_dir, f"{args.clip_uuid}.pt")
    raw_path = os.path.join(args.alpamayo_clips_dir, f"{args.clip_uuid}.pt")
    print(f"Loading source CoC outputs: {coc_path}", flush=True)
    d = torch.load(coc_path, weights_only=False)
    print(f"Loading source raw clip:    {raw_path}", flush=True)
    raw_src = torch.load(raw_path, weights_only=False)

    prompt_ids = d["prompt_input_ids"].to(device)         # (1, 3005), with placeholders
    orig_output_ids = d["output_token_ids"].to(device)
    pixel_values = d["pixel_values"].to(dt_torch).to(device)
    image_grid_thw = d["image_grid_thw"].to(device)
    P = prompt_ids.shape[1]

    n_placeholders = int((prompt_ids[0] == TRAJ_HISTORY_PLACEHOLDER).sum().item())
    print(f"prompt has {n_placeholders} traj_history placeholder tokens (id={TRAJ_HISTORY_PLACEHOLDER})")

    # Source ego_history (for v0 estimate + src_hist variant)
    src_hist_xyz_t = raw_src["data"]["ego_history_xyz"].to(device)  # (1,1,16,3)
    src_hist_rot_t = raw_src["data"]["ego_history_rot"].to(device)  # (1,1,16,3,3)
    src_hist_xyz_np = src_hist_xyz_t[0, 0].float().cpu().numpy()
    v_src, travel_src = history_speed_summary(src_hist_xyz_np)
    v0 = max(v_src, 1.0)
    print(f"Source ego_history: v_recent={v_src:.2f} m/s, travel={travel_src:.1f} m -> v0={v0:.2f} m/s")

    # Pre-load donor traj_data
    donor_traj = {}
    for label, uuid, _note in DONORS:
        if uuid is None or uuid == "SRC":
            continue
        donor_path = os.path.join(args.alpamayo_clips_dir, f"{uuid}.pt")
        if not os.path.exists(donor_path):
            print(f"  [skip {label}] donor raw not found: {donor_path}")
            continue
        donor_raw = torch.load(donor_path, weights_only=False)
        hist_xyz = donor_raw["data"]["ego_history_xyz"].to(device)
        hist_rot = donor_raw["data"]["ego_history_rot"].to(device)
        v, travel = history_speed_summary(hist_xyz[0, 0].float().cpu().numpy())
        donor_traj[label] = {
            "ego_history_xyz": hist_xyz,
            "ego_history_rot": hist_rot,
            "_v": v, "_travel": travel,
        }
        print(f"  [{label}] loaded donor {uuid[:8]} v={v:.2f} travel={travel:.1f}")

    # Source's own CoC output gets reused for every variant — we keep the text identical
    # so the only change between variants is the trajectory history fed to the model.
    src_full_output = orig_output_ids  # already includes <|cot_start|> ... <|cot_end|> ...
    print()

    actions_per = {}
    for label, uuid, note in DONORS:
        # Build input_ids (with traj_history placeholders still in)
        full_ids = torch.cat([
            prompt_ids[0],
            src_full_output,
            torch.tensor([TRAJ_FUTURE_START], device=device),
        ]).unsqueeze(0).clone()

        # Fuse history if requested
        if uuid is None:
            traj_data = None
            tag = "no fusion"
        elif uuid == "SRC":
            traj_data = {"ego_history_xyz": src_hist_xyz_t, "ego_history_rot": src_hist_rot_t}
            tag = f"SRC v={v_src:.1f}"
        else:
            if label not in donor_traj:
                print(f"  [skip {label}] donor not loaded")
                continue
            traj_data = {
                "ego_history_xyz": donor_traj[label]["ego_history_xyz"],
                "ego_history_rot": donor_traj[label]["ego_history_rot"],
            }
            tag = f"DONOR v={donor_traj[label]['_v']:.1f}"

        if traj_data is not None:
            full_ids = target_model.fuse_traj_tokens(full_ids, traj_data)
            n_left = int((full_ids[0] == TRAJ_HISTORY_PLACEHOLDER).sum().item())
            assert n_left == 0, f"{n_left} placeholders remaining after fuse"

        traj_pos = torch.tensor([P + src_full_output.shape[0]], device=device)
        cache, rd = vlm_prefill(target_model, full_ids, pixel_values, image_grid_thw)
        action = run_diffusion(target_model, cache, rd, traj_pos, seed=args.seed)
        action_np = action[0].float().cpu().numpy()
        actions_per[label] = action_np

        a05 = action_np[:6].round(4).tolist()
        print(f"  [{label:18s}] {tag:18s} {note}")
        print(f"      action[0..5] = {a05}")

    # Pairwise diff vs. baselines
    print("\n=== |Δ action[0..63]| (mean L2) vs. no_fuse baseline ===")
    if "no_fuse" in actions_per:
        base = actions_per["no_fuse"]
        for label in actions_per:
            if label == "no_fuse":
                continue
            diff = actions_per[label] - base
            mean_l2 = float(np.linalg.norm(diff, axis=1).mean())
            max_l2 = float(np.linalg.norm(diff, axis=1).max())
            # split by accel/curv
            mean_da = float(np.abs(diff[:, 0]).mean())
            mean_dc = float(np.abs(diff[:, 1]).mean())
            print(f"  [{label:18s}] mean_L2={mean_l2:.5f} max_L2={max_l2:.5f} | mean|Δaccel|={mean_da:.5f} mean|Δcurv|={mean_dc:.5f}")

    print("\n=== |Δ action| vs. src_hist baseline ===")
    if "src_hist" in actions_per:
        base = actions_per["src_hist"]
        for label in actions_per:
            if label == "src_hist":
                continue
            diff = actions_per[label] - base
            mean_l2 = float(np.linalg.norm(diff, axis=1).mean())
            max_l2 = float(np.linalg.norm(diff, axis=1).max())
            mean_da = float(np.abs(diff[:, 0]).mean())
            mean_dc = float(np.abs(diff[:, 1]).mean())
            print(f"  [{label:18s}] mean_L2={mean_l2:.5f} max_L2={max_l2:.5f} | mean|Δaccel|={mean_da:.5f} mean|Δcurv|={mean_dc:.5f}")

    # Save raw actions
    out = os.path.join(args.out_dir, f"{args.clip_uuid}_traj_disturb.pt")
    torch.save({
        "clip_uuid": args.clip_uuid,
        "v0": v0,
        "donors": [(l, u, n) for l, u, n in DONORS],
        "actions": actions_per,
    }, out)
    print(f"\nSaved actions to {out}")


if __name__ == "__main__":
    main()
