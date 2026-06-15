"""Generate a GIF animation showing:
  - Left:  front-camera frame (most recent timestep) — fixed throughout
  - Right: BEV trajectory plot showing GT + predicted-under-each-CoC-variant trajectories,
           with current vehicle position highlighted at each waypoint frame.

Pulls raw images from alpamayo_clips/ and the prompt/CoC structure from target_coc_outputs/.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import einops

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, "/home/ubuntu/alpamayo_code/src")
sys.path.insert(0, "/home/ubuntu/dflash_code")

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from transformers import AutoProcessor
from transformers.cache_utils import DynamicCache

TRAJ_FUTURE_START = 155681
COT_START = 155677
COT_END = 155678

# Default donors: real in-distribution CoCs from training clips, picked for semantic diversity.
# Each entry is (label, donor_clip_uuid) — at runtime we extract the donor's actual GT CoC text
# and substitute it into the source clip's output_token_ids between <|cot_start|> and <|cot_end|>.
DEFAULT_DONORS = [
    ("orig",            None),  # no modification — use source clip's own CoC
    ("donor_stop",      "001ec074-2570-4f8e-9920-8660d54a83e0"),    # "Stop at the stop line..."
    ("donor_turn_right","0014feab-2f1a-4cbf-bac7-d1ea793f7d93"),    # "Turn right at the intersection..."
    ("donor_accelerate","00326722-3980-4049-91e8-7850a2ca4495"),    # "Accelerate to proceed through..."
    ("donor_nudge_left","000da9de-0ee5-465a-9a2d-e7e91d3016bb"),    # "Nudge to the left to increase clearance..."
]


@torch.no_grad()
def vlm_prefill(target_model, input_ids, pixel_values, image_grid_thw):
    vlm = target_model.vlm
    past = DynamicCache()
    out = vlm(
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


def integrate_trajectory(action_array, dt=0.1, v0=10.0):
    n = action_array.shape[0]
    accel = action_array[:, 0]; curv = action_array[:, 1]
    x = np.zeros(n + 1); y = np.zeros(n + 1); theta = np.zeros(n + 1)
    v = v0
    for t in range(n):
        theta[t + 1] = theta[t] + v * curv[t] * dt
        x[t + 1] = x[t] + v * np.cos(theta[t]) * dt
        y[t + 1] = y[t] + v * np.sin(theta[t]) * dt
        v = max(0.0, v + accel[t] * dt)
    return x, y, theta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_path", required=True)
    ap.add_argument("--processor_path", default=None)
    ap.add_argument("--target_outputs_dir", required=True)
    ap.add_argument("--alpamayo_clips_dir", required=True)
    ap.add_argument("--clip_uuid", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--camera_idx", type=int, default=0,
                    help="Which camera index to display (0..3). Default 0 = first listed camera.")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = "cuda"
    dt_torch = torch.bfloat16

    print(f"Loading target...", flush=True)
    target_model = AlpamayoR1.from_pretrained(args.target_path, dtype=dt_torch).to(device).eval()
    target_model.action_in_proj.to(torch.float32)
    target_model.action_out_proj.to(torch.float32)

    proc_path = args.processor_path or args.target_path
    print(f"Loading processor from {proc_path}", flush=True)
    processor = AutoProcessor.from_pretrained(proc_path, trust_remote_code=True)
    tokenizer = processor.tokenizer

    coc_path = os.path.join(args.target_outputs_dir, f"{args.clip_uuid}.pt")
    raw_path = os.path.join(args.alpamayo_clips_dir, f"{args.clip_uuid}.pt")
    print(f"Loading {coc_path}", flush=True)
    d = torch.load(coc_path, weights_only=False)
    print(f"Loading {raw_path}", flush=True)
    raw = torch.load(raw_path, weights_only=False)

    prompt_ids = d["prompt_input_ids"].to(device)
    orig_output_ids = d["output_token_ids"].to(device)
    pixel_values = d["pixel_values"].to(dt_torch).to(device)
    image_grid_thw = d["image_grid_thw"].to(device)
    P = prompt_ids.shape[1]

    # Raw camera frames: (n_cams, n_timesteps, 3, H, W) uint8
    image_frames = raw["data"]["image_frames"]
    camera_indices = raw["data"]["camera_indices"]
    print(f"image_frames shape: {tuple(image_frames.shape)} | camera_indices: {camera_indices.tolist()}")
    # Use the most recent frame from chosen camera
    cam_idx = args.camera_idx
    last_t = -1  # most recent timestep
    front_img = image_frames[cam_idx, last_t].permute(1, 2, 0).cpu().numpy()  # (H, W, 3) uint8

    # Ground truth ego trajectory (xyz in ego frame)
    ego_future_xyz = raw["data"]["ego_future_xyz"][0, 0].cpu().numpy()  # (64, 3)
    ego_history_xyz = raw["data"]["ego_history_xyz"][0, 0].cpu().numpy()  # (16, 3)

    # Estimate initial velocity from ego_history (last 2 points → speed)
    if ego_history_xyz.shape[0] >= 2:
        dxy = ego_history_xyz[-1, :2] - ego_history_xyz[-2, :2]
        v0 = float(np.linalg.norm(dxy) / 0.1)  # assuming dt=0.1s between history samples
        v0 = max(v0, 1.0)
    else:
        v0 = 10.0
    print(f"Initial speed estimate v0 = {v0:.2f} m/s")

    # Decode original CoC
    cot_start_idx = (orig_output_ids == COT_START).nonzero(as_tuple=True)[0]
    cot_end_idx = (orig_output_ids == COT_END).nonzero(as_tuple=True)[0]
    s = int(cot_start_idx[0]) if len(cot_start_idx) > 0 else -1
    e = int(cot_end_idx[0]) if len(cot_end_idx) > 0 else -1
    if s >= 0 and e >= 0:
        orig_coc_text = tokenizer.decode(orig_output_ids[s+1:e].cpu().tolist())
    else:
        orig_coc_text = tokenizer.decode(orig_output_ids.cpu().tolist())
    print(f"Original CoC: {orig_coc_text}")

    # Resolve each donor uuid → its actual CoC text by loading the donor clip
    label_to_text = {}
    for label, donor_uuid in DEFAULT_DONORS:
        if donor_uuid is None:
            label_to_text[label] = orig_coc_text  # source clip's own GT CoC
            continue
        donor_path = os.path.join(args.target_outputs_dir, f"{donor_uuid}.pt")
        if not os.path.exists(donor_path):
            print(f"  [skip {label}] donor clip not found: {donor_path}")
            continue
        donor_d = torch.load(donor_path, weights_only=False)
        donor_oids = donor_d["output_token_ids"]
        ds = (donor_oids == COT_START).nonzero(as_tuple=True)[0]
        de = (donor_oids == COT_END).nonzero(as_tuple=True)[0]
        if len(ds) == 0 or len(de) == 0:
            print(f"  [skip {label}] donor has no CoC tokens")
            continue
        donor_text = tokenizer.decode(donor_oids[int(ds[0])+1:int(de[0])].tolist())
        label_to_text[label] = donor_text
    print("\nResolved CoC texts per variant:")
    for label, txt in label_to_text.items():
        prefix = "(SOURCE GT)" if label == "orig" else "(DONOR)"
        print(f"  [{label}] {prefix}: {txt}")

    # Run target VLM+diffusion under each CoC variant
    actions_per_label = {}
    integrated_per_label = {}
    print()
    for label, mod_text in label_to_text.items():
        if label == "orig":
            new_output = orig_output_ids
        elif s < 0 or e < 0:
            print(f"  Skip [{label}]: cannot find <|cot_start|>/<|cot_end|> in source")
            continue
        else:
            mod_tok = tokenizer(mod_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
            new_output = torch.cat([orig_output_ids[:s+1], mod_tok, orig_output_ids[e:]])

        full_ids = torch.cat([prompt_ids[0], new_output, torch.tensor([TRAJ_FUTURE_START], device=device)]).unsqueeze(0)
        traj_pos = torch.tensor([P + new_output.shape[0]], device=device)
        cache, rd = vlm_prefill(target_model, full_ids, pixel_values, image_grid_thw)
        action = run_diffusion(target_model, cache, rd, traj_pos, seed=args.seed)
        action_np = action[0].float().cpu().numpy()  # (64, 2)
        actions_per_label[label] = action_np
        integrated_per_label[label] = integrate_trajectory(action_np, dt=0.1, v0=v0)
        print(f"  [{label}] action[0]={action_np[0].round(3).tolist()}")

    # ---- Build animation ----
    # Layout: gridspec — top row 4 camera images, bottom row BEV (left) + CoC-text panel (right)
    n_cams = image_frames.shape[0]
    fig = plt.figure(figsize=(22, 11))
    gs = fig.add_gridspec(2, max(4, n_cams), height_ratios=[1.0, 2.2], hspace=0.25, wspace=0.1)

    # Top row: all camera views at most recent timestep
    cam_axes = []
    for ci in range(n_cams):
        ax = fig.add_subplot(gs[0, ci])
        img = image_frames[ci, last_t].permute(1, 2, 0).cpu().numpy()
        ax.imshow(img)
        ax.set_title(f"Camera idx={int(camera_indices[ci])} (t=now)", fontsize=9)
        ax.axis("off")
        cam_axes.append(ax)

    # Bottom-left: BEV plot
    ax_bev = fig.add_subplot(gs[1, : max(2, n_cams // 2)])
    ax_bev.set_title("Predicted vehicle trajectory (BEV)\nGT and per-CoC predictions",
                     fontsize=11)

    color_map = {"orig": "C0", "donor_stop": "C1", "donor_turn_right": "C2",
                 "donor_accelerate": "C3", "donor_nudge_left": "C4"}

    # Bottom-right: CoC text panel
    ax_text = fig.add_subplot(gs[1, max(2, n_cams // 2):])
    ax_text.axis("off")
    ax_text.set_xlim(0, 1); ax_text.set_ylim(0, 1)
    ax_text.text(0.02, 0.98, "CoC variants fed to model (real donor CoCs from other clips):",
                 transform=ax_text.transAxes,
                 verticalalignment="top", fontsize=11, fontweight="bold")
    y_text = 0.92
    for label, _ in DEFAULT_DONORS:
        if label not in label_to_text:
            continue
        text = label_to_text[label]
        if len(text) > 95:
            text = text[:92] + "..."
        col = color_map.get(label, "gray")
        prefix = "GT CoC" if label == "orig" else "DONOR"
        header = f"●  [{label}]   ← {prefix}"
        ax_text.text(0.02, y_text, header, transform=ax_text.transAxes,
                     verticalalignment="top", fontsize=10, fontweight="bold", color=col)
        ax_text.text(0.06, y_text - 0.04, f'"{text}"', transform=ax_text.transAxes,
                     verticalalignment="top", fontsize=9, family="monospace",
                     wrap=True)
        y_text -= 0.18

    # Plot GT future (in xy plane). Note ego_future_xyz: column 0 = forward, column 1 = lateral, column 2 = up.
    gt_x = ego_future_xyz[:, 0]
    gt_y = ego_future_xyz[:, 1]
    ax_bev.plot(gt_x, gt_y, "k--", linewidth=2, alpha=0.7, label="GT future")

    # Static lines for each prediction (color_map already defined above)
    for label, (px, py, _) in integrated_per_label.items():
        ax_bev.plot(px, py, color=color_map.get(label, "gray"),
                    linewidth=1.5, alpha=0.6, label=label)

    # Mark START
    ax_bev.scatter([0], [0], s=200, c="black", marker="*", zorder=10, label="START (vehicle now)")

    # Animated dots: one per CoC variant
    moving_dots = {}
    for label in integrated_per_label:
        (dot,) = ax_bev.plot([], [], "o", color=color_map.get(label, "gray"),
                              markersize=12, markeredgecolor="black", markeredgewidth=0.5,
                              zorder=11)
        moving_dots[label] = dot

    gt_dot, = ax_bev.plot([], [], "o", color="black", markersize=10,
                          markeredgecolor="white", markeredgewidth=1, zorder=11)

    # Determine plot range
    all_x = np.concatenate([gt_x] + [v[0] for v in integrated_per_label.values()])
    all_y = np.concatenate([gt_y] + [v[1] for v in integrated_per_label.values()])
    margin = 5.0
    ax_bev.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax_bev.set_ylim(all_y.min() - margin, all_y.max() + margin)
    ax_bev.set_xlabel("x (m, forward)")
    ax_bev.set_ylabel("y (m, lateral, +=left)")
    ax_bev.set_aspect("equal")
    ax_bev.grid(True, alpha=0.3)
    ax_bev.legend(loc="best", fontsize=8)

    title_text = ax_bev.text(0.02, 0.98, "", transform=ax_bev.transAxes,
                              verticalalignment="top", fontsize=10,
                              bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    n_frames = ego_future_xyz.shape[0]  # 64

    def animate(t):
        # Move GT dot
        gt_dot.set_data([gt_x[t]], [gt_y[t]])
        # Move each CoC dot to position at waypoint t
        for label, (px, py, _) in integrated_per_label.items():
            # px has length 65 (n+1); waypoint t corresponds to index t+1
            idx = min(t + 1, len(px) - 1)
            moving_dots[label].set_data([px[idx]], [py[idx]])
        title_text.set_text(f"t = {t * 0.1:.1f} s  (waypoint {t}/63)")
        return list(moving_dots.values()) + [gt_dot, title_text]

    print(f"Building animation ({n_frames} frames)...")
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=100, blit=False)
    out_path = os.path.join(args.out_dir, f"{args.clip_uuid}_coc_animated.gif")
    anim.save(out_path, writer=PillowWriter(fps=10))
    print(f"\nSaved animation to {out_path}")


if __name__ == "__main__":
    main()
