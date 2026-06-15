"""Modify the chain-of-context (CoC) tokens significantly (e.g., 'turn right 90 degrees')
and visualize the predicted vehicle trajectory under each modified CoC.

Pipeline per clip:
  1. Load clip (prompt_input_ids, output_token_ids, pixel_values, image_grid_thw)
  2. Decode original CoC to text
  3. Define a list of modified CoC variants
  4. For each variant: tokenize → build (prompt + CoC + traj_future_start) → VLM prefill → diffusion → (64, 2) action
  5. Convert (accel, curvature) waypoints → 2D trajectory via simple bicycle-model integration
  6. Save plot per clip + a JSON summary
"""
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import einops

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/ubuntu/alpamayo_code/src")
sys.path.insert(0, "/home/ubuntu/dflash_code")

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from transformers import AutoProcessor
from transformers.cache_utils import DynamicCache

TRAJ_FUTURE_START = 155681
COT_START = 155677
COT_END = 155678
IM_END = 151645

# A few modifications to try
MODIFICATIONS = [
    # (label, replacement_text) — replacement CoC will be everything between <|cot_start|> and <|cot_end|>
    ("orig", None),  # no modification, just sanity
    ("turn_right_90", "I will make a sharp right turn, approximately 90 degrees, immediately."),
    ("turn_left_90",  "I will make a sharp left turn, approximately 90 degrees, immediately."),
    ("hard_brake",    "I will brake hard and come to a complete stop immediately."),
    ("accelerate",    "I will accelerate strongly and continue straight ahead at maximum throttle."),
]


@torch.no_grad()
def vlm_prefill(target_model, input_ids, pixel_values, image_grid_thw):
    vlm = target_model.vlm
    past = DynamicCache()
    out = vlm(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        past_key_values=past,
        use_cache=True,
        return_dict=True,
    )
    rope_deltas = vlm.model.rope_deltas
    return past, rope_deltas


@torch.no_grad()
def run_diffusion(target_model, prompt_cache, rope_deltas, traj_future_start_pos, seed):
    device = rope_deltas.device
    B = traj_future_start_pos.shape[0]
    n_diffusion_tokens = target_model.action_space.get_action_space_dims()[0]
    prefill_seq_len = prompt_cache.get_seq_length()

    position_ids = torch.arange(n_diffusion_tokens, device=device)
    position_ids = einops.repeat(position_ids, "l -> 3 b l", b=B).clone()
    offset = traj_future_start_pos + 1
    delta = rope_deltas + offset[:, None]
    position_ids += delta.to(position_ids.device)

    expert_dtype = next(target_model.expert.parameters()).dtype
    attention_mask = torch.zeros(
        (B, 1, n_diffusion_tokens, prefill_seq_len + n_diffusion_tokens),
        dtype=expert_dtype, device=device,
    )
    for i in range(B):
        attention_mask[i, :, :, offset[i]:-n_diffusion_tokens] = torch.finfo(attention_mask.dtype).min

    forward_kwargs = {}
    if target_model.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    in_dtype = next(target_model.action_in_proj.parameters()).dtype
    out_dtype = next(target_model.action_out_proj.parameters()).dtype

    torch.manual_seed(seed)

    initial_seq_len = prompt_cache.get_seq_length()

    def step_fn(x, t):
        # Crop cache back to the prefill length each call so it doesn't grow
        # across diffusion euler steps (which would mismatch attention_mask).
        if prompt_cache.get_seq_length() > initial_seq_len:
            prompt_cache.crop(initial_seq_len)
        fte = target_model.action_in_proj(
            x.to(in_dtype),
            t.to(in_dtype) if torch.is_tensor(t) else t,
        )
        n_dt = fte.shape[1]
        exp_out = target_model.expert(
            inputs_embeds=fte.to(expert_dtype),
            past_key_values=prompt_cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            **forward_kwargs,
        )
        lh = exp_out.last_hidden_state[:, -n_dt:].to(out_dtype)
        return target_model.action_out_proj(lh).to(x.dtype)

    sampled = target_model.diffusion.sample(
        batch_size=B, step_fn=step_fn, device=device,
        return_all_steps=False,
    )
    return sampled  # (B, n_dt, action_dim)


def integrate_trajectory(action_array, dt=0.1, v0=10.0):
    """Simple bicycle-model integration of (accel, curvature) waypoints.
    accel in m/s^2, curvature in 1/m.
    Returns (x, y) trajectory (T+1 points starting from origin).
    """
    n = action_array.shape[0]
    accel = action_array[:, 0]
    curv = action_array[:, 1]
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    theta = np.zeros(n + 1)
    v = v0
    for t in range(n):
        # Update heading (bicycle model: dtheta/dt = v * curvature)
        theta[t + 1] = theta[t] + v * curv[t] * dt
        # Update position
        x[t + 1] = x[t] + v * np.cos(theta[t]) * dt
        y[t + 1] = y[t] + v * np.sin(theta[t]) * dt
        # Update velocity
        v = max(0.0, v + accel[t] * dt)
    return x, y, theta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_path", required=True)
    ap.add_argument("--processor_path", default=None,
                    help="Path to Qwen3-VL-2B processor (for tokenizer). If None, use target_path.")
    ap.add_argument("--target_outputs_dir", required=True)
    ap.add_argument("--clip_uuid", required=True, help="Clip UUID stem (without .pt)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = "cuda"
    dt_torch = torch.bfloat16

    print(f"Loading target from {args.target_path}", flush=True)
    target_model = AlpamayoR1.from_pretrained(args.target_path, dtype=dt_torch).to(device).eval()
    # Force action_in_proj and action_out_proj to fp32 so the diffusion noise (fp32)
    # matches the projection weights. The trunk inside action_in_proj contains
    # nn.Linear layers loaded in bf16 by default — they need to be fp32 to match
    # x.to(fp32) coming out of the FlowMatching sampler.
    target_model.action_in_proj.to(torch.float32)
    target_model.action_out_proj.to(torch.float32)

    proc_path = args.processor_path or args.target_path
    print(f"Loading processor from {proc_path}", flush=True)
    processor = AutoProcessor.from_pretrained(proc_path, trust_remote_code=True)
    tokenizer = processor.tokenizer

    clip_path = os.path.join(args.target_outputs_dir, f"{args.clip_uuid}.pt")
    print(f"Loading clip {clip_path}", flush=True)
    d = torch.load(clip_path, weights_only=False)
    prompt_ids = d["prompt_input_ids"].to(device)              # (1, P)
    orig_output_ids = d["output_token_ids"].to(device)         # (N,)
    pixel_values = d["pixel_values"].to(dt_torch).to(device)
    image_grid_thw = d["image_grid_thw"].to(device)

    # Decode original CoC
    cot_start_idx = (orig_output_ids == COT_START).nonzero(as_tuple=True)
    cot_end_idx = (orig_output_ids == COT_END).nonzero(as_tuple=True)
    if len(cot_start_idx[0]) > 0 and len(cot_end_idx[0]) > 0:
        s = cot_start_idx[0][0].item()
        e = cot_end_idx[0][0].item()
        orig_coc_text = tokenizer.decode(orig_output_ids[s+1:e].cpu().tolist())
    else:
        s, e = -1, -1
        orig_coc_text = tokenizer.decode(orig_output_ids.cpu().tolist())

    print(f"\n=== Original CoC text ({len(orig_output_ids)} tokens) ===")
    print(f"{orig_coc_text}")

    # Build modified output_token_ids variants.
    # Each variant has the same prefix (everything up to <|cot_start|>) and suffix (after <|cot_end|>),
    # but the body between them is replaced by the new text.
    P = prompt_ids.shape[1]
    results = {}
    actions_collected = {}

    for label, mod_text in MODIFICATIONS:
        if mod_text is None:
            new_output_ids = orig_output_ids
            mod_text_for_record = orig_coc_text
        else:
            if s < 0 or e < 0:
                print(f"  [{label}] cannot find <|cot_start|>/<|cot_end|> in clip, skipping.")
                continue
            mod_token_ids = tokenizer(mod_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
            new_output_ids = torch.cat([
                orig_output_ids[:s+1],          # ... <|cot_start|>
                mod_token_ids,                   # new CoC body
                orig_output_ids[e:],             # <|cot_end|> ... <|im_end|>
            ])
            mod_text_for_record = mod_text

        # Build full sequence (prompt + output + traj_future_start)
        full_ids = torch.cat([prompt_ids[0], new_output_ids,
                              torch.tensor([TRAJ_FUTURE_START], device=device)])
        full_ids = full_ids.unsqueeze(0)

        traj_pos = torch.tensor([P + new_output_ids.shape[0]], device=device)

        cache, rd = vlm_prefill(target_model, full_ids, pixel_values, image_grid_thw)
        action = run_diffusion(target_model, cache, rd, traj_pos, seed=args.seed)
        action_np = action[0].float().cpu().numpy()  # (n_dt, 2)

        actions_collected[label] = action_np
        results[label] = {
            "coc_text": mod_text_for_record,
            "n_tokens": int(new_output_ids.shape[0]),
            "action_first10": action_np[:10].tolist(),
        }

        # Print quick summary
        print(f"\n=== [{label}] CoC = {mod_text_for_record[:100]!r}{'...' if len(mod_text_for_record) > 100 else ''} ===")
        print(f"  num_tokens={new_output_ids.shape[0]}")
        print(f"  action [0..5]: {action_np[:6].round(3).tolist()}")
        print(f"  action [60..63]: {action_np[60:64].round(3).tolist()}")

    # Plot (accel, curvature) profiles + integrated 2D trajectories
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Subplot 1: acceleration over time
    for label, action in actions_collected.items():
        axes[0].plot(action[:, 0], label=label)
    axes[0].set_xlabel("waypoint")
    axes[0].set_ylabel("acceleration (m/s²)")
    axes[0].set_title("Acceleration profile")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: curvature over time
    for label, action in actions_collected.items():
        axes[1].plot(action[:, 1], label=label)
    axes[1].set_xlabel("waypoint")
    axes[1].set_ylabel("curvature (1/m)")
    axes[1].set_title("Curvature profile")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Subplot 3: integrated 2D trajectory
    for label, action in actions_collected.items():
        x, y, _ = integrate_trajectory(action, dt=0.1, v0=10.0)
        axes[2].plot(x, y, marker='o', markersize=2, label=label)
    axes[2].set_xlabel("x (m, forward)")
    axes[2].set_ylabel("y (m, lateral)")
    axes[2].set_title(f"Integrated 2D trajectory (dt=0.1s, v0=10 m/s)")
    axes[2].legend()
    axes[2].set_aspect("equal")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f"Clip {args.clip_uuid[:8]} — CoC modification → predicted vehicle trajectory")
    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, f"{args.clip_uuid}_coc_modified.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved plot to {plot_path}")

    # Save JSON summary
    json_path = os.path.join(args.out_dir, f"{args.clip_uuid}_summary.json")
    with open(json_path, "w") as f:
        json.dump({"clip_uuid": args.clip_uuid, "results": results}, f, indent=2)
    print(f"Saved summary to {json_path}")


if __name__ == "__main__":
    main()
