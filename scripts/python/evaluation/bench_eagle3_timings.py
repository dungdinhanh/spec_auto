"""Bench EAGLE-3 draft kernel timings for theoretical speedup.

Reuses target-side numbers from bench_isolated_timings.py (T_target_step,
T_target_verify) — they're target-only and identical across drafts. This
script only measures the draft side:

  - T_draft_prefill : draft midlayer step 0 over the prefix (q_len = L)
  - T_draft_chain   : one chain step at virtual position (q_len = 1)
  - Then T_draft_total = T_draft_prefill + gamma * T_draft_chain

Final theoretical = avg_iter * T_target_step / (T_draft_total + T_target_verify)

Writes the same JSON schema (with extra fields) so the aggregator can pick
either DFlash or EAGLE-3 drafts.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper
from alpamayo_r1.models.dflash_draft import get_qwen3vl_embed_and_head
from alpamayo_r1.models.eagle3_draft import (
    Eagle3DraftModel, Eagle3DraftConfig, load_eagle3_ckpt,
)
from transformers.cache_utils import DynamicCache


def _build_draft(target, ckpt, dtype=torch.bfloat16):
    sd = ckpt["state_dict"]
    cfg_dict = ckpt.get("config", {})
    target_layer_ids = ckpt.get("target_layer_ids") or [1, 17, 32]
    rollout_length = ckpt.get("rollout_length", 7)
    text_cfg = target.config.get_text_config()
    cfg = Eagle3DraftConfig(
        hidden_size=cfg_dict.get("hidden_size", text_cfg.hidden_size),
        intermediate_size=cfg_dict.get("intermediate_size", text_cfg.intermediate_size),
        num_hidden_layers=1,
        num_attention_heads=cfg_dict.get("num_attention_heads", text_cfg.num_attention_heads),
        num_key_value_heads=cfg_dict.get("num_key_value_heads", text_cfg.num_key_value_heads),
        vocab_size=cfg_dict.get("vocab_size", text_cfg.vocab_size),
        rms_norm_eps=cfg_dict.get("rms_norm_eps", text_cfg.rms_norm_eps),
        rope_theta=cfg_dict.get("rope_theta", getattr(text_cfg, "rope_theta", 10000)),
        max_position_embeddings=cfg_dict.get("max_position_embeddings",
                                              getattr(text_cfg, "max_position_embeddings", 8192)),
        target_layer_ids=target_layer_ids,
        rollout_length=rollout_length,
    )
    m = Eagle3DraftModel(cfg).to(dtype=dtype)
    m.load_state_dict(sd, strict=False)
    return m


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_path", required=True)
    ap.add_argument("--draft_path", required=True)
    ap.add_argument("--clip_path", required=True)
    ap.add_argument("--gamma", type=int, default=15)
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()
    device = "cuda"
    block_size = args.gamma + 1  # match DFlash convention (bs = gamma+1)

    print(f"loading target from {args.target_path}")
    model = AlpamayoR1.from_pretrained(args.target_path, dtype=torch.bfloat16)
    vlm = model.vlm.to(device).eval()
    embed_tokens, lm_head = get_qwen3vl_embed_and_head(vlm)
    tokenizer = model.tokenizer
    processor = helper.get_processor(tokenizer)

    print(f"loading EAGLE-3 draft from {args.draft_path}")
    ckpt = load_eagle3_ckpt(args.draft_path, map_location=device)
    draft = _build_draft(vlm, ckpt).to(device).eval()

    print(f"loading clip {args.clip_path}")
    clip = torch.load(args.clip_path, weights_only=False)
    messages = clip["messages"] if "messages" in clip else helper.create_message(
        clip["data"]["image_frames"].flatten(0, 1))
    prompt_messages = [m for m in messages if m["role"] != "assistant"]
    inputs = processor.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    # --- target prefill once to get the 3 fused hiddens at the prompt range ---
    past = DynamicCache()
    pkw = dict(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
               past_key_values=past, use_cache=True, output_hidden_states=True,
               return_dict=True)
    if "pixel_values" in inputs: pkw["pixel_values"] = inputs["pixel_values"]
    if "image_grid_thw" in inputs: pkw["image_grid_thw"] = inputs["image_grid_thw"]
    prefill = vlm(**pkw)
    target_h = [prefill.hidden_states[i + 1] for i in draft.target_layer_ids]
    bonus = prefill.logits[:, -1:, :].argmax(dim=-1)
    L = target_h[0].shape[1]
    print(f"prefix L = {L}")

    # --- T_target_step (AR decode, single token) ---
    cache_pos = torch.tensor([L], device=device)
    next_tok = bonus
    times = []
    for _ in range(args.warmup):
        out = vlm(input_ids=next_tok, past_key_values=past,
                  cache_position=cache_pos, use_cache=True, return_dict=True)
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        cache_pos = cache_pos + 1
    torch.cuda.synchronize()
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        out = vlm(input_ids=next_tok, past_key_values=past,
                  cache_position=cache_pos, use_cache=True, return_dict=True)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        cache_pos = cache_pos + 1
    times.sort()
    ar_step_ms = times[len(times) // 2]

    # Truncate target's KV back to L+1 (just the bonus added, like real iter 1).
    past.crop(L + 1)

    # --- T_target_verify at q_len = gamma+1 ---
    # Snapshot lengths so we can rewind cache after each measurement.
    snap_len = past.get_seq_length()
    block = torch.full((1, args.gamma + 1), 0, dtype=torch.long, device=device)
    block[:, 0] = bonus[0, 0].item()
    cpos_v = torch.arange(L, L + args.gamma + 1, device=device)
    times = []
    for _ in range(args.warmup):
        _ = vlm(input_ids=block, past_key_values=past, cache_position=cpos_v,
                use_cache=True, return_dict=True)
        past.crop(snap_len)
    torch.cuda.synchronize()
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        _ = vlm(input_ids=block, past_key_values=past, cache_position=cpos_v,
                use_cache=True, return_dict=True)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        past.crop(snap_len)
    times.sort()
    verify_ms = times[len(times) // 2]

    # --- T_draft_prefill (step 0 over L positions) ---
    fused = torch.cat(target_h, dim=-1)
    fc_in = draft.fc(fused)
    shifted = torch.cat([inputs["input_ids"][:, 1:L], bonus], dim=1)
    pos_ids = torch.arange(L, device=device).unsqueeze(0)
    attn_mask_pre = draft._prepare_decoder_attention_mask(
        torch.ones((1, L), dtype=torch.bool, device=device),
        (1, L), fc_in.dtype, device, past_kv_len=0,
    )
    times = []
    for _ in range(args.warmup):
        cache_hidden = [[], []]
        emb = embed_tokens(shifted).to(fc_in.dtype)
        _ = draft.midlayer(input_emb=emb, hidden_states=fc_in,
                           cache_hidden=cache_hidden, attention_mask=attn_mask_pre,
                           position_ids=pos_ids)
    torch.cuda.synchronize()
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        cache_hidden = [[], []]
        emb = embed_tokens(shifted).to(fc_in.dtype)
        _ = draft.midlayer(input_emb=emb, hidden_states=fc_in,
                           cache_hidden=cache_hidden, attention_mask=attn_mask_pre,
                           position_ids=pos_ids)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    draft_prefill_ms = times[len(times) // 2]

    # --- T_draft_chain (gamma chain steps starting from a fresh prefill) ---
    # Build prefill state once.
    cache_hidden_seed = [[], []]
    emb = embed_tokens(shifted).to(fc_in.dtype)
    h0, cache_hidden_seed = draft.midlayer(
        input_emb=emb, hidden_states=fc_in, cache_hidden=cache_hidden_seed,
        attention_mask=attn_mask_pre, position_ids=pos_ids,
    )
    chain_hidden_seed = h0[:, -1:, :].contiguous()
    virtual_pos = torch.tensor([[L - 1]], device=device, dtype=torch.long)
    attn_mask_chain = torch.zeros((1, 1, 1, L), dtype=fc_in.dtype, device=device)

    def run_chain(seed_chain_hidden, seed_cache_hidden):
        ch = seed_chain_hidden
        cache = [list(seed_cache_hidden[0]), list(seed_cache_hidden[1])]
        next_in = bonus
        for _ in range(args.gamma):
            embc = embed_tokens(next_in).to(ch.dtype)
            ch, cache = draft.midlayer(
                input_emb=embc, hidden_states=ch, cache_hidden=cache,
                attention_mask=attn_mask_chain, position_ids=virtual_pos,
            )
            next_in = lm_head(draft.norm(ch)).argmax(-1)
        return ch

    for _ in range(args.warmup):
        run_chain(chain_hidden_seed, cache_hidden_seed)
    torch.cuda.synchronize()
    times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        run_chain(chain_hidden_seed, cache_hidden_seed)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    draft_chain_total_ms = times[len(times) // 2]
    draft_chain_step_ms = draft_chain_total_ms / args.gamma

    out = {
        "ar_step_ms": ar_step_ms,
        "verify_ms": verify_ms,
        "draft_prefill_ms": draft_prefill_ms,
        "draft_chain_total_ms": draft_chain_total_ms,
        "draft_chain_step_ms": draft_chain_step_ms,
        "draft_total_per_iter_ms": draft_prefill_ms + draft_chain_total_ms,
        "block_size": block_size,
        "gamma": args.gamma,
        "prefill_len": L,
        "clip_id": Path(args.clip_path).stem,
        "target_path": args.target_path,
        "draft_path": args.draft_path,
        "kind": "eagle3",
    }
    print(json.dumps(out, indent=2))
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
