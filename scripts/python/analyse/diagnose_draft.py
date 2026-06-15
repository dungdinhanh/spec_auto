"""Diagnose draft predictions: show actual tokens predicted vs target and GT."""
import torch
import torch.nn.functional as F
import sys, glob, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper
from alpamayo_r1.models.dflash_draft import (
    build_dflash_draft_for_qwen3vl,
    get_qwen3vl_embed_and_head,
    extract_context_feature,
)

device = "cuda"
block_size = 16

# Load target
print("Loading target...")
model = AlpamayoR1.from_pretrained("/mnt/resv-harry-6f72s/dungda/models/Alpamayo-R1-10B", dtype=torch.bfloat16)
target = model.vlm.to(device).eval()
embed_tokens, lm_head = get_qwen3vl_embed_and_head(target)
tokenizer = model.tokenizer
processor = helper.get_processor(tokenizer)

# Load draft
print("Building draft...")
draft = build_dflash_draft_for_qwen3vl(
    target, num_draft_layers=5, block_size=block_size, mask_token_id=151669,
).to(torch.bfloat16).to(device).eval()

state_dict = torch.load(
    "/mnt/resv-harry-6f72s/dungda/runs/selfdistill_kl_pretrained_s3/draft_epoch_1.pt",
    map_location=device, weights_only=False
)
draft.load_state_dict(state_dict, strict=False)

# Load one clip
clip_files = sorted(glob.glob("/mnt/resv-harry-6f72s/dungda/data/alpamayo_clips/*.pt"))
clip = torch.load(clip_files[0], weights_only=False)

# Process clip
inputs = processor.apply_chat_template(
    clip["messages"], tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
)
inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
input_ids = inputs["input_ids"]
seq_len = input_ids.shape[1]
print(f"\nSeq len: {seq_len}")

# Target forward
target_kwargs = dict(input_ids=input_ids, use_cache=False, output_hidden_states=True, return_dict=True)
if inputs.get("pixel_values_videos") is not None:
    target_kwargs["pixel_values_videos"] = inputs["pixel_values_videos"]
if inputs.get("video_grid_thw") is not None:
    target_kwargs["video_grid_thw"] = inputs["video_grid_thw"]

with torch.no_grad():
    target_out = target(**target_kwargs)

target_hidden = extract_context_feature(target_out.hidden_states, draft.target_layer_ids)
target_logits = target_out.logits
target_preds = target_logits.argmax(dim=-1)

# Test draft on first 3 blocks
SEP = "=" * 80
print(f"\n{SEP}")
print(f"BLOCK-BY-BLOCK ANALYSIS (block_size={block_size})")
print(f"{SEP}")

for b_idx in range(3):
    start = b_idx * block_size
    end = start + block_size
    if end >= seq_len:
        break

    block_ids = input_ids[:, start:end].clone()
    block_ids[:, 1:] = draft.mask_token_id  # mask all except first

    noise_embedding = embed_tokens(block_ids)
    ctx_hidden = target_hidden[:, :end, :]
    ctx_len = ctx_hidden.shape[1]
    pos_ids = torch.arange(ctx_len + block_size, device=device).unsqueeze(0)

    with torch.no_grad():
        draft_hidden = draft(
            target_hidden=ctx_hidden,
            noise_embedding=noise_embedding,
            position_ids=pos_ids,
            past_key_values=None,
            use_cache=False,
        )
    draft_logits = lm_head(draft_hidden[:, -(block_size - 1):, :])
    draft_preds = draft_logits.argmax(dim=-1)

    # Draft top-5 probabilities for position 1
    draft_probs = F.softmax(draft_logits[0, 0, :].float(), dim=-1)
    top5_probs, top5_ids = draft_probs.topk(5)

    gt_next = input_ids[0, start + 1:end]
    tgt_next = target_preds[0, start:end - 1]

    print(f"\n--- Block {b_idx} (positions {start}-{end-1}) ---")
    first_tok = input_ids[0, start].item()
    print(f"  Input token [0]: id={first_tok} = {repr(tokenizer.decode([first_tok]))}")

    for k in range(min(8, block_size - 1)):
        d = draft_preds[0, k].item()
        g = gt_next[k].item()
        t = tgt_next[k].item()
        d_text = repr(tokenizer.decode([d]))
        g_text = repr(tokenizer.decode([g]))
        t_text = repr(tokenizer.decode([t]))
        match_gt = "Y" if d == g else "N"
        match_tgt = "Y" if d == t else "N"
        print(f"  pos {k+1}: draft={d:>6} {d_text:>25} | gt={g:>6} {g_text:>25} {match_gt} | target={t:>6} {t_text:>25} {match_tgt}")

    print(f"\n  Draft top-5 for position 1:")
    for rank_i in range(5):
        prob = top5_probs[rank_i].item()
        tid = top5_ids[rank_i].item()
        is_gt = " <-- GT" if tid == gt_next[0].item() else ""
        is_tgt = " <-- TGT" if tid == tgt_next[0].item() else ""
        print(f"    #{rank_i+1}: id={tid:>6} {repr(tokenizer.decode([tid])):>25} prob={prob:.4f}{is_gt}{is_tgt}")

# Draft output stats
print(f"\n{SEP}")
print(f"DRAFT OUTPUT STATS")
print(f"{SEP}")

print(f"Draft logits shape: {draft_logits.shape}")
print(f"Draft logits range: [{draft_logits.min().item():.2f}, {draft_logits.max().item():.2f}]")
print(f"Draft logits mean: {draft_logits.mean().item():.4f}, std: {draft_logits.float().std().item():.4f}")

# Check if draft always predicts the same token
all_draft_preds = draft_logits.argmax(dim=-1)[0]
unique_preds = all_draft_preds.unique()
print(f"Unique draft predictions in last block: {unique_preds.tolist()}")
print(f"  decoded: {[repr(tokenizer.decode([t.item()])) for t in unique_preds]}")

# Entropy of draft distribution (how confident/peaked is it?)
for pos in [0, 3, 7, 14]:
    if pos < draft_logits.shape[1]:
        probs = F.softmax(draft_logits[0, pos, :].float(), dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum().item()
        max_prob = probs.max().item()
        print(f"  Position {pos+1}: entropy={entropy:.2f} max_prob={max_prob:.4f}")
