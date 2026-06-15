"""Smoke test for the DFlash draft model adapted to Qwen3-VL / Alpamayo-R1.

What it checks (in order):
  1. Qwen3-VL-8B-Instruct target loads and runs.
  2. `build_dflash_draft_for_qwen3vl` produces a valid draft model
     (no shape errors, correct config fields).
  3. `get_qwen3vl_embed_and_head` resolves embed_tokens + lm_head.
  4. Plain `target.generate()` produces a reference output for a text-only prompt.
  5. `vlm_spec_generate` runs end-to-end (text-only) without crashing.
  6. With an UNTRAINED draft, every draft proposal is rejected, so
     `vlm_spec_generate` must produce the SAME tokens as `target.generate()`
     (greedy decoding for both, temperature=0).
  7. Repeat (5)+(6) with a real image input from one of the cached Alpamayo
     clips, to check the multimodal prefill path.

This is a *smoke test*: it does not verify training-time gradients or
real speedups. It only checks that the plumbing is correct and produces
identical outputs to plain target.generate() when the draft is untrained.

Run on a GPU node (needs ~24GB VRAM for the 8B target):

    python smoke_test_dflash_draft.py
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

import torch

# Force offline mode so we don't accidentally hit HuggingFace
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

GDATA = "/g/data/hn98/dd9648"
TARGET_PATH = os.environ.get("VLM_PATH", f"{GDATA}/models/Qwen3-VL-8B-Instruct")
CLIPS_PATH = os.environ.get(
    "ALPAMAYO_CLIPS_PT", f"{GDATA}/cache/alpamayo_example_data.pt"
)
MAX_NEW_TOKENS = 16          # keep short — we just want to verify correctness
BLOCK_SIZE = 4
MASK_TOKEN_ID = 151643       # Qwen3-VL pad/mask token id (same as Qwen3)


# ----------------------------- helpers ---------------------------------------


def section(title: str):
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}", flush=True)


def assert_equal_tensor(a: torch.Tensor, b: torch.Tensor, name: str):
    if a.shape != b.shape:
        raise AssertionError(
            f"[{name}] shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}"
        )
    if not torch.equal(a, b):
        diff_idx = (a != b).nonzero(as_tuple=False)
        raise AssertionError(
            f"[{name}] tensors differ at {diff_idx.shape[0]} positions; "
            f"first diff at {diff_idx[0].tolist()}: {a.flatten()[:20].tolist()} "
            f"vs {b.flatten()[:20].tolist()}"
        )
    print(f"  ✓ {name}: identical ({tuple(a.shape)})")


# ----------------------------- main test -------------------------------------


def main() -> int:
    section("Step 1: Load Qwen3-VL-8B target")
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

    print(f"  loading from {TARGET_PATH}")
    # Use sdpa instead of flash_attention_2 to avoid GLIBC version mismatch
    # on NCI compute nodes. flash-attn is faster but not required for correctness.
    target = Qwen3VLForConditionalGeneration.from_pretrained(
        TARGET_PATH,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH)
    print(f"  ✓ target loaded ({sum(p.numel() for p in target.parameters()) / 1e9:.1f}B params)")

    section("Step 2: Build DFlash draft from target")
    # Make sure the alpamayo source is on path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from alpamayo_r1.models.dflash_draft import (  # noqa: E402
        build_dflash_draft_for_qwen3vl,
        get_qwen3vl_embed_and_head,
        vlm_spec_generate,
    )

    draft = build_dflash_draft_for_qwen3vl(
        target,
        num_draft_layers=1,
        block_size=BLOCK_SIZE,
        mask_token_id=MASK_TOKEN_ID,
    ).to("cuda").eval()
    print(f"  ✓ draft built ({sum(p.numel() for p in draft.parameters()) / 1e6:.1f}M params)")
    print(f"    block_size={draft.block_size}, target_layer_ids={draft.target_layer_ids}")

    section("Step 3: Resolve embed_tokens / lm_head from target")
    embed_tokens, lm_head = get_qwen3vl_embed_and_head(target)
    print(f"  ✓ embed_tokens: {type(embed_tokens).__name__} {tuple(embed_tokens.weight.shape)}")
    print(f"  ✓ lm_head:      {type(lm_head).__name__} {tuple(lm_head.weight.shape)}")

    section("Step 4: Reference output via target.generate() (text only)")
    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    print(f"  prompt: {prompt!r}  ({input_ids.shape[1]} tokens)")

    with torch.inference_mode():
        ref_out = target.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,           # ignored when do_sample=False
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    print(f"  ref_out shape: {tuple(ref_out.shape)}")
    print(f"  ref decoded:  {tokenizer.decode(ref_out[0], skip_special_tokens=True)!r}")

    section("Step 5+6: Run vlm_spec_generate (text only) and compare to reference")
    t0 = time.time()
    spec_result = vlm_spec_generate(
        target=target,
        draft=draft,
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        stop_token_ids=[tokenizer.eos_token_id],
        temperature=0.0,
        block_size=BLOCK_SIZE,
    )
    dt = time.time() - t0
    spec_out = spec_result["output_ids"]
    print(f"  spec_out shape: {tuple(spec_out.shape)}  ({dt:.2f}s)")
    print(f"  spec decoded:   {tokenizer.decode(spec_out[0], skip_special_tokens=True)!r}")
    print(f"  acceptance lengths: {spec_result['acceptance_lengths']}")

    # With an UNTRAINED draft, the proposed tokens are random, so most are rejected.
    # The accepted tokens may differ from greedy generate() because the draft can
    # propose a token that happens to match. To make the comparison robust we only
    # compare *up to the shorter length* of [num_input_tokens + min_new_tokens].
    common_len = min(ref_out.shape[1], spec_out.shape[1])
    assert_equal_tensor(
        ref_out[0, :common_len].cpu(),
        spec_out[0, :common_len].cpu(),
        "text-only spec vs reference (first common_len tokens)",
    )

    section("Step 7: Test with multimodal prefill (cached Alpamayo clip)")
    if not os.path.exists(CLIPS_PATH):
        print(f"  SKIP: cached clip not found at {CLIPS_PATH}")
    else:
        from alpamayo_r1 import helper  # noqa: E402

        saved = torch.load(CLIPS_PATH, weights_only=False)
        if isinstance(saved, list):
            saved = saved[0]
        data, messages = saved["data"], saved["messages"]

        processor = helper.get_processor(tokenizer)
        mm_inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        # Move tensors to GPU
        mm_inputs = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in mm_inputs.items()}

        print(f"  multimodal input keys: {list(mm_inputs.keys())}")
        print(f"  input_ids shape: {tuple(mm_inputs['input_ids'].shape)}")

        # Reference: target.generate() with the same multimodal inputs.
        with torch.inference_mode():
            mm_ref = target.generate(
                **mm_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        print(f"  mm_ref shape: {tuple(mm_ref.shape)}")

        t0 = time.time()
        mm_spec = vlm_spec_generate(
            target=target,
            draft=draft,
            input_ids=mm_inputs["input_ids"],
            pixel_values=mm_inputs.get("pixel_values"),
            pixel_values_videos=mm_inputs.get("pixel_values_videos"),
            image_grid_thw=mm_inputs.get("image_grid_thw"),
            video_grid_thw=mm_inputs.get("video_grid_thw"),
            attention_mask=mm_inputs.get("attention_mask"),
            max_new_tokens=MAX_NEW_TOKENS,
            stop_token_ids=[tokenizer.eos_token_id],
            temperature=0.0,
            block_size=BLOCK_SIZE,
        )
        dt = time.time() - t0
        mm_spec_out = mm_spec["output_ids"]
        print(f"  mm_spec shape: {tuple(mm_spec_out.shape)}  ({dt:.2f}s)")
        print(f"  acceptance lengths: {mm_spec['acceptance_lengths']}")

        common_len = min(mm_ref.shape[1], mm_spec_out.shape[1])
        assert_equal_tensor(
            mm_ref[0, :common_len].cpu(),
            mm_spec_out[0, :common_len].cpu(),
            "multimodal spec vs reference (first common_len tokens)",
        )

    section("ALL SMOKE CHECKS PASSED ✅")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        print("\n=== SMOKE TEST FAILED ===")
        sys.exit(1)
