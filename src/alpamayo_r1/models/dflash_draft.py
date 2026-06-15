"""DFlash draft model integration for the Qwen3-VL-8B-Instruct backbone used
inside Alpamayo-R1.

This module:
  1. Builds a `DFlashDraftModel` whose hyperparameters are derived from a
     Qwen3-VL target (so the draft shares hidden_size / head_dim / vocab and can
     reuse the target's `embed_tokens` + `lm_head`).
  2. Loads DFlash draft weights (HF folder, safetensors file, or .bin/.pt
     state_dict) into that draft model.

Typical usage:

    from transformers import Qwen3VLForConditionalGeneration
    from alpamayo_r1.models.dflash_draft import (
        build_dflash_draft_for_qwen3vl,
        load_dflash_weights,
    )

    target = Qwen3VLForConditionalGeneration.from_pretrained(
        "/g/data/hn98/dd9648/models/Qwen3-VL-8B-Instruct",
        dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    ).cuda().eval()

    draft = build_dflash_draft_for_qwen3vl(
        target, num_draft_layers=1, block_size=4, mask_token_id=151643,
    ).to(torch.bfloat16).cuda().eval()

    load_dflash_weights(draft, "/path/to/dflash_qwen3vl_draft")
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from transformers import DynamicCache

from dflash.model import (
    DFlashDraftModel,
    build_target_layer_ids,
    extract_context_feature,
    sample,
)


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------

# Fields the DFlash modules read off the Qwen3Config.
_QWEN3_FIELDS = (
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "hidden_act",
    "max_position_embeddings",
    "rms_norm_eps",
    "rope_theta",
    "rope_scaling",
    "attention_bias",
    "attention_dropout",
    "vocab_size",
    "tie_word_embeddings",
    "initializer_range",
    "use_sliding_window",
    "sliding_window",
    "max_window_layers",
)


def _extract_qwen3_text_config(target: nn.Module):
    """Return the underlying text config (Qwen3VLTextConfig) from a Qwen3-VL model."""
    cfg = getattr(target, "config", None)
    if cfg is None:
        raise ValueError("target has no .config")
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is None:
        # Already a text-only Qwen3 config.
        return cfg
    return text_cfg


def build_qwen3_draft_config(
    target: nn.Module,
    num_draft_layers: int = 1,
    block_size: int = 4,
    mask_token_id: Optional[int] = None,
    target_layer_ids: Optional[list[int]] = None,
) -> Qwen3Config:
    """Build a Qwen3Config for the DFlash draft from a Qwen3-VL target model."""
    text_cfg = _extract_qwen3_text_config(target)

    kwargs = {}
    for f in _QWEN3_FIELDS:
        if hasattr(text_cfg, f):
            kwargs[f] = getattr(text_cfg, f)

    # Draft has its own (small) layer count.
    kwargs["num_hidden_layers"] = num_draft_layers

    draft_cfg = Qwen3Config(**kwargs)

    # Qwen3 in transformers uses `layer_types` (per-layer
    # "full_attention"/"sliding_attention"). For the draft we use full
    # attention everywhere — DFlash attends jointly over (target context, noise)
    # so sliding-window doesn't make sense.
    draft_cfg.layer_types = ["full_attention"] * num_draft_layers
    if not hasattr(draft_cfg, "sliding_window") or draft_cfg.sliding_window is None:
        draft_cfg.sliding_window = None

    # DFlash-specific extras read by DFlashDraftModel.__init__.
    num_target_layers = getattr(text_cfg, "num_hidden_layers")
    draft_cfg.num_target_layers = num_target_layers
    draft_cfg.block_size = block_size
    if target_layer_ids is None:
        target_layer_ids = build_target_layer_ids(num_target_layers, num_draft_layers)
    draft_cfg.dflash_config = {
        "target_layer_ids": target_layer_ids,
        "mask_token_id": mask_token_id,
    }
    return draft_cfg


def build_dflash_draft_for_qwen3vl(
    target: nn.Module,
    num_draft_layers: int = 1,
    block_size: int = 4,
    mask_token_id: Optional[int] = None,
    target_layer_ids: Optional[list[int]] = None,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
) -> DFlashDraftModel:
    """Instantiate (uninitialized) a DFlash draft model matched to a Qwen3-VL target."""
    cfg = build_qwen3_draft_config(
        target,
        num_draft_layers=num_draft_layers,
        block_size=block_size,
        mask_token_id=mask_token_id,
        target_layer_ids=target_layer_ids,
    )
    cfg._attn_implementation = attn_implementation
    cfg.dtype = dtype
    draft = DFlashDraftModel(cfg)
    return draft.to(dtype=dtype)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def load_draft_checkpoint(path: str, map_location="cpu") -> dict:
    """Load a draft checkpoint, returning a dict with keys:
      - state_dict
      - mask_token_id (int or None)
      - num_draft_layers (int or None)
      - block_size (int or None)
    Accepts both the new wrapped format {"state_dict": ..., "mask_token_id": ..., ...}
    and the legacy format (plain state_dict). For legacy, all metadata is None.
    """
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path, device=str(map_location) if map_location != "cpu" else "cpu")
        return {"state_dict": sd, "mask_token_id": None,
                "num_draft_layers": None, "block_size": None}
    raw = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        return {
            "state_dict": raw["state_dict"],
            "mask_token_id": raw.get("mask_token_id"),
            "num_draft_layers": raw.get("num_draft_layers"),
            "block_size": raw.get("block_size"),
        }
    # legacy plain state_dict
    return {"state_dict": raw, "mask_token_id": None,
            "num_draft_layers": None, "block_size": None}



def _load_state_dict_from_path(path: Union[str, os.PathLike]) -> dict:
    """Load a state_dict from one of: HF folder, .safetensors, or .bin/.pt."""
    p = Path(path)
    if p.is_dir():
        # Prefer safetensors index / shards, then single safetensors, then .bin.
        idx = p / "model.safetensors.index.json"
        if idx.exists():
            from safetensors.torch import load_file

            with open(idx) as f:
                index = json.load(f)
            shards = sorted(set(index["weight_map"].values()))
            state = {}
            for shard in shards:
                state.update(load_file(p / shard))
            return state
        single_st = p / "model.safetensors"
        if single_st.exists():
            from safetensors.torch import load_file

            return load_file(single_st)
        bin_path = p / "pytorch_model.bin"
        if bin_path.exists():
            return torch.load(bin_path, map_location="cpu", weights_only=True)
        raise FileNotFoundError(f"No model weights found under {p}")
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(p)
    return torch.load(p, map_location="cpu", weights_only=True)


def load_dflash_weights(
    draft: DFlashDraftModel,
    weights: Union[str, os.PathLike, dict],
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Load DFlash draft weights into `draft`.

    `weights` may be a path (HF folder / safetensors / bin) or a state_dict.
    Returns (missing_keys, unexpected_keys) from `load_state_dict`.

    Some checkpoints prefix keys with "model." or "draft_model." — we strip
    common prefixes so the load is forgiving.
    """
    state = weights if isinstance(weights, dict) else _load_state_dict_from_path(weights)

    # Strip common prefixes.
    cleaned: dict = {}
    for k, v in state.items():
        nk = k
        for prefix in ("draft_model.", "draft.", "model.draft.", "model."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
                break
        cleaned[nk] = v

    missing, unexpected = draft.load_state_dict(cleaned, strict=strict)
    return list(missing), list(unexpected)


# ---------------------------------------------------------------------------
# Convenience: pull embed_tokens / lm_head off a Qwen3-VL target
# ---------------------------------------------------------------------------


def get_target_3d_position_ids(
    target_vlm: nn.Module,
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute target's 3D M-RoPE position_ids for a given prompt.

    The target Qwen3-VL model uses 3D positions (height, width, temporal) with
    rope_deltas accounting for the vision-token block. The DFlash draft must
    use the SAME 3D positions so that its cross-attention queries are rotated
    under the same angular regime as target's keys/values.

    Args:
        target_vlm:    Qwen3VLForConditionalGeneration (or its `.vlm` if wrapped).
        input_ids:     (B, T) — same input passed to target's prefill.
        image_grid_thw: (n_images, 3) — temporal/height/width of each image's
                       vision tokens, exactly as supplied to target.
        attention_mask: (B, T) — defaults to all-ones if None.

    Returns:
        position_ids: (3, B, T) tensor matching target's M-RoPE.
                       For text-only sequences (no image_grid_thw) this returns
                       the trivial arange broadcast on all three axes.
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    # Drill through .vlm if a wrapper was passed.
    vlm = getattr(target_vlm, "vlm", target_vlm)
    B, T = input_ids.shape
    device = input_ids.device

    if image_grid_thw is None or not hasattr(vlm.model, "get_rope_index"):
        p = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        return p.unsqueeze(0).expand(3, -1, -1).contiguous()

    # Qwen3VLModel.get_rope_index iterates `image_grid_thw` globally and assumes
    # the count matches the vision tokens across the entire batched input. With
    # B > 1 and per-sample concatenated image_grid_thw, the indexing diverges
    # and `t = image_grid_thw[idx][0]` ends up shaped (3,) instead of scalar,
    # raising "Tensor with 3 elements cannot be converted to Scalar".
    # Workaround: compute per sample (B=1) and stack along the batch axis.
    if B == 1:
        pos_3d, _ = vlm.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )
        return pos_3d  # (3, 1, T)

    # image_grid_thw can come in two shapes from different collate functions:
    #   (n_images_total, 3) — flat-concat (single-sample / non-batched)
    #   (B, n_images_per_sample, 3) — stacked-batched (v2/v3 dataloader)
    # Branch on dim accordingly.
    text_cfg = vlm.config.get_text_config()
    vis_start = getattr(text_cfg, "vision_start_token_id", None)
    if vis_start is None:
        vis_start = 151652  # Qwen3-VL canonical
    images_per_sample = (input_ids == vis_start).sum(dim=1).tolist()

    pieces = []
    cum = 0
    for b in range(B):
        n_b = int(images_per_sample[b])
        if n_b == 0:
            igt_b = None
        elif image_grid_thw.dim() == 3:
            # Stacked: (B, n, 3). Take this sample's row, keep n_b rows in case
            # of variable-image samples (defensive — batched data is usually
            # uniform).
            igt_b = image_grid_thw[b][:n_b]
        else:
            # Flat-concat: (total, 3). Slice contiguously.
            igt_b = image_grid_thw[cum:cum + n_b]
            cum += n_b
        pos_b, _ = vlm.model.get_rope_index(
            input_ids=input_ids[b:b + 1],
            image_grid_thw=igt_b,
            video_grid_thw=None,
            attention_mask=attention_mask[b:b + 1] if attention_mask is not None else None,
        )                                       # (3, 1, T)
        pieces.append(pos_b)
    return torch.cat(pieces, dim=1)             # (3, B, T)


def get_qwen3vl_embed_and_head(target: nn.Module) -> tuple[nn.Embedding, nn.Linear]:
    """Return (embed_tokens, lm_head) from a Qwen3-VL ForConditionalGeneration model.

    DFlash's spec_generate uses `target.model.embed_tokens` and `target.lm_head`,
    which is true for plain Qwen3 CausalLM but not for Qwen3-VL where the text
    decoder is nested. This helper resolves the right modules.
    """
    # Qwen3VLForConditionalGeneration.lm_head exists at the top level.
    lm_head = getattr(target, "lm_head", None)
    if lm_head is None:
        raise AttributeError("target has no lm_head")

    # Embeddings live on the language model: target.model.language_model.embed_tokens
    # (transformers >= 4.49 Qwen3-VL layout). Fall back to a few alternatives.
    candidates = [
        ("model", "language_model", "embed_tokens"),
        ("model", "embed_tokens"),
        ("language_model", "model", "embed_tokens"),
    ]
    for path in candidates:
        obj = target
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and isinstance(obj, nn.Embedding):
            return obj, lm_head
    raise AttributeError("Could not locate embed_tokens on Qwen3-VL target")


# ---------------------------------------------------------------------------
# VLM-aware speculative decoding
# ---------------------------------------------------------------------------


def _cosine_reveal_schedule(num_maskable: int, num_steps: int) -> list[int]:
    """MaskGIT-style cosine schedule. Returns a list of per-step reveal counts
    summing to `num_maskable`. Zero-reveal steps are filtered out, so the
    returned list length may be ≤ `num_steps`.
    """
    if num_steps <= 1 or num_maskable <= 1:
        return [num_maskable]
    masked_after = [num_maskable] + [
        int(round(num_maskable * math.cos(math.pi / 2 * t / num_steps)))
        for t in range(1, num_steps)
    ] + [0]
    reveals = [masked_after[t - 1] - masked_after[t] for t in range(1, num_steps + 1)]
    reveals = [max(0, r) for r in reveals]
    diff = num_maskable - sum(reveals)
    if diff != 0:
        reveals[-1] += diff
    return [r for r in reveals if r > 0]


def _linear_reveal_schedule(num_maskable: int, num_steps: int) -> list[int]:
    """Linear (equal-step) discretization — the DDIM-uniform / standard
    flow-matching solver schedule. For B=16, T=4: mask states after each step
    are 75% → 50% → 25% → 0% (positions still masked: 11 → 8 → 4 → 0).
    Matches v7's discrete training mask set {15, 11, 8, 4} when T=4.
    """
    if num_steps <= 1 or num_maskable <= 1:
        return [num_maskable]
    masked_after = [
        int(round(num_maskable * (num_steps - t) / num_steps))
        for t in range(0, num_steps + 1)
    ]
    masked_after[0] = num_maskable
    masked_after[-1] = 0
    reveals = [masked_after[t - 1] - masked_after[t] for t in range(1, num_steps + 1)]
    reveals = [max(0, r) for r in reveals]
    diff = num_maskable - sum(reveals)
    if diff != 0:
        reveals[-1] += diff
    return [r for r in reveals if r > 0]


def _reveal_schedule(num_maskable: int, num_steps: int, schedule: str) -> list[int]:
    if schedule == "cosine":
        return _cosine_reveal_schedule(num_maskable, num_steps)
    if schedule == "linear":
        return _linear_reveal_schedule(num_maskable, num_steps)
    raise ValueError(f"Unknown refinement schedule: {schedule!r}")


@torch.inference_mode()
def vlm_spec_generate(
    target: nn.Module,
    draft: DFlashDraftModel,
    input_ids: torch.LongTensor,
    *,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 512,
    stop_token_ids: Optional[list[int]] = None,
    temperature: float = 0.0,
    block_size: Optional[int] = None,
) -> dict:
    """Speculative decoding for Qwen3-VL targets with a DFlash draft.

    Strategy
    --------
    1. **Multimodal prefill** through the Qwen3-VL target with `pixel_values` /
       grids — this consumes both visual and text tokens, builds the target KV
       cache, and gives us the first sampled token + per-layer hidden states
       over the *entire* prefill (used as draft cross-attention context).
    2. **Text-only spec decode tail**: every iteration the draft proposes a
       block of `block_size-1` continuation tokens (no new visual tokens), the
       target verifies them in a single forward pass, and we accept the longest
       matching prefix (standard DFlash protocol).

    Notes
    -----
    - During the decode tail we pass `cache_position` to the target and let it
      rebuild M-RoPE position_ids internally from `rope_deltas` cached on the
      model after prefill (this is how `Qwen3VLForConditionalGeneration.generate`
      handles continuation steps).
    - The draft uses plain 1D positions; this is fine because the draft never
      sees image tokens as queries — visual context only enters as cached K/V
      via `target_hidden`.
    - Returns a dict with `output_ids`, `num_input_tokens`, `num_output_tokens`,
      and per-step `acceptance_lengths`.
    """
    target.eval()
    draft.eval()
    device = input_ids.device
    if block_size is None:
        block_size = draft.block_size
    mask_token_id = draft.mask_token_id
    if mask_token_id is None:
        raise ValueError("draft.mask_token_id is not set; pass mask_token_id when building the draft")

    embed_tokens, lm_head = get_qwen3vl_embed_and_head(target)

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    # Buffer for the full sequence (prefill + generated), padded with mask tokens.
    output_ids = torch.full(
        (1, max_length + block_size), mask_token_id, dtype=torch.long, device=device,
    )
    output_ids[:, :num_input_tokens] = input_ids

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    # ---------- Multimodal prefill ----------
    prefill_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values_target,
        use_cache=True,
        output_hidden_states=True,
        logits_to_keep=1,
        return_dict=True,
    )
    if pixel_values is not None:
        prefill_kwargs["pixel_values"] = pixel_values
    if image_grid_thw is not None:
        prefill_kwargs["image_grid_thw"] = image_grid_thw
    if pixel_values_videos is not None:
        prefill_kwargs["pixel_values_videos"] = pixel_values_videos
    if video_grid_thw is not None:
        prefill_kwargs["video_grid_thw"] = video_grid_thw

    output = target(**prefill_kwargs)

    first_token = sample(output.logits, temperature)
    output_ids[:, num_input_tokens : num_input_tokens + 1] = first_token

    # Per-layer hidden states over the full prefill (visual + text). DFlash
    # cross-attends over these in every draft step.
    target_hidden = extract_context_feature(output.hidden_states, draft.target_layer_ids)

    # ---------- Spec-decode tail (text only) ----------
    # The draft uses plain 1D positions starting at the prefill length.
    draft_position_ids = torch.arange(
        max_length + block_size, device=device, dtype=torch.long
    ).unsqueeze(0)

    acceptance_lengths: list[int] = []
    start = num_input_tokens
    while start < max_length:
        # Build a noise block: [first_token, mask, mask, ..., mask] (length=block_size).
        block_output_ids = output_ids[:, start : start + block_size].clone()
        noise_embedding = embed_tokens(block_output_ids)

        # Draft forward — attends over (target_hidden in cached K/V) ++ noise.
        draft_hidden = draft(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=draft_position_ids[
                :, past_key_values_draft.get_seq_length() : start + block_size
            ],
            past_key_values=past_key_values_draft,
            use_cache=True,
        )
        draft_logits = lm_head(draft_hidden[:, -block_size + 1 :, :])
        # Drop the noise K/V we just appended; keep only the verified prefix.
        past_key_values_draft.crop(start)

        block_output_ids[:, 1:] = sample(draft_logits, temperature=0.0)

        # ---------- Target verification (text-only forward) ----------
        cache_position = torch.arange(
            start, start + block_size, device=device, dtype=torch.long
        )
        verify_out = target(
            input_ids=block_output_ids,
            past_key_values=past_key_values_target,
            cache_position=cache_position,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

        posterior = sample(verify_out.logits, temperature)
        # Longest matching prefix between draft proposals and target argmax.
        acceptance_length = (
            (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        )
        # Accept the matched prefix and append the bonus token from the target.
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
            :, : acceptance_length + 1
        ]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
        start += acceptance_length + 1

        # Roll target KV back to the accepted position.
        past_key_values_target.crop(start)

        # Update draft cross-attn context with only the *newly verified* tokens'
        # hidden states (the prefill chunk is already in the draft KV cache).
        target_hidden = extract_context_feature(
            verify_out.hidden_states, draft.target_layer_ids
        )[:, : acceptance_length + 1, :]

        acceptance_lengths.append(acceptance_length + 1)

        if stop_token_ids is not None and any(
            sid in output_ids[0, num_input_tokens:start + 1].tolist() for sid in stop_token_ids
        ):
            break

    # ---------- Trim outputs ----------
    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids is not None:
        stop_t = torch.tensor(stop_token_ids, device=output_ids.device)
        idxs = torch.isin(output_ids[0, num_input_tokens:], stop_t).nonzero(as_tuple=True)[0]
        if idxs.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + idxs[0].item() + 1]

    return {
        "output_ids": output_ids,
        "num_input_tokens": num_input_tokens,
        "num_output_tokens": output_ids.shape[1] - num_input_tokens,
        "acceptance_lengths": acceptance_lengths,
    }


@torch.inference_mode()
def vlm_spec_generate_first_ar(
    target: nn.Module,
    draft: DFlashDraftModel,
    input_ids: torch.LongTensor,
    *,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 512,
    stop_token_ids: Optional[list[int]] = None,
    temperature: float = 0.0,
    block_size: Optional[int] = None,
) -> dict:
    """Variant of `vlm_spec_generate` with a 1-token target-AR step BEFORE
    each spec iteration.

    Motivation
    ----------
    Empirically position 1 (= first draft-proposed token after each anchor) is
    the spec-decode bottleneck — for v6-RM L=4 it accounts for ~41% of all
    iteration rejections. This variant uses the target itself to produce that
    first token (`t1`) per iter, so the draft only proposes positions 2..B-1
    conditioned on the now-correct prefix `[anchor, t1]`.

    Per-iter cost: 1 AR target forward (1 token) + 1 draft forward + 1 verify
    target forward (B positions, anchor of the spec block IS `t1`). The AR
    step's FLOPs ~ 1/B of the verify, so total target compute per iter ≈
    `1 + 1/B` vs the baseline's `1`; per-call launch overhead dominates in
    practice.

    Per-iter L: each iter now emits `1 (from AR) + (k + 1) (from spec)` tokens
    where k is the accepted draft length. We record `k + 2` in
    `acceptance_lengths` to keep `avg_iter_tokens` directly comparable to the
    baseline's per-iter token count.
    """
    target.eval()
    draft.eval()
    device = input_ids.device
    if block_size is None:
        block_size = draft.block_size
    mask_token_id = draft.mask_token_id
    if mask_token_id is None:
        raise ValueError("draft.mask_token_id is not set; pass mask_token_id when building the draft")

    embed_tokens, lm_head = get_qwen3vl_embed_and_head(target)

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids = torch.full(
        (1, max_length + block_size), mask_token_id, dtype=torch.long, device=device,
    )
    output_ids[:, :num_input_tokens] = input_ids

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    # ---------- Multimodal prefill ----------
    prefill_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values_target,
        use_cache=True,
        output_hidden_states=True,
        logits_to_keep=1,
        return_dict=True,
    )
    if pixel_values is not None:
        prefill_kwargs["pixel_values"] = pixel_values
    if image_grid_thw is not None:
        prefill_kwargs["image_grid_thw"] = image_grid_thw
    if pixel_values_videos is not None:
        prefill_kwargs["pixel_values_videos"] = pixel_values_videos
    if video_grid_thw is not None:
        prefill_kwargs["video_grid_thw"] = video_grid_thw

    output = target(**prefill_kwargs)

    first_token = sample(output.logits, temperature)
    output_ids[:, num_input_tokens : num_input_tokens + 1] = first_token

    target_hidden = extract_context_feature(output.hidden_states, draft.target_layer_ids)

    draft_position_ids = torch.arange(
        max_length + block_size, device=device, dtype=torch.long
    ).unsqueeze(0)

    acceptance_lengths: list[int] = []
    start = num_input_tokens
    while start < max_length:
        # ========== Step A: 1-token target AR to produce t1 ==========
        ar_cache_pos = torch.arange(start, start + 1, device=device, dtype=torch.long)
        ar_out = target(
            input_ids=output_ids[:, start : start + 1],
            past_key_values=past_key_values_target,
            cache_position=ar_cache_pos,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        t1 = sample(ar_out.logits, temperature)
        if start + 1 >= max_length:
            output_ids[:, start + 1 : start + 2] = t1
            start += 1
            acceptance_lengths.append(1)
            break
        output_ids[:, start + 1 : start + 2] = t1
        # Append the AR token's per-layer hidden states to the draft cross-attn
        # context (target_hidden) — so the draft sees BOTH prior accepted tokens
        # and the freshly-AR'd t1 as conditioning.
        ar_hidden = extract_context_feature(ar_out.hidden_states, draft.target_layer_ids)
        target_hidden = torch.cat([target_hidden, ar_hidden], dim=1)
        start += 1

        # Early stop if t1 is itself a stop token.
        if stop_token_ids is not None and any(
            sid in output_ids[0, num_input_tokens : start].tolist() for sid in stop_token_ids
        ):
            acceptance_lengths.append(1)
            break

        # ========== Step B: draft proposes pos 1..B-1 of new block (anchor=t1) ==========
        block_output_ids = output_ids[:, start : start + block_size].clone()
        noise_embedding = embed_tokens(block_output_ids)

        draft_hidden = draft(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=draft_position_ids[
                :, past_key_values_draft.get_seq_length() : start + block_size
            ],
            past_key_values=past_key_values_draft,
            use_cache=True,
        )
        draft_logits = lm_head(draft_hidden[:, -block_size + 1 :, :])
        past_key_values_draft.crop(start)

        block_output_ids[:, 1:] = sample(draft_logits, temperature=0.0)

        # ========== Step C: target verify ==========
        cache_position = torch.arange(
            start, start + block_size, device=device, dtype=torch.long
        )
        verify_out = target(
            input_ids=block_output_ids,
            past_key_values=past_key_values_target,
            cache_position=cache_position,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

        posterior = sample(verify_out.logits, temperature)
        acceptance_length = (
            (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        )
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
            :, : acceptance_length + 1
        ]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
        start += acceptance_length + 1

        past_key_values_target.crop(start)
        target_hidden = extract_context_feature(
            verify_out.hidden_states, draft.target_layer_ids
        )[:, : acceptance_length + 1, :]

        # Tokens emitted this iter = 1 (AR) + acceptance_length (draft accepts) + 1 (bonus).
        # Recorded as the per-iter token count so avg_iter_tokens is directly
        # comparable to vlm_spec_generate's metric.
        acceptance_lengths.append(acceptance_length + 2)

        if stop_token_ids is not None and any(
            sid in output_ids[0, num_input_tokens:start + 1].tolist() for sid in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids is not None:
        stop_t = torch.tensor(stop_token_ids, device=output_ids.device)
        idxs = torch.isin(output_ids[0, num_input_tokens:], stop_t).nonzero(as_tuple=True)[0]
        if idxs.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + idxs[0].item() + 1]

    return {
        "output_ids": output_ids,
        "num_input_tokens": num_input_tokens,
        "num_output_tokens": output_ids.shape[1] - num_input_tokens,
        "acceptance_lengths": acceptance_lengths,
        "first_ar": True,
    }


@torch.inference_mode()
def vlm_spec_generate_multistep(
    target: nn.Module,
    draft: DFlashDraftModel,
    input_ids: torch.LongTensor,
    *,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 512,
    stop_token_ids: Optional[list[int]] = None,
    temperature: float = 0.0,
    block_size: Optional[int] = None,
    refinement_steps: int = 1,
    refinement_schedule: str = "cosine",
    lock_pos1: bool = False,
) -> dict:
    """Iterative-refinement (MaskGIT-style) variant of `vlm_spec_generate`.

    Identical to `vlm_spec_generate` except the per-block draft proposal is
    refined over up to `refinement_steps` passes. Each step keeps the
    highest-confidence predictions revealed and re-runs the draft with the
    remaining positions still masked. Reveal schedule is cosine
    (`_cosine_reveal_schedule`): few reveals at first, more later, last step
    flushes everything.

    Cost: each block now costs `R * c` draft compute (where R ≤ refinement_steps
    is the schedule length after dropping zero-reveal steps). Theoretical
    speedup S_eff = L / (1 + R * c).

    When `refinement_steps == 1`, behavior matches `vlm_spec_generate` exactly.
    """
    target.eval()
    draft.eval()
    device = input_ids.device
    if block_size is None:
        block_size = draft.block_size
    mask_token_id = draft.mask_token_id
    if mask_token_id is None:
        raise ValueError("draft.mask_token_id is not set; pass mask_token_id when building the draft")

    embed_tokens, lm_head = get_qwen3vl_embed_and_head(target)

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids = torch.full(
        (1, max_length + block_size), mask_token_id, dtype=torch.long, device=device,
    )
    output_ids[:, :num_input_tokens] = input_ids

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    # ---------- Multimodal prefill ----------
    prefill_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values_target,
        use_cache=True,
        output_hidden_states=True,
        logits_to_keep=1,
        return_dict=True,
    )
    if pixel_values is not None:
        prefill_kwargs["pixel_values"] = pixel_values
    if image_grid_thw is not None:
        prefill_kwargs["image_grid_thw"] = image_grid_thw
    if pixel_values_videos is not None:
        prefill_kwargs["pixel_values_videos"] = pixel_values_videos
    if video_grid_thw is not None:
        prefill_kwargs["video_grid_thw"] = video_grid_thw

    output = target(**prefill_kwargs)

    first_token = sample(output.logits, temperature)
    output_ids[:, num_input_tokens : num_input_tokens + 1] = first_token

    target_hidden = extract_context_feature(output.hidden_states, draft.target_layer_ids)

    draft_position_ids = torch.arange(
        max_length + block_size, device=device, dtype=torch.long
    ).unsqueeze(0)

    # Precompute the reveal schedule once (depends only on block_size + refinement_steps).
    num_maskable = block_size - 1
    reveal_schedule = _reveal_schedule(num_maskable, refinement_steps, refinement_schedule)
    effective_steps = len(reveal_schedule)

    acceptance_lengths: list[int] = []
    start = num_input_tokens
    while start < max_length:
        # Initial block: [anchor, mask, mask, ..., mask].
        block_output_ids = output_ids[:, start : start + block_size].clone()
        # Track which of the B-1 positions (indices 1..B-1 of the block) are still masked.
        still_masked = torch.ones(num_maskable, dtype=torch.bool, device=device)

        # Iterative refinement: each step refines confidence-ranked positions.
        # On step 0 we attach the fresh cross-attn context (target_hidden) and bring
        # the draft KV cache from `past_key_values_draft.get_seq_length()` up to
        # `start`. On steps ≥ 1 the KV already contains everything up to `start`,
        # so we pass an EMPTY cross-attn context and only process the noise block.
        empty_ctx = target_hidden.new_zeros(
            (target_hidden.shape[0], 0, target_hidden.shape[-1])
        )
        for step_idx, n_reveal in enumerate(reveal_schedule):
            noise_embedding = embed_tokens(block_output_ids)
            if step_idx == 0:
                cur_ctx = target_hidden
                cur_pos = draft_position_ids[
                    :, past_key_values_draft.get_seq_length() : start + block_size
                ]
            else:
                cur_ctx = empty_ctx
                cur_pos = draft_position_ids[:, start : start + block_size]
            draft_hidden = draft(
                target_hidden=cur_ctx,
                noise_embedding=noise_embedding,
                position_ids=cur_pos,
                past_key_values=past_key_values_draft,
                use_cache=True,
            )
            draft_logits = lm_head(draft_hidden[:, -num_maskable:, :])  # [1, B-1, V]
            # Drop the noise K/V we just appended; keep only context up to `start`.
            past_key_values_draft.crop(start)

            # Confidence = max softmax prob per position (greedy / temp=0).
            probs = F.softmax(draft_logits.float(), dim=-1)
            conf, pred = probs.max(dim=-1)  # [1, B-1], [1, B-1]
            conf = conf[0]
            pred = pred[0]

            # Restrict reveal candidates to still-masked positions only.
            conf_masked = conf.masked_fill(~still_masked, -1.0)
            # For varB-style training (--lock_pos1): block position 1 is kept
            # masked through every refinement step except the last, mirroring the
            # training distribution (where pos 1 is in the mask set at every
            # discrete level). still_masked index 0 == block position 1.
            if lock_pos1 and step_idx < effective_steps - 1:
                conf_masked[0] = -1.0
            n_actual = min(n_reveal, int(still_masked.sum().item()))
            if n_actual <= 0:
                continue
            topk = torch.topk(conf_masked, k=n_actual)
            reveal_idx = topk.indices  # in [0, B-2]
            # block_output_ids indices are reveal_idx + 1 (skip anchor at position 0).
            block_output_ids[0, reveal_idx + 1] = pred[reveal_idx]
            still_masked[reveal_idx] = False

        # ---------- Target verification ----------
        cache_position = torch.arange(
            start, start + block_size, device=device, dtype=torch.long
        )
        verify_out = target(
            input_ids=block_output_ids,
            past_key_values=past_key_values_target,
            cache_position=cache_position,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

        posterior = sample(verify_out.logits, temperature)
        acceptance_length = (
            (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        )
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
            :, : acceptance_length + 1
        ]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
        start += acceptance_length + 1

        past_key_values_target.crop(start)
        target_hidden = extract_context_feature(
            verify_out.hidden_states, draft.target_layer_ids
        )[:, : acceptance_length + 1, :]

        acceptance_lengths.append(acceptance_length + 1)

        if stop_token_ids is not None and any(
            sid in output_ids[0, num_input_tokens:start + 1].tolist() for sid in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids is not None:
        stop_t = torch.tensor(stop_token_ids, device=output_ids.device)
        idxs = torch.isin(output_ids[0, num_input_tokens:], stop_t).nonzero(as_tuple=True)[0]
        if idxs.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + idxs[0].item() + 1]

    return {
        "output_ids": output_ids,
        "num_input_tokens": num_input_tokens,
        "num_output_tokens": output_ids.shape[1] - num_input_tokens,
        "acceptance_lengths": acceptance_lengths,
        "refinement_steps_requested": refinement_steps,
        "refinement_steps_effective": effective_steps,
        "reveal_schedule": reveal_schedule,
        "refinement_schedule_kind": refinement_schedule,
        "lock_pos1": lock_pos1,
    }


@torch.inference_mode()
def vlm_spec_generate_3d(
    target: nn.Module,
    draft: DFlashDraftModel,
    input_ids: torch.LongTensor,
    *,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 512,
    stop_token_ids: Optional[list[int]] = None,
    temperature: float = 0.0,
    block_size: Optional[int] = None,
) -> dict:
    """3D-M-RoPE variant of `vlm_spec_generate` for DFlashDraftMRoPE3DModel drafts.

    Identical to `vlm_spec_generate` except the draft's `position_ids` are 3D
    (shape `(3, 1, T)`) and are derived from target's `get_rope_index` over the
    prompt, then extended by simple `arange` along each axis during the decode
    tail. The decode-tail extension is the same scheme used by
    `eval_ckpt_sweep_vt.py` for 3D drafts: `dpos = cat([prompt_3d, last + arange])`.
    """
    target.eval()
    draft.eval()
    device = input_ids.device
    if block_size is None:
        block_size = draft.block_size
    mask_token_id = draft.mask_token_id
    if mask_token_id is None:
        raise ValueError("draft.mask_token_id is not set; pass mask_token_id when building the draft")

    embed_tokens, lm_head = get_qwen3vl_embed_and_head(target)

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids = torch.full(
        (1, max_length + block_size), mask_token_id, dtype=torch.long, device=device,
    )
    output_ids[:, :num_input_tokens] = input_ids

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    # ---------- Multimodal prefill ----------
    prefill_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values_target,
        use_cache=True,
        output_hidden_states=True,
        logits_to_keep=1,
        return_dict=True,
    )
    if pixel_values is not None:
        prefill_kwargs["pixel_values"] = pixel_values
    if image_grid_thw is not None:
        prefill_kwargs["image_grid_thw"] = image_grid_thw
    if pixel_values_videos is not None:
        prefill_kwargs["pixel_values_videos"] = pixel_values_videos
    if video_grid_thw is not None:
        prefill_kwargs["video_grid_thw"] = video_grid_thw

    output = target(**prefill_kwargs)

    first_token = sample(output.logits, temperature)
    output_ids[:, num_input_tokens : num_input_tokens + 1] = first_token

    target_hidden = extract_context_feature(output.hidden_states, draft.target_layer_ids)

    # ---------- Build 3D position_ids over prompt + future decode tail ----------
    prompt_3d = get_target_3d_position_ids(
        target.vlm if hasattr(target, "vlm") else target,
        input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )                                                       # (3, 1, num_input_tokens)
    last = prompt_3d[:, :, -1:]                             # (3, 1, 1)
    tail_len = max_length + block_size - num_input_tokens
    tail_arange = torch.arange(1, tail_len + 1, device=device).view(1, 1, -1)
    draft_position_ids = torch.cat(
        [prompt_3d, last + tail_arange], dim=-1,
    )                                                       # (3, 1, max_length + block_size)

    acceptance_lengths: list[int] = []
    start = num_input_tokens
    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        noise_embedding = embed_tokens(block_output_ids)

        draft_hidden = draft(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=draft_position_ids[
                :, :, past_key_values_draft.get_seq_length() : start + block_size
            ],
            past_key_values=past_key_values_draft,
            use_cache=True,
        )
        draft_logits = lm_head(draft_hidden[:, -block_size + 1 :, :])
        past_key_values_draft.crop(start)

        block_output_ids[:, 1:] = sample(draft_logits, temperature=0.0)

        # ---------- Target verification ----------
        cache_position = torch.arange(
            start, start + block_size, device=device, dtype=torch.long
        )
        verify_out = target(
            input_ids=block_output_ids,
            past_key_values=past_key_values_target,
            cache_position=cache_position,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

        posterior = sample(verify_out.logits, temperature)
        acceptance_length = (
            (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        )
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
            :, : acceptance_length + 1
        ]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
        start += acceptance_length + 1

        past_key_values_target.crop(start)

        target_hidden = extract_context_feature(
            verify_out.hidden_states, draft.target_layer_ids
        )[:, : acceptance_length + 1, :]

        acceptance_lengths.append(acceptance_length + 1)

        if stop_token_ids is not None and any(
            sid in output_ids[0, num_input_tokens:start + 1].tolist() for sid in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids is not None:
        stop_t = torch.tensor(stop_token_ids, device=output_ids.device)
        idxs = torch.isin(output_ids[0, num_input_tokens:], stop_t).nonzero(as_tuple=True)[0]
        if idxs.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + idxs[0].item() + 1]

    return {
        "output_ids": output_ids,
        "num_input_tokens": num_input_tokens,
        "num_output_tokens": output_ids.shape[1] - num_input_tokens,
        "acceptance_lengths": acceptance_lengths,
    }


@torch.no_grad()
def warm_start_draft_from_target(
    draft: nn.Module,
    target_layers: list[nn.Module],
    target_layer_ids: list[int],
    verbose: bool = True,
) -> dict:
    """Initialize `draft.layers[i]` from `target_layers[target_layer_ids[i]]`.

    Copies every parameter that exists in both modules with matching shape
    (self_attn q/k/v/o projections + q/k norms, MLP gate/up/down, input + post
    attention layernorms). Leaves draft-only parameters untouched — specifically
    `draft.fc` (context projection) and `draft.hidden_norm`, which have no
    target counterpart.

    Typical caller site (inside train script, after draft is built):

        tgt_layers = vlm.language_model.layers   # Qwen3-VL path
        warm_start_draft_from_target(draft_module, tgt_layers,
                                      draft_module.target_layer_ids)
    """
    n_draft = len(draft.layers)
    if len(target_layer_ids) != n_draft:
        raise ValueError(
            f"draft has {n_draft} layers but target_layer_ids has "
            f"{len(target_layer_ids)} entries; they must match"
        )

    total_copied = 0
    total_skipped = 0
    total_mismatched = 0
    for draft_idx, tgt_idx in enumerate(target_layer_ids):
        dl = draft.layers[draft_idx]
        tl = target_layers[tgt_idx]

        draft_params = dict(dl.named_parameters())
        target_params = dict(tl.named_parameters())

        copied_here = []
        skipped_here = []
        mismatched_here = []
        for name, p_draft in draft_params.items():
            p_tgt = target_params.get(name)
            if p_tgt is None:
                skipped_here.append(name)
                continue
            if p_draft.shape != p_tgt.shape:
                mismatched_here.append(
                    f"{name}: draft{tuple(p_draft.shape)} vs target{tuple(p_tgt.shape)}"
                )
                continue
            p_draft.data.copy_(p_tgt.data.to(p_draft.dtype).to(p_draft.device))
            copied_here.append(name)

        if verbose:
            print(f"[warm_start] draft.layers[{draft_idx}] <- target.layers[{tgt_idx}]: "
                  f"copied={len(copied_here)} skipped={len(skipped_here)} "
                  f"mismatched={len(mismatched_here)}")
            for m in mismatched_here:
                print(f"  MISMATCH: {m}")
            for s in skipped_here:
                print(f"  SKIPPED (not in target): {s}")

        total_copied += len(copied_here)
        total_skipped += len(skipped_here)
        total_mismatched += len(mismatched_here)

    print(f"[warm_start_draft_from_target] total copied={total_copied} "
          f"skipped={total_skipped} mismatched={total_mismatched}")
    return {"copied": total_copied, "skipped": total_skipped,
            "mismatched": total_mismatched}
