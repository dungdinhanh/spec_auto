"""M-RoPE variant of the DFlash draft.

The author's `DFlashDraftModel` uses 1D RoPE via `Qwen3RotaryEmbedding`. This
file provides a subclass that swaps in `Qwen3VLTextRotaryEmbedding` so the
draft applies 3D multimodal rotary (same RoPE variant the Qwen3-VL target
uses), matching the target model's positional-encoding regime.

Mechanism: Qwen3VL's `Qwen3VLTextRotaryEmbedding.forward(x, position_ids)`
returns cos/sin tensors shaped exactly like the plain 1D version, so we only
need to replace `self.rotary_emb` — the downstream `apply_rotary_pos_emb` in
`Qwen3DFlashAttention` consumes them identically. For text-only position
streams (all three M-RoPE axes equal), M-RoPE degenerates to 1D RoPE.

The distinction matters for cross-attention over `target_hidden`: the target's
K/V tensors were produced with the target's 3D M-RoPE rotations; applying the
same class of rotation in the draft at least keeps the frequency geometry
consistent, whereas 1D RoPE mixes two different RoPE flavours in the same
attention.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers.cache_utils import Cache

from dflash.model import DFlashDraftModel


def _build_mrope_rotary(config) -> nn.Module:
    """Return a Qwen3VLTextRotaryEmbedding configured from the draft's config.

    The config passed here is the DFlash draft's Qwen3Config augmented with
    rope_scaling['mrope_section']. Qwen3VLTextRotaryEmbedding reads:
      - config.rope_scaling (including mrope_section)
      - config.max_position_embeddings
      - config.rope_theta
    which Qwen3Config already provides.
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding
    return Qwen3VLTextRotaryEmbedding(config)


class DFlashDraftMRoPEModel(DFlashDraftModel):
    """DFlash draft with 3D multimodal rotary embedding (M-RoPE).

    Identical to `DFlashDraftModel` except:
      1. `self.rotary_emb` is `Qwen3VLTextRotaryEmbedding` (M-RoPE 3D) instead
         of `Qwen3RotaryEmbedding` (1D).
      2. `forward` accepts `position_ids` of shape (B, L) or (3, B, L). If (B, L)
         is given, the rotary module auto-expands to (3, B, L).

    Weights are compatible with `DFlashDraftModel` for all attention/MLP
    parameters — only the rotary-embedding buffer differs. This means a
    pretrained 1D-draft checkpoint can be loaded into this class (with some
    shape differences handled by `strict=False`), useful for fine-tuning.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        # Make sure rope_scaling has an mrope_section entry; if not, default to
        # the canonical Qwen3-VL-8B split.
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        if "mrope_section" not in rope_scaling:
            rope_scaling = {**rope_scaling, "mrope_section": [24, 20, 20]}
            if "rope_type" not in rope_scaling:
                rope_scaling["rope_type"] = "default"
            config.rope_scaling = rope_scaling
        # Swap the 1D rotary for M-RoPE
        self.rotary_emb = _build_mrope_rotary(config)

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        # Qwen3VLTextRotaryEmbedding accepts (B, L) or (3, B, L); returns cos/sin
        # in the same shape as 1D RoPE, so the rest of the stack is untouched.
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)


def build_dflash_draft_mrope_for_qwen3vl(
    target: nn.Module,
    num_draft_layers: int = 1,
    block_size: int = 4,
    mask_token_id: Optional[int] = None,
    target_layer_ids=None,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
    propagate_mrope_interleaved: bool = False,
) -> DFlashDraftMRoPEModel:
    """Build an M-RoPE DFlash draft matched to a Qwen3-VL target.

    `propagate_mrope_interleaved`: when True, copies target's `mrope_interleaved`
    flag (typically True for Qwen3-VL-8B) into the draft's rope_scaling. Default
    False preserves the original behavior, which omits the flag and so makes
    the draft use the non-interleaved rotation channel layout. Old SFT
    checkpoints were trained under the False regime; setting True here changes
    the rotary geometry and is incompatible with those ckpts."""
    # Reuse the 1D-draft config builder — it already sets rope_theta,
    # max_position_embeddings, attention dims, etc.
    from alpamayo_r1.models.dflash_draft import build_qwen3_draft_config
    cfg = build_qwen3_draft_config(
        target,
        num_draft_layers=num_draft_layers,
        block_size=block_size,
        mask_token_id=mask_token_id,
        target_layer_ids=target_layer_ids,
    )
    cfg._attn_implementation = attn_implementation
    cfg.dtype = dtype

    # Propagate the target's mrope_section if present; otherwise default.
    target_cfg = target.config.get_text_config() if hasattr(target.config, "get_text_config") \
        else getattr(target.config, "text_config", target.config)
    target_rope_scaling = getattr(target_cfg, "rope_scaling", None)
    if target_rope_scaling and "mrope_section" in target_rope_scaling:
        rs = {
            "rope_type": target_rope_scaling.get("rope_type", "default"),
            "mrope_section": target_rope_scaling["mrope_section"],
        }
        # Opt-in: propagate target's mrope_interleaved (changes rotation layout).
        if propagate_mrope_interleaved and "mrope_interleaved" in target_rope_scaling:
            rs["mrope_interleaved"] = target_rope_scaling["mrope_interleaved"]
        cfg.rope_scaling = rs

    draft = DFlashDraftMRoPEModel(cfg)
    return draft.to(dtype=dtype)


# ---------------------------------------------------------------------------
# v3 / "full M-RoPE" draft: subclass that bakes in target's interleave channel
# layout AND requires 3D position_ids of shape (3, B, T). This is the
# architecturally-aligned variant — the existing DFlashDraftMRoPEModel above
# is kept unchanged so old 1D-trained ckpts continue to load and behave
# identically under that class.
# ---------------------------------------------------------------------------


class DFlashDraftMRoPE3DModel(DFlashDraftMRoPEModel):
    """DFlash draft fully aligned with target's M-RoPE regime.

    Differences from `DFlashDraftMRoPEModel`:
      1. `mrope_interleaved=True` is forced into rope_scaling at __init__
         time (Qwen3-VL-8B target uses this; the parent class omits the flag,
         which silently changes the rotation channel layout).
      2. `forward` REQUIRES `position_ids` of shape (3, B, L). Passing (B, L)
         raises — that path used to silently degenerate to 1D RoPE in the
         parent and is the bug this class is built to prevent.

    Architecture is otherwise byte-identical to `DFlashDraftMRoPEModel`, so
    weight tensors are interchangeable. But ckpts trained under the parent
    (1D-pos regime) are NOT directly usable here — the rotary geometry differs.
    """

    def __init__(self, config) -> None:
        # Force mrope_interleaved=True before parent __init__ so the rotary
        # buffer is built under the right channel layout.
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        if rope_scaling.get("mrope_interleaved", False) is not True:
            rope_scaling = {**rope_scaling, "mrope_interleaved": True}
            if "rope_type" not in rope_scaling:
                rope_scaling["rope_type"] = "default"
            if "mrope_section" not in rope_scaling:
                rope_scaling["mrope_section"] = [24, 20, 20]
            config.rope_scaling = rope_scaling
        super().__init__(config)

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        # Hard requirement: 3D positions (3, B, L). Reject the silent
        # 1D-degeneration path that the parent class still allows.
        if position_ids.dim() != 3 or position_ids.shape[0] != 3:
            raise ValueError(
                f"DFlashDraftMRoPE3DModel requires 3D position_ids of shape "
                f"(3, B, L); got shape {tuple(position_ids.shape)}. Use "
                f"alpamayo_r1.models.dflash_draft.get_target_3d_position_ids "
                f"to compute it."
            )
        return super().forward(
            position_ids=position_ids,
            attention_mask=attention_mask,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )


def build_dflash_draft_mrope3d_for_qwen3vl(
    target: nn.Module,
    num_draft_layers: int = 1,
    block_size: int = 4,
    mask_token_id: Optional[int] = None,
    target_layer_ids=None,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
) -> DFlashDraftMRoPE3DModel:
    """Build a DFlashDraftMRoPE3DModel matched to a Qwen3-VL target.

    Differences from `build_dflash_draft_mrope_for_qwen3vl`:
      - Returns a `DFlashDraftMRoPE3DModel` (the 3D-required subclass).
      - Always propagates target's `mrope_interleaved` into rope_scaling
        (the new class also forces it on at __init__).
    """
    from alpamayo_r1.models.dflash_draft import build_qwen3_draft_config
    cfg = build_qwen3_draft_config(
        target,
        num_draft_layers=num_draft_layers,
        block_size=block_size,
        mask_token_id=mask_token_id,
        target_layer_ids=target_layer_ids,
    )
    cfg._attn_implementation = attn_implementation
    cfg.dtype = dtype

    target_cfg = target.config.get_text_config() if hasattr(target.config, "get_text_config") \
        else getattr(target.config, "text_config", target.config)
    target_rope_scaling = getattr(target_cfg, "rope_scaling", None)
    if target_rope_scaling and "mrope_section" in target_rope_scaling:
        cfg.rope_scaling = {
            "rope_type": target_rope_scaling.get("rope_type", "default"),
            "mrope_section": target_rope_scaling["mrope_section"],
            "mrope_interleaved": target_rope_scaling.get("mrope_interleaved", True),
        }

    draft = DFlashDraftMRoPE3DModel(cfg)
    return draft.to(dtype=dtype)
