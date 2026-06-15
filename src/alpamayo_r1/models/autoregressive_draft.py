"""EAGLE-3-style autoregressive draft for Qwen3-VL target.

At each position t the draft receives a fusion of two H-dim vectors:
  - `embed_tokens(token_t)`: the token embedding (shared weights with target),
  - `context_hidden_t`: an H-dim "hidden" vector that carries target info.

During training, `context_hidden_t` is the target's hidden state at position
t from the target layer `target_layer_idx` — typically a late layer that
captures visual + textual features from target's vision tower.

During spec-decoding inference, `context_hidden_t` is:
  * target's hidden state at t for prompt and already-committed positions, and
  * the draft's own hidden output from the previous step (chaining) for newly
    proposed positions.

The chain works because the draft's output hidden is the same H-dim tensor as
target's hidden — the training-time signal pushes the draft's output toward
the distribution of target's hidden, so feeding it back as context at the
next step is a well-defined approximation.

Output: final hidden (B, T, H). Apply `lm_head(hidden)` externally for logits.
"""
from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
)


class ARDraftConfig(Qwen3Config):
    model_type = "ar_draft_eagle3"

    def __init__(self, *, target_layer_ids: Optional[list[int]] = None, **kwargs):
        super().__init__(**kwargs)
        # EAGLE-3 fuses multiple target hiddens (low/mid/high). Default = single
        # last layer for backward compat.
        self.target_layer_ids = list(target_layer_ids) if target_layer_ids else [-1]


class ARDraftModel(PreTrainedModel):
    """EAGLE-3-style AR draft: N Qwen3-VL text decoder layers with multi-layer
    feature fusion at the input. The draft is conditioned on a concatenation of
    the token embedding and several target-layer hiddens (low/mid/high in the
    EAGLE-3 paper)."""

    config_class = ARDraftConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_cache_class = True

    def __init__(self, config: ARDraftConfig):
        super().__init__(config)
        self.config = config
        self.target_layer_ids = list(config.target_layer_ids)
        n_ctx = len(self.target_layer_ids)

        # Fuse (1 + n_ctx) × H into H at input.
        # EAGLE-3 paper: concat(token_embed, h_low, h_mid, h_high) -> H.
        self.input_proj = nn.Linear(
            (1 + n_ctx) * config.hidden_size, config.hidden_size, bias=False
        )
        self.input_norm = Qwen3VLTextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layers = nn.ModuleList([
            Qwen3VLTextDecoderLayer(config, i)
            for i in range(config.num_hidden_layers)
        ])
        self.final_norm = Qwen3VLTextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config)

    def forward(
        self,
        input_embeds: torch.Tensor,
        context_hidden: torch.Tensor | list[torch.Tensor],
        position_ids: torch.LongTensor,
        past_key_values: Optional[DynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Args:
          input_embeds:   (B, T, H) — token embeddings.
          context_hidden: list of N (B, T, H) tensors — target hiddens at the
                          configured layer ids (low/mid/high). For backward
                          compatibility, a single tensor is also accepted (will
                          be wrapped in a 1-element list).
          position_ids:   (3, B, T) for M-RoPE, or (B, T) 1D.
        Returns: (B, T, H) — final hidden (caller applies lm_head for logits).
        """
        # Normalise input. If a single tensor came in, treat as a 1-element list.
        if isinstance(context_hidden, torch.Tensor):
            ctx_list = [context_hidden]
        else:
            ctx_list = list(context_hidden)
        if len(ctx_list) != len(self.target_layer_ids):
            raise ValueError(
                f"got {len(ctx_list)} context hiddens, expected "
                f"{len(self.target_layer_ids)} (one per target_layer_ids entry)"
            )

        # Fuse token + multi-layer context at input.
        fused = torch.cat([input_embeds, *ctx_list], dim=-1)   # (B, T, (1+n_ctx)*H)
        hidden = self.input_norm(self.input_proj(fused))       # (B, T, H)

        position_embeddings = self.rotary_emb(hidden, position_ids)
        for layer in self.layers:
            out = layer(
                hidden,
                attention_mask=None,  # causal baked into layer
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden = out[0] if isinstance(out, tuple) else out
        return self.final_norm(hidden)


def build_ar_draft_for_qwen3vl(
    target,
    num_draft_layers: int,
    target_layer_ids: Optional[list[int]] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> ARDraftModel:
    """Build an EAGLE-3-style AR draft from a Qwen3-VL target.

    target_layer_ids: list of target layer indices whose hiddens are fused as
    input to the draft. Paper-faithful EAGLE-3 uses three layers (low/mid/high).
    Defaults to a single last-layer entry [num_hidden_layers - 1] for backward
    compat with the original 1-layer fusion variant.
    """
    # `target` may be either the bare VLM (Qwen3VLForConditionalGeneration) or the
    # AlpamayoR1 wrapper. The Qwen3-VL text config lives on the VLM.
    vlm_cfg = target.vlm.config if hasattr(target, "vlm") else target.config
    target_text_cfg = vlm_cfg.get_text_config()
    draft_cfg_dict = target_text_cfg.to_dict()
    draft_cfg_dict["num_hidden_layers"] = num_draft_layers
    draft_cfg_dict["_attn_implementation"] = "sdpa"

    # M-RoPE scaling (always non-None — Qwen3VLTextRotaryEmbedding reads it)
    target_rope_scaling = getattr(target_text_cfg, "rope_scaling", None) or {}
    draft_cfg_dict["rope_scaling"] = {
        "rope_type": target_rope_scaling.get("rope_type", "default"),
        "mrope_section": target_rope_scaling.get("mrope_section", [24, 20, 20]),
    }

    n_target = target_text_cfg.num_hidden_layers
    if target_layer_ids is None:
        target_layer_ids = [n_target - 1]
    # Normalise negative indices.
    target_layer_ids = [i if i >= 0 else n_target + i for i in target_layer_ids]
    draft_cfg_dict["target_layer_ids"] = target_layer_ids

    config = ARDraftConfig(**draft_cfg_dict)
    model = ARDraftModel(config)
    return model.to(dtype=dtype)


@torch.no_grad()
def warm_start_ar_draft_from_target(
    draft: ARDraftModel,
    target_layers: list[nn.Module],
    layer_ids: list[int],
    verbose: bool = True,
) -> dict:
    """Copy target layer weights into draft layers (shape-safe)."""
    if len(draft.layers) != len(layer_ids):
        raise ValueError(
            f"draft has {len(draft.layers)} layers but layer_ids has "
            f"{len(layer_ids)} entries"
        )
    copied = skipped = mismatched = 0
    for draft_idx, tgt_idx in enumerate(layer_ids):
        dl, tl = draft.layers[draft_idx], target_layers[tgt_idx]
        dp, tp = dict(dl.named_parameters()), dict(tl.named_parameters())
        cc, ss, mm = [], [], []
        for name, p_d in dp.items():
            p_t = tp.get(name)
            if p_t is None:
                ss.append(name); continue
            if p_d.shape != p_t.shape:
                mm.append(name); continue
            p_d.data.copy_(p_t.data.to(p_d.dtype).to(p_d.device))
            cc.append(name)
        if verbose:
            print(f"[ar-eagle warm_start] draft.layers[{draft_idx}] <- target.layers"
                  f"[{tgt_idx}]: copied={len(cc)} skipped={len(ss)} mismatched={len(mm)}")
        copied += len(cc); skipped += len(ss); mismatched += len(mm)
    print(f"[warm_start_ar_draft_from_target] copied={copied} skipped={skipped} "
          f"mismatched={mismatched}")
    return {"copied": copied, "skipped": skipped, "mismatched": mismatched}


def _wrap_ar_ckpt(state_dict, num_draft_layers: int,
                   target_layer_ids: list[int]) -> dict:
    return {
        "state_dict": state_dict,
        "num_draft_layers": num_draft_layers,
        "target_layer_ids": list(target_layer_ids),
        "kind": "ar_draft_eagle3",
    }


def load_ar_draft_checkpoint(path: str, map_location=None) -> dict:
    obj = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        # Back-compat: old ckpts saved a single target_layer_idx.
        if "target_layer_ids" not in obj and "target_layer_idx" in obj:
            obj["target_layer_ids"] = [obj.pop("target_layer_idx")]
        return obj
    return {"state_dict": obj, "num_draft_layers": None,
            "target_layer_ids": None, "kind": "ar_draft_eagle3"}
