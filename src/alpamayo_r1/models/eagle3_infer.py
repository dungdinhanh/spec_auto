"""EAGLE-3 inference draft (tree decoding compatible).

Same trained weights as `Eagle3DraftModel` (training) but uses:
- Standard causal self-attention (no `cache_hidden` multi-step).
- Optional `tree_mask` attribute for tree-decoded inference.
- Standard `DynamicCache` for autoregressive draft tree generation.

State-dict keys are identical to `Eagle3DraftModel`, so a saved ckpt loads
into either class. The choice of class determines attention semantics:
  - `Eagle3DraftModel` → multi-step rollout (training)
  - `Eagle3InferModel` → tree decoding inference

Mirrors `EAGLE/eagle/model/cnets.py` (the inference-side Model class).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from alpamayo_r1.models.eagle3_draft import (
    LlamaRotaryEmbedding, LlamaRMSNorm, LlamaMLP, Eagle3DraftConfig,
    apply_rotary_pos_emb, repeat_kv, _make_causal_mask, _expand_mask,
)


class LlamaInferAttention(nn.Module):
    """Standard causal self-attention with optional tree_mask.

    Shares the 2H input convention with training (concat of input_emb +
    hidden_states), so the trained Q/K/V/O weights load directly.
    """

    def __init__(self, config: Eagle3DraftConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.q_proj = nn.Linear(self.hidden_size * 2, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim, max_position_embeddings=self.max_position_embeddings,
            base=getattr(config, "rope_theta", 10000),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,            # (B, q_len, 2H)
        attention_mask: Optional[torch.Tensor], # (B, 1, q_len, kv_len) additive mask
        position_ids: torch.LongTensor,         # (B, q_len)
        past_key_value: Optional[DynamicCache] = None,
        layer_idx: int = 0,
    ):
        bsz, q_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            kv_seq_len = past_key_value.get_seq_length(layer_idx) + q_len
        else:
            kv_seq_len = q_len
        # Rotary cache must cover the highest position we actually USE in
        # position_ids. position_ids may be sparse (tree positions) so size
        # by max position_ids value, not seq_len.
        max_pos = int(position_ids.max().item()) + 1
        cos, sin = self.rotary_emb(q, seq_len=max(max_pos, kv_seq_len))
        cos, sin = cos.to(q.device), sin.to(q.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        if past_key_value is not None:
            k, v = past_key_value.update(k, v, layer_idx, cache_kwargs={"cache_position": None})

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        return self.o_proj(out)


class LlamaInferDecoderLayer(nn.Module):
    """Single decoder layer for inference. Same weight layout as training:
    `hidden_norm`, `input_layernorm`, `self_attn`, `post_attention_layernorm`,
    `mlp`. Tree mask injection happens via `attention_mask` passed in."""

    def __init__(self, config: Eagle3DraftConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaInferAttention(config)
        self.mlp = LlamaMLP(config)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_emb, hidden_states, attention_mask, position_ids,
                past_key_value=None, layer_idx=0):
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)
        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)  # (B, q_len, 2H)
        attn_out = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask,
            position_ids=position_ids, past_key_value=past_key_value,
            layer_idx=layer_idx,
        )
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Eagle3InferModel(nn.Module):
    """Inference-style EAGLE-3 draft.

    Loads weights from a `Eagle3DraftModel` ckpt — same key layout
    (`fc`, `midlayer.*`, `norm`).

    Forward signature differs from training: takes a single `hidden_states`
    tensor (either target's fused 3H hiddens, or H-dim from a prior step)
    and applies `self.fc` only when the input is 3H.
    """

    def __init__(self, config: Eagle3DraftConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.target_layer_ids = list(config.target_layer_ids)
        self.fc = nn.Linear(self.hidden_size * len(self.target_layer_ids),
                            self.hidden_size, bias=False)
        self.midlayer = LlamaInferDecoderLayer(config)
        self.norm = LlamaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.tree_mask: Optional[torch.Tensor] = None  # set by caller for tree decode

    def _build_attention_mask(self, q_len, kv_len, dtype, device, past_kv_len=0,
                              tree_mask: Optional[torch.Tensor] = None):
        """Return additive 4D causal mask (1, 1, q_len, kv_len). `tree_mask`
        can be rectangular `(q_len, overlay_kv_len)` — the overlay covers the
        TRAILING `overlay_kv_len` columns of the kv axis, leaving any earlier
        columns governed by causal mask only (e.g. the root token in EAGLE's
        tree topology)."""
        mask = _make_causal_mask((1, q_len), dtype, device, past_kv_len)  # (1, 1, q_len, kv_len)
        if tree_mask is not None:
            if tree_mask.dim() == 2:
                tm = tree_mask
            else:
                tm = tree_mask.view(*tree_mask.shape[-2:])
            tm_q, tm_kv = tm.shape
            if tm_q != q_len:
                raise ValueError(f"tree_mask q-dim {tm_q} != q_len {q_len}")
            tm = tm.to(device=device, dtype=torch.bool)
            min_val = torch.finfo(dtype).min
            # Overlay on the trailing tm_kv columns; leftmost (kv_len - tm_kv)
            # columns remain causal-only.
            mask[..., -tm_kv:][:, :, ~tm] = min_val
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,            # (B, q_len, H) or (B, q_len, 3H)
        input_ids: torch.LongTensor,            # (B, q_len)
        embed_tokens: nn.Module,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        use_cache: bool = True,
        tree_mask: Optional[torch.Tensor] = None,   # (q_len, q_len) bool — None = causal only
    ) -> Tuple[torch.Tensor, DynamicCache]:
        device = hidden_states.device
        bsz, q_len = input_ids.shape

        # Apply fc only if hidden_states comes in as 3H (target's fused hiddens).
        if hidden_states.shape[-1] == self.hidden_size * len(self.target_layer_ids):
            hidden_states = self.fc(hidden_states.to(self.fc.weight.dtype))

        if past_key_values is None:
            past_key_values = DynamicCache()
        past_kv_len = past_key_values.get_seq_length(0) if past_key_values is not None else 0

        if position_ids is None:
            position_ids = torch.arange(past_kv_len, past_kv_len + q_len,
                                         device=device).unsqueeze(0)

        attn_mask = self._build_attention_mask(
            q_len, q_len + past_kv_len, hidden_states.dtype, device,
            past_kv_len=past_kv_len, tree_mask=tree_mask,
        )

        input_emb = embed_tokens(input_ids).to(hidden_states.dtype)
        hidden_out = self.midlayer(
            input_emb=input_emb, hidden_states=hidden_states,
            attention_mask=attn_mask, position_ids=position_ids,
            past_key_value=past_key_values, layer_idx=0,
        )
        return hidden_out, past_key_values


def build_eagle3_infer_from_ckpt(target, ckpt, dtype=torch.bfloat16) -> Eagle3InferModel:
    """Reconstruct Eagle3InferModel matching the ckpt shape; load weights.
    `target` is a Qwen3-VL VLM (the unwrapped target, e.g. AlpamayoR1.vlm)."""
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
    model = Eagle3InferModel(cfg).to(dtype=dtype)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[eagle3-infer-load] target_layer_ids={target_layer_ids} "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    return model
