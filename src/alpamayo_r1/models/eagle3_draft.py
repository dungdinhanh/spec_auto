"""EAGLE-3 paper-faithful draft for Qwen3-VL targets.

Direct port of `EAGLE/eagle/traineagle3/cnets.py`, adapted for Qwen3-VL:
- 1 transformer layer (`config.num_hidden_layers = 1`).
- `fc = Linear(3*H, H, bias=False)` fuses three target hiddens (low/mid/high).
- Per-layer: `hidden_norm` + `input_layernorm` (separate RMSNorms), then concat
  along feature dim → 2H → q/k/v projections (each `2H -> n_heads*head_dim`).
- Custom `LlamaAttention.forward` with `cache_hidden` mechanism: at rollout
  step i, attention sees current step's full K/V (over the sequence) plus
  diagonal element-wise products with steps 0..i-1's K — this is the channel
  by which the draft's previous-step hidden flows into the next step.
- `length=7` multi-step rollout in `Model.forward`.

We use 1D rotary in the draft (same as EAGLE-3 with Llama). The Qwen3-VL
target's M-RoPE 3D is internal to its own forward; we just consume the hidden
states it returns.

Vocab: full target vocab (no t2d/d2t subset).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_causal_mask(input_shape, dtype, device, past_key_values_length=0):
    bsz, tgt_len = input_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask, dtype, tgt_len=None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inv = 1.0 - expanded
    return inv.masked_fill(inv.to(torch.bool), torch.finfo(dtype).min)


def repeat_kv(hidden_states, n_rep):
    batch, num_kv, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv * n_rep, slen, head_dim)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings,
                                device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        v = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(v + self.variance_epsilon)
        return self.weight * x.to(in_dtype)


class LlamaAttention(nn.Module):
    """EAGLE-3 attention with cache_hidden multi-step diagonal mechanism.

    Q/K/V projections take 2H input (concat of input_emb and hidden_states from
    the per-layer fusion). At rollout step i, attention computes:
      - standard scaled-dot-product over current step's K/V (full sequence)
      - diagonal element-wise (q[p] * k_step_j[p]).sum(-1) for j in 0..i-1,
        appended as additional logits before softmax. v_step_j[p] is then
        weighted by the corresponding softmax weight and added to the output.
    """

    def __init__(self, config):
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

        self.use_mrope_3d = bool(getattr(config, "use_mrope_3d", False))
        if self.use_mrope_3d:
            # M-RoPE 3D: import the Qwen3-VL rotary so the draft sees the
            # same rotational layout as target. Positions arrive as (3, B, T).
            from transformers.models.qwen3_vl.modeling_qwen3_vl import (
                Qwen3VLTextRotaryEmbedding,
            )
            class _Cfg:
                pass
            cfg = _Cfg()
            cfg.rope_scaling = {
                "rope_type": "default",
                "mrope_section": list(config.mrope_section),
                "mrope_interleaved": bool(config.mrope_interleaved),
            }
            cfg.max_position_embeddings = self.max_position_embeddings
            cfg.head_dim = self.head_dim
            cfg.rope_theta = getattr(config, "rope_theta", 10000)
            cfg.hidden_size = config.hidden_size
            cfg.num_attention_heads = config.num_attention_heads
            self.rotary_emb = Qwen3VLTextRotaryEmbedding(cfg)
        else:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim, max_position_embeddings=self.max_position_embeddings,
                base=getattr(config, "rope_theta", 10000),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_hidden: List[List[torch.Tensor]],
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,  # (B, T) for 1D; (3, B, T) for 3D M-RoPE
    ) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        lck = len(cache_hidden[0])  # number of previous rollout steps cached

        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.use_mrope_3d:
            # position_ids: (3, B, q_len) for the three M-RoPE sections.
            # Shift all 3 axes by lck so each rollout step occupies its own
            # rotary window — same semantics as the 1D `+lck`.
            shifted = position_ids + lck                                  # (3, B, q_len)
            # Qwen3VL rotary returns (cos, sin) already interleaved per M-RoPE
            # section. Standard apply_rotary_pos_emb (q, k, cos, sin) applies it.
            cos, sin = self.rotary_emb(q, shifted)
            from transformers.models.qwen3_vl.modeling_qwen3_vl import (
                apply_rotary_pos_emb as _qwen3vl_apply_rope,
            )
            q, k = _qwen3vl_apply_rope(q, k, cos, sin)
        else:
            # 1D rotary. cache must cover the largest position we'll index.
            max_pos = int(position_ids.max().item()) + lck
            cos, sin = self.rotary_emb(q, seq_len=max_pos + 1)
            cos, sin = cos.to(q.device), sin.to(q.device)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids + lck)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        # Avoid in-place mutation for grad-checkpointing compatibility.
        local_k = list(cache_hidden[0]) + [k]
        local_v = list(cache_hidden[1]) + [v]
        cache_k, cache_v = local_k, local_v

        k0 = cache_k[0]
        v0 = cache_v[0]
        attn = torch.matmul(q, k0.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn = attn + attention_mask
        # Diagonal logits to prior rollout steps' K (per-position dot product).
        for i in range(1, len(cache_k)):
            ki = cache_k[i]
            ai = (q * ki).sum(-1) / math.sqrt(self.head_dim)
            attn = torch.cat((attn, ai[..., None]), dim=-1)

        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        # Step-0 portion of attn: first k0.shape[2] entries (= kv_len of step 0).
        # At training kv_len=q_len=T so this matches the original `:q_len` slice.
        # At inference q_len can be 1 while kv_len is the prefill length.
        k0_len = k0.shape[2]
        attn0 = attn[..., :k0_len]
        out = torch.matmul(attn0, v0)
        for i in range(1, len(cache_v)):
            vi = cache_v[i]
            # Diagonal weight: the i-th cross-step entry sits at index k0_len+i-1.
            ai = attn[..., k0_len + i - 1]
            out = out + ai[..., None] * vi

        out = out.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        out = self.o_proj(out)
        return out, [local_k, local_v]


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    """Per-layer concat of `input_emb` + `hidden_states` with separate norms.

    `hidden_states` here is either the projected target fused hidden (step 0)
    or the previous rollout step's draft hidden output (steps 1..length-1).
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_emb, hidden_states, cache_hidden, attention_mask, position_ids):
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)
        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)  # (B, T, 2H)

        attn_out, new_cache_hidden = self.self_attn(
            hidden_states=hidden_states, cache_hidden=cache_hidden,
            attention_mask=attention_mask, position_ids=position_ids,
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, new_cache_hidden


def _shift_left(t):
    """Drop position 0, append zero at the end. Mirrors EAGLE's `padding(left=False)`."""
    z = torch.zeros_like(t[:, -1:])
    return torch.cat((t[:, 1:], z), dim=1)


class Eagle3DraftConfig:
    """Lightweight config object — we don't need PretrainedConfig machinery.

    `mrope_section` (list of 3 ints summing to head_dim/2) enables M-RoPE 3D
    in the draft attention; positions must then be passed as (3, B, T).
    `mrope_section=None` (default) keeps 1D rotary — our paper's contribution.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        vocab_size: int,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 8192,
        rope_scaling=None,
        target_layer_ids: Optional[List[int]] = None,
        rollout_length: int = 7,
        mrope_section: Optional[List[int]] = None,
        mrope_interleaved: bool = True,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.rope_scaling = rope_scaling
        self.target_layer_ids = list(target_layer_ids) if target_layer_ids is not None else [1, 17, 32]
        self.rollout_length = rollout_length
        self.mrope_section = list(mrope_section) if mrope_section is not None else None
        self.mrope_interleaved = mrope_interleaved
        self.use_mrope_3d = mrope_section is not None


class Eagle3DraftModel(nn.Module):
    """Trainable EAGLE-3 draft. Owns:
       - `fc`: Linear(3H, H, bias=False)
       - `midlayer`: a single LlamaDecoderLayer (config.num_hidden_layers must be 1)
       - `norm`: final RMSNorm

    Does NOT own embed_tokens or lm_head — those are passed in (frozen, from
    target). At training, runs `length` rollout steps with cache_hidden
    accumulating across steps; returns per-step logits + per-step accuracy
    counters for the multi-step KL loss.
    """

    def __init__(self, config: Eagle3DraftConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.length = config.rollout_length
        self.target_layer_ids = list(config.target_layer_ids)
        # 3 target hiddens fused → H. EAGLE-3: bias=False.
        self.fc = nn.Linear(self.hidden_size * len(self.target_layer_ids), self.hidden_size, bias=False)
        if config.num_hidden_layers != 1:
            raise ValueError("EAGLE-3 paper-faithful uses num_hidden_layers=1 (single midlayer).")
        self.midlayer = LlamaDecoderLayer(config)
        self.norm = LlamaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, dtype, device, past_kv_len=0):
        combined = None
        if input_shape[-1] > 1:
            combined = _make_causal_mask(input_shape, dtype, device, past_kv_len)
        if attention_mask is not None:
            expanded = _expand_mask(attention_mask, dtype, tgt_len=input_shape[-1]).to(device)
            combined = expanded if combined is None else expanded + combined
        return combined

    def forward(
        self,
        target_hiddens: List[torch.Tensor],   # list of N (B, T, H), N == len(target_layer_ids)
        input_ids: torch.LongTensor,           # (B, T)
        embed_tokens: nn.Module,               # frozen, from target
        lm_head: nn.Module,                    # frozen, from target
        target_logits: torch.Tensor,           # (B, T, V) target's logits at all positions
        loss_mask: torch.Tensor,               # (B, T) — 1 at output positions, 0 elsewhere
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        """Returns (plosses: list[Tensor], acces: list[float]).

        plosses[k] = soft-KL distillation loss at rollout step k.
        acces[k] = fraction of position-correct argmax matches at step k.
        """
        B, T = input_ids.shape
        if len(target_hiddens) != len(self.target_layer_ids):
            raise ValueError(f"got {len(target_hiddens)} target_hiddens, expected "
                             f"{len(self.target_layer_ids)}")
        device = target_hiddens[0].device
        dtype = target_hiddens[0].dtype

        # 1. Fuse target hiddens via fc, project to H.
        fused = torch.cat(target_hiddens, dim=-1).to(dtype)
        hidden_states = self.fc(fused)  # (B, T, H)

        # 2. EAGLE-3 dataprepare: shift target_logits and input_ids LEFT once;
        # leave loss_mask un-shifted. After each rollout step (except last),
        # all three are shifted by 1. At step k:
        #   - input_ids and target are shifted (k+1) times
        #   - loss_mask is shifted k times
        # → step 0 trains "given embed(token[p+1]) + target_hidden[p], predict
        #   target's distribution at p+1", masked by original loss_mask.
        target = _shift_left(target_logits)              # (B, T, V)
        ids = _shift_left(input_ids)                    # (B, T)
        lmask = loss_mask.to(dtype)[..., None]           # (B, T, 1) — NOT pre-shifted

        # 3. Attention mask
        if attention_mask is None:
            attention_mask = torch.ones((B, T), dtype=torch.bool, device=device)
        attn_mask = self._prepare_decoder_attention_mask(
            attention_mask, (B, T), dtype, device, past_kv_len=0,
        )

        if position_ids is None:
            base_pos = torch.arange(0, T, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
            if self.midlayer.self_attn.use_mrope_3d:
                # Default to flat-3 for the 3D path (caller should override
                # with target's true 3D positions for the paper-baseline run).
                position_ids = base_pos.unsqueeze(0).expand(3, -1, -1).contiguous()
            else:
                position_ids = base_pos

        plosses, acces = [], []
        cache_hidden: List[List[torch.Tensor]] = [[], []]

        for step in range(self.length):
            inputs_embeds = embed_tokens(ids).to(dtype)
            hidden_out, cache_hidden = self.midlayer(
                input_emb=inputs_embeds,
                hidden_states=hidden_states,
                cache_hidden=cache_hidden,
                attention_mask=attn_mask,
                position_ids=position_ids,
            )

            # Soft-KL distillation. Most positions in our sequences are
            # prompt (loss_mask=0); only ~20/~3000 are output. Subset to
            # masked positions BEFORE projecting through the V-dim lm_head /
            # softmax — keeps memory at ~20×V per step instead of T×V.
            mask_bool = lmask.squeeze(-1).bool()      # (B, T)
            n_valid = int(mask_bool.sum().item())
            if n_valid > 0:
                hidden_masked = self.norm(hidden_out)[mask_bool]      # (N, H)
                logits_masked = lm_head(hidden_masked).float()        # (N, V)
                target_masked = target[mask_bool].float()             # (N, V)
                with torch.no_grad():
                    target_p = F.softmax(target_masked, dim=-1)
                log_p = F.log_softmax(logits_masked, dim=-1)
                ploss = -(target_p * log_p).sum(dim=-1).mean()
                with torch.no_grad():
                    acc = (logits_masked.argmax(-1) == target_masked.argmax(-1)).float().mean().item()
            else:
                # Degenerate batch (no output positions). Keep loss on graph
                # but zero-valued so DDP doesn't deadlock.
                ploss = (self.fc.weight.float().sum() * 0.0)
                acc = 0.0
            plosses.append(ploss)
            acces.append(acc)

            # Roll forward: draft's hidden becomes next step's hidden_states.
            hidden_states = hidden_out
            if step != self.length - 1:
                ids = _shift_left(ids)
                target = _shift_left(target)
                lmask = _shift_left(lmask)

        return plosses, acces


def build_eagle3_draft_for_qwen3vl(
    target,
    target_layer_ids: Optional[List[int]] = None,
    rollout_length: int = 7,
    dtype: torch.dtype = torch.bfloat16,
    use_mrope_3d: bool = False,
) -> Eagle3DraftModel:
    """Build a paper-faithful EAGLE-3 draft from a Qwen3-VL Alpamayo target.

    Default `target_layer_ids = [1, 17, 32]` matches EAGLE-3's
    {idx==2, idx==len/2, idx==len-3} layer choice for a 36-layer text model.
    `use_mrope_3d=True` propagates target's `mrope_section` /
    `mrope_interleaved` so the draft's rotary mirrors target's M-RoPE 3D
    (the natural "baseline" framing per project_eagle3_1d_vs_3d_claim.md).
    Default `use_mrope_3d=False` is the 1D arange variant — our paper claim.
    """
    text_cfg = target.vlm.config.get_text_config()
    n_tgt = text_cfg.num_hidden_layers
    if target_layer_ids is None:
        target_layer_ids = [2 - 1, n_tgt // 2 - 1, n_tgt - 3 - 1]  # [1, 17, 32] for 36

    mrope_section = None
    mrope_interleaved = True
    if use_mrope_3d:
        rs = getattr(text_cfg, "rope_scaling", None) or {}
        mrope_section = rs.get("mrope_section", [24, 20, 20])
        mrope_interleaved = rs.get("mrope_interleaved", True)

    cfg = Eagle3DraftConfig(
        hidden_size=text_cfg.hidden_size,
        intermediate_size=text_cfg.intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=text_cfg.num_attention_heads,
        num_key_value_heads=text_cfg.num_key_value_heads,
        vocab_size=text_cfg.vocab_size,
        rms_norm_eps=text_cfg.rms_norm_eps,
        rope_theta=getattr(text_cfg, "rope_theta", 10000),
        max_position_embeddings=getattr(text_cfg, "max_position_embeddings", 8192),
        target_layer_ids=target_layer_ids,
        rollout_length=rollout_length,
        mrope_section=mrope_section,
        mrope_interleaved=mrope_interleaved,
    )
    model = Eagle3DraftModel(cfg).to(dtype=dtype)
    return model


def save_eagle3_ckpt(model: Eagle3DraftModel, path: str):
    """Save with metadata so eval can rebuild without manual config args."""
    torch.save({
        "kind": "eagle3",
        "state_dict": model.state_dict(),
        "target_layer_ids": list(model.target_layer_ids),
        "rollout_length": model.length,
        "config": {
            "hidden_size": model.config.hidden_size,
            "intermediate_size": model.config.intermediate_size,
            "num_attention_heads": model.config.num_attention_heads,
            "num_key_value_heads": model.config.num_key_value_heads,
            "vocab_size": model.config.vocab_size,
            "rms_norm_eps": model.config.rms_norm_eps,
            "rope_theta": model.config.rope_theta,
            "max_position_embeddings": model.config.max_position_embeddings,
            "mrope_section": model.config.mrope_section,
            "mrope_interleaved": model.config.mrope_interleaved,
        },
    }, path)


def load_eagle3_ckpt(path: str, map_location=None):
    return torch.load(path, weights_only=False, map_location=map_location)
