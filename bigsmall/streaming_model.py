"""Proof-of-concept streaming inference wrapper for GPT-2.

Demonstrates running a model with `StreamingLoader`: non-layer tensors load
upfront, each transformer layer is decompressed, executed via a manual GPT-2
forward built from torch primitives, then freed before the next layer is
fetched. Output is bit-identical to a normal full load.

We deliberately avoid `transformers.models.gpt2.modeling_gpt2.GPT2Block` so the
streaming wrapper is robust to transformers version differences. The math is
GPT-2's standard architecture: pre-LN, multi-head causal self-attention via
HF's `Conv1D` layout (`weight` stored (in, out)), GELU MLP, residual sums.
"""
from __future__ import annotations

import gc
import math
from typing import Optional

import torch


def _gelu_new(x: torch.Tensor) -> torch.Tensor:
    """The 'gelu_new' approximation used by GPT-2 (Hendrycks-Gimpel)."""
    return 0.5 * x * (
        1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3)))
    )


class StreamingGPT2:
    """Run GPT-2 layer-by-layer using a BigSmall StreamingLoader.

    Non-layer weights (wte, wpe, ln_f) live in memory at all times. Layer
    weights are read from the loader, used by the manual forward, then freed.

    Output ids are byte-identical to `GPT2LMHeadModel.from_pretrained("gpt2")`
    given the same input and decoding settings.

    Args:
        loader: a `StreamingLoader` that has already been initialised.
        config: a `transformers.GPT2Config`. If None, loaded from "gpt2".
        device: target device for all activations and weights.
        dtype: optional torch dtype to cast everything to (default: keep fp32).
    """

    def __init__(self,
                 loader,
                 config=None,
                 device: str = "cpu",
                 dtype: Optional[torch.dtype] = None):
        from transformers import GPT2Config

        self.loader = loader
        self.device = device
        self.dtype = dtype
        self.config = config or GPT2Config.from_pretrained("gpt2")
        self.n_head = self.config.n_head
        self.n_embd = self.config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.ln_eps = self.config.layer_norm_epsilon

        # Materialise non-layer weights upfront. These are small.
        nl = loader.load_non_layer_tensors()
        self.wte_weight = self._cast(nl["wte.weight"])
        self.wpe_weight = self._cast(nl["wpe.weight"])
        self.ln_f_weight = self._cast(nl["ln_f.weight"])
        self.ln_f_bias = self._cast(nl["ln_f.bias"])
        del nl

    # ---------------------- helpers -----------------------------------------

    def _cast(self, t: torch.Tensor) -> torch.Tensor:
        if self.dtype is not None and t.is_floating_point():
            t = t.to(self.dtype)
        return t.to(self.device)

    @staticmethod
    def _layer_norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
                    eps: float) -> torch.Tensor:
        return torch.nn.functional.layer_norm(
            x, (x.shape[-1],), weight=w, bias=b, eps=eps,
        )

    def _block_forward(self, x: torch.Tensor,
                       L: dict[str, torch.Tensor],
                       layer_idx: int) -> torch.Tensor:
        """Run one GPT-2 transformer block on hidden state x using layer-idx
        tensors loaded from BigSmall."""
        # Tensor names follow the safetensors layout: 'h.<i>.<sub>.<weight>'.
        p = f"h.{layer_idx}."
        ln1_w = L[p + "ln_1.weight"]
        ln1_b = L[p + "ln_1.bias"]
        attn_qkv_w = L[p + "attn.c_attn.weight"]   # (n_embd, 3*n_embd)
        attn_qkv_b = L[p + "attn.c_attn.bias"]     # (3*n_embd,)
        attn_proj_w = L[p + "attn.c_proj.weight"]  # (n_embd, n_embd)
        attn_proj_b = L[p + "attn.c_proj.bias"]
        ln2_w = L[p + "ln_2.weight"]
        ln2_b = L[p + "ln_2.bias"]
        mlp_fc_w = L[p + "mlp.c_fc.weight"]        # (n_embd, 4*n_embd)
        mlp_fc_b = L[p + "mlp.c_fc.bias"]
        mlp_proj_w = L[p + "mlp.c_proj.weight"]    # (4*n_embd, n_embd)
        mlp_proj_b = L[p + "mlp.c_proj.bias"]

        # Attention
        h = self._layer_norm(x, ln1_w, ln1_b, self.ln_eps)
        # Conv1D layout: y = x @ weight + bias  (no transpose).
        qkv = h @ attn_qkv_w + attn_qkv_b                       # (B, T, 3*C)
        B, T, _ = qkv.shape
        q, k, v = qkv.split(self.n_embd, dim=2)
        # (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # Causal self-attention
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        if self.dtype is not None:
            att = att.to(self.dtype)
        ctx = att @ v                                            # (B, nH, T, dH)
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        attn_out = ctx @ attn_proj_w + attn_proj_b
        x = x + attn_out

        # MLP
        h = self._layer_norm(x, ln2_w, ln2_b, self.ln_eps)
        fc = h @ mlp_fc_w + mlp_fc_b
        fc = _gelu_new(fc)
        mlp_out = fc @ mlp_proj_w + mlp_proj_b
        x = x + mlp_out
        return x

    # ---------------------- forward + generate ------------------------------

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (B, T, vocab) for input_ids of shape (B, T)."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)
        B, T = input_ids.shape
        positions = torch.arange(T, device=self.device).unsqueeze(0).expand(B, T)

        x = self.wte_weight[input_ids] + self.wpe_weight[positions]
        if self.dtype is not None:
            x = x.to(self.dtype)

        n_layers = self.loader.layer_count()
        for i in range(n_layers):
            L = self.loader.load_layer(i)
            # Ensure tensors live on the right device with right dtype.
            if self.dtype is not None:
                L = {k: (v.to(self.dtype) if v.is_floating_point() else v)
                     for k, v in L.items()}
            L = {k: v.to(self.device) for k, v in L.items()}
            x = self._block_forward(x, L, i)
            del L
            gc.collect()
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

        x = self._layer_norm(x, self.ln_f_weight, self.ln_f_bias, self.ln_eps)
        logits = x @ self.wte_weight.t()
        return logits

    @torch.no_grad()
    def generate_greedy(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Greedy decode `max_new_tokens` continuation tokens."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        seq = input_ids.to(self.device)
        for _ in range(max_new_tokens):
            logits = self.forward(seq)
            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            seq = torch.cat([seq, next_tok], dim=1)
        return seq
