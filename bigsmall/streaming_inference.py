"""Streaming inference: decompress one transformer layer at a time on demand.

`BigSmallStreamingModel.from_pretrained(repo_id_or_path, device='cuda')` loads
a BigSmall-compressed model in a streaming fashion:

  - Non-layer weights (embeddings, final norm, lm_head) are decompressed
    upfront and pinned on `device`.
  - Each transformer layer's weights are loaded on demand from the
    StreamingLoader during forward, used, then freed (`torch.cuda.empty_cache`
    after the layer's forward call returns).

Peak VRAM is bounded by:
  non_layer_weights + activations + ONE layer's weights

This is a working architecture-agnostic wrapper — it uses the underlying
HuggingFace nn.Module structure (built with `init_empty_weights`) and
patches `forward` on each transformer block to perform the load/run/free
cycle. No per-architecture math is re-implemented here.

Performance caveat (v3.2.0): CPU AC decoding bottlenecks this path at
~17 MB/s on constriction or ~1-2 MB/s on the rANS-based bf16_parallel
codec. The Triton GPU path (`bigsmall.kernels`) accelerates SE-decode
~2x but mantissa-on-CPU still dominates. Streaming inference is currently
correct but slow; future kernel work (warp-cooperative decode,
mantissa-on-GPU) is on the V4+ roadmap.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Optional

from .streaming import StreamingLoader


def _get_param_by_dotted_name(module, dotted_name: str):
    """torch.nn.Module deep parameter lookup by dot-separated name."""
    obj = module
    parts = dotted_name.split(".")
    for p in parts[:-1]:
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    leaf = parts[-1]
    return getattr(obj, leaf)


def _set_param_data(module, dotted_name: str, tensor) -> None:
    """Replace the data of an nn.Parameter (or buffer) in-place by name.

    Works whether the parameter currently lives on the 'meta' device (no
    storage) or a real device — in both cases we swap the underlying tensor
    rather than copying into it.
    """
    obj = module
    parts = dotted_name.split(".")
    for p in parts[:-1]:
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    leaf = parts[-1]
    import torch
    existing = getattr(obj, leaf)
    if isinstance(existing, torch.nn.Parameter):
        setattr(obj, leaf, torch.nn.Parameter(tensor, requires_grad=False))
    else:
        # Buffer
        setattr(obj, leaf, tensor)


def _layer_prefix(layer_idx: int, layer_template: str = "model.layers.{}") -> str:
    return layer_template.format(layer_idx) + "."


class BigSmallStreamingModel:
    """Inference wrapper that decompresses one transformer layer at a time.

    Usage:
        model = BigSmallStreamingModel.from_pretrained("path/to/bs_model",
                                                       hf_config_path=hf_id_or_path,
                                                       device="cuda")
        out = model.generate(input_ids, max_new_tokens=20)

    Args:
        bs_path: directory with bigsmall.index.json + *.bs shards, OR a single .bs file.
        hf_config_path: path / repo id with the matching HF model config
                        (we need this to construct the empty nn.Module shape).
        device: device for all tensors during forward.
        dtype: torch dtype for materialised tensors. Default torch.bfloat16.
        layer_template: tensor-name template for transformer layers.
                        Phi3 / Llama use "model.layers.{}". GPT-2 uses "h.{}".
    """

    def __init__(self,
                 bs_path: str | Path,
                 hf_config_path: str | Path,
                 device: str = "cuda",
                 dtype=None,
                 layer_template: str = "model.layers.{}",
                 prefetch: int = 0):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
        from accelerate import init_empty_weights

        self.device = device
        self.dtype = dtype if dtype is not None else torch.bfloat16
        self.layer_template = layer_template

        self.config = AutoConfig.from_pretrained(str(hf_config_path))

        # Build empty model structure — no memory allocated yet.
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(self.config)
        self.model.eval()

        self.loader = StreamingLoader(bs_path, device=device, dtype=self.dtype)
        self.prefetch = max(0, int(prefetch))
        self._prefetch_state = None  # populated lazily on first forward

        # Materialise non-layer weights upfront.
        non_layer_tensors = self.loader.load_non_layer_tensors()
        for name, tensor in non_layer_tensors.items():
            _set_param_data(self.model, name, tensor.to(self.device).to(self.dtype))

        self.n_layers = self.loader.layer_count()

        # Pre-allocate placeholders for layer params so HF can call .to(device)
        # or .generate() without tripping over meta tensors. The streaming
        # forward swaps the real tensor in just before forward, swaps back
        # to an empty placeholder afterwards.
        import torch
        for i in range(self.n_layers):
            prefix = _layer_prefix(i, self.layer_template)
            layer_module = _get_dotted_module(self.model, prefix.rstrip("."))
            for tname, p in list(layer_module.named_parameters(recurse=True)):
                placeholder = torch.empty(0, dtype=self.dtype, device=self.device)
                _set_param_data(layer_module, tname, placeholder)
            for tname, b in list(layer_module.named_buffers(recurse=True)):
                # Materialise empty buffer on device — HF buffers in Phi3 are scalars.
                if b.device.type == "meta":
                    new_b = torch.zeros_like(b, device=self.device)
                    _set_param_data(layer_module, tname, new_b)

        # Patch each transformer layer's forward.
        for i in range(self.n_layers):
            self._patch_layer(i)

    # ------------------------------------------------------------------------

    def _patch_layer(self, layer_idx: int) -> None:
        """Wrap layer.forward so it loads weights on demand and frees afterwards."""
        prefix = _layer_prefix(layer_idx, self.layer_template)
        layer_module = _get_dotted_module(self.model, prefix.rstrip("."))
        original_forward = layer_module.forward
        loader = self.loader
        dtype = self.dtype
        device = self.device
        n_layers = self.n_layers
        outer = self

        def streaming_forward(*args, **kwargs):
            import torch
            # Load this layer's tensors and patch into the nn.Module.
            # If prefetch is enabled, take the layer from the prefetch
            # queue instead of decoding inline.
            tensors = outer._consume_prefetched_layer(layer_idx)
            if tensors is None:
                tensors = loader.load_layer(layer_idx)
            for full_name, t in tensors.items():
                if not full_name.startswith(prefix):
                    continue
                local_name = full_name[len(prefix):]
                if t.device.type != "cuda" and "cuda" in device:
                    t = t.to(device)
                if t.is_floating_point() and t.dtype != dtype:
                    t = t.to(dtype)
                _set_param_data(layer_module, local_name, t)

            out = original_forward(*args, **kwargs)

            # Reset to meta so VRAM doesn't accumulate.
            # We swap parameters to small placeholder tensors (1-element)
            # rather than meta so the module stays usable for subsequent
            # batches without re-allocating placeholders.
            for full_name in tensors:
                if not full_name.startswith(prefix):
                    continue
                local_name = full_name[len(prefix):]
                placeholder = torch.empty(0, dtype=dtype, device=device)
                _set_param_data(layer_module, local_name, placeholder)
            del tensors
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            return out

        layer_module.forward = streaming_forward

    # ------------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls,
                        bs_path: str | Path,
                        hf_config_path: Optional[str | Path] = None,
                        device: str = "cuda",
                        dtype=None,
                        layer_template: str = "model.layers.{}") -> "BigSmallStreamingModel":
        """Construct a streaming model. `hf_config_path` defaults to bs_path
        if a `config.json` lives next to the .bs shards (typical HF model layout)."""
        bs_path = Path(bs_path)
        if not bs_path.exists():
            raise FileNotFoundError(
                f"BigSmallStreamingModel.from_pretrained: path {bs_path!r} "
                f"does not exist. Pass a path to a .bs file, or to a directory "
                f"containing bigsmall.index.json + *.bs shards. "
                f"To compress a HuggingFace model, use "
                f"`bigsmall.compress_from_hub(repo_id, output_dir)`."
            )
        if hf_config_path is None:
            cand = bs_path if bs_path.is_dir() else bs_path.parent
            if (cand / "config.json").exists():
                hf_config_path = cand
            else:
                raise FileNotFoundError(
                    f"BigSmallStreamingModel.from_pretrained: no config.json "
                    f"found next to bs_path ({cand!r}). "
                    f"Pass hf_config_path=<path-or-repo-id> explicitly, or "
                    f"place the HF config alongside the .bs files. "
                    f"Tip: `huggingface-cli download <repo-id> --include 'config.json' "
                    f"--local-dir {cand}` will fetch just the config."
                )
        return cls(bs_path, hf_config_path, device=device, dtype=dtype,
                   layer_template=layer_template)

    # ------------------------------------------------------------------------

    def forward(self, *args, **kwargs):
        import torch
        with torch.no_grad():
            return self.model(*args, **kwargs)

    def generate(self, input_ids, max_new_tokens: int = 20, **kwargs):
        """Greedy decode wrapper. Falls back to HF .generate() under the hood."""
        import torch
        kwargs.setdefault("do_sample", False)
        kwargs.setdefault("max_new_tokens", max_new_tokens)
        with torch.no_grad():
            return self.model.generate(input_ids.to(self.device), **kwargs)

    def close(self) -> None:
        self._stop_prefetch()
        self.loader.close()

    # ---- Async prefetch -----------------------------------------------------
    #
    # Optional prefetch worker thread that decompresses layer N+1..N+prefetch
    # ahead of the GPU forward on layer N. Useful for first-pass amortisation
    # of layer decompression cost when CPU decode time > GPU forward time.
    #
    # PRACTICAL NOTE: with the v3.6.0 bf16_se_single_kernel codec, per-layer
    # decompress is ~3 s on Phi-3.5-mini while a GPU forward pass for the
    # whole prompt is ~85 ms total. The CPU decompression is the bottleneck
    # and prefetching cannot speed up the worst-case (decoder is the
    # critical path). Prefetch DOES help when:
    #   - The first token's forward latency overlaps with later layer
    #     decompressions (amortises startup).
    #   - The user wants to start GPU compute as soon as layer 0 lands.
    # It does NOT magically multiply throughput.

    def _ensure_prefetch_started(self) -> None:
        """Spin up the prefetch worker thread on first forward call."""
        if self.prefetch <= 0:
            return
        if self._prefetch_state is not None:
            return
        import queue
        import threading
        state = {
            "queue": queue.Queue(maxsize=self.prefetch),
            "next_idx": 0,
            "stop": threading.Event(),
            "thread": None,
        }

        def _worker():
            li = 0
            while not state["stop"].is_set() and li < self.n_layers:
                try:
                    tensors = self.loader.load_layer(li)
                except Exception:
                    # Surface as None — the main thread falls back to inline
                    # decode if a prefetched layer is missing.
                    tensors = None
                # Wait until the consumer has room (bounded queue is the
                # backpressure mechanism).
                while not state["stop"].is_set():
                    try:
                        state["queue"].put((li, tensors), timeout=0.5)
                        break
                    except queue.Full:
                        continue
                li += 1

        state["thread"] = threading.Thread(target=_worker, name="bigsmall-prefetch",
                                           daemon=True)
        state["thread"].start()
        self._prefetch_state = state

    def _consume_prefetched_layer(self, layer_idx: int):
        """Return the prefetched tensors dict for layer_idx, or None if
        prefetch is disabled / not aligned with the requested index."""
        if self.prefetch <= 0:
            return None
        self._ensure_prefetch_started()
        state = self._prefetch_state
        if state is None:
            return None
        q = state["queue"]
        # The worker emits layers in order 0, 1, 2, ... For a standard HF
        # forward pass that's exactly the access pattern. If the model
        # ever accesses out of order we fall back gracefully.
        try:
            got_idx, got_tensors = q.get(timeout=30.0)
        except Exception:
            return None
        if got_idx != layer_idx or got_tensors is None:
            # Mismatch: don't try to recover (rare; would need re-seek).
            # Returning None makes the caller fall back to inline decode.
            return None
        return got_tensors

    def _stop_prefetch(self) -> None:
        state = self._prefetch_state
        if state is None:
            return
        state["stop"].set()
        # Drain the queue so the worker isn't blocked on put().
        import queue as _q
        try:
            while True:
                state["queue"].get_nowait()
        except _q.Empty:
            pass
        state["thread"].join(timeout=2.0)
        self._prefetch_state = None


def _get_dotted_module(model, dotted_name: str):
    """Walk dotted_name into nested modules / module-lists."""
    obj = model
    for p in dotted_name.split("."):
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    return obj
