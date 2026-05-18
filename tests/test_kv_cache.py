"""KV cache compression tests."""
from __future__ import annotations

import hashlib

import pytest


def _md5_tensor(t) -> str:
    import torch
    raw = t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.md5(raw).hexdigest()


def test_kv_roundtrip_lossless():
    """compress_kv_entry → decompress_kv_entry returns bit-identical K, V."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    from bigsmall.codecs.kv_cache import compress_kv_entry, decompress_kv_entry

    torch.manual_seed(0)
    B, H, T, D = 1, 32, 64, 96
    k = torch.randn(B, H, T, D, dtype=torch.float32).to(torch.bfloat16)
    v = torch.randn(B, H, T, D, dtype=torch.float32).to(torch.bfloat16)

    blob = compress_kv_entry(k, v)
    k2, v2 = decompress_kv_entry(blob, device="cpu")
    assert _md5_tensor(k) == _md5_tensor(k2), "K mismatch after roundtrip"
    assert _md5_tensor(v) == _md5_tensor(v2), "V mismatch after roundtrip"


def test_kv_compression_ratio_below_75pct():
    """On synthetic Gaussian KV-shaped tensors the codec should hit <75% of raw."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    from bigsmall.codecs.kv_cache import compress_kv_entry

    torch.manual_seed(0)
    B, H, T, D = 1, 32, 512, 96
    k = (torch.randn(B, H, T, D) * 0.5).to(torch.bfloat16)
    v = (torch.randn(B, H, T, D) * 0.5).to(torch.bfloat16)
    raw = (k.element_size() * k.numel()) + (v.element_size() * v.numel())
    blob = compress_kv_entry(k, v)
    ratio = len(blob) / raw
    assert ratio < 0.75, f"KV compression ratio {ratio*100:.2f}% exceeds 75% gate"


def test_compressed_kv_cache_api():
    """CompressedKVCache set/get/memory_usage/compression_ratio."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    from bigsmall.kv_cache_manager import CompressedKVCache

    torch.manual_seed(1)
    cache = CompressedKVCache(device="cpu")
    assert len(cache) == 0
    assert cache.memory_usage() == 0
    assert cache.compression_ratio() is None

    k = torch.randn(1, 32, 64, 96).to(torch.bfloat16)
    v = torch.randn(1, 32, 64, 96).to(torch.bfloat16)
    cache.set(0, k, v)
    assert cache.has(0)
    assert len(cache) == 1
    k2, v2 = cache.get(0)
    assert _md5_tensor(k) == _md5_tensor(k2)
    assert _md5_tensor(v) == _md5_tensor(v2)

    cr = cache.compression_ratio()
    assert cr is not None and 0.0 < cr < 1.0
    assert cache.memory_usage() > 0


def test_compressed_kv_cache_multi_layer():
    """Multiple layers stored independently; get returns the right pair."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    from bigsmall.kv_cache_manager import CompressedKVCache

    cache = CompressedKVCache(device="cpu")
    layers = {}
    torch.manual_seed(2)
    for li in range(4):
        k = torch.randn(1, 32, 32, 96).to(torch.bfloat16)
        v = torch.randn(1, 32, 32, 96).to(torch.bfloat16)
        layers[li] = (k, v)
        cache.set(li, k, v)
    for li in range(4):
        k2, v2 = cache.get(li)
        k_orig, v_orig = layers[li]
        assert _md5_tensor(k_orig) == _md5_tensor(k2), f"K mismatch layer {li}"
        assert _md5_tensor(v_orig) == _md5_tensor(v2), f"V mismatch layer {li}"


def test_compressed_kv_cache_clear():
    """clear() resets the cache to empty state."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    from bigsmall.kv_cache_manager import CompressedKVCache

    cache = CompressedKVCache(device="cpu")
    k = torch.randn(1, 4, 8, 16).to(torch.bfloat16)
    v = torch.randn(1, 4, 8, 16).to(torch.bfloat16)
    cache.set(0, k, v)
    assert len(cache) == 1
    cache.clear()
    assert len(cache) == 0
    assert cache.memory_usage() == 0
    assert cache.compression_ratio() is None
