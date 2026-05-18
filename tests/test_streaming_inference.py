"""Smoke tests for BigSmallStreamingModel.

The streaming inference wrapper is verified on Phi-3.5-mini in a separate
benchmark script (research/gpu_kernel/streaming_smoke2.py) — that path
takes ~10 minutes per 2-token generation and is unsuitable for the unit
test suite. These tests cover the wrapper's core helper functions and
import-level correctness.
"""
from __future__ import annotations

import pytest


def test_streaming_inference_imports():
    """The module must import without optional deps in the import path."""
    import bigsmall.streaming_inference  # noqa: F401


def test_set_param_data_swaps_parameter():
    """_set_param_data should replace an nn.Parameter cleanly."""
    try:
        import torch
        from torch import nn
    except ImportError:
        pytest.skip("torch not installed")
    from bigsmall.streaming_inference import _set_param_data

    m = nn.Linear(4, 4)
    new_w = torch.zeros(4, 4)
    _set_param_data(m, "weight", new_w)
    assert torch.equal(m.weight, torch.zeros(4, 4))
    assert isinstance(m.weight, nn.Parameter)


def test_set_param_data_nested_index():
    """_set_param_data should walk into ModuleLists by integer index."""
    try:
        import torch
        from torch import nn
    except ImportError:
        pytest.skip("torch not installed")
    from bigsmall.streaming_inference import _set_param_data

    m = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
    target_w = torch.eye(2)
    _set_param_data(m, "1.weight", target_w)
    assert torch.equal(m[1].weight, target_w)


def test_layer_prefix_format():
    """Template substitution must produce expected dotted prefixes."""
    from bigsmall.streaming_inference import _layer_prefix

    assert _layer_prefix(0) == "model.layers.0."
    assert _layer_prefix(31) == "model.layers.31."
    assert _layer_prefix(5, "h.{}") == "h.5."
