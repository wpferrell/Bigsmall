"""Codec regression test.

We bake a synthetic safetensors fixture's joint-entropy lower bound into
`research/baselines/synthetic_v220_baseline.json` and then assert on every CI
run that the freshly-computed bound on the same fixture matches the recorded
value to a tight tolerance. If a future change to `tensor_analysis._entropy_block_bf16`
or its callers drifts the lower bound (e.g. by changing the histogram alphabet
or the H(m|e) formula) this test catches it.

To intentionally accept a new baseline, re-run
`python research/baselines/build_baselines.py` and commit the updated JSON.
"""
import json
import tempfile
from pathlib import Path

import pytest


BASELINE_PATH = Path(__file__).resolve().parents[1] / \
    "research" / "baselines" / "synthetic_v220_baseline.json"

# Tolerance for the assertion. The synthetic fixture is deterministic
# (`torch.manual_seed(42)`) and the entropy computation is pure floating
# point, so a 0.05 pp tolerance is generous -- it absorbs any future
# refactor that doesn't actually change the math.
LB_TOLERANCE_PP = 0.05


def _build_synthetic_fixture(out_path: Path) -> Path:
    """Match `research/baselines/build_baselines.py:build_synthetic_baseline`."""
    import torch
    from safetensors.torch import save_file
    torch.manual_seed(42)
    tensors = {
        "model.embed_tokens.weight": torch.randn(1024, 256, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(256, 256, dtype=torch.bfloat16),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(256, 256, dtype=torch.bfloat16),
        "model.layers.0.mlp.up_proj.weight": torch.randn(512, 256, dtype=torch.bfloat16),
        "model.layers.0.mlp.down_proj.weight": torch.randn(256, 512, dtype=torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.randn(256, dtype=torch.bfloat16),
        "model.norm.weight": torch.randn(256, dtype=torch.bfloat16),
    }
    save_file(tensors, str(out_path))
    return out_path


def test_synthetic_entropy_lower_bound_unchanged():
    """Re-computing the synthetic-fixture lower bound must not drift below
    the v2.2.0 recorded baseline. A drift up is an improvement (we relax
    `LB_TOLERANCE_PP` in that direction so improvements don't break CI);
    a drift down beyond tolerance is a codec regression."""
    if not BASELINE_PATH.exists():
        pytest.skip(f"baseline not found at {BASELINE_PATH}; "
                    f"run `python research/baselines/build_baselines.py`")
    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")

    from bigsmall.tensor_analysis import deep_entropy_analysis

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    baseline_lb = baseline["summary"]["aggregate"]["bigsmall_lower_bound_pct"]
    baseline_df11 = baseline["summary"]["aggregate"]["dfloat11_lower_bound_pct"]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _build_synthetic_fixture(td / "model.safetensors")
        out = td / "_entropy.json"
        data = deep_entropy_analysis(str(td), str(out),
                                     measure_compressed=False, progress=False)
    current_lb = data["summary"]["aggregate"]["bigsmall_lower_bound_pct"]
    current_df11 = data["summary"]["aggregate"]["dfloat11_lower_bound_pct"]

    # Tighten guard: the synthetic fixture is fully deterministic so any drift
    # at all is suspicious. We allow exact match plus floating-point slack.
    assert abs(current_lb - baseline_lb) < LB_TOLERANCE_PP, (
        f"BigSmall lower bound drifted: baseline={baseline_lb:.4f} "
        f"now={current_lb:.4f}. If this is intentional, re-mint via "
        f"`python research/baselines/build_baselines.py`."
    )
    assert abs(current_df11 - baseline_df11) < LB_TOLERANCE_PP, (
        f"DFloat11 lower bound drifted: baseline={baseline_df11:.4f} "
        f"now={current_df11:.4f}."
    )


def test_phi35_baseline_summary_is_compact():
    """The Phi-3.5 baseline JSON is committed for audit; keep it small enough
    that the repo doesn't bloat. The full per-tensor analysis lives at
    `research/phi35_entropy.json` and is gitignored."""
    phi_path = BASELINE_PATH.parent / "phi35_v220_baseline.json"
    if not phi_path.exists():
        pytest.skip(f"Phi baseline not yet built at {phi_path}")
    size = phi_path.stat().st_size
    assert size < 5 * 1024, f"phi35 baseline grew to {size} bytes -- summarise instead"
    payload = json.loads(phi_path.read_text(encoding="utf-8"))
    assert payload["model"] == "microsoft/Phi-3.5-mini-instruct"
    assert "summary" in payload
    assert "aggregate" in payload["summary"]
