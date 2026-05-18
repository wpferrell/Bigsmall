"""B5: status CLI extension tests.

We test the local-scan + helper functions without hitting the HF API. The
remote-listing path is exercised in an integration scenario that requires
real network access, so we keep the unit tests focused on local logic.
"""
import json
import sys
import tempfile
from pathlib import Path
from io import StringIO

import pytest


def _make_local_compressed_model(work: Path, name: str = "synthetic-bigsmall") -> Path:
    """Build a 1-shard local compressed BigSmall model directory."""
    import torch
    from safetensors.torch import save_file
    import bigsmall

    src = work / f"{name}-src"
    src.mkdir()
    save_file({"w": torch.randn(64, 64, dtype=torch.bfloat16)},
              str(src / "model.safetensors"))
    (src / "config.json").write_text(json.dumps({"model_type": "synthetic"}), encoding="utf-8")

    out = work / name
    bigsmall.compress_for_hub(str(src), output_dir=out, overwrite=True, workers=1)
    return out


def test_scan_local_compressed_models():
    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    from bigsmall.cli import _scan_local_compressed_models

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _make_local_compressed_model(root, "alpha-bigsmall")
        _make_local_compressed_model(root, "beta-bigsmall")
        found = _scan_local_compressed_models([str(root)])

    paths = {f["path"] for f in found}
    assert any(p.endswith("alpha-bigsmall") for p in paths)
    assert any(p.endswith("beta-bigsmall") for p in paths)
    for f in found:
        assert f["shards"] >= 1
        assert f["total_bytes"] > 0


def test_diff_local_vs_remote():
    from bigsmall.cli import _diff_local_vs_remote
    local = ["model-00001-of-00002.bs", "model-00002-of-00002.bs"]
    remote = {"model-00001-of-00002.bs": 12345}
    missing = _diff_local_vs_remote(local, remote)
    assert len(missing) == 1
    assert missing[0]["name"] == "model-00002-of-00002.bs"
    assert missing[0]["reason"] == "absent_remote"


def test_estimate_upload_seconds():
    from bigsmall.cli import _estimate_upload_seconds
    # 100 MB at 10 MB/s = ~10 s
    s = _estimate_upload_seconds(100 * 1024 * 1024, mb_per_sec=10.0)
    assert 9.0 < s < 11.0
    assert _estimate_upload_seconds(0) == 0.0


def test_status_json_argparse_round_trip(monkeypatch, capsys, tmp_path):
    """`bigsmall status --json --local-dirs <td>` returns parseable JSON."""
    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
        import huggingface_hub  # noqa: F401
    except ImportError:
        pytest.skip("dependencies not installed")

    # Build the local compressed model BEFORE capsys begins capturing the
    # status output. compress_for_hub prints progress lines that would
    # otherwise pollute the captured JSON.
    _make_local_compressed_model(tmp_path, "gamma-bigsmall")

    # Stub HfApi so the test doesn't hit the network.
    from bigsmall import cli as cli_mod

    class _FakeApi:
        def __init__(self, *a, **kw): pass
        def list_models(self, *a, **kw): return []
        def repo_info(self, *a, **kw):  # pragma: no cover (not hit)
            class _R:
                siblings = []
            return _R()

    import huggingface_hub as hf
    monkeypatch.setattr(hf, "HfApi", _FakeApi)

    # Clear anything captured during model build, then run the CLI.
    capsys.readouterr()
    argv = ["bigsmall", "status", "--user", "wpferrell",
            "--local-dirs", str(tmp_path), "--json"]
    monkeypatch.setattr(sys, "argv", argv)
    cli_mod.main(argv[1:])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["user"] == "wpferrell"
    assert payload["remote"] == []
    assert any(p["path"].endswith("gamma-bigsmall") for p in payload["local"])
