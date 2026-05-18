"""Pipeline resumability tests.

The Pipeline class wraps compress -> upload with a JSON checkpoint. We test:
  - First run compresses and writes a `done` stage for compress.
  - Restart re-uses the existing output; compress is skipped.
  - Upload is skipped when `repo_id is None`.
  - `summary()` returns a JSON-serialisable copy of state.
"""
import json
import tempfile
from pathlib import Path

import pytest


def _make_src(work: Path) -> Path:
    """Synthetic 1-shard HF dir."""
    import torch
    from safetensors.torch import save_file

    src = work / "src"
    src.mkdir()
    save_file({
        "embed.weight": torch.randn(1024, 128, dtype=torch.bfloat16),
        "layer.0.weight": torch.randn(256, 128, dtype=torch.bfloat16),
    }, str(src / "model.safetensors"))
    (src / "config.json").write_text(json.dumps({"model_type": "synthetic"}), encoding="utf-8")
    return src


def test_pipeline_compress_then_resume():
    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    from bigsmall.pipeline import Pipeline, CHECKPOINT_FILENAME

    work = Path(tempfile.mkdtemp(prefix="bigsmall_pipeline_"))
    try:
        src = _make_src(work)
        dst = work / "out"
        p = Pipeline(source=str(src), dst_dir=dst, repo_id=None, workers=1)
        p.run(do_compress=True, do_upload=False)

        # After compress: checkpoint exists, compress=done, no upload attempted.
        cp = dst / CHECKPOINT_FILENAME
        assert cp.exists()
        state = json.loads(cp.read_text(encoding="utf-8"))
        assert state["stages"]["compress"] == "done"
        assert state["stages"]["upload"] == "pending"
        assert state["stages"]["download"] == "done"

        # There's exactly one .bs shard with a recorded byte count.
        shards = list(state["shards"].values())
        assert len(shards) == 1
        assert shards[0]["compressed"] is True
        assert shards[0]["compressed_bytes"] > 0
        first_size = (dst / "model.bs").stat().st_size

        # Restart: the second Pipeline instance must not recompress.
        # We delete the .bs to assert "would have recompressed" if checkpoint
        # wasn't respected -- absence of the file after run() proves skip.
        # (We use mtime instead so we don't have to brittle-time the IO.)
        first_mtime = (dst / "model.bs").stat().st_mtime
        p2 = Pipeline(source=str(src), dst_dir=dst, repo_id=None, workers=1)
        p2.run(do_compress=True, do_upload=False)
        second_mtime = (dst / "model.bs").stat().st_mtime
        assert second_mtime == first_mtime, "shard was recompressed despite done checkpoint"
        assert (dst / "model.bs").stat().st_size == first_size
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)


def test_pipeline_no_repo_skips_upload():
    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError:
        pytest.skip("torch/safetensors not installed")
    from bigsmall.pipeline import Pipeline

    work = Path(tempfile.mkdtemp(prefix="bigsmall_pipeline_noup_"))
    try:
        src = _make_src(work)
        dst = work / "out"
        p = Pipeline(source=str(src), dst_dir=dst, repo_id=None, workers=1)
        p.run()  # full run; upload step is a no-op when repo_id is None
        s = p.summary()
        assert s["stages"]["compress"] == "done"
        assert s["stages"]["upload"] == "pending"
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)


def test_checkpoint_is_json_serialisable():
    from bigsmall.pipeline import Pipeline
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "out"
        p = Pipeline(source="ignored", dst_dir=dst, repo_id=None)
        # Round-trip the summary through json -- this guards against any
        # non-serialisable types (e.g. Path) sneaking into state.
        json.dumps(p.summary())
