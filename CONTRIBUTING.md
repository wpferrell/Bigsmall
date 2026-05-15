# Contributing to BigSmall

Thanks for your interest in BigSmall. This document describes how to set up a
development environment, how the codebase is laid out, and the conventions we
follow when adding codecs, integrations, or fixing bugs.

## Development setup

BigSmall targets Python 3.9+ and PyTorch 2.0+.

```bash
git clone https://github.com/wpferrell/Bigsmall.git
cd Bigsmall

# Create a virtualenv (any of venv, conda, uv is fine)
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
.\.venv\Scripts\activate           # Windows

pip install -e ".[all]"
pip install pytest tqdm
```

The `[all]` extra pulls in the optional HuggingFace Hub, diffusion, and vLLM
integrations. The core compressor only needs `numpy`, `safetensors`, `zstandard`,
`constriction`, and `torch`.

## Running tests

```bash
pytest tests/ -v
```

The test suite covers:

- `test_roundtrip.py` - bit-identical compress/decompress for all five float formats.
- `test_delta.py` - fine-tune delta compression against a base model.
- `test_diffusion.py` - Stable Diffusion UNet (FP16) and VAE (FP32) round-trips.
- `test_inference.py` - decoded weights produce identical model output.
- `test_streaming.py` - StreamingLoader matches full-load output and peak RAM is bounded.
- `test_hf_integration.py` - HuggingFace Hub `compress_for_hub` / `from_pretrained`.

Some tests skip themselves automatically if the underlying model is not in
your HuggingFace cache (e.g. GPT-2). Use
`HF_HUB_DOWNLOAD_TIMEOUT=120 huggingface-cli download gpt2` to pre-cache them.

## Adding a new codec

Codecs live in `bigsmall/codecs/`. A codec is a Python module with two
top-level functions:

```python
def encode(raw_bytes: bytes) -> tuple[bytes, dict]:
    """Return (compressed_blob, extras_dict). `extras` is stored verbatim
    in the container header and passed back to decode()."""

def decode(blob: bytes, extras: dict, n_weights: int) -> bytes:
    """Return the original raw bytes for `n_weights` weights."""
```

To wire a new format codec into the encoder/decoder:

1. Add the module under `bigsmall/codecs/<name>.py`.
2. Register it in the `_FORMAT_CODECS` dict in both
   `bigsmall/encoder.py` and `bigsmall/decoder.py`.
3. Teach `bigsmall/formats.py::detect_format_from_dtype` how to map a
   safetensors dtype string to the new format name.
4. Add a row to the format-support table in `README.md`.
5. Add a round-trip test in `tests/test_roundtrip.py` that compresses a
   small tensor in the new format and asserts byte-identical reconstruction.

If your codec is purely "compress these bytes" (no per-element entropy model),
prefer reusing `bigsmall/codecs/generic.py::encode_zstd` rather than adding a
new module.

## Adding a new model integration

Integrations live in `bigsmall/integrations/`. Each integration is a single
file that adapts a third-party loader (transformers, diffusers, vllm, ...) to
read BigSmall containers.

Conventions:

- A `from_pretrained(bs_path, model_class=None, ...)` function that returns a
  loaded model object.
- An `install_hook()` function for transparent monkey-patching, with a
  matching `uninstall_hook()` that restores everything it touched.
- Surface the integration from `bigsmall/__init__.py` if it should be part of
  the public API.

The transformers integration in `bigsmall/integrations/huggingface.py` is the
canonical example. Note in particular how `install_hook()` records originals
in `_HOOK_STATE` so `uninstall_hook()` can put them back - never throw the
originals away.

## Pull-request guidelines

- Branch from `main`. Keep PRs focused - one feature or one bug fix per PR.
- Add or update tests for any behaviour change. PRs that touch a codec or
  the container format must add a round-trip test.
- Run `pytest tests/ -v` and make sure everything passes (or document why a
  skip is expected).
- Update `CHANGELOG.md` under the `[Unreleased]` section.
- Don't reformat unrelated files. Keep diffs small and reviewable.
- For larger changes, open an issue first to discuss the approach.

## Code style

- Format with [black](https://github.com/psf/black) at default settings.
- Type annotations are **not required**. Add them where they clarify intent,
  skip them where they just add noise. The existing codebase mixes both
  styles intentionally.
- Prefer explicit names over abbreviations: `tensor_index` not `ti`,
  `compressed_bytes` not `cb`.
- Imports: standard library, then third-party, then `bigsmall` (relative).
- Strings: double quotes everywhere unless escaping forces single quotes.
- Don't add comments that restate the code. Comments are for the *why* -
  invariants, surprising constraints, gotchas - not the *what*.
- Public functions get docstrings. Private helpers usually don't.

## Reporting bugs

Please open a GitHub issue with:

- BigSmall version (`python -c "import bigsmall; print(bigsmall.__version__)"`)
- Python and PyTorch versions
- A minimal reproduction (ideally a small `.safetensors` file or repo ID)
- The full traceback

If you suspect a lossless-correctness bug (decompressed weights don't match
the source), please attach the output of `bigsmall verify <file>.bs` against
your source file. That's the fastest signal we have.

## License

By contributing you agree that your contributions are licensed under
Apache 2.0, the same license as the rest of the project.
