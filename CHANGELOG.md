## [2.2.0] - 2026-05-17

### Added
- **Cross-shard tied-weight deduplication** (`compress_for_hub(dedupe_cross_shard=True)`).
  Scans every tensor across every shard for md5 duplicates, stores only the
  master copy, and records aliases in `bigsmall.index.json:metadata.duplicate_map`.
  `from_pretrained` and `StreamingLoader` materialise the duplicates
  transparently. Biggest win is on models where `lm_head.weight` is tied to
  `embed_tokens.weight` but split into different shards (frees ~120 MB on a
  Mistral-7B-class model).
- **Raw codec for tiny tensors** (`encoder.RAW_TINY_THRESHOLD = 512`).
  Tensors below 512 bytes skip the entropy coder entirely and are stored
  uncompressed under `codec="raw"`. Saves the per-tensor codec header
  overhead on bias / norm scales.
- **Resumable pipeline** (`bigsmall/pipeline.py`, `bigsmall pipeline run` CLI).
  JSON checkpoint at `<dst_dir>/.bigsmall_pipeline.json` tracks per-stage and
  per-shard completion so a crashed multi-GB upload restarts from the right
  place instead of from scratch.
- **`bigsmall status` extension** -- now also reports local compressed
  models, disk usage, ETA, and a missing-shard diff vs. the HF remote.
  Adds `--json` for machine-readable output and `--local-dirs` to override
  the scan paths.
- **Windows-aware default workers**. `_default_workers()` now returns 1 on
  Windows and `min(cpu_count, 8)` on Linux/macOS by default, with the
  `BIGSMALL_WORKERS` env override unchanged. `multiprocessing.freeze_support()`
  added to the CLI entry point so frozen builds work correctly.
- **`research/compare_dfloat11.py`** -- internal competitive-analysis tool.
  Computes both the BigSmall and the DFloat11 theoretical lower bounds per
  layer type and prints a comparison table.

### Tests
- `tests/test_cross_shard_dedup.py` (5 tests): roundtrip, index map, byte
  savings, StreamingLoader resolution.
- `tests/test_raw_tiny.py` (3 tests): tiny tensor codec selection, byte-
  identical roundtrip, threshold boundary behaviour.
- `tests/test_workers.py` (4 tests): parametrised over `workers=1,2`,
  platform-default expectations, env override.
- `tests/test_pipeline.py` (3 tests): compress + restart skip, no-upload
  path, JSON-serialisability of state.
- `tests/test_status_cli.py` (4 tests): local scan, missing-shard diff,
  ETA estimator, `--json` round-trip with stubbed HfApi.

### Not in this release (deferred to dedicated sessions)
- A2 cross-shard probability table sharing (.bs format change)
- A3 per-row delta for embedding tensors (.bs format change)
- A4 block dedup on fused QKV (.bs format change)
- A5 sparsity-aware codec for high-kurtosis MLP layers (.bs format change)
- B3 GPU-accelerated compression (multi-day port)
- B4 per-model codec auto-selection (cross-cutting refactor)

## [2.1.0] - 2026-05-17

### Added
- `bigsmall status` CLI command -- shows all HF repos with shard count, size, card status
- Upload resume logic -- skips already-uploaded shards automatically
- `upload_to_hub_lfs()` -- git LFS push method that bypasses Python API freeze on >2GB files
- Auto-cleanup option after successful upload (`cleanup=True`)

### Fixed
- Test fixtures now use synthetic tensors -- no real model files in repo
- `*.bs` and `V3_ROADMAP.md` added to .gitignore
# Changelog

## [2.0.1] - 2026-05-15
### Added
- vLLM integration: `bigsmall.vllm_serve()` and `bigsmall.vllm_decompress()` now accept HuggingFace repo IDs and multi-shard models

## [2.0.0] - 2026-05-15
### Added
- AutoModel transparent hook — `bigsmall.install_hook()` makes all `AutoModel.from_pretrained()` calls work with BigSmall-compressed repos
- Progress bars on compress, decompress, from_pretrained, StreamingLoader
- Enhanced `bigsmall info` — per-tensor ratios, streaming RAM estimate, format breakdown
- `paper.pdf` — technical paper
- CONTRIBUTING.md
- Pre-compressed models: Llama 3.1 8B and Qwen 2.5 14B on HuggingFace

## [1.0.1] - 2026-05-14
### Changed
- Updated PyPI description and classifiers

## [1.0.0] - 2026-05-14
### Added
- Initial release: compress, decompress, streaming loader, HF integration, delta compression
- Support for FP32, BF16, FP16, FP8, FP4
- CLI: compress, decompress, info, verify

