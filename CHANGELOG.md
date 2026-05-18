## [3.0.0] - 2026-05-18

v3.0.0 closes the V3 per-tensor structural-codec research arc.  The
per-tensor arithmetic coder is now confirmed at the joint-entropy lower
bound on real LLM weight tensors; all five candidate structural codecs
investigated this cycle returned negative or zero-gain on real models.
The release ships the work that wires those investigations into a
production surface (per-tensor auto-selection, container format v2,
migrate tool) and stamps the milestone with a major version bump.

### Added
- **`bigsmall migrate` tool** (`bigsmall/migrate.py`, `bigsmall migrate`
  CLI).  Re-encodes an existing `.bs` file against the current
  `auto_select_codec` registry.  Decodes each tensor blob, offers it to
  every candidate codec registered for the tensor's dtype, and replaces
  the original blob only when a smaller alternative exists.  By
  construction the migrated file is never larger than the original; every
  tensor's decompressed bytes are unchanged (md5-verified).  Supports
  `--dry-run` (compute savings, write nothing) and `--no-backup`
  (skip the default `<file>.bs.bak`).  Delta containers
  (`model_type="delta"`) are out of scope and return early without
  modification.
- **CLI `migrate` subcommand** wired into `bigsmall.cli`.  Prints
  per-codec change counts and the blob savings percentage.

### Tests
- `tests/test_migrate.py` (5 tests): output is a valid `.bs` readable by
  the decoder, file never grows, every tensor decompresses to the same
  bytes, `--dry-run` writes nothing, `--no-backup` skips the `.bs.bak`.

### Milestone notes
This release is the public V3 cut.  Internally it sits on top of:

- Per-tensor codec auto-selection (2.5.0).
- A5 sparsity-aware BF16 codec + container format v2 framework (2.4.0).
- Version-check warning, `BigSmallVersionError`, model-card patcher
  (2.4.1).
- Cross-shard tied-weight dedup, raw codec for tiny tensors, resumable
  pipeline, Windows-aware multiprocessing (2.2.0).

The full per-tensor structural-codec research arc is documented in the
companion DONE files (`A2_DONE.md`, `A3_DONE.md`, `A4_DONE.md`,
`A5_DONE.md`, `B3_DONE.md`, `EMB_REORDER_DONE.md`) and was summarised in
the 2.5.0 release notes.  No further intra-tensor lossless gains are
available on real LLM weight tensors under the current coding model;
future work moves to cross-tensor / quantize-plus-residual research.

## [2.5.0] - 2026-05-18

### Added
- **`bigsmall/codec_registry.py`** — a per-tensor codec auto-selection
  layer.  Registers each built-in codec under a name (`bf16_se_ac`,
  `bf16_sparsity_v1`, `fp32_se_ac`, `fp16_se_ac`, `fp8_cat_ac`,
  `fp4_cat_ac`, `zstd`) and ships an ordered candidate list per dtype.
  `auto_select_codec(raw, fmt, dtype, ...)` tries every candidate, returns
  the smallest blob, never raises, and is deterministic (first candidate
  wins ties — so the historical default always wins when other candidates
  match its size).  New codecs plug in via `register_codec(name, enc, dec)`
  with no encoder changes.
- **Encoder rewired** (`bigsmall/encoder.py:_encode_worker`).  The generic
  float / raw path now hands off to `auto_select_codec` instead of the
  fixed `_FORMAT_CODECS[fmt]` dispatch.  Tied-weight and special-codec
  (lowcard, wpe_delta) handling stays where it was; the `RAW_TINY_THRESHOLD`
  tiny-tensor short-circuit still bypasses auto-select.  The previous
  `a5_hint` worker argument is gone — auto_select runs the sparsity check
  internally.
- **`codec_stats` header key** (`bigsmall/encoder.py` + `bigsmall/container.py`).
  Optional dict of `codec_name -> count` written to the `.bs` header so
  users can audit which codec was picked for how many tensors per model.
  Old readers ignore the key.  `container.info()` exposes the same map
  (falling back to a per-tensor re-tally for older files), and the
  `bigsmall info` CLI prints a `codec_breakdown` section.

### Tests
- `tests/test_b4_auto_select.py` (7 tests):
  - auto_select never produces a blob larger than the historical default
  - auto_select picks the smallest of `bf16` / `bf16_sparsity_v1` / `zstd`
    on a high-kurtosis tensor
  - auto_select skips A5 on well-behaved Gaussian BF16 tensors
  - `register_codec` extends the candidate list at runtime
  - auto_select never raises (zstd fallback on malformed input)
  - compress() stamps `codec_stats` that sum to `tensor_count`
  - `bigsmall info` CLI emits a `codec_breakdown` section

### Per-tensor codec confirmed at joint entropy floor
This release closes the V3 per-tensor structural-codec research arc.
Five candidate codecs were investigated against real models in dedicated
sessions; all five came back negative or zero-gain.  The per-tensor
arithmetic coder is at the joint-entropy lower bound on real LLM weight
tensors and no further loseless gains are available without crossing the
boundary into cross-tensor reordering, weight-value clustering, or lossy
quantisation.  Documented decisions:

- **A5 sparsity split** (shipped 2.4.0 as a research artefact;
  `research/a5_benchmark.md`): 0.000 pp impact.  Mask cost exactly
  cancels the per-population `H(e)` reduction (chain-rule identity).
- **A3 embedding row-XOR delta** (`A3_DONE.md`,
  `research/a3_entropy_check.md`): Step-0 gate rejected.  XOR of
  uncorrelated BF16 distributions approaches uniform; on Phi-3.5-mini
  H(delta) is 0.49 bits/el HIGHER than H(rows).
- **A2 shared probability tables** (`A2_DONE.md`,
  `research/a2_entropy_check.md`): Step-0 net -5.4 MB on Phi-3.5-mini.
  Per-tensor SE tables are already only ~47 KB total; KL penalty from
  pooling same-layer-type SE distributions dwarfs the table savings.
- **A4 QKV block dedup** (`A4_DONE.md`, `research/a4_qkv_check.md`):
  max pair cosine 0.17 (Phi-3.5-mini, fused qkv_proj) / 0.002
  (Qwen3-8B, GQA K vs V).  Q/K/V are nearly orthogonal in weight space —
  training differentiates them by design.
- **Embedding row reordering** (`EMB_REORDER_DONE.md`,
  `research/emb_reorder_check.md`): nearest-neighbour cosine averages
  0.29 on Phi-3.5-mini embed_tokens — far below the ~0.85 needed for
  XOR-delta to win.  L2-sort, PC1-sort, and greedy-NN-chain orderings
  all make the delta entropy ~0.47 bits/el WORSE than the rows.
- **B3 GPU compression** (`B3_DONE.md`,
  `research/b3_gpu_benchmark.md`): not recommended.  Bottleneck is I/O
  + byte-decomposition + container assembly (~70% of wall-clock); no
  GPU-accelerable phase exists.  No drop-in GPU AC library for Python.

### Tooling / scaffolding
- `bigsmall.codec_registry.CodecStats` — tiny per-run accumulator the
  encoder uses to populate the header.

## [2.4.1] - 2026-05-18

### Added
- **Background PyPI version check on import** (`bigsmall/_version_check.py`).
  Once per 24 h (cached under `~/.cache/bigsmall/version_check.json`) the
  package fetches the latest version from PyPI on a daemon thread and prints
  a one-line warning to stderr if an upgrade is available. Failures (DNS,
  timeout, parse errors) are silent. Disable with
  `BIGSMALL_DISABLE_VERSION_CHECK=1`.
- **`BigSmallVersionError`** (`bigsmall/exceptions.py`) — actionable
  upgrade-instruction error raised when a `.bs` file or codec requires a
  newer release. Replaces the legacy `ValueError("Unsupported BigSmall
  version")` and the generic `ValueError("Unknown codec")`. Embeds the
  exact `pip install --upgrade bigsmall` instruction in the message.
  Exported from `bigsmall`.

### Tests
- `tests/test_version_check.py` (6 tests): warning fires when newer
  available, silent when same/newer locally, silent on network failure,
  fresh cache skips PyPI, env-disable short-circuits, daemon thread.
- `tests/test_version_errors.py` (4 tests): top-level export,
  unsupported-version path, unknown-codec path, message format.

### Docs / model cards
- `README.md` pre-compressed-models table refreshed against ground truth
  from the HF API (20 currently-live `-bigsmall` repos). Removed
  `deepseek-v4-flash` row (not yet live). Added `phi-3.5-mini-instruct`.
  Highlighted Llama-3.2-1B's exceptional **60.4 %** ratio with a footnote
  explaining the codec-version independence.
- `README.md` hardware-guide section is now a proper markdown table
  covering 2 GB - 24 GB + CPU, including Phi-3.5 Mini, Gemma 3 12B / 27B
  and Qwen 2.5 32B. The previous ASCII-art block had become unreadable
  Unicode artefacts.
- `README.md` adds a version-compat callout under the install command so
  users on `< 2.4.0` know to upgrade before reading models compressed
  with the 2.4 codec family.
- Fixed broken single-backtick code-fences in the README (4 blocks).
- **Patched HF model cards on 19 stable `-bigsmall` repos** via
  `patch_hf_cards.py`. Targeted patches only — broken single-backtick
  code fences swapped for proper triple-backtick fences, the outdated
  GPU-VRAM hardware table replaced with the 6-row 2.4.0 table, and the
  same `bigsmall >= 2.4.0` version-compat callout inserted after the
  first `pip install` block. 5 repos targeted by the in-progress upload
  job (qwen3-8b, gemma-3-{4b,12b,27b}-it, qwen2.5-32b-instruct) were
  deliberately skipped to avoid racing with the uploader's commits.

## [2.4.0] - 2026-05-17

### Added
- **A5 sparsity-aware BF16 codec** (`bigsmall/codecs/bf16_sparsity.py`).
  Detects high-kurtosis BF16 tensors (kurtosis ≥ 2.0 OR near-zero fraction
  ≥ 0.05 %), splits them into a near-zero and an outlier population by
  threshold `T = mean(|w|) * 0.25`, codes each sub-population with the
  existing per-tensor joint AC coder, and codes the 0/1 selector mask with
  the same machinery on a two-symbol alphabet.
  - Lossless on every roundtrip tested (md5-verified).
  - Containers using the codec stamp as format v2.
- **Encoder dispatch + safety net** in `bigsmall/encoder.py:_encode_worker`.
  The dispatch runs both A5 and plain bf16 and keeps A5 only when it
  produced fewer bytes. This guarantees no regression even when A5 isn't
  beneficial for a given tensor.
- **`bigsmall.tensor_analysis.compute_sparsity_stats()`** — O(n) sampled
  kurtosis + near-zero scan that drives the encoder dispatcher's
  decision. Uses numpy (no scipy dependency added).

### Tests
- `tests/test_a5_sparsity.py` (6 tests): direct-codec roundtrip,
  full-pipeline roundtrip, normal-distribution non-firing, safety-net
  fall-through, v2 container stamping via codec module, and a guard on
  the size-equivalence finding.

### Research finding (also documented in `A5_DONE.md`)
- **A5 does not beat plain BF16 on real Qwen3 weights**, despite the
  CLAUDE.md spec's prediction of ≥ 0.3 pp improvement. Measured impact on
  Qwen3-8B MLP gate_proj / up_proj / down_proj layers: within 1 KB on
  100 MB tensors (0.0001 % drift). The plain BF16 codec is already at the
  joint-entropy floor (`H(s,e) + H(m|e)`); any partition determined by
  `(s,e,m)` satisfies `H(s,e,m) = H(mask) + H(s,e,m|mask)`, so the
  `H(s,e|mask)` reduction A5 exploits is exactly cancelled by the mask
  cost. The codec ships as a research artifact and a foundation for
  hybrid (cross-tensor + sparsity) experiments; the safety net ensures it
  has zero ratio impact today.
- Full benchmark in `research/a5_benchmark.md`.

### Default-writer behaviour
- Container default version stays at v1. v2 only stamps when a tensor
  actually uses a v2-only codec (currently A5, which the safety net
  almost never keeps), so the 2.4.0 default output remains byte-readable
  by every 2.0.x consumer.

## [2.3.0] - 2026-05-17

### Added
- **`.bs` container format v2 (reader-side only)**. The reader now accepts
  both v1 (legacy) and v2 stamped containers; v2 reserves room for upcoming
  codec extensions (shared probability tables, row-delta, sparsity, QKV
  block references) without forcing existing v1 files to be re-encoded.
  The default writer continues to emit v1 until a release ships an actual
  v2-only codec feature — so every model produced by 2.3.0 stays readable
  by any 2.0.x consumer pinned in the wild.
  - `bigsmall.formats.BS_FORMAT_VERSION` exported (= 1 in this release).
  - `BS_FORMAT_VERSION_V1`, `BS_FORMAT_VERSION_V2`,
    `BS_SUPPORTED_FORMAT_VERSIONS` constants added.
  - `container.write_container(format_version=...)` lets callers stamp v2
    explicitly for forward-compat tests.
  - `bigsmall.index.json:metadata.format_version` records the bundled
    container version for consumer diagnostics.
- **Codec-regression test** at `tests/test_codec_regression.py`.
  Recomputes the joint-entropy lower bound on a deterministic synthetic
  fixture (`torch.manual_seed(42)`) and asserts it matches the v2.2.0
  baseline JSON in `research/baselines/synthetic_v220_baseline.json` to
  0.05 pp. If a future codec change drifts the floor, CI fails. The
  Phi-3.5-mini summary lives at
  `research/baselines/phi35_v220_baseline.json` for human audit (compact:
  ~1 KB, no per-tensor list).
- **DFloat11 paper-claim verification** in
  `research/compare_dfloat11.py:verify_dfloat11_paper_claims()` and the
  matching CLI subcommand `python research/compare_dfloat11.py verify`.
  Cross-checks our measured theoretical bounds against the published
  DFloat11 numbers from arXiv:2504.11651. Findings recorded in
  `research/dfloat11_verification.md`: the paper's 63.7 % on Qwen-2.5-14B
  is 2.79 pp below the DFloat11 theoretical floor for that model family
  under their stated coding model, i.e. inconsistent — until we can
  download Qwen-2.5-14B-Instruct and measure directly, treat that number
  as unverified rather than a benchmark we have to beat.

### Backward compatibility
- v1 files written by 2.0.x / 2.1.x / 2.2.x remain readable by 2.3.0
  unchanged. **This guarantee holds for all future BigSmall releases.**
- v2 files are rejected by any pre-2.3.0 reader with a clear error
  (`"Unsupported BigSmall version"`). 2.3.0 itself does not yet emit v2
  files (default writer still produces v1).

### Deferred to dedicated sessions (priority order from sprint planning)
1. **A5** sparsity-aware codec for high-kurtosis MLP tensors — biggest
   real compression gain (Qwen3 early MLP layers drag headline ratio
   0.4–0.6 pp; A5 targets that distribution).
2. **A3** embedding row delta — needs explicit uint16 XOR-delta design
   (bf16 subtraction is lossy).
3. **B3** GPU-accelerated histogram — benchmark first; histogram is
   currently a small fraction of encode time so the win is unclear.
4. **A2** shared probability tables — after format v2 emits actual v2
   files.
5. **A4** QKV block dedup — lowest priority; unlikely to find duplicates
   on trained models.
6. **B4** per-model codec auto-selection — last, wires everything
   together once the underlying codecs exist.
7. **Pipeline auto-profile integration** — depends on B4.
8. **`bigsmall migrate` v1→v2 tool** — external action workflow,
   dedicated session.

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

