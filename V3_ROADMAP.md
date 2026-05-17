# BigSmall V3 Roadmap

Created May 17 2026 -- learned from mass model deployment (19 models shipped in one session)

---

## Context

V2.0.1 is live on PyPI. 19 pre-compressed models published to HuggingFace in May 2026.
This document captures everything learned from that deployment and maps it to V3 improvements.

---

## 1. Upload infrastructure (fix first)

### Problems found
- HuggingFace LFS two-phase commit drops TCP connections silently on files >2GB
- Phase 1 (byte transfer) works fine. Phase 2 (SHA256 finalization + git commit) hangs indefinitely
- Python HF library has no timeout on phase 2 -- freeze is silent, looks like progress bar stall
- upload_folder() and bigsmall.upload_to_hub() load entire file into RAM before upload
- Parallel shard uploads compound freeze probability -- always one at a time
- Current workaround: kill + retry loop -- HF caches bytes server-side so retries pick up where they left off

### V3 fixes
- Fix Windows git LFS credential passing -- eliminate freeze+retry loop entirely
- Add checkpointing -- save progress after each shard, detect already-uploaded shards on HF, skip them
- Build upload resume logic -- query HF first, diff against local, only upload missing
- Stream file directly to HF without loading into RAM -- use chunked HTTP upload
- Never run parallel shard uploads -- enforce single-file upload in the API

---

## 2. Codec improvements

### Problems found
- Llama 3.2-1B only compressed to 60.4% -- well below expected ~65%
- DFloat11 beat us on Qwen 2.5-14B: 63.7% vs our 66.1% -- their pure exponent coding wins on that weight distribution
- GPT-2 FP32 hits 75.5% -- FP32 has more entropy to exploit than BF16
- Compression ratios vary more than expected across architectures (60.4% to 70.9%)
- Compression is single-threaded on Windows (BIGSMALL_WORKERS=1 required) -- very slow on 70B+ models

### V3 fixes
- Investigate Llama 3.2-1B low ratio -- architecture or codec issue?
- Investigate DFloat11 winning on Qwen 14B -- per-model codec tuning opportunity
- Per-model codec selection -- auto-detect optimal coding strategy per architecture
- GPU-accelerated compression -- massive speedup for 70B+ models
- Fix Windows multiprocessing -- BIGSMALL_WORKERS=1 is a crutch, fix freeze_support properly
- Investigate mixed precision codec -- FP32 tensors in BF16 models could get better ratios

---

## 3. Pipeline improvements

### Problems found
- Download + compress + upload has no checkpointing -- crash on 70B model = start over completely
- HF cache accumulated 107GB before we noticed
- Test artifacts (.bs files) accumulated 10GB in tests/ folder -- not in .gitignore
- Auto-delete after upload is essential -- without it hundreds of GB accumulate
- No monitoring of what is uploaded vs local -- had to manually diff every session

### V3 fixes
- Add *.bs and *.safetensors to .gitignore immediately
- Build pipeline with checkpointing: download_state, compress_state, upload_state per model
- Auto-cleanup HF cache after each model upload
- Build bigsmall status CLI command -- shows all HF repos, which shards live, which missing
- Scheduled cleanup job -- clear tmp and cache weekly
- Upload manifest -- JSON tracking what has been compressed and uploaded with checksums

---

## 4. Discoverability and model cards

### Findings
- Models without cards get 0 downloads -- card is the product page, non-negotiable
- BigSmall vs DFloat11 distinction needs to be in first 2 lines
- Streaming loader is the #1 differentiator but was buried -- "run Mistral 7B on a 4GB GPU" is the hook
- Smaller models (GPT-2, 0.5B) get disproportionate bot/crawler downloads -- not meaningful signal
- Qwen 7B (36 downloads) and 14B (31 downloads) in 2 days without marketing = promising organic signal
- Two categories of lossless compression exist: storage/distribution (BigSmall, ZipNN) vs runtime (DFloat11, ZipServ)

### V3 improvements
- Auto-generate model cards from compression stats -- bigsmall upload --generate-card
- Add download tracking dashboard on Pi -- daily email with per-model download counts
- Build BigSmall model hub page on resonance-layer.com listing all pre-compressed models
- Add trending score tracking -- monitor which models are gaining traction
- Add verified lossless badge system -- automated md5 verification report in card

---

## 5. Competitive positioning

### Findings vs DFloat11
- Better compression ratio (65% vs 70% BF16) on most models
- DFloat11 wins on Qwen 14B (63.7% vs 66.1%) -- specific weight distribution advantage
- BigSmall: storage/distribution compression -- decompress at load, run native speed, any hardware
- DFloat11: runtime compression -- weights stay compressed in VRAM during inference, CUDA only, ~2x overhead at batch=1
- Different use cases, not direct competition -- messaging must reflect this clearly

### Findings vs ZipNN
- Better compression ratio (65% vs 67% BF16)
- BigSmall has streaming loader (peak RAM under 2GB for any model) -- ZipNN loads full model
- BigSmall has more pre-compressed models on HF (19 vs ~5)
- ZipNN is mainly a library, not model hub -- we own the distribution angle

### V3 opportunities
- First to market with pre-compressed 70B+ models -- nobody else has these losslessly compressed on HF
- Streaming loader is unique -- make it the headline feature everywhere
- Build benchmark page -- automated comparison vs DFloat11, ZipNN on same models
- Investigate per-model codec tuning to consistently beat DFloat11 on all models

---

## 6. New model targets (priority order)

### Already done (May 17 2026)
GPT-2, Mistral 7B v0.2+v0.3, Llama 3/3.1/3.2 (1B/3B/8B), Qwen 2.5 (0.5B/1.5B/3B/7B/14B), Qwen3-4B, Gemma 2 (2B/2B-it/9B-it), Gemma 3 (270M/270M-it/1B-it)

### Overnight queue (May 17-18 2026)
Qwen 2.5-32B, Llama 3.1-70B, Qwen 2.5-72B, DeepSeek V4-Flash (148GB)

### Next priority
- Qwen3-8B (hot, 1M+ downloads)
- Gemma 3-4B, 12B, 27B (need Gemma license, huge download counts)
- DeepSeek-R1 variants
- Phi-3.5 Mini (Microsoft, popular edge use case)
- Mistral-Small, Mistral-Large

---

## 7. Technical debt

- *.bs in .gitignore -- test artifacts accumulated 10GB, deleted May 17 2026
- Fix compress_for_hub() Windows multiprocessing -- BIGSMALL_WORKERS=1 is a workaround
- Add proper logging to upload pipeline -- currently only stdout in bat windows
- Build proper test fixtures -- small synthetic tensors, not real 7B model shards
- Version the .bs format explicitly -- future codec changes need backward compat
- Add format version header to .bs files

---

## Summary priority order for V3

1. Fix Windows multiprocessing (BIGSMALL_WORKERS=1 workaround)
2. Build resumable upload pipeline with checkpointing
3. Add bigsmall status CLI command
4. Investigate low-ratio models (Llama 3.2-1B at 60.4%)
5. GPU-accelerated compression for 70B+ models
6. Per-model codec auto-selection
7. Auto-generate model cards
8. Build model hub page on resonance-layer.com
9. Download tracking dashboard
10. Benchmark page vs DFloat11 and ZipNN
