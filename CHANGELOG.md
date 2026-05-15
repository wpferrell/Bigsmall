# Changelog

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
