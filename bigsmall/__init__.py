"""BigSmall - Lossless neural network weight compression.

Public API:
    bigsmall.compress(src, dst, mode="balanced")          - compress safetensors -> .bs
    bigsmall.decompress(src, dst=None) -> dict[str, ndarray]
    bigsmall.load(src, device="cpu") -> dict[str, torch.Tensor]
    bigsmall.info(src) -> dict
    bigsmall.verify(src) -> bool
    bigsmall.compress_delta(finetune, base, dst, mode="balanced")
    bigsmall.decompress_delta(delta_src, base_src, dst=None) -> dict[str, ndarray]

HuggingFace Hub round-trip (Phase 4):
    bigsmall.compress_for_hub(source, output_dir)   - compress any HF model
    bigsmall.upload_to_hub(output_dir, repo_id)     - push to the Hub
    bigsmall.from_pretrained(repo_or_path)          - download + decompress -> state_dict
    bigsmall.install_hook()                         - monkey-patch safetensors.load_file

Streaming loader (Phase 4 cont.):
    bigsmall.StreamingLoader(path, device="cuda")   - layer-by-layer decompression
"""
__version__ = "3.3.0"

from . import _version_check
_version_check.check_version_async(__version__)

from .encoder import compress, compress_delta
from .decoder import decompress, decompress_delta, load
from .verify import verify
from .container import info
from .exceptions import BigSmallVersionError
from .hub import compress_for_hub, upload_to_hub, upload_to_hub_lfs, from_pretrained
from .integrations.huggingface import install_hook, uninstall_hook
from .integrations.vllm import (
    decompress_to_temp as vllm_decompress,
    bigsmall_vllm_serve as vllm_serve,
)
from .streaming import StreamingLoader

__all__ = [
    "compress",
    "decompress",
    "load",
    "info",
    "verify",
    "compress_delta",
    "decompress_delta",
    "compress_for_hub",
    "upload_to_hub",
    "upload_to_hub_lfs",
    "from_pretrained",
    "install_hook",
    "uninstall_hook",
    "vllm_decompress",
    "vllm_serve",
    "StreamingLoader",
    "BigSmallVersionError",
    "__version__",
]
