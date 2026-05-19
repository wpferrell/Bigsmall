"""md5 verification: round-trip a .bs file or compare against a source safetensors."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import codec_registry, container
from .decoder import decompress


def verify(src: str | Path, source_safetensors: str | Path | None = None) -> bool:
    """Verify a .bs file.

    If source_safetensors is provided, compare each decoded tensor's md5 to the
    md5 of the source. Otherwise verify each tensor's decoded bytes match the
    md5 stored in the .bs header.
    """
    src = Path(src)
    header, _ = container.read_header(src)
    out = decompress(src)

    all_ok = True
    if source_safetensors is not None:
        from safetensors import safe_open
        import torch
        with safe_open(str(source_safetensors), framework="pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                src_md5 = hashlib.md5(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                arr = out.get(name)
                if arr is None:
                    print(f"  MISSING {name}")
                    all_ok = False
                    continue
                dec_md5 = hashlib.md5(arr.tobytes()).hexdigest()
                if src_md5 != dec_md5:
                    print(f"  MISMATCH {name} src={src_md5} dec={dec_md5}")
                    all_ok = False
        return all_ok

    # Header md5 verification
    for t_meta in header["tensors"]:
        arr = out[t_meta["name"]]
        dec_md5 = hashlib.md5(arr.tobytes()).hexdigest()
        if dec_md5 != t_meta["md5"]:
            print(f"  MISMATCH {t_meta['name']} hdr={t_meta['md5']} dec={dec_md5}")
            all_ok = False
    return all_ok


def verify_fast(src: str | Path, verbose: bool = False) -> tuple[bool, list[str]]:
    """Structural integrity check that DOES NOT decompress.

    Validates, in order:
      1. Container header parses (magic + version + JSON body).
      2. Every per-tensor blob offset + compressed_bytes falls within the
         file's data section.
      3. No two tensor blobs overlap (would indicate header corruption).
      4. Every codec name referenced by the header is registered in the
         current `codec_registry` (so the file is decodable).
      5. If the file lives next to a `bigsmall.index.json`, the index's
         shard list / tensor count is consistent with the container.

    Returns (ok, problems) — `problems` is a list of human-readable
    diagnostic strings. Empty list on success.

    Designed to run in seconds even on multi-GB shards. Use the full
    `verify()` path when you actually need md5-confirmed lossless decode.
    """
    problems: list[str] = []
    src = Path(src)
    if not src.exists():
        problems.append(f"file does not exist: {src}")
        return False, problems

    try:
        header, data_offset = container.read_header(src)
    except Exception as e:
        problems.append(f"header parse failed: {type(e).__name__}: {e}")
        return False, problems

    file_size = src.stat().st_size
    data_size = file_size - data_offset

    tensors = header.get("tensors") or []
    if not isinstance(tensors, list):
        problems.append("header['tensors'] is not a list")
        return False, problems

    # 2 + 3: blob offsets + overlap detection.
    intervals: list[tuple[int, int, str]] = []  # (start, end, name)
    for t in tensors:
        name = t.get("name", "?")
        off = t.get("offset")
        nbytes = t.get("compressed_bytes")
        if not isinstance(off, int) or off < 0:
            problems.append(f"{name}: invalid offset {off!r}")
            continue
        if not isinstance(nbytes, int) or nbytes < 0:
            problems.append(f"{name}: invalid compressed_bytes {nbytes!r}")
            continue
        if off + nbytes > data_size:
            problems.append(
                f"{name}: blob [{off}, {off + nbytes}) extends past data "
                f"section end ({data_size})"
            )
            continue
        intervals.append((off, off + nbytes, name))

    intervals.sort()
    # tied_ref blobs are zero-length placeholders that share offsets with
    # their master — accept zero-length entries; only flag real overlaps.
    for i in range(1, len(intervals)):
        prev_s, prev_e, prev_name = intervals[i - 1]
        cur_s, cur_e, cur_name = intervals[i]
        if cur_s < prev_e and cur_s != prev_s:
            problems.append(
                f"blob overlap: {prev_name} ends at {prev_e}, {cur_name} starts at {cur_s}"
            )

    # 4: every codec is registered (or one of the special inline codecs).
    known_inline = {"tied_ref", "raw", "special", "zstd_xor_delta"}
    for t in tensors:
        codec = t.get("codec")
        if codec is None:
            problems.append(f"{t.get('name','?')}: missing codec field")
            continue
        if codec in known_inline:
            continue
        if codec_registry.get_codec(codec) is None:
            # Format-suffixed codecs like fp32_se_ac, fp16_se_ac etc. are
            # dispatched directly in decoder._decode_blob, not via the
            # registry — recognise them by suffix.
            if codec.endswith("_se_ac") or codec in ("fp8_cat_ac", "fp4_cat_ac"):
                continue
            problems.append(f"{t.get('name','?')}: unknown codec {codec!r}")

    # 5: sibling bigsmall.index.json consistency (if present).
    index_path = src.parent / "bigsmall.index.json"
    if index_path.exists() and index_path.is_file():
        try:
            idx = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception as e:
            problems.append(f"bigsmall.index.json unreadable: {e}")
        else:
            shards = idx.get("shards") or {}
            # If the index lists this file, sanity-check tensor count.
            this_name = src.name
            if this_name in shards:
                ind_count = shards[this_name].get("tensor_count")
                if isinstance(ind_count, int) and ind_count != len(tensors):
                    problems.append(
                        f"index tensor_count mismatch: index says {ind_count}, "
                        f"header has {len(tensors)}"
                    )

    if verbose and not problems:
        print(f"  tensors:       {len(tensors)}")
        print(f"  data section:  {data_size} bytes")
        print(f"  file:          {file_size} bytes")
        print(f"  codecs:        {sorted({t.get('codec') for t in tensors})}")

    return (not problems), problems
