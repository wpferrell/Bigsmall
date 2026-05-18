"""BigSmall `migrate` — re-encode a `.bs` file with current best codecs.

Reads an existing `.bs` container, walks every tensor, decodes each blob to
raw bytes, and re-encodes it with `codec_registry.auto_select_codec` (the same
"smallest wins" dispatcher the encoder runs on a fresh compress).  If the new
blob is strictly smaller, it replaces the original; otherwise the original is
kept verbatim.  By construction the migrated file is therefore never larger.

Scope of re-encoding
--------------------
Only the "generic float / raw" path goes through auto-select:

  - `tied_ref`          : kept as-is (no payload to re-encode).
  - `raw`               : kept as-is (tiny-tensor short-circuit; already optimal).
  - `special`           : kept as-is (lowcard / wpe_delta pattern codecs).
  - `zstd_xor_delta`    : kept as-is (delta-mode containers are not migrated;
                          `migrate` returns early with model_type="delta").

Everything else (bf16_se_ac, fp32_se_ac, fp16_se_ac, fp8_cat_ac, fp4_cat_ac,
bf16_sparsity_v1, zstd, plus any future codec registered before migrate runs)
is decoded and offered to auto_select.

Format version on the migrated file
-----------------------------------
The output is stamped v2 whenever any tensor uses a v2-only codec (currently
just `bf16_sparsity_v1`); otherwise v1.  This matches `encoder.compress`.
"""
from __future__ import annotations

import io
import shutil
import struct
from pathlib import Path

from . import codec_registry, container, formats
from .decoder import _decode_blob
from .formats import BS_FORMAT_VERSION_V1, BS_FORMAT_VERSION_V2


# Codec names that do NOT go through auto-select — they're either payloadless
# or already at the best codec for their data pattern.
_KEEP_AS_IS_CODECS = frozenset({
    "tied_ref", "raw", "special", "zstd_xor_delta",
})


def _fmt_for_dtype(dtype_str: str) -> str:
    """Return BigSmall format key for a tensor dtype string.

    Falls back to "raw" for any non-float dtype so auto-select uses zstd.
    """
    try:
        return formats.detect_format_from_dtype(dtype_str)
    except ValueError:
        return "raw"


def migrate(src: str | Path, dry_run: bool = False, backup: bool = True) -> dict:
    """Migrate a `.bs` file to the current best codec selection.

    Args:
        src:     path to the `.bs` file to migrate (mutated in place unless
                 `dry_run`).
        dry_run: if True, compute the savings but write nothing.
        backup:  if True (default), write the original to `<src>.bs.bak`
                 before overwriting.  Ignored when `dry_run=True`.

    Returns:
        dict with keys::

            tensors_total       int  # total tensors walked
            tensors_migrated    int  # tensors whose codec changed
            bytes_before        int  # total compressed blob bytes before
            bytes_after         int  # total compressed blob bytes after
            savings_pct         float  # 100 * (1 - after/before) on blob bytes
            file_size_before    int  # full file size before
            file_size_after     int  # full file size after migration (or
                                     # estimated if dry_run)
            format_version      int  # output container version
            dry_run             bool # echoed back
            codec_changes       dict[str, int]
                # old_codec -> new_codec change counts as "old->new"

    Never raises on a structurally valid `.bs` file — bad input from
    `container.read_header` propagates unchanged (file not a `.bs`, version
    unsupported, etc.).
    """
    src = Path(src)
    header, data_offset = container.read_header(src)
    file_size_before = src.stat().st_size

    # Delta containers are out of scope for migrate (they encode XOR payloads,
    # not standalone tensor data) — bail early but still report stats.
    if header.get("model_type") == "delta":
        return {
            "tensors_total": header.get("tensor_count", 0),
            "tensors_migrated": 0,
            "bytes_before": file_size_before - data_offset,
            "bytes_after": file_size_before - data_offset,
            "savings_pct": 0.0,
            "file_size_before": file_size_before,
            "file_size_after": file_size_before,
            "format_version": header.get("_format_version", 1),
            "dry_run": dry_run,
            "codec_changes": {},
            "skipped_reason": "delta_container_not_migrated",
        }

    with open(src, "rb") as f:
        f.seek(data_offset)
        data = f.read()

    new_data_buf = io.BytesIO()
    new_tensors: list[dict] = []
    offset = 0
    bytes_before = 0
    bytes_after = 0
    tensors_migrated = 0
    codec_changes: dict[str, int] = {}
    used_v2 = False
    v2_codecs = {"bf16_sparsity_v1"}

    for t in header["tensors"]:
        blob = data[t["offset"]:t["offset"] + t["compressed_bytes"]]
        old_codec = t["codec"]
        new_blob = blob
        new_codec = old_codec
        new_extras = t.get("extra")

        if old_codec in _KEEP_AS_IS_CODECS:
            pass
        else:
            try:
                raw = _decode_blob(t, blob)
            except Exception:
                raw = None
            if raw is not None and len(raw) > 0:
                fmt = _fmt_for_dtype(t["dtype"])
                try:
                    cand_blob, cand_codec, cand_extras = (
                        codec_registry.auto_select_codec(
                            raw, fmt=fmt, dtype=t["dtype"],
                            tensor_name=t.get("name", ""),
                            shape=tuple(t.get("shape") or ()),
                        )
                    )
                except Exception:
                    cand_blob = None
                if cand_blob is not None and len(cand_blob) < len(blob):
                    new_blob = cand_blob
                    new_codec = cand_codec
                    new_extras = cand_extras or {}
                    tensors_migrated += 1
                    key = f"{old_codec}->{new_codec}"
                    codec_changes[key] = codec_changes.get(key, 0) + 1

        bytes_before += len(blob)
        bytes_after += len(new_blob)
        if new_codec in v2_codecs:
            used_v2 = True

        new_data_buf.write(new_blob)
        new_t = dict(t)  # shallow copy
        # Strip the synthetic `_format_version` key that read_header injects
        # on the header dict — we only want the actual tensor entry fields.
        new_t.pop("_format_version", None)
        new_t["codec"] = new_codec
        new_t["compressed_bytes"] = len(new_blob)
        new_t["offset"] = offset
        new_t["extra"] = new_extras or None
        new_tensors.append(new_t)
        offset += len(new_blob)

    savings = (bytes_before - bytes_after) / bytes_before * 100.0 if bytes_before else 0.0
    target_version = BS_FORMAT_VERSION_V2 if used_v2 else BS_FORMAT_VERSION_V1

    # Rebuild the codec_stats header key from the new tensor list.
    stats: dict[str, int] = {}
    for nt in new_tensors:
        stats[nt["codec"]] = stats.get(nt["codec"], 0) + 1

    out_header = {k: v for k, v in header.items()
                  if not k.startswith("_") and k != "tensors" and k != "codec_stats"}
    out_header["tensors"] = new_tensors
    out_header["tensor_count"] = len(new_tensors)
    out_header["codec_stats"] = stats

    file_size_after = file_size_before
    if not dry_run:
        if backup:
            shutil.copy2(src, src.with_suffix(src.suffix + ".bak"))
        # Atomic-ish: write to sibling, then replace.
        tmp = src.with_suffix(src.suffix + ".tmp")
        container.write_container(tmp, out_header, new_data_buf.getvalue(),
                                  format_version=target_version)
        tmp.replace(src)
        file_size_after = src.stat().st_size

    return {
        "tensors_total": len(header["tensors"]),
        "tensors_migrated": tensors_migrated,
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
        "savings_pct": savings,
        "file_size_before": file_size_before,
        "file_size_after": file_size_after,
        "format_version": target_version,
        "dry_run": dry_run,
        "codec_changes": codec_changes,
    }
