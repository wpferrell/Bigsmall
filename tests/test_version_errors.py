"""Tests for `BigSmallVersionError` -- the actionable upgrade-instruction
error raised by the decoder when it can't read a newer .bs file.
"""
import json
import struct
import tempfile
from pathlib import Path

import pytest


def test_version_error_importable_from_package():
    """`from bigsmall import BigSmallVersionError` must work."""
    import bigsmall
    assert hasattr(bigsmall, "BigSmallVersionError")
    assert issubclass(bigsmall.BigSmallVersionError, Exception)


def test_unsupported_container_version_raises_versionerror(tmp_path):
    """A .bs file stamped with an impossibly-future version must raise
    `BigSmallVersionError`, not the legacy `ValueError`."""
    from bigsmall import container, BigSmallVersionError

    path = tmp_path / "future.bs"
    header_bytes = b"{}"
    with open(path, "wb") as f:
        f.write(container.MAGIC)
        f.write(struct.pack("<H", 99))             # future version
        f.write(struct.pack("<I", len(header_bytes)))
        f.write(header_bytes)

    with pytest.raises(BigSmallVersionError) as excinfo:
        container.read_header(path)
    msg = str(excinfo.value)
    assert "pip install --upgrade bigsmall" in msg
    assert "requires bigsmall >= 2.4.0" in msg


def test_unknown_codec_in_decoder_raises_versionerror():
    """A header that references an unknown codec name (e.g. a v3-only codec
    in a v2 file) must raise `BigSmallVersionError` from the decoder
    dispatch, not the generic ValueError."""
    from bigsmall.decoder import _decode_blob
    from bigsmall import BigSmallVersionError

    tensor_meta = {
        "name": "x",
        "shape": [4],
        "dtype": "BF16",
        "codec": "future_codec_v9",
        "special": None,
        "compressed_bytes": 8,
        "offset": 0,
        "md5": "0" * 32,
        "extra": None,
    }
    with pytest.raises(BigSmallVersionError) as excinfo:
        _decode_blob(tensor_meta, b"\x00" * 8)
    assert "pip install --upgrade bigsmall" in str(excinfo.value)
    assert "future_codec_v9" in str(excinfo.value)


def test_versionerror_message_format():
    """Construct directly and check the message format."""
    from bigsmall import BigSmallVersionError
    err = BigSmallVersionError(required="3.0.0", installed="2.4.0",
                               detail="hypothetical")
    msg = str(err)
    assert "requires bigsmall >= 3.0.0" in msg
    assert "you have 2.4.0" in msg
    assert "pip install --upgrade bigsmall" in msg
    assert "hypothetical" in msg
