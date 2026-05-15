"""BigSmall command-line interface."""
import argparse
import sys
import time
from pathlib import Path


def _cmd_compress(args):
    from . import encoder
    src = Path(args.src)
    if args.output:
        dst = Path(args.output)
    else:
        dst = src.with_suffix(".bs")
    mode = "balanced"
    if args.storage:
        mode = "storage"
    elif args.inference:
        mode = "inference"

    t0 = time.perf_counter()
    if args.base:
        encoder.compress_delta(src, args.base, dst, mode=mode)
    else:
        encoder.compress(src, dst, mode=mode)
    elapsed = time.perf_counter() - t0
    src_size = src.stat().st_size
    dst_size = dst.stat().st_size
    pct = (dst_size / src_size * 100) if src_size > 0 else 0
    print(f"compressed {src} -> {dst}", flush=True)
    print(f"  source:     {src_size:,} bytes", flush=True)
    print(f"  compressed: {dst_size:,} bytes ({pct:.2f}%)", flush=True)
    print(f"  saved:      {src_size - dst_size:,} bytes", flush=True)
    print(f"  elapsed:    {elapsed:.1f}s", flush=True)


def _cmd_decompress(args):
    from . import decoder
    src = Path(args.src)
    if args.output:
        dst = Path(args.output)
    else:
        dst = src.with_suffix(".safetensors")

    t0 = time.perf_counter()
    if args.base:
        decoder.decompress_delta(src, args.base, dst)
    else:
        decoder.decompress(src, dst)
    elapsed = time.perf_counter() - t0
    print(f"decompressed {src} -> {dst}  ({elapsed:.1f}s)", flush=True)


def _fmt_bytes(n):
    if n is None:
        return "-"
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PiB"


def _cmd_info(args):
    from .container import info
    i = info(args.src)

    def line(k, v):
        print(f"  {k:26s} {v}")

    print(f"BigSmall container: {i['path']}")
    line("format", i["format"])
    line("mode", i["mode"])
    line("model_type", i["model_type"])
    line("base_model", i["base_model"])
    line("container_version", i["version"])
    line("tensor_count", i["tensor_count"])
    line("file_size", f"{i['file_size']:,} bytes ({_fmt_bytes(i['file_size'])})")
    line("estimated_raw_bytes", f"{i['estimated_raw_bytes']:,} bytes ({_fmt_bytes(i['estimated_raw_bytes'])})")
    line("overall ratio_pct", f"{i['ratio_pct']:.2f}%")
    line("layer_count", i["layer_count"])
    line("non_layer_raw_bytes", _fmt_bytes(i["non_layer_raw_bytes"]))
    line("largest_layer_raw_bytes", _fmt_bytes(i["largest_layer_raw_bytes"]))
    line("streaming_peak_ram_est", _fmt_bytes(i["streaming_peak_ram_bytes"]))

    if i["format_breakdown"]:
        print("  format_breakdown")
        for k, v in sorted(i["format_breakdown"].items(), key=lambda x: -x[1]):
            print(f"    {k:8s} {v} tensors")
    if i["special_counts"]:
        print("  special tensors")
        for k, v in sorted(i["special_counts"].items()):
            print(f"    {k:12s} {v}")

    if i["top5_best"]:
        print("  top 5 best-compressed tensors (lower ratio = better)")
        for pt in i["top5_best"]:
            print(f"    {pt['ratio_pct']:6.2f}%  {pt['name']}  "
                  f"({_fmt_bytes(pt['raw_bytes'])} -> {_fmt_bytes(pt['compressed_bytes'])})")
    if i["top5_worst"]:
        print("  top 5 worst-compressed tensors")
        for pt in i["top5_worst"]:
            print(f"    {pt['ratio_pct']:6.2f}%  {pt['name']}  "
                  f"({_fmt_bytes(pt['raw_bytes'])} -> {_fmt_bytes(pt['compressed_bytes'])})")


def _cmd_verify(args):
    from .verify import verify
    ok = verify(args.src, source_safetensors=args.source)
    if ok:
        print("OK", flush=True)
        sys.exit(0)
    print("FAIL", flush=True)
    sys.exit(1)


def _cmd_benchmark(args):
    from . import encoder, decoder
    src = Path(args.src)
    dst = src.with_suffix(".bs")
    print(f"Benchmarking {src.name}...")
    t0 = time.perf_counter(); encoder.compress(src, dst); te = time.perf_counter() - t0
    t0 = time.perf_counter(); _ = decoder.decompress(dst); td = time.perf_counter() - t0
    src_size = src.stat().st_size
    dst_size = dst.stat().st_size
    pct = (dst_size / src_size * 100) if src_size > 0 else 0
    print(f"  encode:     {te:.1f}s  ({src_size / te / 1024 / 1024:.1f} MiB/s)")
    print(f"  decode:     {td:.1f}s  ({src_size / td / 1024 / 1024:.1f} MiB/s)")
    print(f"  ratio:      {pct:.2f}% ({src_size:,} -> {dst_size:,})")


def main(argv=None):
    p = argparse.ArgumentParser(prog="bigsmall", description="BigSmall lossless NN weight compression")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compress", help="Compress a .safetensors file to .bs")
    c.add_argument("src")
    c.add_argument("-o", "--output", default=None)
    c.add_argument("--base", default=None, help="Base safetensors path - enables delta mode")
    grp = c.add_mutually_exclusive_group()
    grp.add_argument("--storage", action="store_true", help="Maximum compression mode")
    grp.add_argument("--balanced", action="store_true", help="Balanced ratio+speed (default)")
    grp.add_argument("--inference", action="store_true", help="Fastest decode mode")
    c.set_defaults(func=_cmd_compress)

    d = sub.add_parser("decompress", help="Decompress a .bs file to .safetensors")
    d.add_argument("src")
    d.add_argument("-o", "--output", default=None)
    d.add_argument("--base", default=None, help="Base file path for delta decompression")
    d.set_defaults(func=_cmd_decompress)

    i = sub.add_parser("info", help="Show metadata for a .bs file")
    i.add_argument("src")
    i.set_defaults(func=_cmd_info)

    v = sub.add_parser("verify", help="Verify md5 round-trip of a .bs file")
    v.add_argument("src")
    v.add_argument("--source", default=None, help="Compare against original .safetensors")
    v.set_defaults(func=_cmd_verify)

    b = sub.add_parser("benchmark", help="Encode/decode benchmark for a model")
    b.add_argument("src")
    b.set_defaults(func=_cmd_benchmark)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
