"""BigSmall command-line interface."""
import argparse
import multiprocessing
import os
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
    if i.get("codec_stats"):
        print("  codec_breakdown")
        for k, v in sorted(i["codec_stats"].items(), key=lambda x: -x[1]):
            print(f"    {k:20s} {v} tensors")
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


def _cmd_migrate(args):
    from . import migrate as _migrate
    result = _migrate.migrate(args.src, dry_run=args.dry_run, backup=args.backup)
    def _line(k, v):
        print(f"  {k:22s} {v}", flush=True)
    if result.get("skipped_reason"):
        print(f"migrate skipped: {result['skipped_reason']} ({args.src})", flush=True)
        return
    print(f"migrate{' (dry-run)' if args.dry_run else ''}: {args.src}", flush=True)
    _line("tensors_total",    result["tensors_total"])
    _line("tensors_migrated", result["tensors_migrated"])
    _line("bytes_before",     f"{result['bytes_before']:,}")
    _line("bytes_after",      f"{result['bytes_after']:,}")
    _line("blob_savings_pct", f"{result['savings_pct']:.3f}%")
    _line("format_version",   result["format_version"])
    if result["codec_changes"]:
        print("  codec_changes")
        for k, v in sorted(result["codec_changes"].items(), key=lambda x: -x[1]):
            print(f"    {k:32s} {v}")


def _cmd_pipeline_run(args):
    from .pipeline import Pipeline
    p = Pipeline(
        source=args.source,
        dst_dir=args.dst_dir,
        repo_id=args.repo_id,
        mode=args.mode,
        token=args.token,
        workers=args.workers,
        use_lfs_upload=args.lfs,
    )
    p.run(do_compress=args.compress, do_upload=args.upload)


def _cmd_pipeline_status(args):
    import json as _json
    from .pipeline import Pipeline, CHECKPOINT_FILENAME
    cp = Path(args.dst_dir) / CHECKPOINT_FILENAME
    if not cp.exists():
        print(f"No pipeline checkpoint at {cp}")
        sys.exit(1)
    print(cp.read_text(encoding="utf-8"))


def _resolve_hf_token(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    token_path = Path.home() / ".huggingface" / "token"
    if token_path.exists():
        try:
            return token_path.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None
    return None


def _scan_local_compressed_models(local_dirs):
    """Find directories under each local_dir that contain bigsmall.index.json."""
    found = []
    for d in local_dirs:
        root = Path(d)
        if not root.exists() or not root.is_dir():
            continue
        for idx in root.rglob("bigsmall.index.json"):
            model_dir = idx.parent
            shards = sorted(model_dir.glob("*.bs"))
            total_bytes = sum(s.stat().st_size for s in shards)
            found.append({
                "path": str(model_dir),
                "shards": len(shards),
                "shard_names": [s.name for s in shards],
                "total_bytes": total_bytes,
            })
    return found


def _diff_local_vs_remote(local_shard_names, remote_shard_sizes):
    """Return shard names present locally but missing or size-mismatched remotely."""
    missing = []
    for name in local_shard_names:
        if name not in remote_shard_sizes:
            missing.append({"name": name, "reason": "absent_remote"})
    return missing


def _estimate_upload_seconds(total_bytes, mb_per_sec=10.0):
    """Coarse ETA at a conservative HF upload throughput."""
    if total_bytes <= 0:
        return 0.0
    return total_bytes / (mb_per_sec * 1024 * 1024)


def _cmd_status(args):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub is required: pip install huggingface_hub", flush=True)
        sys.exit(1)

    token = _resolve_hf_token(args.token)
    api = HfApi(token=token)
    user = args.user

    try:
        models = list(api.list_models(author=user))
    except Exception as e:
        print(f"Failed to list models for {user!r}: {e}", flush=True)
        sys.exit(1)

    suffix = args.suffix
    matches = []
    for m in models:
        repo_id = getattr(m, "id", None) or getattr(m, "modelId", None)
        if not repo_id:
            continue
        if not repo_id.endswith(suffix):
            continue
        matches.append(repo_id)
    matches.sort()

    # Build remote info dictionary
    remote: dict[str, dict] = {}
    for repo_id in matches:
        shard_sizes: dict[str, int] = {}
        has_readme = False
        err = None
        try:
            info = api.repo_info(repo_id=repo_id, repo_type="model", files_metadata=True)
            siblings = getattr(info, "siblings", None) or []
            for s in siblings:
                name = getattr(s, "rfilename", None) or ""
                size = getattr(s, "size", None) or 0
                if name.endswith(".bs"):
                    shard_sizes[name] = int(size) if size else 0
                if name.lower() == "readme.md":
                    has_readme = True
        except Exception as e:
            err = str(e)
        remote[repo_id] = {
            "shards": shard_sizes,
            "readme": has_readme,
            "error": err,
        }

    # Scan local compressed dirs
    if args.local_dirs is None:
        default_local = os.environ.get("BIGSMALL_TMP", r"C:\tmp\bs_out")
        local_dirs = [default_local]
    else:
        local_dirs = list(args.local_dirs)
    local_models = _scan_local_compressed_models(local_dirs)

    if args.as_json:
        import json as _json
        report = {
            "user": user,
            "suffix": suffix,
            "remote": [
                {
                    "repo_id": rid,
                    "shards": len(info["shards"]),
                    "total_bytes": sum(info["shards"].values()),
                    "readme": info["readme"],
                    "error": info["error"],
                }
                for rid, info in sorted(remote.items())
            ],
            "local": local_models,
        }
        # Add a missing-shards diff: for each remote repo, what local model
        # has shards not yet on the Hub?
        diffs = []
        for lm in local_models:
            best_match = None
            for rid in remote:
                # Heuristic match: trailing path segment of lm['path']
                # matches the trailing path of rid (everything after '/')
                if Path(lm["path"]).name.replace("_", "-").lower() in rid.lower():
                    best_match = rid
                    break
            if best_match is None:
                continue
            missing = _diff_local_vs_remote(lm["shard_names"], remote[best_match]["shards"])
            if missing:
                diffs.append({
                    "repo_id": best_match,
                    "local_path": lm["path"],
                    "missing_shards": missing,
                    "eta_seconds": _estimate_upload_seconds(
                        sum(lm["total_bytes"] for _ in [None])  # whole-model proxy
                    ),
                })
        report["pending_uploads"] = diffs
        print(_json.dumps(report, indent=2))
        return

    # Text table mode
    if not matches and not local_models:
        print(f"No repos matching {user}/*{suffix} and no local compressed models.")
        return

    if matches:
        rows = []
        for repo_id in matches:
            info = remote[repo_id]
            if info["error"] is not None:
                rows.append((repo_id, "?", "?", "?", f"err: {info['error']}"))
                continue
            n = len(info["shards"])
            gb = sum(info["shards"].values()) / (1024 ** 3)
            rows.append((repo_id, str(n), f"{gb:.2f}",
                         "yes" if info["readme"] else "no", ""))
        name_w = max(len("repo"), max(len(r[0]) for r in rows))
        shard_w = max(len("shards"), max(len(r[1]) for r in rows))
        gb_w = max(len("GB"), max(len(r[2]) for r in rows))
        readme_w = max(len("readme"), max(len(r[3]) for r in rows))
        header = f"{'repo':<{name_w}}  {'shards':>{shard_w}}  {'GB':>{gb_w}}  {'readme':>{readme_w}}"
        print(header)
        print("-" * len(header))
        for repo_id, shards, gb, readme, err in rows:
            line = f"{repo_id:<{name_w}}  {shards:>{shard_w}}  {gb:>{gb_w}}  {readme:>{readme_w}}"
            if err:
                line += f"  {err}"
            print(line)

    if local_models:
        print("")
        print(f"Local compressed models ({len(local_models)} found in: {', '.join(local_dirs)})")
        print("-" * 70)
        for lm in local_models:
            gb = lm["total_bytes"] / (1024 ** 3)
            eta = _estimate_upload_seconds(lm["total_bytes"])
            print(f"  {lm['path']}")
            print(f"    {lm['shards']} shards  {gb:.2f} GB  upload ETA ~{eta/60:.1f} min")


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

    s = sub.add_parser("status", help="List BigSmall repos on HuggingFace")
    s.add_argument("--user", default="wpferrell", help="HF username (default: wpferrell)")
    s.add_argument("--suffix", default="-bigsmall",
                   help="Only list repos whose name ends with this suffix (default: -bigsmall)")
    s.add_argument("--token", default=None, help="HF token override")
    s.add_argument("--local-dirs", default=None, nargs="*",
                   help="Directories to scan for local compressed models (default: $BIGSMALL_TMP or C:/tmp/bs_out)")
    s.add_argument("--json", dest="as_json", action="store_true",
                   help="Emit a machine-readable JSON report instead of a table")
    s.set_defaults(func=_cmd_status)

    m = sub.add_parser("migrate", help="Re-encode a .bs file with current best codecs")
    m.add_argument("src", help="Path to the .bs file to migrate (mutated in place)")
    m.add_argument("--dry-run", action="store_true",
                   help="Compute savings, do not write any files")
    m.add_argument("--no-backup", dest="backup", action="store_false", default=True,
                   help="Skip writing <src>.bs.bak before overwriting (default: backup is on)")
    m.set_defaults(func=_cmd_migrate)

    p_pipe = sub.add_parser("pipeline", help="Resumable compress + upload pipeline")
    pp_sub = p_pipe.add_subparsers(dest="pipeline_cmd", required=True)
    p_run = pp_sub.add_parser("run", help="Run the pipeline (resumable)")
    p_run.add_argument("source", help="Local model directory OR HF repo id")
    p_run.add_argument("dst_dir", help="Output directory for compressed shards")
    p_run.add_argument("--repo-id", default=None, help="Target HF repo id; required for upload")
    p_run.add_argument("--no-upload", dest="upload", action="store_false", default=True)
    p_run.add_argument("--no-compress", dest="compress", action="store_false", default=True)
    p_run.add_argument("--mode", default="balanced", choices=("storage", "balanced", "inference"))
    p_run.add_argument("--lfs", action="store_true", help="Use upload_to_hub_lfs")
    p_run.add_argument("--token", default=None, help="HF token override")
    p_run.add_argument("--workers", type=int, default=None)
    p_run.set_defaults(func=_cmd_pipeline_run)

    p_st = pp_sub.add_parser("status", help="Show pipeline checkpoint for a dst_dir")
    p_st.add_argument("dst_dir")
    p_st.set_defaults(func=_cmd_pipeline_status)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    # Required for Windows spawn-based multiprocessing when the CLI is frozen
    # (PyInstaller / cx_Freeze). No-op for a standard `pip install` invocation.
    multiprocessing.freeze_support()
    main()
