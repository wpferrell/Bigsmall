"""Resumable BigSmall pipeline: download -> compress -> upload.

Crashing during a multi-day 70B upload is the most common operational pain
point for the BigSmall hub flow. This module wraps `compress_for_hub` and
`upload_to_hub` with a JSON checkpoint that records exactly which stages have
completed for each shard, so a restarted run skips work that was already done.

Checkpoint file: `{dst_dir}/.bigsmall_pipeline.json`
Layout:
    {
      "schema": 1,
      "source": "<source path or HF repo id>",
      "dst_dir": "<absolute path>",
      "repo_id": "<HF repo id or null>",
      "stages": {
        "download": "done" | "pending",
        "compress": "done" | "pending",
        "upload":   "done" | "pending"
      },
      "shards": {
        "<shard filename>": {
          "compressed": bool,
          "uploaded":   bool,
          "compressed_bytes": int | null
        }
      }
    }

CLI:
    bigsmall pipeline run <source> <dst_dir> [--repo-id ID]
                          [--no-upload] [--no-compress] [--mode balanced]

The pipeline is idempotent: running it again after partial completion picks
up where it left off without redoing finished stages.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

CHECKPOINT_FILENAME = ".bigsmall_pipeline.json"
SCHEMA_VERSION = 1


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class Pipeline:
    """Resumable download -> compress -> upload orchestrator."""

    def __init__(self,
                 source: str | Path,
                 dst_dir: str | Path,
                 repo_id: Optional[str] = None,
                 mode: str = "balanced",
                 token: Optional[str] = None,
                 workers: Optional[int] = None,
                 use_lfs_upload: bool = False):
        self.source = str(source)
        self.dst_dir = Path(dst_dir).resolve()
        self.repo_id = repo_id
        self.mode = mode
        self.token = token
        self.workers = workers
        self.use_lfs_upload = use_lfs_upload
        self.dst_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.dst_dir / CHECKPOINT_FILENAME
        self.state = self._load_or_init()

    # ---------------------- checkpoint i/o ----------------------------------

    def _initial_state(self) -> dict:
        return {
            "schema": SCHEMA_VERSION,
            "source": self.source,
            "dst_dir": str(self.dst_dir),
            "repo_id": self.repo_id,
            "created_at": _now(),
            "updated_at": _now(),
            "stages": {
                "download": "pending",
                "compress": "pending",
                "upload": "pending",
            },
            "shards": {},
        }

    def _load_or_init(self) -> dict:
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            if state.get("schema") != SCHEMA_VERSION:
                raise ValueError(
                    f"Checkpoint schema {state.get('schema')} unsupported "
                    f"(expected {SCHEMA_VERSION}). Move {self.checkpoint_path}."
                )
            return state
        return self._initial_state()

    def _save(self) -> None:
        self.state["updated_at"] = _now()
        tmp = self.checkpoint_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
        os.replace(tmp, self.checkpoint_path)

    # ---------------------- stages ------------------------------------------

    def _shards_in_dst(self) -> list[Path]:
        return sorted(self.dst_dir.glob("*.bs"))

    def run_compress(self) -> None:
        """Compress the source model (download is implicit via compress_for_hub)."""
        from . import hub  # local import keeps CLI startup fast

        if self.state["stages"]["compress"] == "done":
            print(f"[pipeline] compress already done -> {self.dst_dir}", flush=True)
            return

        # Mark download as done once we've kicked off compress_for_hub, which
        # internally handles materialising the source (local or HF repo).
        self.state["stages"]["download"] = "done"
        self._save()

        hub.compress_for_hub(
            self.source,
            output_dir=self.dst_dir,
            mode=self.mode,
            workers=self.workers,
            overwrite=False,
        )

        for shard in self._shards_in_dst():
            entry = self.state["shards"].setdefault(shard.name, {
                "compressed": False, "uploaded": False, "compressed_bytes": None,
            })
            entry["compressed"] = True
            entry["compressed_bytes"] = shard.stat().st_size
        self.state["stages"]["compress"] = "done"
        self._save()
        print(f"[pipeline] compress complete: {len(self._shards_in_dst())} shards", flush=True)

    def run_upload(self) -> None:
        """Push the compressed directory to the HF Hub. Skips files already up."""
        if not self.repo_id:
            print("[pipeline] no repo_id set; skipping upload", flush=True)
            return
        if self.state["stages"]["upload"] == "done":
            print(f"[pipeline] upload already done -> {self.repo_id}", flush=True)
            return
        from . import hub

        if self.use_lfs_upload:
            hub.upload_to_hub_lfs(self.dst_dir, repo_id=self.repo_id,
                                  token=self.token)
        else:
            hub.upload_to_hub(self.dst_dir, repo_id=self.repo_id,
                              token=self.token)

        for shard_name, entry in self.state["shards"].items():
            entry["uploaded"] = True
        self.state["stages"]["upload"] = "done"
        self._save()
        print(f"[pipeline] upload complete -> {self.repo_id}", flush=True)

    def run(self, do_compress: bool = True, do_upload: bool = True) -> None:
        if do_compress:
            self.run_compress()
        if do_upload:
            self.run_upload()

    # ---------------------- introspection -----------------------------------

    def summary(self) -> dict:
        """Return a copy of the current checkpoint state for reporting."""
        return json.loads(json.dumps(self.state))
