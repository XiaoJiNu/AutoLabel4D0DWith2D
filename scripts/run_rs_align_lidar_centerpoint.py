#!/usr/bin/env python3
"""Run CenterPoint on all aligned-frame hs64 bins (batches of <=4)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(
    "/data/code/location/pointFuse/output/runtime/openpcdet-centerpoint-8cacccec/run_in_runtime.sh"
)
INNER = _REPO_ROOT / "scripts" / "run_rs_repr_lidar_centerpoint.py"
DEFAULT_MANIFEST = Path(
    "/data/data/automomous/autolabel4d/manifests/rs_align_frames_v0.json"
)
DEFAULT_CACHE = Path(
    "/data/data/automomous/autolabel4d/pseudo_labels/rs_align_lidar"
)
DEFAULT_VIZ = _REPO_ROOT / "outputs" / "rs_align_lidar"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--viz-dir", type=Path, default=DEFAULT_VIZ)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--score-thresh", type=float, default=0.1)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    man = json.loads(args.manifest.read_text(encoding="utf-8"))
    bins = [str(fr["hs64_path"]) for fr in (man.get("frames") or [])]
    if not bins:
        print("FAIL: no frames")
        return 1
    args.cache_root.mkdir(parents=True, exist_ok=True)
    args.viz_dir.mkdir(parents=True, exist_ok=True)
    bs = max(2, min(int(args.batch_size), 4))
    rc = 0
    for i in range(0, len(bins), bs):
        batch = bins[i : i + bs]
        # pad to >=2 if last batch is size 1 (script requires n>=2)
        if len(batch) == 1:
            batch = batch + [batch[0]]
        cmd = [
            str(RUNTIME),
            str(INNER),
            "--n",
            str(len(batch)),
            "--bins",
            *batch,
            "--cache-root",
            str(args.cache_root),
            "--viz-dir",
            str(args.viz_dir),
            "--score-thresh",
            str(args.score_thresh),
            "--device",
            args.device,
            "--status-md",
            str(args.cache_root / f"status_batch_{i:02d}.md"),
        ]
        print("[align_cp] batch", i, "n=", len(set(batch)), "cmd tail=", cmd[-8:])
        r = subprocess.run(cmd, cwd=str(_REPO_ROOT))
        if r.returncode != 0:
            rc = r.returncode
            print(f"[align_cp] batch {i} rc={r.returncode}")
    # summarize
    oks = list(args.cache_root.glob("lidar_centerpoint_*.json"))
    print(f"[align_cp] caches={len(oks)} under {args.cache_root}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
