#!/usr/bin/env python3
"""Stage D interface smoke: lidar_stub only on nuScenes mini scene-0103, N<=2 frames.

No heavy teacher weights / no torch model load. Prints nvidia-smi VRAM query if available.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
for p in (_SRC, _REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data.nuscenes_mini import (  # noqa: E402
    list_sample_tokens,
    load_nuscenes,
    sample_meta,
)
from data.paths import load_paths  # noqa: E402
from teachers.base import SerialTeacherGuard, assert_serial_idle  # noqa: E402
from teachers.lidar_stub import DEFAULT_CACHE_ROOT, LidarTeacherStub  # noqa: E402


def _parse_args() -> argparse.Namespace:
    paths = load_paths()
    default_dataroot = paths.get(
        "nuscenes_mini", "/data/data/automomous/nuscenes/v1.0-mini"
    )
    ap = argparse.ArgumentParser(description="Stage D teacher serial interface smoke")
    ap.add_argument("--dataroot", default=default_dataroot)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--scene", default="scene-0103")
    ap.add_argument("--max-samples", type=int, default=2)
    ap.add_argument(
        "--cache-root",
        default=str(DEFAULT_CACHE_ROOT),
        help="teacher_cache_smoke directory",
    )
    return ap.parse_args()


def query_vram() -> Optional[str]:
    """nvidia-smi query without loading torch."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
        return out.strip()
    except Exception as exc:  # noqa: BLE001
        return f"(nvidia-smi unavailable: {exc})"


def main() -> int:
    args = _parse_args()
    n = max(1, min(int(args.max_samples), 2))  # hard cap N<=2 for smoke
    print(f"[smoke] scene={args.scene} N={n} cache={args.cache_root}")
    print(f"[smoke] VRAM before: {query_vram()}")

    SerialTeacherGuard.reset()
    nusc = load_nuscenes(args.dataroot, version=args.version, verbose=False)
    tokens = list_sample_tokens(nusc, args.scene, max_samples=n)
    if not tokens:
        print("FAIL: no sample tokens")
        return 1

    teacher = LidarTeacherStub(cache_root=args.cache_root)
    teacher.set_nusc(nusc)
    cache_paths: List[str] = []

    try:
        teacher.load()
        if SerialTeacherGuard.resident() != teacher.name:
            print("FAIL: serial guard did not acquire lidar_stub")
            return 1
        for i, tok in enumerate(tokens):
            meta = sample_meta(nusc, tok, args.scene)
            frame: Dict[str, Any] = {
                "sample_token": tok,
                "scene_name": args.scene,
                "sample_idx": i,
                "lidar_sd_token": meta["lidar_sd_token"],
                "nusc": nusc,
            }
            result = teacher.infer_frame(frame)
            assert result["meta"]["is_fake"] is True
            assert len(result["objects"]) >= 1
            cache_paths.append(result["cache_path"])
            print(
                f"  frame[{i}] token={tok[:8]}… "
                f"n_obj={len(result['objects'])} cache={result['cache_path']}"
            )
    finally:
        teacher.unload()

    assert_serial_idle()
    print(f"[smoke] VRAM after unload: {query_vram()}")
    print(f"[smoke] resident={SerialTeacherGuard.resident()!r}")
    print(f"[smoke] cache_files={cache_paths}")
    print("PASS: Stage D lidar_stub serial smoke OK (interface only; no teacher weights)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
