#!/usr/bin/env python3
"""Representative-frame RoboSense hs64 CenterPoint teacher (N=2..4).

LiDAR-only; skip DINO/SAM; no multipart; no train; no kitti; no git push.
Always writes JSON + BEV PNG + status md. Prefer real inference.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
for p in (_SRC, _REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from teachers.base import SerialTeacherGuard, assert_serial_idle  # noqa: E402
from teachers.lidar_centerpoint import (  # noqa: E402
    DEFAULT_CACHE_ROOT,
    DEFAULT_CKPT,
    DEFAULT_CFG,
    LidarCenterPointTeacher,
    load_hs64_bin,
    render_bev_png,
)

CST = timezone(timedelta(hours=8))
DEFAULT_HS64_DIR = Path(
    "/data/data/automomous/robosense/subset/lidar_occ_trainval/"
    "processed_data_20230906/SWEEPER-001/hs64"
)
DEFAULT_VIZ_DIR = _REPO_ROOT / "outputs" / "rs_repr_lidar"
DEFAULT_STATUS_MD = Path("/home/yr/grok-bot-work/autolabel4d_stageB/RS_REPR_LIDAR_STATUS.md")


def query_vram() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT, text=True, timeout=15,
        )
        return out.strip()
    except Exception as exc:
        return f"(nvidia-smi unavailable: {exc})"


def nvidia_smi_full() -> str:
    try:
        return subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, text=True, timeout=20)
    except Exception as exc:
        return f"(nvidia-smi failed: {exc})"


def pick_bins(hs64_dir: Path, n: int, explicit: Optional[List[str]] = None) -> List[Path]:
    if explicit:
        return [Path(p) for p in explicit]
    bins = sorted(hs64_dir.glob("*.bin"))
    if not bins:
        raise FileNotFoundError(f"no *.bin under {hs64_dir}")
    if len(bins) <= n:
        return bins
    step = max(1, len(bins) // n)
    picked, seen = [], set()
    for i in range(n):
        p = bins[min(i * step, len(bins) - 1)]
        if p not in seen:
            seen.add(p)
            picked.append(p)
    for p in bins:
        if len(picked) >= n:
            break
        if p not in seen:
            seen.add(p)
            picked.append(p)
    return picked[:n]


def write_fail_json(cache_root: Path, sample_token: str, scene_name: str, lidar_path: str, reason: str) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    out = cache_root / f"lidar_centerpoint_{sample_token}.json"
    payload = {
        "version": "1.0",
        "sample_token": sample_token,
        "scene_name": scene_name,
        "timestamp": "",
        "lidar_sd_token": "",
        "lidar_path": lidar_path,
        "objects": [],
        "status": "BLOCKED_INFER",
        "meta": {
            "generator": "scripts/run_rs_repr_lidar_centerpoint.py",
            "is_fake": False,
            "is_stub": False,
            "success": False,
            "notes": reason,
            "n_objects": 0,
        },
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def write_status_md(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RS Repr LiDAR CenterPoint Status",
        "",
        f"- Generated (CST/UTC+8): {report['generated_cst']}",
        f"- Success: **{report['success']}**",
        f"- Runtime: `{report.get('runtime', '')}`",
        f"- Ckpt: `{report.get('ckpt', '')}`",
        f"- Cfg: `{report.get('cfg', '')}`",
        f"- N frames: {report.get('n_frames')}",
        f"- VRAM before: `{report.get('vram_before')}`",
        f"- VRAM after unload: `{report.get('vram_after')}`",
        f"- Peak allocated MB: {report.get('peak_allocated_mb')}",
        f"- Blocker: {report.get('blocker') or '(none)'}",
        "",
        "## Per-frame",
        "",
    ]
    for fr in report.get("frames", []):
        lines.append(
            f"- `{fr['sample_token']}`: success={fr.get('success')} "
            f"n_obj={fr.get('n_objects')} n_pts={fr.get('n_points_in_range')} "
            f"json=`{fr.get('json_path')}` bev=`{fr.get('bev_path')}` "
            f"err={fr.get('error') or '-'}"
        )
    lines += [
        "",
        "## Outputs",
        "",
        f"- Pseudo JSON: `{report.get('cache_root')}`",
        f"- BEV PNG: `{report.get('viz_dir')}`",
        "",
        "## nvidia-smi before",
        "",
        "```",
        report.get("nvidia_smi_before", ""),
        "```",
        "",
        "## nvidia-smi after",
        "",
        "```",
        report.get("nvidia_smi_after", ""),
        "```",
        "",
        "## Notes",
        "",
        "- LiDAR-only; DINO/SAM skipped.",
        "- hs64 = float64 XYZ; intensity/timestamp padded 0.",
        "- Boxes in LiDAR frame (no global calib this round).",
        "- Model unloaded; SerialTeacherGuard released.",
        "- Runtime so3.py shimmed to avoid kornia/torch.jit import crash.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hs64-dir", type=Path, default=DEFAULT_HS64_DIR)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--bins", nargs="*", default=None)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    ap.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    ap.add_argument("--viz-dir", type=Path, default=DEFAULT_VIZ_DIR)
    ap.add_argument("--status-md", type=Path, default=DEFAULT_STATUS_MD)
    ap.add_argument("--score-thresh", type=float, default=0.1)
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    n = max(2, min(int(args.n), 4))
    report: Dict[str, Any] = {
        "generated_cst": datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "success": False,
        "n_frames": n,
        "ckpt": str(args.ckpt),
        "cfg": str(args.cfg),
        "cache_root": str(args.cache_root),
        "viz_dir": str(args.viz_dir),
        "runtime": "openpcdet-centerpoint-8cacccec + mv2d via run_in_runtime.sh",
        "frames": [],
        "blocker": None,
        "peak_allocated_mb": None,
    }
    print(f"[rs_repr] N={n} hs64_dir={args.hs64_dir}")
    report["nvidia_smi_before"] = nvidia_smi_full()
    report["vram_before"] = query_vram()
    print(f"[rs_repr] VRAM before: {report['vram_before']}")
    args.viz_dir.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)

    try:
        bins = pick_bins(args.hs64_dir, n, args.bins)
    except Exception as exc:
        report["blocker"] = f"bin_select: {exc}"
        write_status_md(args.status_md, report)
        return 2

    SerialTeacherGuard.reset()
    teacher = LidarCenterPointTeacher(
        ckpt=args.ckpt, cfg_file=args.cfg, cache_root=args.cache_root,
        score_thresh=args.score_thresh, device=args.device,
    )
    load_ok = False
    try:
        teacher.load()
        load_ok = True
        print(f"[rs_repr] loaded VRAM_MB={teacher.vram_allocated_mb()}")
    except Exception as exc:
        report["blocker"] = f"model_load: {type(exc).__name__}: {exc}"
        traceback.print_exc()

    peak = teacher.vram_allocated_mb()
    pc_range = (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0)

    for i, bin_path in enumerate(bins):
        token = bin_path.stem
        scene = "SWEEPER-001"
        fr: Dict[str, Any] = {
            "sample_token": token, "lidar_path": str(bin_path), "success": False,
            "n_objects": 0, "n_points_in_range": None, "json_path": None,
            "bev_path": None, "error": None,
        }
        points_for_bev = None
        objects: List[Dict[str, Any]] = []
        try:
            xyz = load_hs64_bin(bin_path)
            import numpy as np
            pr = np.array(pc_range, dtype=np.float32)
            m = (
                (xyz[:, 0] >= pr[0]) & (xyz[:, 0] <= pr[3])
                & (xyz[:, 1] >= pr[1]) & (xyz[:, 1] <= pr[4])
                & (xyz[:, 2] >= pr[2]) & (xyz[:, 2] <= pr[5])
            )
            points_for_bev = xyz[m]
            fr["n_points_in_range"] = int(points_for_bev.shape[0])
            if load_ok:
                result = teacher.infer_frame({
                    "lidar_path": str(bin_path), "sample_token": token,
                    "scene_name": scene, "frame_id": i, "timestamp": token,
                })
                objects = result["objects"]
                points_for_bev = result.get("points_xyz_in_range", points_for_bev)
                fr["n_objects"] = len(objects)
                fr["n_points_in_range"] = result["meta"].get("n_points_in_range", fr["n_points_in_range"])
                fr["json_path"] = result["cache_path"]
                fr["success"] = True
                v = teacher.vram_allocated_mb()
                if v is not None:
                    peak = max(peak or 0, v)
            else:
                jp = write_fail_json(args.cache_root, token, scene, str(bin_path), report["blocker"] or "not loaded")
                fr["json_path"] = str(jp)
                fr["error"] = report["blocker"]
        except Exception as exc:
            fr["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            jp = write_fail_json(args.cache_root, token, scene, str(bin_path), fr["error"])
            fr["json_path"] = str(jp)

        bev_path = args.viz_dir / f"bev_{token}.png"
        try:
            import numpy as np
            render_bev_png(
                points_for_bev if points_for_bev is not None else np.zeros((0, 3)),
                objects, bev_path,
                title=f"{token} n_obj={len(objects)} success={fr['success']}",
                pc_range=pc_range,
            )
            fr["bev_path"] = str(bev_path)
        except Exception as exc:
            fr["error"] = (fr["error"] or "") + f" | bev: {exc}"

        report["frames"].append(fr)
        print(f"  [{i}] {token} success={fr['success']} n_obj={fr['n_objects']} pts={fr['n_points_in_range']}")

    try:
        teacher.unload()
    except Exception:
        SerialTeacherGuard.reset()
    try:
        assert_serial_idle()
    except Exception as exc:
        print("WARN serial:", exc)

    report["peak_allocated_mb"] = peak
    report["vram_after"] = query_vram()
    report["nvidia_smi_after"] = nvidia_smi_full()
    report["success"] = bool(report["frames"]) and all(f.get("success") for f in report["frames"])
    if report["success"]:
        report["blocker"] = None
    write_status_md(args.status_md, report)
    args.status_md.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[rs_repr] VRAM after: {report['vram_after']}")
    print(f"[rs_repr] status: {args.status_md} success={report['success']}")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
