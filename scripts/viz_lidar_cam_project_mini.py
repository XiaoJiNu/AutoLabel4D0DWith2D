#!/usr/bin/env python3
"""Stage C: project LIDAR_TOP onto CAM_FRONT (+ optional cam) on nuScenes mini; save overlays."""
from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
for p in (_SRC, _REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data.nuscenes_mini import (  # noqa: E402
    cam_filepath,
    lidar_filepath,
    list_sample_tokens,
    load_nuscenes,
)
from data.paths import load_paths, repo_root  # noqa: E402
from geometry.project import project_nuscenes_lidar_to_cam  # noqa: E402
from geometry.sensors import get_nuscenes_cam_lidar_preset, load_rs_4f_1l  # noqa: E402
from geometry.time_pose import check_nuscenes_lidar_cam_sync  # noqa: E402

_CST = timezone(timedelta(hours=8))


def _parse_args() -> argparse.Namespace:
    paths = load_paths()
    default_dataroot = paths.get("nuscenes_mini", "/data/data/automomous/nuscenes/v1.0-mini")
    default_viz = str(repo_root() / "outputs" / "geometry_proj_mini")
    default_log_dir = "/data/data/automomous/autolabel4d/logs"
    ap = argparse.ArgumentParser(description="Stage C LiDAR→cam projection viz (nuScenes mini)")
    ap.add_argument("--dataroot", default=default_dataroot)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--scene", default="scene-0103")
    ap.add_argument("--n", type=int, default=4, help="Number of frames to visualize")
    ap.add_argument(
        "--sample-mode",
        choices=("fixed", "random"),
        default="fixed",
        help="fixed: first N keyframes; random: sample N without replacement",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cam", default="CAM_FRONT", help="Primary camera channel")
    ap.add_argument(
        "--extra-cam",
        default="CAM_FRONT_LEFT",
        help="Optional second camera (empty string to disable)",
    )
    ap.add_argument("--max-points", type=int, default=80000, help="Subsample LiDAR for overlay")
    ap.add_argument("--min-depth", type=float, default=0.5)
    ap.add_argument("--out-viz", default=default_viz)
    ap.add_argument("--log-dir", default=default_log_dir)
    return ap.parse_args()


def load_lidar_points(bin_path: Path) -> np.ndarray:
    pts = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 5)[:, :3]
    return pts


def depth_to_bgr(depths: np.ndarray) -> np.ndarray:
    """Map depth to BGR using a simple jet-like scale (near=red-ish, far=blue-ish)."""
    d = np.asarray(depths, dtype=np.float64)
    if d.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    d_clip = np.clip(d, 1.0, 60.0)
    t = (d_clip - 1.0) / (60.0 - 1.0)
    # BGR: far -> blue, near -> red/yellow
    b = (255 * t).astype(np.uint8)
    g = (255 * (1.0 - np.abs(t - 0.5) * 2)).astype(np.uint8)
    r = (255 * (1.0 - t)).astype(np.uint8)
    return np.stack([b, g, r], axis=1)


def overlay_points(
    img_bgr: np.ndarray,
    uv: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray,
    point_radius: int = 1,
) -> Tuple[np.ndarray, int]:
    out = img_bgr.copy()
    m = mask.astype(bool)
    pts = uv[m]
    cols = depth_to_bgr(depth[m])
    n = 0
    for (u, v), c in zip(pts, cols):
        cv2.circle(out, (int(round(u)), int(round(v))), point_radius, (int(c[0]), int(c[1]), int(c[2])), -1)
        n += 1
    return out, n


def select_tokens(tokens: Sequence[str], n: int, mode: str, seed: int) -> List[str]:
    if n <= 0:
        raise ValueError("--n must be > 0")
    if len(tokens) == 0:
        raise RuntimeError("no sample tokens in scene")
    n_use = min(n, len(tokens))
    if mode == "fixed":
        return list(tokens[:n_use])
    rng = random.Random(seed)
    return rng.sample(list(tokens), n_use)


def main() -> int:
    args = _parse_args()
    out_viz = Path(args.out_viz)
    out_viz.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Preset smoke: nuScenes + ensure RS YAML still loads
    nusc_preset = get_nuscenes_cam_lidar_preset()
    try:
        rs_preset = load_rs_4f_1l()
        rs_note = f"RS preset OK name={rs_preset.name} cams={rs_preset.camera_channels}"
    except Exception as e:  # noqa: BLE001
        rs_note = f"RS preset load failed (non-fatal for mini viz): {e}"

    cams: List[str] = [args.cam]
    if args.extra_cam and str(args.extra_cam).strip():
        cams.append(str(args.extra_cam).strip())

    print(f"NuScenesCamLidar cams={nusc_preset.camera_channels[:3]}... lidar={nusc_preset.lidar_channel}")
    print(rs_note)
    print(f"Loading {args.version} from {args.dataroot}")
    nusc = load_nuscenes(args.dataroot, version=args.version, verbose=False)
    all_tokens = list_sample_tokens(nusc, args.scene, max_samples=None)
    tokens = select_tokens(all_tokens, args.n, args.sample_mode, args.seed)
    print(f"scene={args.scene} total_keyframes={len(all_tokens)} selected={len(tokens)} mode={args.sample_mode}")

    saved: List[Path] = []
    sync_lines: List[str] = []
    for i, tok in enumerate(tokens):
        lidar_path = lidar_filepath(nusc, tok)
        pts = load_lidar_points(lidar_path)
        if args.max_points > 0 and len(pts) > args.max_points:
            idx = np.linspace(0, len(pts) - 1, args.max_points).astype(np.int64)
            pts = pts[idx]

        for cam in cams:
            sync = check_nuscenes_lidar_cam_sync(nusc, tok, cam_channel=cam)
            sync_lines.append(
                f"{tok[:8]} {cam} delta_ms={sync.delta_ms:.2f} ok={sync.ok} | {sync.messages[0]}"
            )
            cam_path, _sd = cam_filepath(nusc, tok, channel=cam)
            img = cv2.imread(str(cam_path))
            if img is None:
                raise RuntimeError(f"failed to read {cam_path}")
            h, w = img.shape[:2]
            uv, mask, depth = project_nuscenes_lidar_to_cam(
                nusc,
                tok,
                pts,
                cam_channel=cam,
                image_size=(w, h),
                min_depth=args.min_depth,
            )
            over, n_vis = overlay_points(img, uv, mask, depth, point_radius=1)
            label = f"{args.scene} {tok[:8]} {cam} vis={n_vis}/{len(pts)} dt={sync.delta_ms:.1f}ms"
            cv2.putText(over, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            out_name = f"{args.scene}_{tok[:12]}_{cam}_proj.png"
            out_path = out_viz / out_name
            cv2.imwrite(str(out_path), over)
            saved.append(out_path)
            print(f"  wrote {out_path} ({n_vis} points)")

    now = datetime.now(_CST).strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"geometry_proj_mini_{now}.log"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"Stage C geometry projection viz\n")
        f.write(f"time_CST={now}\n")
        f.write(f"dataroot={args.dataroot}\n")
        f.write(f"scene={args.scene} n={len(tokens)} mode={args.sample_mode} seed={args.seed}\n")
        f.write(f"cams={cams}\n")
        f.write(f"out_viz={out_viz}\n")
        f.write(f"{rs_note}\n")
        f.write("--- sync ---\n")
        for line in sync_lines:
            f.write(line + "\n")
        f.write("--- outputs ---\n")
        for p in saved:
            f.write(str(p) + "\n")
    print(f"log: {log_path}")
    print(f"PASS: wrote {len(saved)} overlays under {out_viz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
