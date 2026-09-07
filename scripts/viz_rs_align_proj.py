#!/usr/bin/env python3
"""Project hs64 LiDAR onto RoboSense images for aligned frames; save overlays.

Uses manifests/rs_align_frames_v0.json + PKL calib (sensor2ego, hs2global, ego2global).
Prefers CAM_FRONT (folder 0, pinhole) for smoke; OV cams need fisheye unwrap later.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from geometry.project import project_lidar_to_image, invert_rt, transform_points  # noqa: E402

CST = timezone(timedelta(hours=8))


def cst_now() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S CST")


def load_hs64_bin(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.float32)
    if raw.size % 4 == 0:
        pts = raw.reshape(-1, 4)[:, :3]
    elif raw.size % 3 == 0:
        pts = raw.reshape(-1, 3)
    else:
        # try 5-dim like nuScenes
        if raw.size % 5 == 0:
            pts = raw.reshape(-1, 5)[:, :3]
        else:
            raise ValueError(f"unexpected bin size {raw.size} at {path}")
    return pts.astype(np.float64)


def depth_to_bgr(depths: np.ndarray) -> np.ndarray:
    if depths.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    d = depths.copy()
    lo, hi = np.percentile(d, [5, 95]) if d.size > 10 else (d.min(), d.max() + 1e-6)
    hi = max(hi, lo + 1e-3)
    x = np.clip((d - lo) / (hi - lo), 0, 1)
    # simple jet-ish without matplotlib
    r = np.clip(1.5 - np.abs(x - 0.75) * 4, 0, 1)
    g = np.clip(1.5 - np.abs(x - 0.5) * 4, 0, 1)
    b = np.clip(1.5 - np.abs(x - 0.25) * 4, 0, 1)
    return (np.stack([b, g, r], axis=1) * 255).astype(np.uint8)


def rt_from_ego2global(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.asarray(R, dtype=np.float64).reshape(3, 3), np.asarray(t, dtype=np.float64).reshape(3)


def hs_to_cam(
    pts_hs: np.ndarray,
    hs2global: np.ndarray,
    ego2global_R: np.ndarray,
    ego2global_t: np.ndarray,
    sensor2ego_R: np.ndarray,
    sensor2ego_t: np.ndarray,
) -> np.ndarray:
    """points_hs -> global -> ego -> cam."""
    T = np.asarray(hs2global, dtype=np.float64).reshape(4, 4)
    R_hg, t_hg = T[:3, :3], T[:3, 3]
    pts_g = transform_points(pts_hs, R_hg, t_hg)
    R_eg, t_eg = rt_from_ego2global(ego2global_R, ego2global_t)
    R_ge, t_ge = invert_rt(R_eg, t_eg)
    pts_ego = transform_points(pts_g, R_ge, t_ge)
    R_se = np.asarray(sensor2ego_R, dtype=np.float64).reshape(3, 3)
    t_se = np.asarray(sensor2ego_t, dtype=np.float64).reshape(3)
    R_es, t_es = invert_rt(R_se, t_se)
    return transform_points(pts_ego, R_es, t_es)


def find_pkl_sample_for_stem(
    samples: List[dict], stem: str, cam_name: str = "CAM_FRONT"
) -> Optional[dict]:
    """Match PKL sample by hs64 or image filename stem."""
    for s in samples:
        hs = Path(str(s.get("hs64_path", ""))).stem
        if hs == stem:
            return s
        cams = s.get("images", {}).get("cams", {})
        cv = cams.get(cam_name) or {}
        if Path(str(cv.get("data_path", ""))).stem == stem:
            return s
        # also try any cam
        for cv in cams.values():
            if isinstance(cv, dict) and Path(str(cv.get("data_path", ""))).stem == stem:
                return s
    return None


def index_pkl_by_stem(samples: List[dict]) -> Dict[str, dict]:
    idx: Dict[str, dict] = {}
    for s in samples:
        hs = Path(str(s.get("hs64_path", ""))).stem
        if hs:
            idx[hs] = s
        for cv in s.get("images", {}).get("cams", {}).values():
            if isinstance(cv, dict):
                st = Path(str(cv.get("data_path", ""))).stem
                if st and st not in idx:
                    idx[st] = s
    return idx


def overlay_points(img: np.ndarray, uv: np.ndarray, mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
    out = img.copy()
    if mask.sum() == 0:
        return out
    cols = depth_to_bgr(depth[mask])
    uvs = uv[mask].astype(np.int32)
    for (u, v), c in zip(uvs, cols):
        if 0 <= u < out.shape[1] and 0 <= v < out.shape[0]:
            cv2.circle(out, (int(u), int(v)), 1, (int(c[0]), int(c[1]), int(c[2])), -1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/data/data/automomous/autolabel4d/manifests/rs_align_frames_v0.json")
    ap.add_argument("--val-pkl", default="/data/data/automomous/robosense/subset/splits/robosense_local_val.pkl")
    ap.add_argument("--train-pkl", default="/data/data/automomous/robosense/subset/splits/robosense_local_train.pkl")
    ap.add_argument("--cam-folder", default="0", help="numeric folder id (0=CAM_FRONT pinhole)")
    ap.add_argument("--cam-name", default="CAM_FRONT")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-points", type=int, default=80000)
    ap.add_argument("--min-depth", type=float, default=0.5)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "rs_align_proj"))
    ap.add_argument("--log-dir", default="/data/data/automomous/autolabel4d/logs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "rs_align_proj.log"

    lines: List[str] = [f"=== viz_rs_align_proj {cst_now()} ==="]
    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    frames = man.get("frames") or []
    n_aligned = int(man.get("n_aligned_frames") or len(frames))
    lines.append(f"manifest n_aligned_frames={n_aligned}")

    if n_aligned == 0 or not frames:
        msg = (
            "SKIP projection: no aligned image↔hs64 frames on disk. "
            "See rs_align_frames_v0 coverage.reasons / download estimate (>=10.74GB next part)."
        )
        lines.append(msg)
        # write placeholder note
        note = out_dir / "NO_ALIGNED_FRAMES.txt"
        note.write_text("\n".join(lines) + "\n", encoding="utf-8")
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return 0

    # Load PKLs for calib (val first, then train if needed)
    samples: List[dict] = []
    for p in (args.val_pkl, args.train_pkl):
        pp = Path(p)
        if pp.is_file():
            with pp.open("rb") as f:
                samples.extend(pickle.load(f))
    idx = index_pkl_by_stem(samples)
    lines.append(f"pkl index size={len(idx)}")

    selected = frames[: max(1, args.n)]
    written = []
    for i, fr in enumerate(selected):
        stem = fr.get("hs_stem") or fr.get("stem")
        img_stem = fr.get("stem")
        hs_path = Path(fr["hs64_path"])
        img_path = fr.get("images", {}).get(str(args.cam_folder))
        if not img_path:
            lines.append(f"[{i}] missing cam folder {args.cam_folder} for {stem}")
            continue
        img_path = Path(img_path)
        sample = idx.get(stem) or idx.get(img_stem)
        if sample is None:
            lines.append(f"[{i}] no PKL calib for stem={stem}")
            continue
        cam = sample["images"]["cams"].get(args.cam_name)
        if cam is None:
            lines.append(f"[{i}] cam {args.cam_name} missing in PKL")
            continue

        pts = load_hs64_bin(hs_path)
        if pts.shape[0] > args.max_points:
            rng = np.random.default_rng(0)
            sel = rng.choice(pts.shape[0], size=args.max_points, replace=False)
            pts = pts[sel]

        pts_cam = hs_to_cam(
            pts,
            sample["hs2global"],
            sample["ego2global_rotation"],
            sample["ego2global_translation"],
            cam["sensor2ego_rotation"],
            cam["sensor2ego_translation"],
        )
        K = np.asarray(cam["cam_intrinsic"], dtype=np.float64)
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            lines.append(f"[{i}] failed to read image {img_path}")
            continue
        h, w = img.shape[:2]
        # identity R,t already in cam frame
        uv, mask, depth = project_lidar_to_image(
            pts_cam,
            K,
            np.eye(3),
            np.zeros(3),
            image_size=(w, h),
            min_depth=args.min_depth,
        )
        # Actually pts_cam already in camera frame; project_points_pinhole path:
        from geometry.project import project_points_pinhole

        uv, mask, depth = project_points_pinhole(pts_cam, K, image_size=(w, h), min_depth=args.min_depth)
        vis = overlay_points(img, uv, mask, depth)
        n_valid = int(mask.sum())
        out_name = f"proj_{i:02d}_{img_stem}_{args.cam_name}.jpg"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), vis)
        written.append(str(out_path))
        lines.append(
            f"[{i}] wrote {out_path} pts={pts.shape[0]} valid={n_valid} "
            f"frac={n_valid/max(pts.shape[0],1):.4f}"
        )

    lines.append(f"done n_written={len(written)} at {cst_now()}")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "viz_list.json").write_text(
        json.dumps({"written": written, "log": str(log_path)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
