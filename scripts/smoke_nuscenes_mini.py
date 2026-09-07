#!/usr/bin/env python3
"""Stage A4: nuScenes mini smoke — read samples, write fake pseudo labels, validate, viz."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion

# Allow running as scripts/smoke_nuscenes_mini.py with PYTHONPATH=src or repo root
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
    sample_meta,
)
from data.paths import load_paths, repo_root  # noqa: E402
from export.pseudo_label_io import (  # noqa: E402
    list_label_files,
    read_label,
    validate_label,
    write_label,
)


def _parse_args() -> argparse.Namespace:
    paths = load_paths()
    default_dataroot = paths.get("nuscenes_mini", "/data/data/automomous/nuscenes/v1.0-mini")
    default_pseudo = str(Path(paths.get("pseudo_labels", "/data/data/automomous/autolabel4d/pseudo_labels")) / "smoke_nuscenes_mini")
    default_viz = str(repo_root() / "outputs" / "smoke_nuscenes_mini")
    ap = argparse.ArgumentParser(description="AutoLabel4D Stage A4 nuScenes mini smoke")
    ap.add_argument("--dataroot", default=default_dataroot)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--scene", default="scene-0103")
    ap.add_argument("--max-samples", type=int, default=8)
    ap.add_argument("--out-pseudo", default=default_pseudo)
    ap.add_argument("--out-viz", default=default_viz)
    return ap.parse_args()


def load_lidar_points(bin_path: Path) -> np.ndarray:
    pts = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 5)[:, :3]
    return pts


def lidar_to_global(nusc, sample_token: str, xyz_lidar: np.ndarray) -> np.ndarray:
    """Transform Nx3 points from LIDAR_TOP frame to global."""
    sample = nusc.get("sample", sample_token)
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    q_cs = Quaternion(cs["rotation"])
    t_cs = np.array(cs["translation"], dtype=np.float64)
    q_ep = Quaternion(ep["rotation"])
    t_ep = np.array(ep["translation"], dtype=np.float64)
    xyz = np.asarray(xyz_lidar, dtype=np.float64)
    # lidar -> ego -> global
    xyz_ego = (q_cs.rotation_matrix @ xyz.T).T + t_cs
    xyz_global = (q_ep.rotation_matrix @ xyz_ego.T).T + t_ep
    return xyz_global


def yaw_quat_wxyz(yaw: float) -> List[float]:
    q = Quaternion(axis=[0, 0, 1], radians=yaw)
    return [float(q.w), float(q.x), float(q.y), float(q.z)]


def make_synthetic_objects(sample_idx: int) -> List[Dict[str, Any]]:
    """Invent 2–4 boxes at fixed offsets in LIDAR frame (returned as lidar-local)."""
    # (dx, dy, dz, w, l, h, yaw, category, quality)
    catalog = [
        (8.0, 2.0, 0.0, 1.8, 4.5, 1.6, 0.1, "car", "C"),
        (12.0, -3.5, 0.0, 1.9, 4.8, 1.7, -0.2, "car", "C"),
        (5.0, 6.0, 0.0, 0.7, 0.8, 1.7, 1.2, "pedestrian", "C"),
        (15.0, 1.0, 0.0, 2.0, 5.5, 2.2, 0.0, "truck", "C"),
    ]
    n = 2 + (sample_idx % 3)  # 2..4
    objs_lidar = []
    for i, row in enumerate(catalog[:n]):
        dx, dy, dz, w, l, h, yaw, cat, qual = row
        objs_lidar.append(
            {
                "track_id": f"smoke_trk_{i:03d}",
                "category": cat,
                "translation_lidar": [dx, dy, dz],
                "size": [w, l, h],
                "yaw": yaw,
                "score": float(0.95 - 0.05 * i),
                "quality": qual,
                "valid_fields": ["translation", "size", "rotation", "category"],
                "ignore": False,
                "source": "smoke_synthetic",
            }
        )
    return objs_lidar


def objects_to_global(nusc, sample_token: str, objs_lidar: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for o in objs_lidar:
        xyz_l = np.array([o["translation_lidar"]], dtype=np.float64)
        xyz_g = lidar_to_global(nusc, sample_token, xyz_l)[0]
        # rotate yaw into global: lidar yaw + ego/cs yaw approx via transforming forward vector
        yaw_l = o["yaw"]
        fwd_l = np.array([[np.cos(yaw_l), np.sin(yaw_l), 0.0]], dtype=np.float64)
        origin = np.zeros((1, 3), dtype=np.float64)
        fwd_g = lidar_to_global(nusc, sample_token, fwd_l)[0] - lidar_to_global(nusc, sample_token, origin)[0]
        yaw_g = float(np.arctan2(fwd_g[1], fwd_g[0]))
        out.append(
            {
                "track_id": o["track_id"],
                "category": o["category"],
                "translation": [float(xyz_g[0]), float(xyz_g[1]), float(xyz_g[2])],
                "size": list(o["size"]),
                "rotation": yaw_quat_wxyz(yaw_g),
                "score": o["score"],
                "quality": o["quality"],
                "valid_fields": list(o["valid_fields"]),
                "ignore": bool(o["ignore"]),
                "source": o["source"],
            }
        )
    return out


def build_label(nusc, sample_token: str, scene_name: str, sample_idx: int) -> Dict[str, Any]:
    meta_s = sample_meta(nusc, sample_token, scene_name)
    objs_l = make_synthetic_objects(sample_idx)
    objs_g = objects_to_global(nusc, sample_token, objs_l)
    return {
        "version": "1.0",
        "sample_token": meta_s["sample_token"],
        "scene_name": meta_s["scene_name"],
        "timestamp": meta_s["timestamp"],
        "lidar_sd_token": meta_s["lidar_sd_token"],
        "objects": objs_g,
        "meta": {
            "generator": "scripts/smoke_nuscenes_mini.py",
            "is_fake": True,
            "notes": "Synthetic boxes for Stage A I/O smoke only; not for training.",
        },
        # keep lidar-frame copies for viz convenience (not required by schema)
        "_smoke_lidar_objects": objs_l,
    }


def box_corners_bev(cx: float, cy: float, w: float, l: float, yaw: float) -> np.ndarray:
    """Return 5x2 closed polygon corners in BEV (xy). size is w,l,h."""
    hw, hl = w / 2.0, l / 2.0
    corners = np.array(
        [
            [hl, hw],
            [hl, -hw],
            [-hl, -hw],
            [-hl, hw],
            [hl, hw],
        ],
        dtype=np.float64,
    )
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (corners @ R.T) + np.array([cx, cy], dtype=np.float64)


def draw_bev(points: np.ndarray, objs_lidar: List[Dict[str, Any]], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    if len(points):
        # subsample for speed
        step = max(1, len(points) // 40000)
        pts = points[::step]
        ax.scatter(pts[:, 0], pts[:, 1], s=0.2, c="k", alpha=0.35, linewidths=0)
    colors = {"car": "tab:blue", "truck": "tab:orange", "pedestrian": "tab:red"}
    for o in objs_lidar:
        x, y, _ = o["translation_lidar"]
        w, l, _ = o["size"]
        poly = box_corners_bev(x, y, w, l, o["yaw"])
        ax.plot(poly[:, 0], poly[:, 1], color=colors.get(o["category"], "tab:green"), lw=2)
        ax.text(x, y, o["track_id"], fontsize=7, color=colors.get(o["category"], "g"))
    ax.set_aspect("equal")
    ax.set_xlabel("x (lidar, m)")
    ax.set_ylabel("y (lidar, m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    # focus near ego
    ax.set_xlim(-5, 40)
    ax.set_ylim(-20, 20)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def project_lidar_to_cam(nusc, sample_token: str, xyz_lidar: np.ndarray, cam_channel: str = "CAM_FRONT") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project lidar points to camera pixels. Returns (uv, depths, mask)."""
    sample = nusc.get("sample", sample_token)
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    cam_sd = nusc.get("sample_data", sample["data"][cam_channel])
    lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    lidar_ep = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    cam_ep = nusc.get("ego_pose", cam_sd["ego_pose_token"])

    xyz = np.asarray(xyz_lidar, dtype=np.float64)
    # lidar -> ego(lidar time) -> global -> ego(cam time) -> cam
    q_l = Quaternion(lidar_cs["rotation"])
    t_l = np.array(lidar_cs["translation"])
    q_le = Quaternion(lidar_ep["rotation"])
    t_le = np.array(lidar_ep["translation"])
    xyz_ego_l = (q_l.rotation_matrix @ xyz.T).T + t_l
    xyz_g = (q_le.rotation_matrix @ xyz_ego_l.T).T + t_le

    q_ce = Quaternion(cam_ep["rotation"])
    t_ce = np.array(cam_ep["translation"])
    xyz_ego_c = (q_ce.inverse.rotation_matrix @ (xyz_g - t_ce).T).T
    q_c = Quaternion(cam_cs["rotation"])
    t_c = np.array(cam_cs["translation"])
    xyz_cam = (q_c.inverse.rotation_matrix @ (xyz_ego_c - t_c).T).T

    depths = xyz_cam[:, 2]
    K = np.array(cam_cs["camera_intrinsic"], dtype=np.float64)
    u = K[0, 0] * (xyz_cam[:, 0] / depths) + K[0, 2]
    v = K[1, 1] * (xyz_cam[:, 1] / depths) + K[1, 2]
    uv = np.stack([u, v], axis=1)
    return uv, depths, xyz_cam


def draw_cam_overlay(
    nusc,
    sample_token: str,
    objs_lidar: List[Dict[str, Any]],
    cam_path: Path,
    out_path: Path,
) -> None:
    img = cv2.imread(str(cam_path))
    if img is None:
        raise RuntimeError(f"failed to read image: {cam_path}")
    h, w = img.shape[:2]
    # project a few lidar box centers + corner points
    for o in objs_lidar:
        corners3 = []
        cx, cy, cz = o["translation_lidar"]
        bw, bl, bh = o["size"]
        yaw = o["yaw"]
        # 8 corners in lidar
        for dx in (-bl / 2, bl / 2):
            for dy in (-bw / 2, bw / 2):
                for dz in (0.0, bh):
                    x = cx + dx * np.cos(yaw) - dy * np.sin(yaw)
                    y = cy + dx * np.sin(yaw) + dy * np.cos(yaw)
                    z = cz + dz
                    corners3.append([x, y, z])
        corners3 = np.array(corners3, dtype=np.float64)
        uv, depths, _ = project_lidar_to_cam(nusc, sample_token, corners3)
        mask = (depths > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        pts = uv[mask].astype(np.int32)
        for p in pts:
            cv2.circle(img, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)
        # center
        ctr = np.array([o["translation_lidar"]], dtype=np.float64)
        uv_c, d_c, _ = project_lidar_to_cam(nusc, sample_token, ctr)
        if d_c[0] > 0.5 and 0 <= uv_c[0, 0] < w and 0 <= uv_c[0, 1] < h:
            cv2.putText(
                img,
                o["track_id"],
                (int(uv_c[0, 0]), int(uv_c[0, 1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def strip_smoke_extra(label: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in label.items() if not k.startswith("_")}


def main() -> int:
    args = _parse_args()
    out_pseudo = Path(args.out_pseudo)
    out_viz = Path(args.out_viz)
    out_pseudo.mkdir(parents=True, exist_ok=True)
    out_viz.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "dataroot": args.dataroot,
        "version": args.version,
        "scene": args.scene,
        "max_samples": args.max_samples,
        "out_pseudo": str(out_pseudo),
        "out_viz": str(out_viz),
        "status": "FAIL",
        "n_written": 0,
        "n_validated": 0,
        "viz_pngs": [],
        "errors": [],
    }

    try:
        nusc = load_nuscenes(args.dataroot, version=args.version, verbose=False)
        tokens = list_sample_tokens(nusc, args.scene, max_samples=args.max_samples)
        if not tokens:
            raise RuntimeError(f"no samples found for scene {args.scene}")

        lidar_objs_by_token: Dict[str, List[Dict[str, Any]]] = {}
        for i, tok in enumerate(tokens):
            label = build_label(nusc, tok, args.scene, i)
            lidar_objs_by_token[tok] = label["_smoke_lidar_objects"]
            clean = strip_smoke_extra(label)
            write_label(out_pseudo / f"{tok}.json", clean)
            summary["n_written"] += 1

            # viz
            pts = load_lidar_points(lidar_filepath(nusc, tok))
            bev_path = out_viz / f"{tok}_bev.png"
            draw_bev(pts, lidar_objs_by_token[tok], bev_path, title=f"{args.scene} {tok[:8]}… BEV")
            summary["viz_pngs"].append(str(bev_path.resolve()))

            cam_path, _ = cam_filepath(nusc, tok, "CAM_FRONT")
            cam_out = out_viz / f"{tok}_cam_front.png"
            draw_cam_overlay(nusc, tok, lidar_objs_by_token[tok], cam_path, cam_out)
            summary["viz_pngs"].append(str(cam_out.resolve()))

        # read-back validate only labels written in this run (ignore stale JSONs)
        for tok in tokens:
            fp = out_pseudo / f"{tok}.json"
            lab = read_label(fp)
            validate_label(lab)
            if not lab["meta"].get("is_fake", False):
                raise ValueError(f"{fp} meta.is_fake must be true for smoke")
            summary["n_validated"] += 1

        # index / summary files
        index_lines = [
            f"<html><head><meta charset='utf-8'><title>smoke {args.scene}</title></head><body>",
            f"<h1>AutoLabel4D smoke — {args.scene}</h1>",
            f"<p>samples={len(tokens)} written={summary['n_written']} validated={summary['n_validated']}</p>",
            "<ul>",
        ]
        for p in summary["viz_pngs"]:
            name = Path(p).name
            index_lines.append(f'<li><a href="{name}">{name}</a><br><img src="{name}" style="max-width:480px"/></li>')
        index_lines += ["</ul></body></html>"]
        (out_viz / "index.html").write_text("\n".join(index_lines), encoding="utf-8")
        if summary["n_written"] == summary["n_validated"] == len(tokens) and len(summary["viz_pngs"]) >= 2 * len(tokens):
            summary["status"] = "PASS"
        else:
            summary["errors"].append(
                f"count mismatch written={summary['n_written']} validated={summary['n_validated']} "
                f"tokens={len(tokens)} viz={len(summary['viz_pngs'])}"
            )

        (out_viz / "summary.txt").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as e:
        summary["status"] = "FAIL"
        summary["errors"].append(f"{type(e).__name__}: {e}")
        print(json.dumps(summary, indent=2), flush=True)
        print("FAIL", flush=True)
        return 1

    print(json.dumps(summary, indent=2), flush=True)
    print(summary["status"], flush=True)
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
