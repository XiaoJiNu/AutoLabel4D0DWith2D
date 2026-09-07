#!/usr/bin/env python3
"""Remap fused RS pseudo labels to nuScenes 10-class set + point-cloud/3D-box viz.

Reads rs_fused_export (or --src), remaps/drops non-nuScenes categories, writes
rs_fused_export_nuscenes, regenerates nuScenes-style results JSON, and renders
BEV + perspective wireframe screenshots (matplotlib) showing LiDAR points and
3D boxes with class/score/track_id labels.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

CST = timezone(timedelta(hours=8))

NUSC10 = [
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "traffic_cone",
    "barrier",
]
NUSC10_SET = frozenset(NUSC10)

CATEGORY_REMAP = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "trailer": "trailer",
    "construction_vehicle": "construction_vehicle",
    "construction": "construction_vehicle",
    "pedestrian": "pedestrian",
    "person": "pedestrian",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "cyclist": "bicycle",
    "traffic_cone": "traffic_cone",
    "cone": "traffic_cone",
    "barrier": "barrier",
    "vehicle": "car",
}

CLASS_COLORS = {
    "car": "#1f77b4",
    "truck": "#ff7f0e",
    "bus": "#2ca02c",
    "trailer": "#d62728",
    "construction_vehicle": "#9467bd",
    "pedestrian": "#e377c2",
    "motorcycle": "#8c564b",
    "bicycle": "#17becf",
    "traffic_cone": "#bcbd22",
    "barrier": "#7f7f7f",
}


def now_cst() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S UTC+08:00")


def remap_category(name: str) -> Optional[str]:
    key = str(name or "").strip().lower().replace(" ", "_")
    if key in CATEGORY_REMAP:
        return CATEGORY_REMAP[key]
    if key in NUSC10_SET:
        return key
    return None


def load_hs64_bin(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.float64)
    if raw.size % 3 != 0:
        raise ValueError(f"hs64 bin not float64 xyz: {path} n={raw.size}")
    xyz = raw.reshape(-1, 3).astype(np.float32)
    finite = np.isfinite(xyz).all(1)
    m = finite & (np.abs(xyz[:, 0]) < 200) & (np.abs(xyz[:, 1]) < 200) & (np.abs(xyz[:, 2]) < 50)
    m &= np.linalg.norm(xyz, axis=1) > 1e-3
    return xyz[m]


def yaw_from_obj(obj: Dict[str, Any]) -> float:
    if obj.get("yaw_lidar") is not None:
        return float(obj["yaw_lidar"])
    qw, qx, qy, qz = [float(x) for x in obj["rotation"]]
    return math.atan2(2.0 * (qw * qz), 1.0 - 2.0 * (qz * qz))


def box_corners_3d(obj: Dict[str, Any]) -> np.ndarray:
    """Return 8 corners (8x3) in LiDAR frame. size = w,l,h (nuScenes)."""
    x, y, z = [float(v) for v in obj["translation"]]
    w, l, h = [float(v) for v in obj["size"]]
    yaw = yaw_from_obj(obj)
    # corners in box frame: x forward (l), y left (w), z up (h)
    dx, dy, dz = l / 2.0, w / 2.0, h / 2.0
    corners = np.array(
        [
            [dx, dy, -dz],
            [dx, -dy, -dz],
            [-dx, -dy, -dz],
            [-dx, dy, -dz],
            [dx, dy, dz],
            [dx, -dy, dz],
            [-dx, -dy, dz],
            [-dx, dy, dz],
        ],
        dtype=np.float64,
    )
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return corners @ R.T + np.array([x, y, z], dtype=np.float64)


EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
    (4, 5), (5, 6), (6, 7), (7, 4),  # top
    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical
]


def remap_label(label: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    stats = collections.Counter()
    kept: List[Dict[str, Any]] = []
    for obj in label.get("objects") or []:
        raw = str(obj.get("category", ""))
        mapped = remap_category(raw)
        stats["in"] += 1
        if mapped is None:
            stats["dropped"] += 1
            stats[f"drop:{raw}"] += 1
            continue
        if mapped != raw:
            stats["remapped"] += 1
            stats[f"map:{raw}->{mapped}"] += 1
        else:
            stats["kept_same"] += 1
        o = dict(obj)
        o["category"] = mapped
        o["category_raw"] = raw
        kept.append(o)
        stats[f"out:{mapped}"] += 1
    out = dict(label)
    out["version"] = "1.1"
    out["objects"] = kept
    meta = dict(label.get("meta") or {})
    meta["generator"] = "scripts/remap_nuscenes_classes_viz.py"
    meta["category_whitelist"] = list(NUSC10)
    meta["notes"] = (
        (meta.get("notes") or "")
        + f" | remapped to nuScenes 10-class only @ {now_cst()}"
    ).strip(" |")
    out["meta"] = meta
    return out, dict(stats)


def labels_to_results(labels: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    results: Dict[str, List[Dict[str, Any]]] = {}
    for lab in labels:
        tok = str(lab["sample_token"])
        boxes = []
        for obj in lab.get("objects") or []:
            if bool(obj.get("ignore", False)):
                continue
            name = str(obj["category"])
            score = float(obj.get("score", 0.0))
            tid = str(obj.get("track_id", ""))
            boxes.append(
                {
                    "sample_token": tok,
                    "translation": [float(x) for x in obj["translation"]],
                    "size": [float(x) for x in obj["size"]],
                    "rotation": [float(x) for x in obj["rotation"]],
                    "velocity": [0.0, 0.0],
                    "detection_name": name,
                    "detection_score": score,
                    "attribute_name": "",
                    "tracking_id": tid,
                    "tracking_name": name,
                    "tracking_score": score,
                    "track_id": tid,
                    "quality": str(obj.get("quality", "C")),
                    "source": str(obj.get("source", "")),
                }
            )
        results[tok] = boxes
    return {
        "meta": {
            "use_camera": True,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "generator": "scripts/remap_nuscenes_classes_viz.py",
            "category_whitelist": list(NUSC10),
            "generated_cst": now_cst(),
        },
        "results": results,
    }


def subsample_points(xyz: np.ndarray, max_pts: int = 120000) -> np.ndarray:
    if len(xyz) <= max_pts:
        return xyz
    idx = np.linspace(0, len(xyz) - 1, max_pts).astype(np.int64)
    return xyz[idx]


def render_frame_png(
    points_xyz: np.ndarray,
    objects: Sequence[Dict[str, Any]],
    out_png: Path,
    title: str,
    pc_range: Sequence[float] = (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    pts = subsample_points(np.asarray(points_xyz)[:, :3], 100000)
    # range filter for display
    m = (
        (pts[:, 0] >= pc_range[0])
        & (pts[:, 0] <= pc_range[3])
        & (pts[:, 1] >= pc_range[1])
        & (pts[:, 1] <= pc_range[4])
        & (pts[:, 2] >= pc_range[2])
        & (pts[:, 2] <= pc_range[5])
    )
    pts = pts[m]

    fig = plt.figure(figsize=(16, 8), dpi=130)

    # ---- BEV ----
    ax0 = fig.add_subplot(1, 2, 1)
    if len(pts):
        ax0.scatter(pts[:, 0], pts[:, 1], s=0.12, c=pts[:, 2], cmap="viridis", alpha=0.4, linewidths=0)
    for obj in objects:
        cat = str(obj.get("category", "?"))
        color = CLASS_COLORS.get(cat, "crimson")
        corners = box_corners_3d(obj)
        # bottom rectangle
        bottom = corners[[0, 1, 2, 3, 0]]
        ax0.plot(bottom[:, 0], bottom[:, 1], color=color, lw=1.4)
        # heading
        front = corners[[0, 1]].mean(axis=0)
        center = corners.mean(axis=0)
        ax0.plot([center[0], front[0]], [center[1], front[1]], color=color, lw=1.6)
        score = float(obj.get("score", 0.0))
        tid = str(obj.get("track_id", ""))[-6:]
        ax0.text(
            center[0],
            center[1],
            f"{cat}\n{score:.2f}/{tid}",
            color=color,
            fontsize=6,
            ha="center",
            va="bottom",
            zorder=5,
        )
    ax0.set_xlim(pc_range[0], pc_range[3])
    ax0.set_ylim(pc_range[1], pc_range[4])
    ax0.set_aspect("equal")
    ax0.set_xlabel("x (m, lidar)")
    ax0.set_ylabel("y (m, lidar)")
    ax0.set_title(f"BEV | n_box={len(objects)}")
    ax0.grid(True, alpha=0.25)

    # ---- 3D perspective ----
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    if len(pts):
        # further subsample for 3D speed
        p3 = subsample_points(pts, 40000)
        ax1.scatter(p3[:, 0], p3[:, 1], p3[:, 2], s=0.08, c=p3[:, 2], cmap="viridis", alpha=0.35, linewidths=0)
    for obj in objects:
        cat = str(obj.get("category", "?"))
        color = CLASS_COLORS.get(cat, "crimson")
        corners = box_corners_3d(obj)
        for i, j in EDGES:
            ax1.plot(
                [corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]],
                color=color,
                lw=1.2,
            )
        center = corners.mean(axis=0)
        score = float(obj.get("score", 0.0))
        ax1.text(center[0], center[1], center[2] + 0.3, f"{cat} {score:.2f}", color=color, fontsize=6)
    ax1.set_xlim(pc_range[0], pc_range[3])
    ax1.set_ylim(pc_range[1], pc_range[4])
    ax1.set_zlim(pc_range[2], pc_range[5])
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.view_init(elev=22, azim=-60)
    ax1.set_title("3D wireframe + point cloud")

    # legend
    present = sorted({str(o.get("category")) for o in objects})
    handles = [
        plt.Line2D([0], [0], color=CLASS_COLORS.get(c, "k"), lw=2, label=c) for c in present
    ]
    if handles:
        ax0.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.85)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        default="/data/data/automomous/autolabel4d/pseudo_labels/rs_fused_export",
    )
    ap.add_argument(
        "--dst",
        default="/data/data/automomous/autolabel4d/pseudo_labels/rs_fused_export_nuscenes",
    )
    ap.add_argument(
        "--viz-dir",
        default="/data/code/cv/AutoLabel/AutoLabel4D0DWith2D/outputs/rs_nuscenes_pc_viz",
    )
    ap.add_argument(
        "--copy-viz",
        default="/home/yr/grok-bot-work/autolabel4d_stageB/viz_nuscenes_classes",
    )
    ap.add_argument("--max-frames", type=int, default=12)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    viz_dir = Path(args.viz_dir)
    copy_viz = Path(args.copy_viz)
    dst.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    copy_viz.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("2023*.json"))[: args.max_frames]
    if not files:
        raise SystemExit(f"no frame json in {src}")

    before = collections.Counter()
    after = collections.Counter()
    dropped_total = 0
    remapped_total = 0
    labels_out: List[Dict[str, Any]] = []
    viz_paths: List[str] = []
    per_frame = []

    for jf in files:
        lab = json.loads(jf.read_text())
        for o in lab.get("objects") or []:
            before[str(o.get("category"))] += 1
        out, stats = remap_label(lab)
        for o in out.get("objects") or []:
            after[str(o.get("category"))] += 1
        dropped_total += int(stats.get("dropped", 0))
        remapped_total += int(stats.get("remapped", 0))
        out_path = dst / jf.name
        out_path.write_text(json.dumps(out, indent=2) + "\n")
        labels_out.append(out)

        lidar_path = (out.get("meta") or {}).get("lidar_path")
        if not lidar_path or not Path(lidar_path).is_file():
            per_frame.append({"stem": jf.stem, "error": f"missing lidar {lidar_path}", "n_boxes": len(out["objects"])})
            continue
        xyz = load_hs64_bin(Path(lidar_path))
        title = f"{jf.stem} | nuScenes-10 | boxes={len(out['objects'])} | pts={len(xyz)}"
        png = viz_dir / f"pc3d_{jf.stem}.png"
        render_frame_png(xyz, out["objects"], png, title=title)
        # also BEV-only dense copy name for convenience
        shutil.copy2(png, copy_viz / png.name)
        viz_paths.append(str(png))
        per_frame.append(
            {
                "stem": jf.stem,
                "n_boxes": len(out["objects"]),
                "n_pts": int(len(xyz)),
                "lidar_path": lidar_path,
                "label_path": str(out_path),
                "viz_png": str(png),
                "viz_copy": str(copy_viz / png.name),
                "class_counts": dict(collections.Counter(o["category"] for o in out["objects"])),
            }
        )
        print(f"[ok] {jf.stem} boxes={len(out['objects'])} pts={len(xyz)} -> {png}")

    results = labels_to_results(labels_out)
    results_path = dst / "nuscenes_results_rs_fused_nuscenes.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n")

    report = {
        "generated_cst": now_cst(),
        "class_list_used": list(NUSC10),
        "class_list_note": "Official nuScenes detection/tracking 10-class set",
        "src": str(src),
        "dst": str(dst),
        "results_json": str(results_path),
        "viz_dir": str(viz_dir),
        "viz_copy_dir": str(copy_viz),
        "n_frames": len(files),
        "n_boxes_before": int(sum(before.values())),
        "n_boxes_after": int(sum(after.values())),
        "dropped_total": dropped_total,
        "remapped_total": remapped_total,
        "classes_before": dict(before),
        "classes_after": dict(after),
        "viz_paths": viz_paths,
        "per_frame": per_frame,
        "files_changed_repo": [
            "configs/label_schema.yaml",
            "src/export/nuscenes_tables.py",
            "src/teachers/dino_real.py",
            "scripts/remap_nuscenes_classes_viz.py",
        ],
    }
    report_path = dst / "remap_nuscenes_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    (copy_viz / "remap_nuscenes_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("n_frames", "n_boxes_before", "n_boxes_after", "dropped_total", "classes_after")}, indent=2))
    print("report", report_path)


if __name__ == "__main__":
    main()
