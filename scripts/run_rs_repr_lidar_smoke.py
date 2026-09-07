#!/usr/bin/env python3
"""Representative-frame LiDAR CenterPoint smoke on PARTIAL RoboSense subset.

Picks N hs64 bins from SWEEPER-001, runs CenterPoint (hednet-gpu / OpenPCDet),
writes JSON under pseudo_labels/rs_repr_lidar/ and BEV PNGs under outputs/rs_repr_lidar/.

Marks is_rs_partial=true. Does NOT download GDINO/SAM. No git push / no kitti touch.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

REPO = Path("/data/code/cv/AutoLabel/AutoLabel4D0DWith2D")
HS64_DIR = Path(
    "/data/data/automomous/robosense/subset/lidar_occ_trainval/"
    "processed_data_20230906/SWEEPER-001/hs64"
)
IMG_DIR = Path(
    "/data/data/automomous/robosense/subset/image_trainval/"
    "processed_data_20231011/SWEEPER-001/images/0"
)
OUT_JSON = Path("/data/data/automomous/autolabel4d/pseudo_labels/rs_repr_lidar")
OUT_VIZ = REPO / "outputs" / "rs_repr_lidar"
CST = timezone(timedelta(hours=8))


def nvidia_smi_snapshot() -> str:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=False,
        )
        return (r.stdout or r.stderr or "").strip()
    except Exception as e:
        return f"nvidia-smi failed: {e}"


def pick_frames(n: int = 4) -> list[Path]:
    bins = sorted(HS64_DIR.glob("*.bin"))
    if not bins:
        raise FileNotFoundError(f"No hs64 bins under {HS64_DIR}")
    # Prefer early contiguous frames from first day (same sweep neighborhood)
    idxs = [0, 10, 30, 60][:n]
    idxs = [i for i in idxs if i < len(bins)]
    while len(idxs) < min(n, len(bins)):
        j = len(idxs) * 15
        if j < len(bins) and j not in idxs:
            idxs.append(j)
        else:
            break
    return [bins[i] for i in idxs]


def nearby_image_note(bin_path: Path) -> dict:
    """Images are from a different calendar window on this PARTIAL subset."""
    imgs = sorted(IMG_DIR.glob("*.jpg")) if IMG_DIR.is_dir() else []
    return {
        "lidar_stem": bin_path.stem,
        "n_images_cam0": len(imgs),
        "image_first": imgs[0].name if imgs else None,
        "image_last": imgs[-1].name if imgs else None,
        "time_aligned": False,
        "reason": (
            "PARTIAL part_01: hs64 dates ~2023-09-07..09 vs images ~2023-09-28..10-08; "
            "no timestamp-matched neighbors in this extract."
        ),
    }


def bev_viz(xyz: np.ndarray, objects: list, out_png: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import matplotlib.transforms as transforms

    # downsample for speed
    if xyz.shape[0] > 80000:
        rng = np.random.default_rng(0)
        sel = rng.choice(xyz.shape[0], 80000, replace=False)
        pts = xyz[sel]
    else:
        pts = xyz
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    ax.scatter(pts[:, 0], pts[:, 1], s=0.15, c=pts[:, 2], cmap="viridis", alpha=0.5)
    for o in objects:
        x, y, z = o["translation"]
        w, l, h = o["size"]  # w,l,h
        yaw = o.get("yaw", 0.0)
        # rectangle centered at (x,y), size l x w along yaw
        rect = Rectangle((-l / 2, -w / 2), l, w, fill=False, edgecolor="red", linewidth=1.2)
        t = (
            transforms.Affine2D().rotate(yaw).translate(x, y) + ax.transData
        )
        rect.set_transform(t)
        ax.add_patch(rect)
        ax.text(x, y, o.get("category", "?")[:4], color="yellow", fontsize=7)
    ax.set_aspect("equal")
    ax.set_xlim(-54, 54)
    ax.set_ylim(-54, 54)
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def structural_stub_export(bin_paths: list[Path], reason: str) -> dict:
    """Dry structural export with real point counts when inference blocked."""
    OUT_JSON.mkdir(parents=True, exist_ok=True)
    OUT_VIZ.mkdir(parents=True, exist_ok=True)
    frames = []
    for p in bin_paths:
        xyz = np.fromfile(p, dtype=np.float64).reshape(-1, 3).astype(np.float32)
        token = p.stem
        payload = {
            "version": "1.0",
            "partial_subset": True,
            "is_rs_partial": True,
            "sample_token": token,
            "scene_name": "SWEEPER-001",
            "objects": [],
            "meta": {
                "generator": "scripts.run_rs_repr_lidar_smoke.structural_stub_export",
                "is_fake": True,
                "is_stub": True,
                "inference_ran": False,
                "blocker": reason,
                "n_points": int(xyz.shape[0]),
                "hs64_path": str(p),
                "nearby_images": nearby_image_note(p),
            },
        }
        out = OUT_JSON / f"lidar_centerpoint_STUB_{token}.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        bev_viz(xyz, [], OUT_VIZ / f"bev_stub_{token}.png", f"STUB {token} N={xyz.shape[0]}")
        frames.append({"token": token, "n_points": int(xyz.shape[0]), "json": str(out)})
    return {"mode": "structural_stub", "frames": frames, "blocker": reason}


def run_real(bin_paths: list[Path], score_thresh: float) -> dict:
    sys.path.insert(0, str(REPO / "src"))
    from teachers.lidar_centerpoint import LidarCenterPointTeacher

    teacher = LidarCenterPointTeacher(score_thresh=score_thresh, cache_root=OUT_JSON)
    msg = "LidarCenterPointTeacher (hednet-gpu + runtime site-packages)"
    smi_before = nvidia_smi_snapshot()
    t0 = time.time()
    teacher.load()
    results = []
    try:
        for p in bin_paths:
            frame = {
                "hs64_path": str(p),
                "sample_token": p.stem,
                "scene_name": "SWEEPER-001",
            }
            r = teacher.infer_frame(frame)
            xyz = np.fromfile(p, dtype=np.float64).reshape(-1, 3).astype(np.float32)
            png = OUT_VIZ / f"bev_{p.stem}.png"
            bev_viz(xyz, r["objects"], png, f"CP {p.stem} boxes={len(r['objects'])}")
            r["bev_png"] = str(png)
            r["nearby_images"] = nearby_image_note(p)
            results.append(r)
            print(
                f"FRAME {p.name} points={r.get('num_points') or r.get('meta',{}).get('n_points') or r.get('meta',{}).get('num_points')} "
                f"boxes={len(r['objects'])} -> {r['cache_path']}"
            )
    finally:
        teacher.unload()
        try:
            import torch
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    smi_after = nvidia_smi_snapshot()
    elapsed = time.time() - t0
    return {
        "mode": "real_inference",
        "elapsed_s": elapsed,
        "smi_before": smi_before,
        "smi_after": smi_after,
        "pcdet_msg": msg,
        "frames": [
            {
                "token": r["sample_token"],
                "n_points": r.get("num_points") or r.get("meta", {}).get("n_points") or r.get("meta", {}).get("num_points"),
                "n_boxes": len(r["objects"]),
                "json": r.get("cache_path"),
                "bev": r.get("bev_png"),
                "vram_peak_mb": r["meta"].get("vram_peak_mb"),
                "categories": sorted({o["category"] for o in r["objects"]}),
            }
            for r in results
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--score-thresh", type=float, default=0.1)
    ap.add_argument("--force-stub", action="store_true")
    args = ap.parse_args()

    OUT_JSON.mkdir(parents=True, exist_ok=True)
    OUT_VIZ.mkdir(parents=True, exist_ok=True)

    bins = pick_frames(args.n)
    print("PICKED", [b.name for b in bins])
    print("SMI_BEFORE", nvidia_smi_snapshot())

    report = {
        "artifact_kind": "rs_repr_lidar_smoke",
        "generated_at_cst": datetime.now(CST).isoformat(),
        "is_rs_partial": True,
        "partial_subset": True,
        "repo": str(REPO),
        "hs64_dir": str(HS64_DIR),
        "picked": [str(b) for b in bins],
        "image_alignment": nearby_image_note(bins[0]),
    }

    if args.force_stub:
        body = structural_stub_export(bins, reason="--force-stub")
    else:
        try:
            body = run_real(bins, score_thresh=args.score_thresh)
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            print("REAL_INFERENCE_FAILED", e)
            print(tb)
            body = structural_stub_export(bins, reason=f"{type(e).__name__}: {e}")
            body["traceback"] = tb

    report.update(body)
    report["smi_final"] = nvidia_smi_snapshot()
    report_path = OUT_JSON / "rs_repr_lidar_smoke_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    # also copy report under outputs
    out_rep = OUT_VIZ / "rs_repr_lidar_smoke_report.json"
    with out_rep.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("REPORT", report_path)
    print("MODE", report.get("mode"))
    print("SMI_FINAL", report["smi_final"])
    print("PASS" if report.get("mode") == "real_inference" else "PASS_STUB_FALLBACK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
