#!/usr/bin/env python3
"""LiDAR-primary fusion + image 2D/mask supplement → nuScenes-style export + viz.

Primary: CenterPoint thr>=0.3 on aligned frames.
Secondary: GDINO 2D boxes + SAM masks (CAM_FRONT), matched via projected 2D IoU.
Reuses fusion/* and export/*. Compare summary vs lidar-only export.
"""
from __future__ import annotations

import argparse
import base64
import json
import pickle
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
for p in (_SRC, _REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from export.nuscenes_tables import (  # noqa: E402
    labels_to_results,
    write_fused_pseudo_labels,
    write_results_json,
)
from export.pseudo_label_io import validate_label  # noqa: E402
from fusion.associate import apply_association, associate_lidar_primary  # noqa: E402
from fusion.quality import apply_quality, summarize_quality  # noqa: E402
from fusion.track import assign_tracks  # noqa: E402
from geometry.project import invert_rt, project_points_pinhole, transform_points  # noqa: E402
from teachers.lidar_centerpoint import load_hs64_bin, render_bev_png  # noqa: E402

# Import helpers from lidar-only export without requiring scripts package
import importlib.util as _ilu

_loe_path = _REPO_ROOT / "scripts" / "run_rs_lidar_only_export.py"
_spec = _ilu.spec_from_file_location("run_rs_lidar_only_export", _loe_path)
_loe = _ilu.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_loe)
assign_tracks_with_time_gap = _loe.assign_tracks_with_time_gap
filter_objects_by_score = _loe.filter_objects_by_score
parse_rs_timestamp = _loe.parse_rs_timestamp

CST = timezone(timedelta(hours=8))
DEFAULT_MANIFEST = Path(
    "/data/data/automomous/autolabel4d/manifests/rs_align_frames_v0.json"
)
DEFAULT_CP_DIR = Path(
    "/data/data/automomous/autolabel4d/pseudo_labels/rs_align_lidar"
)
DEFAULT_CP_THR = Path(
    "/data/data/automomous/autolabel4d/pseudo_labels/rs_align_lidar_thr03"
)
DEFAULT_VISION = Path(
    "/data/data/automomous/autolabel4d/pseudo_labels/rs_vision_dino_sam"
)
DEFAULT_OUT = Path(
    "/data/data/automomous/autolabel4d/pseudo_labels/rs_fused_export"
)
DEFAULT_VIZ = _REPO_ROOT / "outputs" / "rs_fused_viz"
DEFAULT_LIDAR_ONLY = Path(
    "/data/data/automomous/autolabel4d/pseudo_labels/rs_lidar_only_export"
)
PC_RANGE = (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0)
DEFAULT_VAL_PKL = Path(
    "/data/data/automomous/robosense/subset/splits/robosense_local_val.pkl"
)
DEFAULT_TRAIN_PKL = Path(
    "/data/data/automomous/robosense/subset/splits/robosense_local_train.pkl"
)


def cst_now() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S %Z")


def index_pkl_by_stem(samples: List[dict]) -> Dict[str, dict]:
    idx: Dict[str, dict] = {}
    for s in samples:
        hs = Path(str(s.get("hs64_path", ""))).stem
        if hs:
            idx[hs] = s
        for cv in (s.get("images") or {}).get("cams", {}).values():
            if isinstance(cv, dict):
                st = Path(str(cv.get("data_path", ""))).stem
                if st and st not in idx:
                    idx[st] = s
    return idx


def load_pkl_index(val_pkl: Path, train_pkl: Path) -> Dict[str, dict]:
    samples: List[dict] = []
    for p in (val_pkl, train_pkl):
        if p.is_file():
            with p.open("rb") as f:
                samples.extend(pickle.load(f))
    return index_pkl_by_stem(samples)


def hs_to_cam(
    pts_hs: np.ndarray,
    hs2global: np.ndarray,
    ego2global_R: np.ndarray,
    ego2global_t: np.ndarray,
    sensor2ego_R: np.ndarray,
    sensor2ego_t: np.ndarray,
) -> np.ndarray:
    T = np.asarray(hs2global, dtype=np.float64).reshape(4, 4)
    R_hg, t_hg = T[:3, :3], T[:3, 3]
    pts_g = transform_points(pts_hs, R_hg, t_hg)
    R_eg = np.asarray(ego2global_R, dtype=np.float64).reshape(3, 3)
    t_eg = np.asarray(ego2global_t, dtype=np.float64).reshape(3)
    R_ge, t_ge = invert_rt(R_eg, t_eg)
    pts_ego = transform_points(pts_g, R_ge, t_ge)
    R_se = np.asarray(sensor2ego_R, dtype=np.float64).reshape(3, 3)
    t_se = np.asarray(sensor2ego_t, dtype=np.float64).reshape(3)
    R_es, t_es = invert_rt(R_se, t_se)
    return transform_points(pts_ego, R_es, t_es)


def box3d_corners_lidar(obj: Dict[str, Any]) -> np.ndarray:
    """8 corners in LiDAR frame from translation/size/yaw (w,l,h + quat wxyz)."""
    from fusion.associate import yaw_from_quat_wxyz

    cx, cy, cz = [float(v) for v in obj["translation"]]
    w, l, h = [float(v) for v in obj["size"]]
    yaw = yaw_from_quat_wxyz(obj.get("rotation", [1, 0, 0, 0]))
    x = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
    z = np.array([0, 0, 0, 0, h, h, h, h]) - h / 2.0
    c, s = np.cos(yaw), np.sin(yaw)
    xr = c * x - s * y + cx
    yr = s * x + c * y + cy
    zr = z + cz
    return np.stack([xr, yr, zr], axis=1)


def project_box_to_xyxy(
    obj: Dict[str, Any],
    sample: dict,
    cam_name: str = "CAM_FRONT",
    image_size: Optional[Tuple[int, int]] = None,
) -> Optional[List[float]]:
    cam = (sample.get("images") or {}).get("cams", {}).get(cam_name)
    if cam is None:
        return None
    corners = box3d_corners_lidar(obj)
    pts_cam = hs_to_cam(
        corners,
        sample["hs2global"],
        sample["ego2global_rotation"],
        sample["ego2global_translation"],
        cam["sensor2ego_rotation"],
        cam["sensor2ego_translation"],
    )
    K = np.asarray(cam["cam_intrinsic"], dtype=np.float64).reshape(3, 3)
    uv, mask, depth = project_points_pinhole(
        pts_cam, K, image_size=image_size, min_depth=0.3
    )
    if mask.sum() < 2:
        return None
    uvs = uv[mask]
    x1, y1 = float(uvs[:, 0].min()), float(uvs[:, 1].min())
    x2, y2 = float(uvs[:, 0].max()), float(uvs[:, 1].max())
    if image_size is not None:
        w, h = image_size
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(w - 1), x2), min(float(h - 1), y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-9 else 0.0


def associate_lidar_with_2d(
    lidar_objects: Sequence[Dict[str, Any]],
    dets_2d: Sequence[Dict[str, Any]],
    sample: Optional[dict],
    *,
    cam_name: str = "CAM_FRONT",
    image_size: Optional[Tuple[int, int]] = None,
    min_iou: float = 0.1,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Project LiDAR boxes to image; greedy 2D IoU match to DINO dets.

    Returns (matches, annotated_lidar_objects).
    """
    lidar = list(lidar_objects or [])
    dets = list(dets_2d or [])
    if sample is None or not dets:
        matches = associate_lidar_primary(lidar, None)
        return matches, [dict(o) for o in lidar]

    proj: List[Optional[List[float]]] = []
    for o in lidar:
        proj.append(project_box_to_xyxy(o, sample, cam_name=cam_name, image_size=image_size))

    used: set[int] = set()
    matches: List[Dict[str, Any]] = []
    annotated: List[Dict[str, Any]] = []
    order = sorted(range(len(lidar)), key=lambda i: float(lidar[i].get("score", 0.0)), reverse=True)
    assigned: Dict[int, Dict[str, Any]] = {}

    for i in order:
        best_j = None
        best_iou = min_iou
        pb = proj[i]
        if pb is not None:
            for j, d in enumerate(dets):
                if j in used:
                    continue
                # category soft gate
                lc = str(lidar[i].get("category", ""))
                sc = str(d.get("category", ""))
                if lc and sc and lc != sc:
                    # allow vehicle group
                    veh = {"car", "truck", "bus", "trailer", "construction_vehicle", "vehicle"}
                    if not (lc in veh and sc in veh):
                        if not (
                            (lc in {"pedestrian", "person"} and sc in {"pedestrian", "person"})
                            or (lc in {"bicycle", "motorcycle", "cyclist"} and sc in {"bicycle", "motorcycle", "cyclist"})
                        ):
                            continue
                iou = iou_xyxy(pb, d["bbox_xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
        m = {
            "lidar_idx": i,
            "secondary_idx": best_j,
            "score": float(best_iou if best_j is not None else 0.0),
            "metric": "image_iou" if best_j is not None else "unmatched",
            "distance_m": None,
            "bev_iou": None,
            "image_iou": float(best_iou) if best_j is not None else None,
            "proj_xyxy": pb,
        }
        if best_j is not None:
            used.add(best_j)
        assigned[i] = m

    for i, o in enumerate(lidar):
        m = assigned.get(i) or {
            "lidar_idx": i,
            "secondary_idx": None,
            "score": 0.0,
            "metric": "unmatched",
            "distance_m": None,
            "bev_iou": None,
        }
        matches.append(m)
        obj = dict(o)
        obj["assoc"] = m
        if m.get("proj_xyxy"):
            obj["proj_xyxy"] = m["proj_xyxy"]
        if m["secondary_idx"] is not None:
            sec = dets[m["secondary_idx"]]
            obj["source"] = str(obj.get("source", "lidar")) + "+dino"
            # mild score fusion
            if "score" in obj and "score" in sec:
                obj["score"] = float(
                    min(1.0, 0.7 * float(obj["score"]) + 0.3 * float(sec["score"]))
                )
            obj["dino_bbox_xyxy"] = list(sec.get("bbox_xyxy") or [])
            obj["dino_score"] = float(sec.get("score", 0.0))
            obj["has_image_support"] = True
        else:
            obj["has_image_support"] = False
        annotated.append(obj)
    return matches, annotated


def attach_masks(
    objects: List[Dict[str, Any]],
    masks: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach SAM mask metadata when dino_idx matches assoc secondary."""
    by_dino: Dict[int, Dict[str, Any]] = {}
    for m in masks or []:
        if "dino_idx" in m:
            by_dino[int(m["dino_idx"])] = m
    out = []
    for o in objects:
        obj = dict(o)
        assoc = obj.get("assoc") or {}
        sj = assoc.get("secondary_idx")
        if sj is not None and int(sj) in by_dino:
            mk = by_dino[int(sj)]
            obj["sam_mask_area"] = mk.get("mask_area")
            obj["sam_mask_bbox_xyxy"] = mk.get("mask_bbox_xyxy") or mk.get("bbox_xyxy")
            obj["source"] = str(obj.get("source", "lidar")) + "+sam"
            obj["has_mask"] = True
        else:
            obj["has_mask"] = False
        out.append(obj)
    return out


def _to_schema_sample(src: Dict[str, Any], objects: List[Dict[str, Any]], notes: str) -> Dict[str, Any]:
    token = str(src["sample_token"])
    meta_src = dict(src.get("meta") or {})
    return {
        "version": str(src.get("version", "1.0")),
        "sample_token": token,
        "scene_name": str(src.get("scene_name", "SWEEPER-001")),
        "timestamp": parse_rs_timestamp(src.get("timestamp", token)),
        "lidar_sd_token": str(src.get("lidar_sd_token") or ""),
        "objects": objects,
        "meta": {
            "generator": "scripts/run_rs_fused_export.py",
            "is_fake": bool(meta_src.get("is_fake", False)),
            "notes": notes,
            "lidar_path": src.get("lidar_path"),
            "use_camera": True,
            "use_lidar": True,
        },
    }


def decode_mask_crop(mask_entry: Dict[str, Any]) -> Optional[Tuple[np.ndarray, List[int]]]:
    b64 = mask_entry.get("mask_packbits_b64")
    bb = mask_entry.get("mask_bbox_xyxy")
    mh = mask_entry.get("mask_h")
    mw = mask_entry.get("mask_w")
    if not b64 or not bb or not mh or not mw:
        return None
    packed = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    flat = np.unpackbits(packed)[: int(mh) * int(mw)]
    crop = flat.reshape(int(mh), int(mw)).astype(np.uint8)
    return crop, [int(v) for v in bb]


def draw_cam_front_viz(
    image_path: Path,
    objects: List[Dict[str, Any]],
    dets_2d: List[Dict[str, Any]],
    masks: List[Dict[str, Any]],
    out_png: Path,
    title: str = "",
) -> Optional[Path]:
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    overlay = img.copy()
    # masks semi-transparent
    for mk in masks or []:
        decoded = decode_mask_crop(mk)
        if decoded is None:
            continue
        crop, (x0, y0, x1, y1) = decoded
        region = overlay[y0:y1, x0:x1]
        if region.shape[:2] != crop.shape:
            continue
        color = np.zeros_like(region)
        color[:, :] = (0, 200, 80)
        mask3 = crop.astype(bool)
        region[mask3] = (0.55 * region[mask3] + 0.45 * color[mask3]).astype(np.uint8)
        overlay[y0:y1, x0:x1] = region
    # DINO boxes (thin cyan)
    for d in dets_2d or []:
        x1, y1, x2, y2 = [int(v) for v in d["bbox_xyxy"]]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 200, 0), 1)
        cv2.putText(
            overlay,
            f"{d.get('category','?')}:{float(d.get('score',0)):.2f}",
            (x1, max(15, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 200, 0),
            1,
            cv2.LINE_AA,
        )
    # Projected / matched LiDAR boxes (green if matched else orange)
    for o in objects or []:
        bb = o.get("proj_xyxy") or o.get("dino_bbox_xyxy")
        if not bb:
            continue
        x1, y1, x2, y2 = [int(v) for v in bb]
        matched = bool(o.get("has_image_support"))
        col = (0, 255, 0) if matched else (0, 140, 255)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), col, 2)
        cv2.putText(
            overlay,
            f"{o.get('category','?')}:{float(o.get('score',0)):.2f}",
            (x1, min(overlay.shape[0] - 4, y2 + 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            col,
            1,
            cv2.LINE_AA,
        )
    if title:
        cv2.putText(
            overlay,
            title[:120],
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), overlay)
    return out_png


def maybe_bev(label: Dict[str, Any], out_png: Path, title: str = "") -> Optional[Path]:
    lidar_path = label.get("lidar_path") or (label.get("meta") or {}).get("lidar_path") or ""
    points = np.zeros((0, 3), dtype=np.float64)
    if lidar_path and Path(lidar_path).is_file():
        try:
            xyz = load_hs64_bin(lidar_path)
            x_min, y_min, z_min, x_max, y_max, z_max = PC_RANGE
            m = (
                (xyz[:, 0] >= x_min)
                & (xyz[:, 0] <= x_max)
                & (xyz[:, 1] >= y_min)
                & (xyz[:, 1] <= y_max)
                & (xyz[:, 2] >= z_min)
                & (xyz[:, 2] <= z_max)
            )
            points = xyz[m]
        except Exception as exc:  # noqa: BLE001
            print(f"[bev] load fail: {exc}")
    try:
        return render_bev_png(
            points,
            list(label.get("objects") or []),
            out_png,
            title=title or f"{label.get('sample_token')} n={len(label.get('objects') or [])}",
            pc_range=PC_RANGE,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[bev] render fail: {exc}")
        return None


def write_thr03_from_cp(cp_dir: Path, out_dir: Path, thresh: float = 0.3) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for p in sorted(cp_dir.glob("lidar_centerpoint_*.json")):
        if "smoke" in p.stem.lower():
            continue
        src = json.loads(p.read_text(encoding="utf-8"))
        if str(src.get("status", "OK")).upper() not in ("OK", "NONE", ""):
            if src.get("status") and str(src.get("status")).upper() != "OK":
                continue
        objs = filter_objects_by_score(src.get("objects") or [], thresh)
        out = dict(src)
        out["objects"] = objs
        meta = dict(src.get("meta") or {})
        meta["score_thresh_post"] = thresh
        meta["n_objects"] = len(objs)
        out["meta"] = meta
        op = out_dir / p.name
        op.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(op)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="RS fused LiDAR+vision export")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--cp-dir", type=Path, default=DEFAULT_CP_DIR)
    ap.add_argument("--cp-thr-dir", type=Path, default=DEFAULT_CP_THR)
    ap.add_argument("--vision-root", type=Path, default=DEFAULT_VISION)
    ap.add_argument("--out-pseudo", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--out-viz", type=Path, default=DEFAULT_VIZ)
    ap.add_argument("--lidar-only-dir", type=Path, default=DEFAULT_LIDAR_ONLY)
    ap.add_argument("--score-thresh", type=float, default=0.3)
    ap.add_argument("--cam-name", default="CAM_FRONT")
    ap.add_argument("--val-pkl", type=Path, default=DEFAULT_VAL_PKL)
    ap.add_argument("--train-pkl", type=Path, default=DEFAULT_TRAIN_PKL)
    ap.add_argument("--max-time-gap-s", type=float, default=5.0)
    ap.add_argument("--no-viz", action="store_true")
    args = ap.parse_args()

    man = json.loads(args.manifest.read_text(encoding="utf-8"))
    frames = list(man.get("frames") or [])
    pkl_idx = load_pkl_index(args.val_pkl, args.train_pkl)

    # Ensure thr03 caches exist for aligned CP
    if args.cp_dir.is_dir():
        write_thr03_from_cp(args.cp_dir, args.cp_thr_dir, args.score_thresh)

    dino_dir = args.vision_root / "dino"
    sam_dir = args.vision_root / "sam"
    args.out_pseudo.mkdir(parents=True, exist_ok=True)
    args.out_viz.mkdir(parents=True, exist_ok=True)

    raw_samples: List[Dict[str, Any]] = []
    count_rows = []
    viz_paths: List[str] = []

    for fr in frames:
        stem = str(fr.get("stem"))
        cp_path = args.cp_thr_dir / f"lidar_centerpoint_{stem}.json"
        if not cp_path.is_file():
            # try raw cp_dir
            cp_path = args.cp_dir / f"lidar_centerpoint_{stem}.json"
        row = {"token": stem, "n_lidar": 0, "n_dino": 0, "n_matched": 0, "ok": False, "error": None}
        if not cp_path.is_file():
            row["error"] = f"missing CP cache for {stem}"
            count_rows.append(row)
            print(f"[fuse] SKIP {stem}: no CP")
            continue
        src = json.loads(cp_path.read_text(encoding="utf-8"))
        objs = filter_objects_by_score(src.get("objects") or [], args.score_thresh)
        row["n_lidar"] = len(objs)

        dino_path = dino_dir / f"dino_{stem}_{args.cam_name}.json"
        sam_path = sam_dir / f"sam_{stem}_{args.cam_name}.json"
        dets = []
        masks = []
        img_size = None
        if dino_path.is_file():
            dino = json.loads(dino_path.read_text(encoding="utf-8"))
            dets = list(dino.get("detections_2d") or [])
            img_size_l = (dino.get("meta") or {}).get("image_size")
            if img_size_l and len(img_size_l) == 2:
                img_size = (int(img_size_l[0]), int(img_size_l[1]))  # w,h
        if sam_path.is_file():
            sam = json.loads(sam_path.read_text(encoding="utf-8"))
            masks = list(sam.get("masks") or [])
        row["n_dino"] = len(dets)

        sample_pkl = pkl_idx.get(stem)
        matches, annotated = associate_lidar_with_2d(
            objs,
            dets,
            sample_pkl,
            cam_name=args.cam_name,
            image_size=img_size,
            min_iou=0.1,
        )
        annotated = attach_masks(annotated, masks)
        row["n_matched"] = sum(1 for m in matches if m.get("secondary_idx") is not None)

        objs_q = apply_quality(
            annotated, is_stub=False, association=matches, min_score_keep=args.score_thresh
        )
        # bump quality slightly when image-supported
        for o in objs_q:
            if o.get("has_image_support") and o.get("quality") == "C" and float(o.get("score", 0)) >= 0.45:
                o["quality"] = "B"

        notes = (
            f"Fused Stage E on RS aligned frame; LiDAR thr>={args.score_thresh} + "
            f"{args.cam_name} DINO/SAM; matched={row['n_matched']}/{row['n_lidar']}."
        )
        sample = _to_schema_sample(src, objs_q, notes)
        # ensure lidar_path
        if not sample["meta"].get("lidar_path"):
            sample["meta"]["lidar_path"] = fr.get("hs64_path") or src.get("lidar_path")
        raw_samples.append(sample)
        row["ok"] = True
        row["quality"] = summarize_quality(objs_q)
        count_rows.append(row)
        print(
            f"[fuse] {stem}: lidar={row['n_lidar']} dino={row['n_dino']} "
            f"matched={row['n_matched']} {row['quality']}"
        )

        if not args.no_viz:
            img_path = (fr.get("images") or {}).get("0")
            if img_path and Path(img_path).is_file():
                vp = draw_cam_front_viz(
                    Path(img_path),
                    objs_q,
                    dets,
                    masks,
                    args.out_viz / f"cam_{stem}_{args.cam_name}.png",
                    title=f"{stem} fused matched={row['n_matched']}/{row['n_lidar']}",
                )
                if vp:
                    viz_paths.append(str(vp))
            # BEV
            label_bev = dict(sample)
            label_bev["lidar_path"] = sample["meta"].get("lidar_path")
            bp = maybe_bev(
                label_bev,
                args.out_viz / f"bev_{stem}.png",
                title=f"fused {stem} n={len(objs_q)}",
            )
            if bp:
                viz_paths.append(str(bp))

    raw_samples.sort(key=lambda s: (s["timestamp"], s["sample_token"]))
    tracked = assign_tracks_with_time_gap(
        raw_samples,
        max_match_dist_m=4.0,
        max_time_gap_us=int(args.max_time_gap_s * 1_000_000),
        prefix="rsf",
    )
    for s in tracked:
        # strip non-schema extras that break validate? keep required only — extras OK
        # But validate requires exact fields; extras fine. Ensure track_id etc present.
        clean_objs = []
        for o in s.get("objects") or []:
            oo = dict(o)
            # remove bulky non-schema keys from written labels? keep small cues
            for k in list(oo.keys()):
                if k.startswith("mask_pack"):
                    del oo[k]
            clean_objs.append(oo)
        s["objects"] = clean_objs
        validate_label(s)

    written = write_fused_pseudo_labels(tracked, args.out_pseudo)
    results = labels_to_results(
        tracked,
        include_tracking=True,
        skip_ignore=False,
        meta={
            "is_fake": False,
            "scene": "SWEEPER-001",
            "stage": "E",
            "mode": "lidar_primary_fused_vision",
            "score_thresh": args.score_thresh,
            "generator": "scripts/run_rs_fused_export.py",
            "use_lidar": True,
            "use_camera": True,
        },
    )
    results_path = args.out_pseudo / "nuscenes_results_rs_fused.json"
    write_results_json(results_path, results)

    # Compare to lidar-only
    compare = {"fused_frames": len(tracked), "fused_boxes": sum(len(s.get('objects') or []) for s in tracked)}
    if args.lidar_only_dir.is_dir():
        lo_files = [p for p in args.lidar_only_dir.glob("*.json") if p.name.startswith("20") or "objects" in p.read_text()[:200]]
        # safer recount
        n_lo_boxes = 0
        n_lo_frames = 0
        for p in sorted(args.lidar_only_dir.glob("*.json")):
            if p.name.startswith("nuscenes") or p.name.startswith("stage_"):
                continue
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if "objects" not in d:
                continue
            n_lo_frames += 1
            n_lo_boxes += len(d.get("objects") or [])
        compare["lidar_only_frames"] = n_lo_frames
        compare["lidar_only_boxes"] = n_lo_boxes
        compare["note"] = (
            "lidar-only export covers earlier repr stems; fused covers aligned "
            "image↔hs64 stems — frame sets differ; compare box-density / quality mixes."
        )

    report = {
        "generated_cst": cst_now(),
        "n_frames": len(tracked),
        "cp_dir": str(args.cp_dir),
        "cp_thr_dir": str(args.cp_thr_dir),
        "vision_root": str(args.vision_root),
        "out_pseudo": str(args.out_pseudo),
        "results_json": str(results_path),
        "written_labels": [str(p) for p in written],
        "viz_dir": str(args.out_viz),
        "viz_paths": viz_paths,
        "per_frame": count_rows,
        "compare_lidar_only": compare,
    }
    rp = args.out_pseudo / "stage_e_fused_report.json"
    rp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[fuse] wrote {len(written)} labels → {args.out_pseudo}")
    print(f"[fuse] compare: {compare}")
    print("PASS" if tracked else "FAIL")
    return 0 if tracked else 1


if __name__ == "__main__":
    raise SystemExit(main())
