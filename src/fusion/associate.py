"""LiDAR-primary multi-modal association (Stage E stub).

Real fusion will project LiDAR boxes to cameras and match 2D dets / masks /
depth cues. This module only exposes a stable API plus cheap IoU / center-
distance placeholders for smoke tests — no GPU, no heavy models.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Default association hyper-params (smoke / placeholders).
DEFAULT_MAX_CENTER_DIST_M = 3.0
DEFAULT_MIN_BEV_IOU = 0.1


def box_bev_corners(translation: Sequence[float], size: Sequence[float], yaw: float) -> np.ndarray:
    """Axis-aligned-ish BEV rectangle corners from (cx,cy), (w,l), yaw (rad).

    size is nuScenes [width, length, height]. Returns (4, 2) XY corners.
    """
    cx, cy = float(translation[0]), float(translation[1])
    w, l = float(size[0]), float(size[1])
    c, s = np.cos(yaw), np.sin(yaw)
    # local corners: length along x, width along y (nuScenes vehicle frame-ish)
    corners = np.array(
        [
            [l / 2.0, w / 2.0],
            [l / 2.0, -w / 2.0],
            [-l / 2.0, -w / 2.0],
            [-l / 2.0, w / 2.0],
        ],
        dtype=np.float64,
    )
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return corners @ rot.T + np.array([cx, cy], dtype=np.float64)


def yaw_from_quat_wxyz(rotation: Sequence[float]) -> float:
    """Extract yaw (Z) from quaternion wxyz; identity → 0."""
    w, x, y, z = [float(v) for v in rotation]
    # yaw from quaternion (ZYX): atan2(2(wz+xy), 1-2(y^2+z^2)) simplified for z-up
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def bev_iou(box_a: Dict[str, Any], box_b: Dict[str, Any]) -> float:
    """Placeholder BEV IoU via axis-aligned bounding boxes of oriented corners.

    Not a true polygon IoU — good enough for association smoke stubs.
    """
    ya = yaw_from_quat_wxyz(box_a.get("rotation", [1, 0, 0, 0]))
    yb = yaw_from_quat_wxyz(box_b.get("rotation", [1, 0, 0, 0]))
    ca = box_bev_corners(box_a["translation"], box_a["size"], ya)
    cb = box_bev_corners(box_b["translation"], box_b["size"], yb)
    min_a, max_a = ca.min(axis=0), ca.max(axis=0)
    min_b, max_b = cb.min(axis=0), cb.max(axis=0)
    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    wh = np.maximum(inter_max - inter_min, 0.0)
    inter = float(wh[0] * wh[1])
    area_a = float(max(max_a[0] - min_a[0], 0.0) * max(max_a[1] - min_a[1], 0.0))
    area_b = float(max(max_b[0] - min_b[0], 0.0) * max(max_b[1] - min_b[1], 0.0))
    union = area_a + area_b - inter
    if union <= 1e-9:
        return 0.0
    return inter / union


def center_distance_xy(box_a: Dict[str, Any], box_b: Dict[str, Any]) -> float:
    """Horizontal center distance (meters) between two boxes."""
    ta = box_a["translation"]
    tb = box_b["translation"]
    dx = float(ta[0]) - float(tb[0])
    dy = float(ta[1]) - float(tb[1])
    return float(np.hypot(dx, dy))


def associate_lidar_primary(
    lidar_objects: Sequence[Dict[str, Any]],
    secondary_objects: Sequence[Dict[str, Any]] | None = None,
    *,
    max_center_dist_m: float = DEFAULT_MAX_CENTER_DIST_M,
    min_bev_iou: float = DEFAULT_MIN_BEV_IOU,
    prefer: str = "distance",
) -> List[Dict[str, Any]]:
    """Associate secondary cues onto each LiDAR box (LiDAR-primary).

    Parameters
    ----------
    lidar_objects :
        Primary 3D boxes (teacher / fused candidates).
    secondary_objects :
        Optional 2D/3D cues (DINO/SAM stubs). None → identity passthrough.
    prefer :
        ``\"distance\"`` (default) or ``\"iou\"`` for greedy matching cost.

    Returns
    -------
    list of dict
        One entry per lidar object::
            {
              \"lidar_idx\": int,
              \"secondary_idx\": int | None,
              \"score\": float,          # similarity (higher better)
              \"metric\": str,          # \"distance\" | \"iou\" | \"passthrough\"
              \"distance_m\": float | None,
              \"bev_iou\": float | None,
            }
    """
    lidar = list(lidar_objects or [])
    secondary = list(secondary_objects or [])
    if not secondary:
        return [
            {
                "lidar_idx": i,
                "secondary_idx": None,
                "score": 1.0,
                "metric": "passthrough",
                "distance_m": None,
                "bev_iou": None,
            }
            for i in range(len(lidar))
        ]

    used: set[int] = set()
    matches: List[Dict[str, Any]] = []
    for i, lobj in enumerate(lidar):
        best_j: Optional[int] = None
        best_score = -1.0
        best_dist: Optional[float] = None
        best_iou: Optional[float] = None
        for j, sobj in enumerate(secondary):
            if j in used:
                continue
            # Category soft gate when both present
            lc = str(lobj.get("category", ""))
            sc = str(sobj.get("category", sobj.get("detection_name", "")))
            if lc and sc and lc != sc and not _compat_cats(lc, sc):
                continue
            dist = center_distance_xy(lobj, sobj) if "translation" in sobj else None
            iou = bev_iou(lobj, sobj) if "translation" in sobj and "size" in sobj else None
            if prefer == "iou":
                if iou is None or iou < min_bev_iou:
                    continue
                score = float(iou)
            else:
                if dist is None or dist > max_center_dist_m:
                    continue
                score = float(1.0 / (1.0 + dist))
                if iou is not None and iou < min_bev_iou and dist > max_center_dist_m * 0.5:
                    continue
            if score > best_score:
                best_score = score
                best_j = j
                best_dist = dist
                best_iou = iou
        if best_j is not None:
            used.add(best_j)
        matches.append(
            {
                "lidar_idx": i,
                "secondary_idx": best_j,
                "score": float(best_score if best_j is not None else 0.0),
                "metric": prefer if best_j is not None else "unmatched",
                "distance_m": best_dist,
                "bev_iou": best_iou,
            }
        )
    return matches


def _compat_cats(a: str, b: str) -> bool:
    """Loose category compatibility for stub association."""
    groups = [
        {"car", "vehicle", "truck", "bus", "trailer", "construction_vehicle"},
        {"pedestrian", "person"},
        {"bicycle", "motorcycle", "cycle"},
    ]
    for g in groups:
        if a in g and b in g:
            return True
    return False


def apply_association(
    lidar_objects: Sequence[Dict[str, Any]],
    secondary_objects: Sequence[Dict[str, Any]] | None = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Return shallow-copied lidar objects annotated with association metadata.

    Attaches ``assoc`` key; does not mutate inputs.
    """
    matches = associate_lidar_primary(lidar_objects, secondary_objects, **kwargs)
    out: List[Dict[str, Any]] = []
    secondary = list(secondary_objects or [])
    for m in matches:
        obj = dict(lidar_objects[m["lidar_idx"]])
        obj["assoc"] = m
        if m["secondary_idx"] is not None and secondary:
            sec = secondary[m["secondary_idx"]]
            # Optional score bump when matched (stub fusion cue).
            if "score" in obj and "score" in sec:
                obj["score"] = float(min(1.0, 0.5 * float(obj["score"]) + 0.5 * float(sec["score"])))
            obj["source"] = str(obj.get("source", "lidar")) + "+assoc"
        out.append(obj)
    return out


__all__ = [
    "DEFAULT_MAX_CENTER_DIST_M",
    "DEFAULT_MIN_BEV_IOU",
    "box_bev_corners",
    "yaw_from_quat_wxyz",
    "bev_iou",
    "center_distance_xy",
    "associate_lidar_primary",
    "apply_association",
]
