"""Export AutoLabel4D pseudo labels → nuScenes-style results dict (+ track_id).

Reuses ``export.pseudo_label_io`` for read/validate/write of per-sample JSON.
Stage E thin skeleton — suitable for smoke / offline eval wiring, not full
nuScenes DB table generation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from export.pseudo_label_io import read_label, validate_label, write_label

# Official nuScenes detection / tracking 10-class set (OpenPCDet CLASS_NAMES order).
NUSCENES_DETECTION_CLASSES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]
NUSCENES_CLASS_SET = frozenset(NUSCENES_DETECTION_CLASSES)

# Free-text / coarse aliases → nuScenes class; missing key means drop when filtering.
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

DEFAULT_ATTR = ""


def remap_category(name: str, *, drop_unmapped: bool = True) -> Optional[str]:
    """Map a free-text / coarse label onto the nuScenes 10-class set."""
    key = str(name or "").strip().lower().replace(" ", "_")
    mapped = CATEGORY_REMAP.get(key)
    if mapped is not None:
        return mapped
    if key in NUSCENES_CLASS_SET:
        return key
    return None if drop_unmapped else key


def object_to_nusc_box(
    obj: Dict[str, Any],
    sample_token: str,
    *,
    include_tracking: bool = True,
    skip_ignore: bool = True,
) -> Optional[Dict[str, Any]]:
    """Convert one pseudo-label object to a nuScenes results box dict."""
    if skip_ignore and bool(obj.get("ignore", False)):
        return None
    translation = [float(x) for x in obj["translation"]]
    size = [float(x) for x in obj["size"]]
    rotation = [float(x) for x in obj["rotation"]]
    score = float(obj.get("score", 0.0))
    name = remap_category(str(obj.get("category", "car")), drop_unmapped=True)
    if name is None:
        return None
    box: Dict[str, Any] = {
        "sample_token": sample_token,
        "translation": translation,
        "size": size,
        "rotation": rotation,
        "velocity": [0.0, 0.0],
        "detection_name": name,
        "detection_score": score,
        "attribute_name": str(obj.get("attribute_name", DEFAULT_ATTR)),
        # Extra AutoLabel4D fields (harmless for custom loaders; strip if needed)
        "quality": str(obj.get("quality", "C")),
        "valid_fields": list(obj.get("valid_fields") or []),
        "ignore": bool(obj.get("ignore", False)),
        "source": str(obj.get("source", "")),
    }
    if include_tracking:
        tid = str(obj.get("track_id", ""))
        box["tracking_id"] = tid
        box["tracking_name"] = name
        box["tracking_score"] = score
        box["track_id"] = tid  # explicit first-class alias per project schema
    return box


def labels_to_results(
    labels: Sequence[Dict[str, Any]],
    *,
    include_tracking: bool = True,
    skip_ignore: bool = False,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a nuScenes-style ``{meta, results}`` dict from pseudo-label samples.

    ``results[sample_token]`` is a list of box dicts with ``track_id`` /
    ``tracking_id`` when ``include_tracking`` is True.
    """
    results: Dict[str, List[Dict[str, Any]]] = {}
    for label in labels:
        validate_label(label)
        tok = str(label["sample_token"])
        boxes: List[Dict[str, Any]] = []
        for obj in label.get("objects") or []:
            box = object_to_nusc_box(
                obj,
                tok,
                include_tracking=include_tracking,
                skip_ignore=skip_ignore,
            )
            if box is not None:
                boxes.append(box)
        results[tok] = boxes

    out_meta = {
        "use_camera": False,
        "use_lidar": True,
        "use_radar": False,
        "use_map": False,
        "use_external": False,
        "generator": "export.nuscenes_tables",
        "notes": "Stage E fusion/track/export smoke; stub teachers may set is_fake.",
    }
    if meta:
        out_meta.update(meta)
    return {"meta": out_meta, "results": results}


def write_results_json(path: str | Path, payload: Dict[str, Any], *, indent: int = 2) -> Path:
    """Write nuScenes-style results JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if "results" not in payload or "meta" not in payload:
        raise ValueError("payload must contain 'meta' and 'results'")
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)
        f.write("\n")
    return out


def read_results_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "results" not in data:
        raise ValueError(f"not a nuScenes results dict: {path}")
    return data


def export_pseudo_dir_to_results(
    pseudo_dir: str | Path,
    results_path: str | Path,
    *,
    include_tracking: bool = True,
    skip_ignore: bool = False,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load all ``*.json`` pseudo labels from a directory and export results."""
    d = Path(pseudo_dir)
    labels = [read_label(p) for p in sorted(d.glob("*.json"))]
    # Skip nested results files if accidentally present
    labels = [lb for lb in labels if "objects" in lb and "sample_token" in lb]
    payload = labels_to_results(
        labels,
        include_tracking=include_tracking,
        skip_ignore=skip_ignore,
        meta=meta,
    )
    write_results_json(results_path, payload)
    return payload


def write_fused_pseudo_labels(
    labels: Sequence[Dict[str, Any]],
    out_dir: str | Path,
) -> List[Path]:
    """Validate + write per-sample fused pseudo labels via pseudo_label_io."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for label in labels:
        tok = label["sample_token"]
        path = out / f"{tok}.json"
        write_label(path, label)
        paths.append(path)
    return paths


__all__ = [
    "object_to_nusc_box",
    "labels_to_results",
    "write_results_json",
    "read_results_json",
    "export_pseudo_dir_to_results",
    "write_fused_pseudo_labels",
]
