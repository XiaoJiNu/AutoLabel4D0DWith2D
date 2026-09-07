#!/usr/bin/env python3
"""Stage B4 skeleton: static RoboSense subset audit → dataset_audit.json.

Reads configs/paths.yaml (robosense_subset + manifests) and rs_subset_v0*
manifests, plus configs/sensors_rs_4f_1l.yaml (RS_4F_1L).

If subset is missing/empty (download+extract not finished): write a pending
audit JSON with status=WAITING_FOR_SUBSET and exit 0 so the pipeline can
poll safely.

When subset has data: heuristic checks for hs64_path, forbid livox /
velodyne_path as top LiDAR (presence OK if hs64 preferred), four fisheye
(named CAM_*_OV or ≥4 numeric **/images/[0-9]* with mapping_unknown), and
GT isolation (pkl annos → PASS_WITH_WARNINGS; generator must not read).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = REPO_ROOT / "configs" / "paths.yaml"
DEFAULT_SENSORS = REPO_ROOT / "configs" / "sensors_rs_4f_1l.yaml"
DEFAULT_SUBSET_CFG = REPO_ROOT / "configs" / "rs_subset_v0.yaml"

# Path / key name heuristics for RoboSense media (refine after first extract).
HS64_KEY_HINTS = ("hs64_path", "hs64", "LIDAR_TOP")
HS64_PATH_GLOBS = ("*hs64*", "*HS64*", "*lidar_top*", "*LIDAR_TOP*")
FORBID_FIELD_HINTS = ("livox", "velodyne_path", "velodyne")
FISHEYE_DIR_HINTS = (
    "CAM_FRONT_OV",
    "CAM_LEFT_OV",
    "CAM_RIGHT_OV",
    "CAM_BACK_OV",
    "front_ov",
    "left_ov",
    "right_ov",
    "back_ov",
    "fisheye",
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
META_EXTS = {".json", ".pkl", ".pickle", ".yaml", ".yml"}
GT_FORBIDDEN_FOR_GENERATOR = ("annos.id", "annos['id']", "gt_boxes", "gt_names")


def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


def load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML required: pip install pyyaml")
    if not path.is_file():
        raise FileNotFoundError(f"YAML not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return data


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dir_nonempty(path: Path) -> bool:
    if not path.exists():
        return False
    if not path.is_dir():
        return False
    try:
        next(path.iterdir())
        return True
    except StopIteration:
        return False


def collect_manifests(manifests_dir: Path) -> Dict[str, Any]:
    """Load rs_subset_v0* draft manifests (txt / json) if present."""
    found: Dict[str, Any] = {"paths": [], "txt_preview": None, "plan": None, "probe": None}
    if not manifests_dir.is_dir():
        return found
    for p in sorted(manifests_dir.glob("rs_subset_v0*")):
        found["paths"].append(str(p))
        name = p.name
        try:
            if name.endswith(".txt"):
                found["txt_preview"] = p.read_text(encoding="utf-8", errors="replace")[:2000]
            elif name.endswith("_plan.json"):
                found["plan"] = json.loads(p.read_text(encoding="utf-8"))
            elif name.endswith("_probe.json"):
                found["probe"] = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            found.setdefault("read_errors", []).append({"path": str(p), "error": str(exc)})
    return found


def _walk_files(root: Path, max_files: int = 50000) -> List[Path]:
    out: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            out.append(Path(dirpath) / fn)
            if len(out) >= max_files:
                return out
    return out


def _collect_keys(obj: Any, prefix: str = "", depth: int = 0, max_depth: int = 4) -> Set[str]:
    keys: Set[str] = set()
    if depth > max_depth:
        return keys
    if isinstance(obj, dict):
        for k, v in obj.items():
            ks = str(k)
            path = f"{prefix}.{ks}" if prefix else ks
            keys.add(ks)
            keys.add(path)
            keys |= _collect_keys(v, path, depth + 1, max_depth)
    elif isinstance(obj, (list, tuple)) and obj and depth < max_depth:
        # sample first few elements
        for i, item in enumerate(obj[:3]):
            keys |= _collect_keys(item, f"{prefix}[{i}]", depth + 1, max_depth)
    return keys


def _safe_load_meta(path: Path, max_bytes: int = 8_000_000) -> Tuple[Optional[Any], Optional[str]]:
    try:
        if path.stat().st_size > max_bytes:
            return None, "skipped_large"
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as f:
                return json.load(f), None
        if path.suffix.lower() in {".pkl", ".pickle"}:
            with path.open("rb") as f:
                return pickle.load(f), None
        if path.suffix.lower() in {".yaml", ".yml"} and yaml is not None:
            with path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    return None, "unsupported"


def _find_numeric_image_cam_dirs(subset_dir: Path) -> Dict[str, Any]:
    """Detect **/images/[0-9]* layout (RoboSense numeric cam folders)."""
    numeric_dirs: Dict[str, Dict[str, Any]] = {}
    # Walk only directories named images, then list numeric children.
    for dirpath, dirnames, _filenames in os.walk(subset_dir):
        base = Path(dirpath).name
        if base != "images":
            continue
        for child in dirnames:
            if re.fullmatch(r"[0-9]+", child):
                cam_path = Path(dirpath) / child
                rel = str(cam_path.relative_to(subset_dir))
                # Count image files quickly (cap sample)
                n_img = 0
                sample = None
                try:
                    for fn in os.listdir(cam_path):
                        if Path(fn).suffix.lower() in IMAGE_EXTS:
                            n_img += 1
                            if sample is None:
                                sample = f"{rel}/{fn}"
                except OSError:
                    pass
                numeric_dirs[child] = {
                    "rel": rel,
                    "image_count": n_img,
                    "sample": sample,
                }
    ids_sorted = sorted(numeric_dirs.keys(), key=lambda x: int(x))
    return {
        "numeric_cam_ids": ids_sorted,
        "numeric_cam_count": len(ids_sorted),
        "dirs": numeric_dirs,
        "ok_four_plus": len(ids_sorted) >= 4,
    }


def _probe_split_pkl_gt(subset_dir: Path) -> Dict[str, Any]:
    """Probe splits/*.pkl for annos (may be large; prefer smaller val)."""
    splits = subset_dir / "splits"
    result: Dict[str, Any] = {
        "pkl_present": False,
        "pkl_paths": [],
        "has_annos": False,
        "has_annos_id": False,
        "sample_keys": [],
        "probed": None,
        "error": None,
    }
    if not splits.is_dir():
        # also accept pkls anywhere under subset
        pkls = sorted(subset_dir.rglob("*.pkl"))
    else:
        pkls = sorted(splits.glob("*.pkl"))
        if not pkls:
            pkls = sorted(subset_dir.rglob("*.pkl"))
    result["pkl_paths"] = [str(p.relative_to(subset_dir)) for p in pkls]
    result["pkl_present"] = bool(pkls)
    if not pkls:
        return result
    # Prefer smaller file for probe
    pkls_sorted = sorted(pkls, key=lambda p: p.stat().st_size)
    probe = pkls_sorted[0]
    result["probed"] = str(probe.relative_to(subset_dir))
    try:
        with probe.open("rb") as f:
            obj = pickle.load(f)
        sample = None
        if isinstance(obj, list) and obj:
            sample = obj[0]
        elif isinstance(obj, dict):
            # dict-of-frames or top-level
            sample = next(iter(obj.values())) if obj else None
            if not isinstance(sample, dict):
                sample = obj
        if isinstance(sample, dict):
            result["sample_keys"] = sorted(str(k) for k in sample.keys())
            if "annos" in sample:
                result["has_annos"] = True
                ann = sample.get("annos")
                if isinstance(ann, dict) and ("id" in ann):
                    result["has_annos_id"] = True
            if any(k in sample for k in ("gt_boxes", "gt_names", "gt_attr_label")):
                result["has_annos"] = True
        # known RoboSense schema also has hs64_path / velodyne_path on frames
        if isinstance(sample, dict):
            result["frame_has_hs64_path"] = "hs64_path" in sample
            result["frame_has_velodyne_path"] = "velodyne_path" in sample
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
        # Presence of official split PKLs still implies GT may exist
        if result["pkl_present"]:
            result["has_annos"] = True
            result["note"] = "pkl present; assume annos may exist (probe failed)"
    return result


def heuristic_scan_subset(
    subset_dir: Path,
    camera_channels: List[str],
    lidar_source: str,
    forbid_fields: List[str],
) -> Dict[str, Any]:
    """Heuristic presence checks once extract has populated subset/."""
    files = _walk_files(subset_dir)
    rels = [str(p.relative_to(subset_dir)) for p in files]
    lower_rels = [r.lower() for r in rels]
    warnings: List[str] = []

    # --- LiDAR: prefer hs64_path / hs64 media; forbid livox/velodyne as top ---
    hs64_path_hits: List[str] = []
    for p, rel in zip(files, rels):
        low = rel.lower()
        if "hs64" in low or "lidar_top" in low:
            hs64_path_hits.append(rel)
    meta_keys: Set[str] = set()
    meta_samples: List[str] = []
    for p in files:
        if p.suffix.lower() not in META_EXTS:
            continue
        if len(meta_samples) >= 40:
            break
        # Split PKLs are huge; skip full key harvest here (dedicated GT probe).
        rel_p = str(p.relative_to(subset_dir))
        if "splits/" in rel_p.replace("\\", "/") and p.suffix.lower() in {".pkl", ".pickle"}:
            meta_samples.append(rel_p + " (deferred_gt_probe)")
            continue
        obj, err = _safe_load_meta(p)
        meta_samples.append(rel_p)
        if obj is None:
            continue
        meta_keys |= _collect_keys(obj)

    gt_probe = _probe_split_pkl_gt(subset_dir)
    if gt_probe.get("frame_has_hs64_path"):
        meta_keys.add("hs64_path")
    if gt_probe.get("frame_has_velodyne_path"):
        meta_keys.add("velodyne_path")
    if gt_probe.get("has_annos"):
        meta_keys.add("annos")
    if gt_probe.get("has_annos_id"):
        meta_keys.add("annos.id")

    has_hs64_key = any(
        ("hs64_path" == k.lower()) or k.lower().endswith(".hs64_path") or k.lower() == "hs64"
        for k in meta_keys
    )
    has_hs64_media = bool(hs64_path_hits)
    hs64_ok = has_hs64_key or has_hs64_media or (lidar_source.lower() == "hs64" and has_hs64_media)

    forbid_hits: Dict[str, Any] = {}
    for field in forbid_fields:
        fl = field.lower()
        path_hits = [r for r in rels if fl in r.lower()]
        key_hits = [k for k in meta_keys if fl in k.lower()]
        # Also catch bare "livox" / "velodyne" path segments
        if not path_hits and fl in ("livox", "velodyne_path", "velodyne"):
            path_hits = [r for r in rels if fl.replace("_path", "") in r.lower()]
        if path_hits or key_hits:
            forbid_hits[field] = {
                "path_samples": path_hits[:10],
                "key_samples": key_hits[:10],
            }

    # Using livox/velodyne as TOP lidar is forbidden; presence OK if hs64 preferred.
    forbid_as_top_ok = True
    forbid_notes: List[str] = []
    for field, hits in forbid_hits.items():
        note = (
            f"Found '{field}' references (paths={len(hits.get('path_samples', []))} "
            f"keys={len(hits.get('key_samples', []))}); marked forbid_as_top — "
            f"must NOT use as top LiDAR (require {lidar_source}/hs64_path)."
        )
        forbid_notes.append(note)
        if has_hs64_key or has_hs64_media:
            warnings.append(f"forbid_as_top:{field}_present_hs64_preferred")
        else:
            # Hard fail only if hs64 missing AND forbidden field present
            forbid_as_top_ok = False

    # --- Four fisheye: named CAM_*_OV OR ≥4 numeric **/images/[0-9]* ---
    cam_presence: Dict[str, Dict[str, Any]] = {}
    for cam in camera_channels:
        cam_l = cam.lower()
        # strip CAM_ / _OV variants for looser path match
        tokens = [t for t in re.split(r"[_\-]+", cam_l) if t and t not in {"cam"}]
        path_hits = []
        for r, rl in zip(rels, lower_rels):
            if cam_l in rl or all(t in rl for t in tokens):
                path_hits.append(r)
        # also match FOV dir hints
        for hint in FISHEYE_DIR_HINTS:
            if hint.lower() in cam_l or cam_l in hint.lower():
                path_hits.extend([r for r, rl in zip(rels, lower_rels) if hint.lower() in rl])
        # unique
        uniq = sorted(set(path_hits))
        img_hits = [r for r in uniq if Path(r).suffix.lower() in IMAGE_EXTS]
        cam_presence[cam] = {
            "path_hits": len(uniq),
            "image_hits": len(img_hits),
            "samples": uniq[:5],
            "present": bool(uniq),
        }
    any_images = any(Path(r).suffix.lower() in IMAGE_EXTS for r in rels)
    named_fisheye_ok = bool(camera_channels) and all(v["present"] for v in cam_presence.values())
    numeric_layout = _find_numeric_image_cam_dirs(subset_dir)
    mapping_unknown = False
    if named_fisheye_ok:
        fisheye_ok = True
        fisheye_mode = "named_CAM_OV"
    elif numeric_layout["ok_four_plus"] and any_images:
        fisheye_ok = True
        mapping_unknown = True
        fisheye_mode = "numeric_images_dirs"
        warnings.append(
            "four_fisheye_mapping_unknown: "
            f">={4} numeric cam folders under **/images/[0-9]* "
            f"(ids={numeric_layout['numeric_cam_ids']}); "
            "CAM_*_OV name mapping not established"
        )
    else:
        fisheye_ok = False
        fisheye_mode = "missing"
    if not any_images:
        fisheye_ok = False  # media not extracted yet

    # --- GT isolation (policy note + split PKL probe) ---
    gt_key_hits = sorted(
        k
        for k in meta_keys
        if "annos" in k.lower()
        or k.lower().endswith(".id")
        or "gt_box" in k.lower()
        or k.lower() in {"id", "annos.id"}
    )
    for k in gt_probe.get("sample_keys") or []:
        if "anno" in k.lower() or k.lower() in {"gt_boxes", "gt_names", "id"}:
            if k not in gt_key_hits:
                gt_key_hits.append(k)
    gt_key_hits = sorted(set(gt_key_hits))
    gt_status = "NOTED"
    if gt_probe.get("pkl_present") and (gt_probe.get("has_annos") or gt_probe.get("has_annos_id")):
        gt_status = "PASS_WITH_WARNING"
        warnings.append(
            "gt_isolation: split PKL has annos(/id) — evaluation-only; "
            "generator/teacher MUST NOT use official GT boxes/IDs"
        )
    elif gt_probe.get("pkl_present"):
        gt_status = "PASS_WITH_WARNING"
        warnings.append(
            "gt_isolation: split PKL present — treat as possibly containing annos; "
            "generator must not read official GT"
        )
    gt_isolation = {
        "policy": (
            "Official GT boxes/IDs (including annos.id) are evaluation-only. "
            "Generator / teacher input manifests MUST NOT read annos.id or official boxes."
        ),
        "forbidden_for_generator": list(GT_FORBIDDEN_FOR_GENERATOR),
        "meta_key_hits_sample": gt_key_hits[:30],
        "generator_must_not_read_annos_id": True,
        "pkl_probe": gt_probe,
        "status": gt_status,
        "ok": True,  # isolation policy noted; never hard-fail presence of GT
    }

    checks = {
        "subset_nonempty": True,
        "hs64_path": {
            "required_lidar_source": lidar_source,
            "has_hs64_key": has_hs64_key,
            "has_hs64_media": has_hs64_media,
            "media_samples": hs64_path_hits[:10],
            "ok": bool(has_hs64_key or has_hs64_media),
            "note": "Force hs64_path / lidar_source=hs64; do not use livox or velodyne_path as top LiDAR.",
        },
        "forbid_livox_velodyne_as_top": {
            "forbid_fields": list(forbid_fields),
            "hits": forbid_hits,
            "ok": forbid_as_top_ok,
            "forbid_as_top": True,  # policy: livox/velodyne may exist but never as top
            "presence_ok_if_hs64_preferred": bool(has_hs64_key or has_hs64_media),
            "notes": forbid_notes,
        },
        "four_fisheye": {
            "required_channels": list(camera_channels),
            "presence": cam_presence,
            "numeric_layout": {
                "numeric_cam_ids": numeric_layout["numeric_cam_ids"],
                "numeric_cam_count": numeric_layout["numeric_cam_count"],
                "ok_four_plus": numeric_layout["ok_four_plus"],
                "dir_samples": {
                    k: {"rel": v["rel"], "image_count": v["image_count"], "sample": v["sample"]}
                    for k, v in list(numeric_layout["dirs"].items())[:12]
                },
            },
            "mapping_unknown": mapping_unknown,
            "mode": fisheye_mode,
            "any_images": any_images,
            "ok": fisheye_ok,
            "note": (
                "OK if named CAM_*_OV present, OR ≥4 numeric subdirs under **/images/[0-9]* "
                "(mapping_unknown recorded when numeric-only)."
            ),
        },
        "gt_isolation": gt_isolation,
    }

    failures = []
    if not checks["hs64_path"]["ok"]:
        failures.append("hs64_path_missing")
    if not checks["forbid_livox_velodyne_as_top"]["ok"]:
        failures.append("forbid_lidar_fields_used_without_hs64")
    if not checks["four_fisheye"]["ok"]:
        failures.append("four_fisheye_incomplete")

    if failures:
        status = "FAIL"
    elif warnings:
        status = "PASS_WITH_WARNINGS"
    else:
        status = "PASS"
    return {
        "file_count": len(files),
        "meta_keys_sample": sorted(meta_keys)[:80],
        "meta_files_sampled": meta_samples,
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
        "status": status,
    }


def build_pending_audit(
    *,
    subset_dir: Path,
    manifests_dir: Path,
    sensors: Dict[str, Any],
    subset_cfg: Optional[Dict[str, Any]],
    manifests: Dict[str, Any],
    reason: str,
) -> Dict[str, Any]:
    return {
        "stage": "B4",
        "name": "dataset_audit",
        "generated_at_utc": utc_now(),
        "status": "WAITING_FOR_SUBSET",
        "pending": True,
        "reason": reason,
        "paths": {
            "robosense_subset": str(subset_dir),
            "manifests": str(manifests_dir),
            "sensors_yaml": str(DEFAULT_SENSORS),
            "subset_cfg": str(DEFAULT_SUBSET_CFG),
        },
        "sensors_preset": {
            "preset_name": sensors.get("preset_name"),
            "camera_channels": sensors.get("camera_channels"),
            "lidar_source": sensors.get("lidar_source"),
            "lidar_channel": sensors.get("lidar_channel"),
            "forbid_lidar_fields": sensors.get("forbid_lidar_fields"),
            "notes": sensors.get("notes"),
        },
        "rs_subset_v0": {
            "config_name": (subset_cfg or {}).get("name"),
            "manifest_files": manifests.get("paths", []),
            "plan_estimated_gb": ((manifests.get("plan") or {}).get("selection") or {}).get(
                "estimated_gb"
            ),
        },
        "checks_planned": [
            "subset_nonempty",
            "hs64_path_present",
            "forbid_livox_and_velodyne_path_as_top_lidar",
            "four_fisheye_CAM_*_OV_presence_heuristics",
            "gt_isolation_generator_must_not_read_annos.id",
        ],
        "gt_isolation": {
            "policy": (
                "Official GT boxes/IDs (including annos.id) are evaluation-only. "
                "Generator / teacher input manifests MUST NOT read annos.id."
            ),
            "status": "PENDING_UNTIL_SUBSET",
        },
    }


def build_full_audit(
    *,
    subset_dir: Path,
    manifests_dir: Path,
    sensors: Dict[str, Any],
    subset_cfg: Optional[Dict[str, Any]],
    manifests: Dict[str, Any],
    scan: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "stage": "B4",
        "name": "dataset_audit",
        "generated_at_utc": utc_now(),
        "status": scan["status"],
        "pending": False,
        "paths": {
            "robosense_subset": str(subset_dir),
            "manifests": str(manifests_dir),
            "sensors_yaml": str(DEFAULT_SENSORS),
            "subset_cfg": str(DEFAULT_SUBSET_CFG),
        },
        "sensors_preset": {
            "preset_name": sensors.get("preset_name"),
            "camera_channels": sensors.get("camera_channels"),
            "lidar_source": sensors.get("lidar_source"),
            "lidar_channel": sensors.get("lidar_channel"),
            "forbid_lidar_fields": sensors.get("forbid_lidar_fields"),
            "notes": sensors.get("notes"),
        },
        "rs_subset_v0": {
            "config_name": (subset_cfg or {}).get("name"),
            "manifest_files": manifests.get("paths", []),
            "plan_estimated_gb": ((manifests.get("plan") or {}).get("selection") or {}).get(
                "estimated_gb"
            ),
        },
        "scan": scan,
        "gt_isolation": scan["checks"]["gt_isolation"],
        "failures": scan.get("failures", []),
        "warnings": scan.get("warnings", []),
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage B4 RoboSense subset static audit skeleton")
    p.add_argument("--paths", type=Path, default=DEFAULT_PATHS, help="configs/paths.yaml")
    p.add_argument(
        "--sensors",
        type=Path,
        default=DEFAULT_SENSORS,
        help="configs/sensors_rs_4f_1l.yaml",
    )
    p.add_argument(
        "--subset-cfg",
        type=Path,
        default=DEFAULT_SUBSET_CFG,
        help="configs/rs_subset_v0.yaml",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON (default: <manifests>/dataset_audit.json)",
    )
    p.add_argument(
        "--subset",
        type=Path,
        default=None,
        help="Override robosense_subset path from paths.yaml",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    paths = load_yaml(args.paths)
    sensors = load_yaml(args.sensors)
    subset_cfg: Optional[Dict[str, Any]] = None
    if args.subset_cfg.is_file():
        try:
            subset_cfg = load_yaml(args.subset_cfg)
        except Exception as exc:  # noqa: BLE001
            eprint(f"WARN: could not load subset cfg: {exc}")

    subset_dir = Path(args.subset) if args.subset else Path(paths["robosense_subset"])
    manifests_dir = Path(
        paths.get("manifests", "/data/data/automomous/autolabel4d/manifests")
    )
    out_path = args.out or (manifests_dir / "dataset_audit.json")

    manifests = collect_manifests(manifests_dir)

    camera_channels = [str(c) for c in (sensors.get("camera_channels") or [])]
    lidar_source = str(sensors.get("lidar_source") or "hs64")
    forbid_fields = [str(x) for x in (sensors.get("forbid_lidar_fields") or [])]

    print(f"[B4] subset_dir={subset_dir}")
    print(f"[B4] manifests_dir={manifests_dir}")
    print(f"[B4] sensors preset={sensors.get('preset_name')} lidar_source={lidar_source}")
    print(f"[B4] cameras={camera_channels}")
    print(f"[B4] forbid={forbid_fields}")
    print(f"[B4] rs_subset_v0 manifests: {len(manifests.get('paths') or [])}")

    if not dir_nonempty(subset_dir):
        reason = (
            f"robosense_subset empty or missing ({subset_dir}); "
            "waiting for download+extract to finish"
        )
        audit = build_pending_audit(
            subset_dir=subset_dir,
            manifests_dir=manifests_dir,
            sensors=sensors,
            subset_cfg=subset_cfg,
            manifests=manifests,
            reason=reason,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"status=WAITING_FOR_SUBSET")
        print(f"wrote {out_path}")
        return 0

    scan = heuristic_scan_subset(subset_dir, camera_channels, lidar_source, forbid_fields)
    audit = build_full_audit(
        subset_dir=subset_dir,
        manifests_dir=manifests_dir,
        sensors=sensors,
        subset_cfg=subset_cfg,
        manifests=manifests,
        scan=scan,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"status={audit['status']}")
    if audit.get("failures"):
        print(f"failures={audit['failures']}")
    warnings = (audit.get("scan") or {}).get("warnings") or audit.get("warnings") or []
    if warnings:
        print(f"warnings={warnings}")
    print(f"wrote {out_path}")
    # Non-zero only on hard FAIL when data is present (pending always 0).
    return 0 if audit["status"] in ("PASS", "PASS_WITH_WARNINGS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
