#!/usr/bin/env python3
"""LiDAR-only Stage E on RoboSense SWEEPER-001 CenterPoint caches.

- Reads lidar_centerpoint_*.json under rs_repr_lidar/
- Filters by score threshold (default 0.3)
- Association passthrough (no GDINO/SAM), quality A/B/C, greedy track_id
- Exports nuScenes-style results + per-frame pseudo labels
- Optional BEV overlays

Also exposes helpers used by threshold post-process.
No GDINO download / multipart / kitti / GPU train.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from fusion.associate import associate_lidar_primary  # noqa: E402
from fusion.quality import apply_quality, summarize_quality  # noqa: E402
from fusion.track import assign_tracks  # noqa: E402
from teachers.lidar_centerpoint import load_hs64_bin, render_bev_png  # noqa: E402

DEFAULT_CP_DIR = Path(
    "/data/data/automomous/autolabel4d/pseudo_labels/rs_repr_lidar"
)
DEFAULT_OUT_PSEUDO = Path(
    "/data/data/automomous/autolabel4d/pseudo_labels/rs_lidar_only_export"
)
DEFAULT_OUT_VIZ = _REPO_ROOT / "outputs" / "rs_lidar_only_export"
DEFAULT_SCENE = "SWEEPER-001"
PC_RANGE = (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0)

_TS_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{3})$"
)


def parse_rs_timestamp(ts: Any) -> int:
    """Convert RS token/timestamp string → int microseconds since epoch (UTC)."""
    if isinstance(ts, int):
        return ts
    if isinstance(ts, float):
        return int(ts)
    s = str(ts or "").strip()
    m = _TS_RE.match(s)
    if m:
        y, mo, d, h, mi, sec, ms = map(int, m.groups())
        dt = datetime(y, mo, d, h, mi, sec, ms * 1000, tzinfo=timezone.utc)
        return int(dt.timestamp() * 1_000_000)
    # fallback: stable hash-ish from digits
    digits = re.sub(r"\D", "", s)
    if digits:
        return int(digits[:18])
    return 0


def list_cp_jsons(
    cp_dir: Path,
    *,
    scene: str = DEFAULT_SCENE,
    skip_smoke: bool = True,
) -> List[Path]:
    files = sorted(cp_dir.glob("lidar_centerpoint_*.json"))
    out: List[Path] = []
    for p in files:
        if skip_smoke and "smoke" in p.stem.lower():
            continue
        try:
            with p.open("r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        if d.get("status") not in (None, "OK", "ok", "Ok"):
            # still allow if status missing but objects present
            if d.get("status") and str(d.get("status")).upper() != "OK":
                continue
        if scene and d.get("scene_name") and d.get("scene_name") != scene:
            continue
        out.append(p)
    return out


def filter_objects_by_score(
    objects: Sequence[Dict[str, Any]], thresh: float
) -> List[Dict[str, Any]]:
    return [
        dict(o)
        for o in (objects or [])
        if float(o.get("score", 0.0)) >= float(thresh)
    ]


def load_cp_label(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_filtered_cp_label(
    src: Dict[str, Any],
    out_path: Path,
    *,
    thresh: float,
    generator: str,
) -> Dict[str, Any]:
    objs = filter_objects_by_score(src.get("objects") or [], thresh)
    out = dict(src)
    out["objects"] = objs
    meta = dict(src.get("meta") or {})
    meta["generator"] = generator
    meta["is_fake"] = bool(meta.get("is_fake", False))
    meta["notes"] = (
        f"Score-threshold filter thr>={thresh} from CenterPoint cache; "
        f"n_before={len(src.get('objects') or [])} n_after={len(objs)}. "
        + str(meta.get("notes") or "")
    )
    meta["score_thresh_post"] = float(thresh)
    meta["n_objects_before"] = len(src.get("objects") or [])
    meta["n_objects"] = len(objs)
    out["meta"] = meta
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return out


def maybe_render_bev(
    label: Dict[str, Any],
    out_png: Path,
    *,
    title: str = "",
) -> Optional[Path]:
    lidar_path = label.get("lidar_path") or ""
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
            print(f"[bev] points load failed for {lidar_path}: {exc}")
    try:
        return render_bev_png(
            points,
            list(label.get("objects") or []),
            out_png,
            title=title
            or f"{label.get('sample_token')} n={len(label.get('objects') or [])}",
            pc_range=PC_RANGE,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[bev] render failed: {exc}")
        return None


def postprocess_thresholds(
    cp_dir: Path,
    *,
    thresholds: Sequence[float] = (0.3, 0.5),
    scene: str = DEFAULT_SCENE,
    pseudo_root: Path = Path("/data/data/automomous/autolabel4d/pseudo_labels"),
    viz_root: Path = _REPO_ROOT / "outputs",
    do_bev: bool = True,
) -> Dict[str, Any]:
    """Filter CP JSONs at given thresholds; write thr dirs + BEV overlays."""
    files = list_cp_jsons(cp_dir, scene=scene, skip_smoke=True)
    summary: Dict[str, Any] = {"frames": [], "thresholds": {}}
    for thr in thresholds:
        # 0.3 -> "03", 0.5 -> "05" (dirs: rs_repr_lidar_thr{03,05})
        tag = f"{int(round(thr * 10)):02d}"

        out_pseudo = pseudo_root / f"rs_repr_lidar_thr{tag}"
        out_viz = viz_root / f"rs_repr_lidar_thr{tag}"
        out_pseudo.mkdir(parents=True, exist_ok=True)
        if do_bev:
            out_viz.mkdir(parents=True, exist_ok=True)

        thr_rows = []
        for p in files:
            src = load_cp_label(p)
            token = str(src.get("sample_token") or p.stem.replace("lidar_centerpoint_", ""))
            n_before = len(src.get("objects") or [])
            out_path = out_pseudo / f"lidar_centerpoint_{token}.json"
            filtered = write_filtered_cp_label(
                src,
                out_path,
                thresh=thr,
                generator="scripts/run_rs_lidar_only_export.py:postprocess_thresholds",
            )
            n_after = len(filtered.get("objects") or [])
            bev_path = None
            if do_bev:
                bev_path = maybe_render_bev(
                    filtered,
                    out_viz / f"bev_{token}.png",
                    title=f"{token} thr>={thr} n={n_after}/{n_before}",
                )
            thr_rows.append(
                {
                    "token": token,
                    "n_before": n_before,
                    "n_after": n_after,
                    "json": str(out_path),
                    "bev": str(bev_path) if bev_path else None,
                }
            )
            print(
                f"[thr {thr}] {token}: {n_before} → {n_after} → {out_path.name}"
            )
        summary["thresholds"][str(thr)] = {
            "tag": tag,
            "pseudo_dir": str(out_pseudo),
            "viz_dir": str(out_viz),
            "frames": thr_rows,
            "n_boxes_before": sum(r["n_before"] for r in thr_rows),
            "n_boxes_after": sum(r["n_after"] for r in thr_rows),
        }
    # also record raw 0.1 baseline counts
    baseline = []
    for p in files:
        src = load_cp_label(p)
        token = str(src.get("sample_token") or p.stem.replace("lidar_centerpoint_", ""))
        n = len(src.get("objects") or [])
        baseline.append({"token": token, "n": n, "path": str(p)})
    summary["baseline_thr0.1"] = {
        "frames": baseline,
        "n_boxes": sum(b["n"] for b in baseline),
        "cp_dir": str(cp_dir),
    }
    return summary


def _to_schema_sample(
    src: Dict[str, Any],
    objects: List[Dict[str, Any]],
    *,
    generator: str,
    notes: str,
) -> Dict[str, Any]:
    token = str(src["sample_token"])
    meta_src = dict(src.get("meta") or {})
    sample = {
        "version": str(src.get("version", "1.0")),
        "sample_token": token,
        "scene_name": str(src.get("scene_name", DEFAULT_SCENE)),
        "timestamp": parse_rs_timestamp(src.get("timestamp", token)),
        "lidar_sd_token": str(src.get("lidar_sd_token") or ""),
        "objects": objects,
        "meta": {
            "generator": generator,
            "is_fake": bool(meta_src.get("is_fake", False)),
            "notes": notes,
            "lidar_path": src.get("lidar_path"),
            "source_status": src.get("status"),
            "partial_subset": src.get("partial_subset"),
            "is_rs_partial": src.get("is_rs_partial"),
        },
    }
    return sample


def assign_tracks_with_time_gap(
    samples: List[Dict[str, Any]],
    *,
    max_match_dist_m: float = 4.0,
    max_time_gap_us: int = 5_000_000,
    prefix: str = "rs",
) -> List[Dict[str, Any]]:
    """Greedy track_id, but start a new track namespace when time gap is large.

    Representative RS frames are sparse across hours/days; linking across a
    multi-hour gap would create spurious multi-frame tracks.
    """
    if not samples:
        return []
    samples = sorted(samples, key=lambda s: (s["timestamp"], s["sample_token"]))
    segments: List[List[Dict[str, Any]]] = [[samples[0]]]
    for s in samples[1:]:
        prev_ts = int(segments[-1][-1]["timestamp"])
        cur_ts = int(s["timestamp"])
        if cur_ts - prev_ts > int(max_time_gap_us):
            segments.append([s])
        else:
            segments[-1].append(s)

    out: List[Dict[str, Any]] = []
    id_offset = 0
    for seg in segments:
        linked = assign_tracks(
            seg, mode="greedy", max_match_dist_m=max_match_dist_m, prefix=prefix
        )
        # Re-prefix track ids to be globally unique across segments
        remapped: Dict[str, str] = {}
        next_local = 0
        for s in linked:
            objs = []
            for o in s.get("objects") or []:
                o = dict(o)
                old_tid = str(o.get("track_id", ""))
                if old_tid not in remapped:
                    remapped[old_tid] = f"{prefix}_{id_offset + next_local:03d}"
                    next_local += 1
                o["track_id"] = remapped[old_tid]
                objs.append(o)
            s = dict(s)
            s["objects"] = objs
            out.append(s)
        id_offset += next_local
    return out


def run_stage_e(
    cp_dir: Path,
    *,
    score_thresh: float = 0.3,
    scene: str = DEFAULT_SCENE,
    out_pseudo: Path = DEFAULT_OUT_PSEUDO,
    out_viz: Path = DEFAULT_OUT_VIZ,
    max_match_dist_m: float = 4.0,
    max_time_gap_us: int = 5_000_000,
    do_bev: bool = True,
    include_empty: bool = True,
) -> Dict[str, Any]:
    files = list_cp_jsons(cp_dir, scene=scene, skip_smoke=True)
    if not files:
        raise FileNotFoundError(f"no CP JSON under {cp_dir} for scene={scene}")

    raw_samples: List[Dict[str, Any]] = []
    count_rows = []
    for p in files:
        src = load_cp_label(p)
        token = str(src["sample_token"])
        objs_all = list(src.get("objects") or [])
        objs = filter_objects_by_score(objs_all, score_thresh)
        if not include_empty and not objs:
            print(f"[E] skip empty after thr {token}")
            continue
        matches = associate_lidar_primary(objs, None)
        objs_q = apply_quality(
            objs, is_stub=False, association=matches, min_score_keep=score_thresh
        )
        sample = _to_schema_sample(
            src,
            objs_q,
            generator="scripts/run_rs_lidar_only_export.py",
            notes=(
                f"LiDAR-only Stage E on RS {scene}; CP thr>={score_thresh}; "
                "assoc passthrough; greedy track_id; no GDINO/SAM."
            ),
        )
        raw_samples.append(sample)
        count_rows.append(
            {
                "token": token,
                "n_raw": len(objs_all),
                "n_thr": len(objs),
                "quality": summarize_quality(objs_q),
            }
        )
        print(
            f"[E] {token}: raw={len(objs_all)} thr>={score_thresh}->{len(objs)} "
            f"{summarize_quality(objs_q)}"
        )

    # Chronological order by timestamp; reset tracks across large time gaps
    raw_samples.sort(key=lambda s: (s["timestamp"], s["sample_token"]))

    tracked = assign_tracks_with_time_gap(
        raw_samples,
        max_match_dist_m=max_match_dist_m,
        max_time_gap_us=max_time_gap_us,
        prefix="rs",
    )
    for s in tracked:
        validate_label(s)

    out_pseudo.mkdir(parents=True, exist_ok=True)
    written = write_fused_pseudo_labels(tracked, out_pseudo)

    results = labels_to_results(
        tracked,
        include_tracking=True,
        skip_ignore=False,
        meta={
            "is_fake": False,
            "scene": scene,
            "stage": "E",
            "mode": "lidar_only",
            "score_thresh": score_thresh,
            "max_time_gap_us": max_time_gap_us,
            "generator": "scripts/run_rs_lidar_only_export.py",
            "use_lidar": True,
            "use_camera": False,
        },
    )
    results_path = out_pseudo / "nuscenes_results_rs_lidar_only.json"
    write_results_json(results_path, results)

    # track_id stats
    all_tids: List[str] = []
    tid_frames: Dict[str, int] = {}
    for s in tracked:
        seen = set()
        for o in s.get("objects") or []:
            tid = str(o.get("track_id", ""))
            if not tid:
                continue
            all_tids.append(tid)
            seen.add(tid)
        for tid in seen:
            tid_frames[tid] = tid_frames.get(tid, 0) + 1

    unique_tids = sorted(set(all_tids))
    multi_frame = {k: v for k, v in tid_frames.items() if v >= 2}
    track_stats = {
        "n_boxes_total": len(all_tids),
        "n_unique_track_ids": len(unique_tids),
        "n_multi_frame_tracks": len(multi_frame),
        "multi_frame_track_ids": sorted(multi_frame.keys()),
        "track_id_frame_counts": dict(sorted(tid_frames.items(), key=lambda kv: (-kv[1], kv[0]))),
        "frames": [
            {
                "token": s["sample_token"],
                "n_objects": len(s.get("objects") or []),
                "track_ids": [o.get("track_id") for o in (s.get("objects") or [])],
                "quality": summarize_quality(s.get("objects") or []),
            }
            for s in tracked
        ],
    }

    viz_paths: List[str] = []
    if do_bev:
        out_viz.mkdir(parents=True, exist_ok=True)
        for s in tracked:
            # attach lidar_path from meta for BEV
            lp = (s.get("meta") or {}).get("lidar_path")
            label_for_bev = dict(s)
            if lp:
                label_for_bev["lidar_path"] = lp
            bp = maybe_render_bev(
                label_for_bev,
                out_viz / f"bev_{s['sample_token']}.png",
                title=(
                    f"StageE lidar-only {s['sample_token']} "
                    f"n={len(s.get('objects') or [])} thr>={score_thresh}"
                ),
            )
            if bp:
                viz_paths.append(str(bp))

    report = {
        "scene": scene,
        "score_thresh": score_thresh,
        "n_frames": len(tracked),
        "cp_dir": str(cp_dir),
        "out_pseudo": str(out_pseudo),
        "results_json": str(results_path),
        "written_labels": [str(p) for p in written],
        "viz_dir": str(out_viz),
        "viz_paths": viz_paths,
        "per_frame_counts": count_rows,
        "track_stats": track_stats,
    }
    report_path = out_pseudo / "stage_e_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    report["report_path"] = str(report_path)
    print(f"[E] wrote {len(written)} labels → {out_pseudo}")
    print(f"[E] results → {results_path}")
    print(
        f"[E] tracks: unique={track_stats['n_unique_track_ids']} "
        f"multi_frame={track_stats['n_multi_frame_tracks']} "
        f"boxes={track_stats['n_boxes_total']}"
    )
    return report


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="RS LiDAR-only Stage E export")
    ap.add_argument("--cp-dir", default=str(DEFAULT_CP_DIR))
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--score-thresh", type=float, default=0.3)
    ap.add_argument("--out-pseudo", default=str(DEFAULT_OUT_PSEUDO))
    ap.add_argument("--out-viz", default=str(DEFAULT_OUT_VIZ))
    ap.add_argument("--max-match-dist-m", type=float, default=4.0)
    ap.add_argument(
        "--max-time-gap-s",
        type=float,
        default=5.0,
        help="Reset track ids when consecutive frames exceed this gap (seconds)",
    )
    ap.add_argument("--no-bev", action="store_true")
    ap.add_argument(
        "--postprocess-thresholds",
        action="store_true",
        help="Also write rs_repr_lidar_thr03/05 filtered caches + BEV",
    )
    ap.add_argument(
        "--thresholds-only",
        action="store_true",
        help="Only run threshold post-process (skip Stage E)",
    )
    ap.add_argument(
        "--skip-empty",
        action="store_true",
        help="Drop frames with zero boxes after threshold",
    )
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    cp_dir = Path(args.cp_dir)
    print(f"[rs_lidar_only] cp_dir={cp_dir} scene={args.scene}")

    thr_summary = None
    if args.postprocess_thresholds or args.thresholds_only:
        thr_summary = postprocess_thresholds(
            cp_dir,
            thresholds=(0.3, 0.5),
            scene=args.scene,
            do_bev=not args.no_bev,
        )
        print(
            "[thr] baseline 0.1 boxes=",
            thr_summary["baseline_thr0.1"]["n_boxes"],
        )
        for t, info in thr_summary["thresholds"].items():
            print(
                f"[thr] {t}: {info['n_boxes_before']} → {info['n_boxes_after']} "
                f"dir={info['pseudo_dir']}"
            )

    if args.thresholds_only:
        out = _REPO_ROOT / "outputs" / "rs_repr_lidar_thr_summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(thr_summary, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"[thr] summary → {out}")
        print("PASS")
        return 0

    report = run_stage_e(
        cp_dir,
        score_thresh=args.score_thresh,
        scene=args.scene,
        out_pseudo=Path(args.out_pseudo),
        out_viz=Path(args.out_viz),
        max_match_dist_m=args.max_match_dist_m,
        max_time_gap_us=int(args.max_time_gap_s * 1_000_000),
        do_bev=not args.no_bev,
        include_empty=not args.skip_empty,
    )
    if thr_summary is not None:
        report["threshold_postprocess"] = thr_summary
        with Path(report["report_path"]).open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            f.write("\n")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
