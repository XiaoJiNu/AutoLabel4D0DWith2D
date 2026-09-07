#!/usr/bin/env python3
"""Stage E smoke: quality + track + nuScenes-style export (N<=2, scene-0103).

Reads teacher_cache_smoke lidar_stub JSONs when present; otherwise regenerates
via LidarTeacherStub. No long GPU / no heavy models.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
for p in (_SRC, _REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data.paths import load_paths, repo_root  # noqa: E402
from export.nuscenes_tables import (  # noqa: E402
    labels_to_results,
    write_fused_pseudo_labels,
    write_results_json,
)
from export.pseudo_label_io import read_label, validate_label  # noqa: E402
from fusion.associate import associate_lidar_primary  # noqa: E402
from fusion.quality import apply_quality, summarize_quality  # noqa: E402
from fusion.track import assign_tracks  # noqa: E402


def _parse_args() -> argparse.Namespace:
    paths = load_paths()
    pseudo_root = Path(
        paths.get("pseudo_labels", "/data/data/automomous/autolabel4d/pseudo_labels")
    )
    default_cache = pseudo_root / "teacher_cache_smoke"
    default_out = pseudo_root / "fusion_export_smoke"
    default_viz = repo_root() / "outputs" / "fusion_export_smoke"
    default_dataroot = paths.get(
        "nuscenes_mini", "/data/data/automomous/nuscenes/v1.0-mini"
    )
    ap = argparse.ArgumentParser(description="Stage E fusion/track/export smoke")
    ap.add_argument("--scene", default="scene-0103")
    ap.add_argument("--max-samples", type=int, default=2)
    ap.add_argument("--cache-root", default=str(default_cache))
    ap.add_argument("--out-pseudo", default=str(default_out))
    ap.add_argument("--out-viz", default=str(default_viz))
    ap.add_argument("--dataroot", default=default_dataroot)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--no-viz", action="store_true")
    ap.add_argument(
        "--regenerate",
        action="store_true",
        help="Force lidar_stub regeneration even if cache exists",
    )
    return ap.parse_args()


def _load_or_regen_cache(
    args: argparse.Namespace, n: int
) -> List[Dict[str, Any]]:
    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(cache_root.glob("lidar_stub_*.json"))

    labels: List[Dict[str, Any]] = []
    if existing and not args.regenerate:
        for p in existing[:n]:
            raw = read_label(p)
            # Cache may omit timestamp — fill later from nuScenes if possible.
            labels.append(raw)
        print(f"[smoke] loaded {len(labels)} cache file(s) from {cache_root}")
        return labels

    # Regenerate via stub
    from data.nuscenes_mini import list_sample_tokens, load_nuscenes, sample_meta
    from teachers.lidar_stub import LidarTeacherStub

    nusc = load_nuscenes(args.dataroot, version=args.version, verbose=False)
    tokens = list_sample_tokens(nusc, args.scene, max_samples=n)
    teacher = LidarTeacherStub(cache_root=cache_root)
    teacher.set_nusc(nusc)
    teacher.load()
    try:
        for i, tok in enumerate(tokens):
            meta = sample_meta(nusc, tok, args.scene)
            frame = {
                "sample_token": tok,
                "scene_name": args.scene,
                "sample_idx": i,
                "lidar_sd_token": meta["lidar_sd_token"],
                "nusc": nusc,
            }
            teacher.infer_frame(frame)
    finally:
        teacher.unload()

    existing = sorted(cache_root.glob("lidar_stub_*.json"))
    for p in existing[:n]:
        labels.append(read_label(p))
    print(f"[smoke] regenerated {len(labels)} cache file(s) via lidar_stub")
    return labels


def _enrich_timestamps(labels: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    """Ensure required schema fields: timestamp (int)."""
    nusc = None
    try:
        from data.nuscenes_mini import load_nuscenes, sample_meta

        nusc = load_nuscenes(args.dataroot, version=args.version, verbose=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] nuScenes optional enrich skipped: {exc}")

    for lb in labels:
        if isinstance(lb.get("timestamp"), int):
            continue
        if nusc is not None:
            try:
                meta = sample_meta(nusc, lb["sample_token"], lb.get("scene_name", args.scene))
                lb["timestamp"] = int(meta["timestamp"])
                if not lb.get("lidar_sd_token"):
                    lb["lidar_sd_token"] = meta["lidar_sd_token"]
                continue
            except Exception:  # noqa: BLE001
                pass
        lb["timestamp"] = int(lb.get("timestamp") or 0)


def _maybe_viz(labels: List[Dict[str, Any]], out_viz: Path) -> List[Path]:
    """Tiny BEV scatter PNGs (optional)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] viz skipped (matplotlib): {exc}")
        return []

    out_viz.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    colors = {"car": "C0", "pedestrian": "C1", "truck": "C2"}
    for lb in labels:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        for o in lb.get("objects") or []:
            if o.get("ignore"):
                continue
            x, y = float(o["translation"][0]), float(o["translation"][1])
            cat = str(o.get("category", "?"))
            ax.scatter([x], [y], c=colors.get(cat, "C3"), s=40)
            ax.text(x, y, f"{o.get('track_id','')}/{o.get('quality','')}", fontsize=7)
        ax.set_aspect("equal")
        ax.set_title(f"{lb.get('scene_name')} {lb['sample_token'][:8]}")
        ax.grid(True, alpha=0.3)
        p = out_viz / f"bev_{lb['sample_token'][:12]}.png"
        fig.tight_layout()
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)
    return paths


def main() -> int:
    args = _parse_args()
    n = max(1, min(int(args.max_samples), 2))
    print(f"[smoke] Stage E fusion/export scene={args.scene} N={n}")

    raw_labels = _load_or_regen_cache(args, n)
    if not raw_labels:
        print("FAIL: no teacher cache labels")
        return 1

    # Filter to requested scene when possible
    filtered = [lb for lb in raw_labels if lb.get("scene_name", args.scene) == args.scene]
    if not filtered:
        filtered = raw_labels[:n]
    filtered = filtered[:n]
    _enrich_timestamps(filtered, args)

    # Association placeholder (no secondary cues in this smoke → passthrough)
    frames_objs = []
    assoc_all = []
    for lb in filtered:
        objs = list(lb.get("objects") or [])
        matches = associate_lidar_primary(objs, None)
        frames_objs.append(objs)
        assoc_all.append(matches)

    # Quality per frame
    quality_frames: List[List[Dict[str, Any]]] = []
    for objs, matches in zip(frames_objs, assoc_all):
        quality_frames.append(
            apply_quality(objs, is_stub=True, association=matches)
        )

    # Build sample dicts then track across frames
    samples: List[Dict[str, Any]] = []
    for lb, objs in zip(filtered, quality_frames):
        sample = {
            "version": str(lb.get("version", "1.0")),
            "sample_token": str(lb["sample_token"]),
            "scene_name": str(lb.get("scene_name", args.scene)),
            "timestamp": int(lb["timestamp"]),
            "lidar_sd_token": str(lb.get("lidar_sd_token") or ""),
            "objects": objs,
            "meta": {
                "generator": "scripts/run_fusion_export_smoke.py",
                "is_fake": True,
                "notes": (
                    "Stage E fusion/track/export smoke from lidar_stub cache; "
                    "not for training."
                ),
            },
        }
        samples.append(sample)

    tracked = assign_tracks(samples, mode="greedy", prefix="fuse")
    for s in tracked:
        validate_label(s)
        print(f"[smoke] {s['sample_token'][:8]}… {summarize_quality(s['objects'])}")

    out_pseudo = Path(args.out_pseudo)
    written = write_fused_pseudo_labels(tracked, out_pseudo)
    results = labels_to_results(
        tracked,
        include_tracking=True,
        skip_ignore=False,
        meta={"is_fake": True, "scene": args.scene, "stage": "E"},
    )
    results_path = out_pseudo / "nuscenes_results_smoke.json"
    write_results_json(results_path, results)

    # Sanity: every box has track_id in results
    n_boxes = 0
    for tok, boxes in results["results"].items():
        for b in boxes:
            assert "track_id" in b and b["track_id"], f"missing track_id in {tok}"
            assert "tracking_id" in b
            n_boxes += 1

    viz_paths: List[Path] = []
    if not args.no_viz:
        viz_paths = _maybe_viz(tracked, Path(args.out_viz))

    print(f"[smoke] wrote {len(written)} pseudo labels → {out_pseudo}")
    print(f"[smoke] results → {results_path} (boxes={n_boxes})")
    if viz_paths:
        print(f"[smoke] viz → {viz_paths[0].parent} ({len(viz_paths)} png)")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
