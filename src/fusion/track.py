"""Thin track_id assignment stub (Stage E).

Smoke modes:
- ``identity``: keep existing track_id / index-stable ids across a single frame
- ``greedy``: nearest-center linking across consecutive frames (fake stable ids)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from fusion.associate import center_distance_xy


def _ensure_track_id(obj: Dict[str, Any], idx: int, prefix: str = "trk") -> str:
    tid = obj.get("track_id")
    if isinstance(tid, str) and tid.strip():
        return tid
    return f"{prefix}_{idx:03d}"


def assign_track_ids_frame(
    objects: Sequence[Dict[str, Any]],
    *,
    prefix: str = "trk",
    force_reindex: bool = False,
) -> List[Dict[str, Any]]:
    """Assign / normalize track_id within one frame (identity stub)."""
    out: List[Dict[str, Any]] = []
    for i, raw in enumerate(objects):
        obj = dict(raw)
        if force_reindex or not (isinstance(obj.get("track_id"), str) and obj["track_id"].strip()):
            obj["track_id"] = f"{prefix}_{i:03d}"
        else:
            obj["track_id"] = str(obj["track_id"])
        out.append(obj)
    return out


def link_tracks_greedy(
    frames: Sequence[Sequence[Dict[str, Any]]],
    *,
    max_match_dist_m: float = 4.0,
    prefix: str = "trk",
) -> List[List[Dict[str, Any]]]:
    """Greedy center-distance linking over frames → stable track_id.

    First frame seeds ids; later frames match to previous unmatched tracks by
    XY distance + category. Unmatched detections spawn new ids.
    """
    if not frames:
        return []

    next_id = 0
    results: List[List[Dict[str, Any]]] = []
    prev: List[Dict[str, Any]] = []

    for fi, frame_objs in enumerate(frames):
        curated: List[Dict[str, Any]] = []
        used_prev: set[int] = set()

        # Sort by score desc for greedy priority
        order = sorted(
            range(len(frame_objs)),
            key=lambda i: float(frame_objs[i].get("score", 0.0)),
            reverse=True,
        )
        assigned: Dict[int, str] = {}

        for i in order:
            obj = frame_objs[i]
            best_j: Optional[int] = None
            best_d = max_match_dist_m + 1.0
            cat = str(obj.get("category", ""))
            for j, pobj in enumerate(prev):
                if j in used_prev:
                    continue
                if cat and str(pobj.get("category", "")) and cat != pobj.get("category"):
                    continue
                d = center_distance_xy(obj, pobj)
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_j is not None and best_d <= max_match_dist_m:
                tid = str(prev[best_j]["track_id"])
                used_prev.add(best_j)
            else:
                tid = f"{prefix}_{next_id:03d}"
                next_id += 1
            assigned[i] = tid

        for i, raw in enumerate(frame_objs):
            obj = dict(raw)
            obj["track_id"] = assigned.get(i, f"{prefix}_{next_id:03d}")
            if i not in assigned:
                next_id += 1
            curated.append(obj)

        results.append(curated)
        prev = curated

    return results


def assign_tracks(
    frames: Sequence[Dict[str, Any]] | Sequence[Sequence[Dict[str, Any]]],
    *,
    mode: str = "greedy",
    max_match_dist_m: float = 4.0,
    prefix: str = "trk",
) -> List[Dict[str, Any]]:
    """High-level helper.

    Accepts either:
    - list of sample dicts with ``objects`` key (mutates copies, returns samples), or
    - list of object-lists (returns list of object-lists wrapped? — see below).

    Always returns list of sample-like dicts when inputs are samples; when inputs
    are raw object lists, returns list[{'objects': ...}].
    """
    if not frames:
        return []

    first = frames[0]
    sample_mode = isinstance(first, dict) and "objects" in first

    if sample_mode:
        samples = [dict(s) for s in frames]  # type: ignore[arg-type]
        obj_frames = [list(s.get("objects") or []) for s in samples]
    else:
        samples = [{"objects": list(fr)} for fr in frames]  # type: ignore[arg-type]
        obj_frames = [list(fr) for fr in frames]  # type: ignore[arg-type]

    if mode == "identity":
        linked = [assign_track_ids_frame(fr, prefix=prefix) for fr in obj_frames]
    else:
        linked = link_tracks_greedy(obj_frames, max_match_dist_m=max_match_dist_m, prefix=prefix)

    out: List[Dict[str, Any]] = []
    for s, objs in zip(samples, linked):
        s = dict(s)
        s["objects"] = objs
        out.append(s)
    return out


__all__ = [
    "assign_track_ids_frame",
    "link_tracks_greedy",
    "assign_tracks",
]
