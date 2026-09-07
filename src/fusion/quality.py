"""Quality A/B/C + valid_fields + ignore helpers (matches configs/label_schema.yaml).

Stage E thin rules — not production calibration. Schema enums: quality ∈ {A,B,C}.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

QUALITY_VALUES = ("A", "B", "C")

# Fields that must be present & well-formed for a box to be non-ignore.
CORE_FIELDS = ("translation", "size", "rotation", "category")

# Default score thresholds (stub).
SCORE_A = 0.85
SCORE_B = 0.55


def list_present_fields(obj: Dict[str, Any]) -> List[str]:
    """Return object keys that look populated for schema ``valid_fields``."""
    present: List[str] = []
    for k in CORE_FIELDS:
        v = obj.get(k)
        if v is None:
            continue
        if k in ("translation", "size") and isinstance(v, (list, tuple)) and len(v) == 3:
            present.append(k)
        elif k == "rotation" and isinstance(v, (list, tuple)) and len(v) == 4:
            present.append(k)
        elif k == "category" and isinstance(v, str) and v.strip():
            present.append(k)
    # Optional extras if present
    for k in ("score", "track_id", "velocity"):
        if k in obj and obj[k] is not None:
            if k not in present:
                present.append(k)
    return present


def missing_core_fields(obj: Dict[str, Any]) -> List[str]:
    present = set(list_present_fields(obj))
    return [k for k in CORE_FIELDS if k not in present]


def should_ignore(
    obj: Dict[str, Any],
    *,
    quality: Optional[str] = None,
    min_score: float = 0.15,
    force_ignore: bool = False,
) -> bool:
    """Ignore when malformed, forced, or below floor score.

    Schema: ``ignore: bool``. Ignored boxes stay in the file for audit but
    should be skipped by student training loaders.
    """
    if force_ignore or bool(obj.get("force_ignore", False)):
        return True
    if missing_core_fields(obj):
        return True
    score = obj.get("score")
    if score is not None and float(score) < min_score:
        return True
    q = quality if quality is not None else obj.get("quality")
    # C-quality alone is NOT auto-ignore (stubs are often C); only explicit flags.
    if obj.get("ignore") is True and quality is None:
        # Preserve caller-set ignore when we are not recomputing quality yet.
        return True
    _ = q
    return False


def assign_quality(
    obj: Dict[str, Any],
    *,
    score_a: float = SCORE_A,
    score_b: float = SCORE_B,
    is_stub: bool = False,
    matched_secondary: bool = False,
) -> str:
    """Assign quality letter A/B/C for one object.

    Heuristic (smoke):
    - Missing core fields → C
    - Stub / fake teacher without secondary match → at best B (usually C if low score)
    - High score + complete fields (+ optional secondary) → A
    - Mid score → B
    - Else → C
    """
    if missing_core_fields(obj):
        return "C"
    score = float(obj.get("score", 0.0))
    if is_stub and not matched_secondary:
        # Interface stubs: cap at B; low score stays C.
        if score >= score_a:
            return "B"
        if score >= score_b:
            return "B"
        return "C"
    if score >= score_a and (matched_secondary or not is_stub):
        return "A"
    if score >= score_b:
        return "B"
    return "C"


def apply_quality(
    objects: Sequence[Dict[str, Any]],
    *,
    is_stub: bool = False,
    association: Optional[Sequence[Dict[str, Any]]] = None,
    score_a: float = SCORE_A,
    score_b: float = SCORE_B,
    min_score_keep: float = 0.15,
) -> List[Dict[str, Any]]:
    """Copy objects and fill ``quality``, ``valid_fields``, ``ignore``.

    ``association`` is optional output from ``associate.associate_lidar_primary``
    (same length / lidar_idx aligned).
    """
    assoc_by_idx: Dict[int, Dict[str, Any]] = {}
    if association:
        for m in association:
            assoc_by_idx[int(m["lidar_idx"])] = m

    out: List[Dict[str, Any]] = []
    for i, raw in enumerate(objects):
        obj = dict(raw)
        matched = False
        m = assoc_by_idx.get(i) or obj.get("assoc")
        if isinstance(m, dict) and m.get("secondary_idx") is not None:
            matched = True
        # Detect stub-ish sources
        src = str(obj.get("source", ""))
        stub = is_stub or "stub" in src or "fake" in src or bool(obj.get("is_stub", False))

        valid = list_present_fields(obj)
        # Schema prefers geometry+category in valid_fields; drop score/track_id extras
        # if you want strict schema example — keep core intersection for stability.
        valid_fields = [f for f in valid if f in CORE_FIELDS]
        if not valid_fields:
            valid_fields = list(CORE_FIELDS)  # still annotate intent; ignore will trip

        q = assign_quality(
            obj,
            score_a=score_a,
            score_b=score_b,
            is_stub=stub,
            matched_secondary=matched,
        )
        ign = should_ignore(obj, quality=q, min_score=min_score_keep)
        # If ignore was already True and malformed, keep True.
        if missing_core_fields(obj):
            ign = True

        obj["quality"] = q
        obj["valid_fields"] = valid_fields
        obj["ignore"] = bool(ign)
        out.append(obj)
    return out


def summarize_quality(objects: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Small counters for smoke logs."""
    counts = {k: 0 for k in QUALITY_VALUES}
    n_ignore = 0
    for o in objects:
        q = str(o.get("quality", "?"))
        if q in counts:
            counts[q] += 1
        if o.get("ignore"):
            n_ignore += 1
    return {"n": len(objects), "by_quality": counts, "n_ignore": n_ignore}


__all__ = [
    "QUALITY_VALUES",
    "CORE_FIELDS",
    "SCORE_A",
    "SCORE_B",
    "list_present_fields",
    "missing_core_fields",
    "should_ignore",
    "assign_quality",
    "apply_quality",
    "summarize_quality",
]
