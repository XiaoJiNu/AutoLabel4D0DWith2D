"""StageRunner stub: register/run stages listed in pipeline_v3_5090.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PIPELINE = _REPO_ROOT / "configs" / "pipeline_v3_5090.yaml"

# Map pipeline stage ids → high-level letter stages (A…F) for listing.
_STAGE_LETTER: Dict[str, str] = {
    "P0_audit": "A/B",
    "P1_geometry": "C",
    "P2_lidar_teacher": "D1",
    "P3_dino": "D2",
    "P3b_sam": "D3",
    "P4_depth": "D4",
    "P5_fusion_track_export": "E",
    "P6_student_pointpillars": "F",
}

_STAGE_D_IDS = ("P2_lidar_teacher", "P3_dino", "P3b_sam", "P4_depth")
_STAGE_E_IDS = ("P5_fusion_track_export",)
_STAGE_F_IDS = ("P6_student_pointpillars",)

StageFn = Callable[[Dict[str, Any]], Any]


class StageRunner:
    """Thin orchestrator: lists stages from YAML and runs registered callables."""

    def __init__(self, pipeline_yaml: str | Path | None = None):
        path = Path(pipeline_yaml) if pipeline_yaml else _DEFAULT_PIPELINE
        with path.open("r", encoding="utf-8") as f:
            self.cfg: Dict[str, Any] = yaml.safe_load(f) or {}
        self.pipeline_path = path
        self._handlers: Dict[str, StageFn] = {}

    @property
    def stage_ids(self) -> List[str]:
        stages = self.cfg.get("stages") or []
        return [s["id"] for s in stages if isinstance(s, dict) and "id" in s]

    @property
    def serial_teachers(self) -> bool:
        return bool(self.cfg.get("serial_teachers", True))

    def stage_d_ids(self) -> List[str]:
        """Stage D teacher serial stages present in this pipeline YAML."""
        return [sid for sid in self.stage_ids if sid in _STAGE_D_IDS]

    def register(self, stage_id: str, fn: StageFn) -> None:
        self._handlers[stage_id] = fn

    def list_stages(self) -> List[Dict[str, Any]]:
        out = []
        for sid in self.stage_ids:
            letter = _STAGE_LETTER.get(sid, "?")
            out.append(
                {
                    "id": sid,
                    "letter": letter,
                    "is_stage_d": sid in _STAGE_D_IDS,
                    "is_stage_e": sid in _STAGE_E_IDS,
                    "is_stage_f": sid in _STAGE_F_IDS,
                    "registered": sid in self._handlers,
                }
            )
        return out

    def list_stage_d(self) -> List[Dict[str, Any]]:
        """Convenience: only D1–D4 teacher stages."""
        return [s for s in self.list_stages() if s["is_stage_d"]]

    def list_stage_e(self) -> List[Dict[str, Any]]:
        """Convenience: Stage E fusion / track / export (P5)."""
        return [
            s
            for s in self.list_stages()
            if s["id"] in _STAGE_E_IDS or s.get("letter") == "E"
        ]

    def list_stage_f(self) -> List[Dict[str, Any]]:
        """Convenience: Stage F student PointPillars (P6)."""
        return [
            s
            for s in self.list_stages()
            if s["id"] in _STAGE_F_IDS or s.get("letter") == "F"
        ]

    def run(self, stage_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run one stage or all registered stages in YAML order. Unregistered stages are skipped."""
        ctx = dict(context or {})
        results: Dict[str, Any] = {}
        ids = [stage_id] if stage_id else self.stage_ids
        for sid in ids:
            if sid not in self.stage_ids:
                raise KeyError(f"unknown stage id '{sid}'; known={self.stage_ids}")
            fn = self._handlers.get(sid)
            if fn is None:
                results[sid] = {"status": "skipped", "reason": "not_registered"}
                continue
            results[sid] = {"status": "ok", "result": fn(ctx)}
        return results
