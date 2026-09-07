"""Grounding DINO stub — real Tiny weights after RS audit / Stage D2."""
from __future__ import annotations

from typing import Any, Dict

from teachers.base import SerialTeacherGuard

_NEXT = (
    "NEXT: wire Grounding DINO Tiny (frozen) after RoboSense subset audit; "
    "serial load→infer→unload; cache under pseudo_labels/teacher_cache/. "
    "Do not download heavy weights in this stub."
)


class DinoTeacherStub:
    name = "dino_stub"

    def __init__(self) -> None:
        self._loaded = False

    def load(self) -> None:
        SerialTeacherGuard.acquire(self)
        self._loaded = True
        # No model download / no torch occupancy.

    def unload(self) -> None:
        self._loaded = False
        SerialTeacherGuard.release(self)

    def infer_frame(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        if not self._loaded:
            raise RuntimeError(f"{self.name}: call load() before infer_frame()")
        # Empty stub result; raise if caller insists on real DINO:
        if frame.get("require_real"):
            raise NotImplementedError(_NEXT)
        return {
            "teacher": self.name,
            "sample_token": frame.get("sample_token"),
            "detections_2d": [],
            "meta": {
                "is_stub": True,
                "is_fake": True,
                "notes": _NEXT,
            },
        }
