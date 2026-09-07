"""Depth teacher stub — OMNI-DC / E3-lite after RS audit / Stage D4."""
from __future__ import annotations

from typing import Any, Dict

from teachers.base import SerialTeacherGuard

_NEXT = (
    "NEXT: wire OMNI-DC v1.0 (or E3-lite fallback); may disable depth and keep E2; "
    "serial load→infer→unload after RoboSense audit. "
    "Do not download heavy weights in this stub."
)


class DepthTeacherStub:
    name = "depth_stub"

    def __init__(self) -> None:
        self._loaded = False

    def load(self) -> None:
        SerialTeacherGuard.acquire(self)
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False
        SerialTeacherGuard.release(self)

    def infer_frame(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        if not self._loaded:
            raise RuntimeError(f"{self.name}: call load() before infer_frame()")
        if frame.get("require_real"):
            raise NotImplementedError(_NEXT)
        return {
            "teacher": self.name,
            "sample_token": frame.get("sample_token"),
            "depth": None,
            "meta": {
                "is_stub": True,
                "is_fake": True,
                "notes": _NEXT,
            },
        }
