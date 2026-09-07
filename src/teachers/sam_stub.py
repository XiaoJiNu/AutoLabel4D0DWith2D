"""SAM 2.1 stub — real Small weights after RS audit / Stage D3."""
from __future__ import annotations

from typing import Any, Dict

from teachers.base import SerialTeacherGuard

_NEXT = (
    "NEXT: wire SAM 2.1 Small (frozen) after RoboSense subset audit; "
    "prefer Tiny fallback if VRAM tight; serial load→infer→unload; "
    "no SAM3. Do not download heavy weights in this stub."
)


class SamTeacherStub:
    name = "sam_stub"

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
            "masks": [],
            "meta": {
                "is_stub": True,
                "is_fake": True,
                "notes": _NEXT,
            },
        }
