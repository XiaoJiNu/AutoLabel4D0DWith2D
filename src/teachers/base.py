"""TeacherStage protocol + serial residency guard (one teacher resident at a time)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class TeacherStage(Protocol):
    """Thin serial teacher interface. Implementations must not hold GPU after unload()."""

    name: str

    def load(self) -> None:
        """Load weights / allocate resources. Must call SerialTeacherGuard.acquire(self)."""
        ...

    def infer_frame(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        """Run one-frame inference. Frame keys are teacher-specific (sample_token, paths, …)."""
        ...

    def unload(self) -> None:
        """Release weights / free VRAM. Must call SerialTeacherGuard.release(self)."""
        ...


class SerialTeacherGuard:
    """Enforce at most one resident TeacherStage (serial teachers on 24GB)."""

    _resident: Optional[TeacherStage] = None
    _resident_name: Optional[str] = None

    @classmethod
    def resident(cls) -> Optional[str]:
        return cls._resident_name

    @classmethod
    def acquire(cls, teacher: TeacherStage) -> None:
        name = getattr(teacher, "name", type(teacher).__name__)
        if cls._resident is not None and cls._resident is not teacher:
            raise RuntimeError(
                f"serial teacher violation: '{cls._resident_name}' still resident; "
                f"call unload() before load() of '{name}'"
            )
        cls._resident = teacher
        cls._resident_name = name

    @classmethod
    def release(cls, teacher: TeacherStage) -> None:
        if cls._resident is teacher:
            cls._resident = None
            cls._resident_name = None

    @classmethod
    def reset(cls) -> None:
        """Test/smoke helper — clear guard without unloading (prefer unload())."""
        cls._resident = None
        cls._resident_name = None


def assert_serial_idle() -> None:
    name = SerialTeacherGuard.resident()
    if name is not None:
        raise RuntimeError(f"expected no resident teacher, found '{name}'")
