"""Serial teachers (Stage D). Stubs + real CenterPoint / DINO / SAM teachers."""
from .base import SerialTeacherGuard, TeacherStage, assert_serial_idle
from .lidar_stub import LidarTeacherStub
from .lidar_centerpoint import LidarCenterPointTeacher
from .dino_stub import DinoTeacherStub
from .dino_real import DinoTeacherReal
from .sam_stub import SamTeacherStub
from .sam_real import SamTeacherReal
from .depth_stub import DepthTeacherStub

__all__ = [
    "TeacherStage",
    "SerialTeacherGuard",
    "assert_serial_idle",
    "LidarTeacherStub",
    "LidarCenterPointTeacher",
    "DinoTeacherStub",
    "DinoTeacherReal",
    "SamTeacherStub",
    "SamTeacherReal",
    "DepthTeacherStub",
]
