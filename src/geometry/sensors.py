"""Sensor preset loading for AutoLabel4D Stage C.

Provides a thin interface over YAML presets (RoboSense RS_4F_1L) and a
hard-coded NuScenesCamLidar preset for mini smoke / projection until RS data
and calibration arrive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RS_YAML = _REPO_ROOT / "configs" / "sensors_rs_4f_1l.yaml"


@dataclass
class SensorPreset:
    """Named camera/LiDAR channel layout + optional notes."""

    name: str
    camera_channels: List[str]
    lidar_channel: str
    lidar_source: Optional[str] = None
    forbid_lidar_fields: List[str] = field(default_factory=list)
    annotation_split: Optional[str] = None
    notes: str = ""
    # Optional per-camera model hints: "pinhole" | "fisheye" | "unknown"
    camera_models: Dict[str, str] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def is_fisheye(self, channel: str) -> bool:
        return self.camera_models.get(channel, "unknown") == "fisheye"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "preset_name": self.name,
            "camera_channels": list(self.camera_channels),
            "lidar_channel": self.lidar_channel,
            "lidar_source": self.lidar_source,
            "forbid_lidar_fields": list(self.forbid_lidar_fields),
            "annotation_split": self.annotation_split,
            "notes": self.notes,
            "camera_models": dict(self.camera_models),
            **dict(self.extra),
        }


def _coerce_list(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    return [str(x) for x in val]


def load_sensor_preset(yaml_path: str | Path | None = None) -> SensorPreset:
    """Load a SensorPreset from YAML (default: configs/sensors_rs_4f_1l.yaml)."""
    path = Path(yaml_path) if yaml_path else _DEFAULT_RS_YAML
    if not path.is_file():
        raise FileNotFoundError(f"sensor preset YAML not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"sensor YAML must be a mapping: {path}")

    name = str(data.get("preset_name") or data.get("name") or path.stem)
    cams = _coerce_list(data.get("camera_channels"))
    lidar_ch = str(data.get("lidar_channel") or "LIDAR_TOP")
    lidar_src = data.get("lidar_source")
    forbid = _coerce_list(data.get("forbid_lidar_fields"))
    notes = str(data.get("notes") or "").strip()
    ann = data.get("annotation_split")

    # RS_4F_1L cameras are fisheye (OV); mark explicitly for Stage C fisheye stub.
    cam_models: Dict[str, str] = {}
    raw_models = data.get("camera_models") or {}
    if isinstance(raw_models, dict):
        cam_models = {str(k): str(v) for k, v in raw_models.items()}
    elif name.upper() in ("RS_4F_1L", "RS4F1L") or "fisheye" in notes.lower() or "鱼眼" in notes:
        for c in cams:
            cam_models.setdefault(c, "fisheye")

    known = {
        "preset_name",
        "name",
        "camera_channels",
        "lidar_channel",
        "lidar_source",
        "forbid_lidar_fields",
        "annotation_split",
        "notes",
        "camera_models",
    }
    extra = {k: v for k, v in data.items() if k not in known}

    return SensorPreset(
        name=name,
        camera_channels=cams,
        lidar_channel=lidar_ch,
        lidar_source=str(lidar_src) if lidar_src is not None else None,
        forbid_lidar_fields=forbid,
        annotation_split=str(ann) if ann is not None else None,
        notes=notes,
        camera_models=cam_models,
        extra=extra,
    )


def load_rs_4f_1l(yaml_path: str | Path | None = None) -> SensorPreset:
    """Convenience: load RoboSense 4-fisheye + 1 LiDAR preset."""
    preset = load_sensor_preset(yaml_path)
    if preset.name.upper() not in ("RS_4F_1L", "RS4F1L") and yaml_path is None:
        # still return what YAML says; caller may override path
        pass
    return preset


def get_nuscenes_cam_lidar_preset(
    camera_channels: Optional[Sequence[str]] = None,
    lidar_channel: str = "LIDAR_TOP",
) -> SensorPreset:
    """Hard-coded nuScenes mini / full camera+LiDAR preset (pinhole cams)."""
    cams = list(
        camera_channels
        or (
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_FRONT_LEFT",
        )
    )
    return SensorPreset(
        name="NuScenesCamLidar",
        camera_channels=cams,
        lidar_channel=lidar_channel,
        lidar_source="LIDAR_TOP",
        forbid_lidar_fields=[],
        annotation_split=None,
        notes=(
            "nuScenes pinhole cameras + LIDAR_TOP. Used as Stage C prior until "
            "RoboSense RS_4F_1L calibration and fisheye unwrap are ready."
        ),
        camera_models={c: "pinhole" for c in cams},
    )


# Alias matching the task name
NuScenesCamLidar = get_nuscenes_cam_lidar_preset


__all__ = [
    "SensorPreset",
    "load_sensor_preset",
    "load_rs_4f_1l",
    "get_nuscenes_cam_lidar_preset",
    "NuScenesCamLidar",
]
