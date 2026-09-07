"""Stage C geometry: sensor presets, LiDAR→cam projection, time/pose hooks, fisheye stub."""
from __future__ import annotations

from geometry.fisheye import VirtualPinhole, fisheye_to_virtual_pinhole, is_fisheye_stub_ready
from geometry.project import (
    invert_rt,
    project_lidar_to_image,
    project_nuscenes_lidar_to_cam,
    project_points_pinhole,
    transform_points,
)
from geometry.sensors import (
    NuScenesCamLidar,
    SensorPreset,
    get_nuscenes_cam_lidar_preset,
    load_rs_4f_1l,
    load_sensor_preset,
)
from geometry.time_pose import (
    TimeSyncReport,
    check_nuscenes_lidar_cam_sync,
    check_rs_time_alignment,
    pose_translation_norm,
    sanity_check_pose_chain,
    timestamp_delta_ms,
)

__all__ = [
    "SensorPreset",
    "load_sensor_preset",
    "load_rs_4f_1l",
    "get_nuscenes_cam_lidar_preset",
    "NuScenesCamLidar",
    "transform_points",
    "invert_rt",
    "project_points_pinhole",
    "project_lidar_to_image",
    "project_nuscenes_lidar_to_cam",
    "TimeSyncReport",
    "timestamp_delta_ms",
    "check_nuscenes_lidar_cam_sync",
    "pose_translation_norm",
    "sanity_check_pose_chain",
    "check_rs_time_alignment",
    "VirtualPinhole",
    "fisheye_to_virtual_pinhole",
    "is_fisheye_stub_ready",
]
