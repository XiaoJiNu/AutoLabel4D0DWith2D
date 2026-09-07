"""LiDAR → camera image projection (pinhole K + extrinsics).

Core Stage C primitive: given points in a LiDAR/sensor frame and camera
intrinsics/extrinsics (or a nuScenes sample), return pixel uv, validity mask,
and depth along the camera optical axis.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from pyquaternion import Quaternion

ArrayLike = Union[np.ndarray, Sequence[float]]


def _as_Nx3(points: ArrayLike) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    pts = pts.reshape(-1, pts.shape[-1])
    if pts.shape[1] < 3:
        raise ValueError(f"points must be Nx3+, got shape {pts.shape}")
    return pts[:, :3].copy()


def transform_points(points: np.ndarray, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Apply R @ p + t to (N,3)."""
    pts = _as_Nx3(points)
    if pts.shape[0] == 0:
        return pts
    R = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    t = np.asarray(trans, dtype=np.float64).reshape(1, 3)
    return (R @ pts.T).T + t


def invert_rt(rot: np.ndarray, trans: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    t = np.asarray(trans, dtype=np.float64).reshape(3)
    r_inv = R.T
    t_inv = -r_inv @ t
    return r_inv, t_inv


def project_points_pinhole(
    points_cam: ArrayLike,
    K: ArrayLike,
    image_size: Optional[Tuple[int, int]] = None,
    min_depth: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project camera-frame points with pinhole K.

    Args:
        points_cam: Nx3 in camera coordinates (OpenCV/nuScenes: x-right, y-down, z-forward).
        K: 3x3 intrinsics.
        image_size: optional (width, height) to build in-bounds mask; if None, mask is depth-only.
        min_depth: reject points with z <= min_depth.

    Returns:
        uv: Nx2 float pixel coordinates (may be outside image).
        mask: N bool — depth valid (and in-bounds if image_size given).
        depth: N float — camera z.
    """
    pts = _as_Nx3(points_cam)
    n = pts.shape[0]
    if n == 0:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0,), dtype=bool),
            np.zeros((0,), dtype=np.float64),
        )

    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    depth = pts[:, 2].copy()
    # Avoid div-by-zero; invalid depths get nan uv then masked out
    z_safe = np.where(depth > min_depth, depth, np.nan)
    x = pts[:, 0] / z_safe
    y = pts[:, 1] / z_safe
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    # optional skew
    if abs(K[0, 1]) > 1e-12:
        u = u + K[0, 1] * y
    uv = np.stack([u, v], axis=1)
    mask = np.isfinite(uv).all(axis=1) & (depth > min_depth)
    if image_size is not None:
        w, h = int(image_size[0]), int(image_size[1])
        mask = (
            mask
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < w)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < h)
        )
    # replace nan with 0 for cleaner downstream
    uv = np.nan_to_num(uv, nan=0.0)
    return uv, mask, depth


def project_lidar_to_image(
    points_lidar: ArrayLike,
    K: ArrayLike,
    R_lidar_to_cam: ArrayLike,
    t_lidar_to_cam: ArrayLike,
    image_size: Optional[Tuple[int, int]] = None,
    min_depth: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project LiDAR-frame points using a single rigid + K.

    points_cam = R @ points_lidar + t
    Returns (uv, mask, depth) same as project_points_pinhole.
    """
    pts = _as_Nx3(points_lidar)
    R = np.asarray(R_lidar_to_cam, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t_lidar_to_cam, dtype=np.float64).reshape(3)
    pts_cam = transform_points(pts, R, t)
    return project_points_pinhole(pts_cam, K, image_size=image_size, min_depth=min_depth)


def _quat_trans(cs_or_ep: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    q = Quaternion(cs_or_ep["rotation"])
    t = np.array(cs_or_ep["translation"], dtype=np.float64)
    return q.rotation_matrix, t


def project_nuscenes_lidar_to_cam(
    nusc: Any,
    sample_token: str,
    points_lidar: ArrayLike,
    cam_channel: str = "CAM_FRONT",
    lidar_channel: str = "LIDAR_TOP",
    image_size: Optional[Tuple[int, int]] = None,
    min_depth: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project LIDAR_TOP points onto a nuScenes camera (handles ego-pose time gap).

    Chain: lidar -> ego(lidar) -> global -> ego(cam) -> cam, then K.
    Returns (uv, mask, depth).
    """
    sample = nusc.get("sample", sample_token)
    lidar_sd = nusc.get("sample_data", sample["data"][lidar_channel])
    cam_sd = nusc.get("sample_data", sample["data"][cam_channel])
    lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    lidar_ep = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    cam_ep = nusc.get("ego_pose", cam_sd["ego_pose_token"])

    pts = _as_Nx3(points_lidar)
    R_l, t_l = _quat_trans(lidar_cs)
    R_le, t_le = _quat_trans(lidar_ep)
    pts_ego_l = transform_points(pts, R_l, t_l)
    pts_g = transform_points(pts_ego_l, R_le, t_le)

    R_ce, t_ce = _quat_trans(cam_ep)
    R_ce_inv, t_ce_inv = invert_rt(R_ce, t_ce)
    pts_ego_c = transform_points(pts_g, R_ce_inv, t_ce_inv)

    R_c, t_c = _quat_trans(cam_cs)
    R_c_inv, t_c_inv = invert_rt(R_c, t_c)
    pts_cam = transform_points(pts_ego_c, R_c_inv, t_c_inv)

    K = np.array(cam_cs["camera_intrinsic"], dtype=np.float64)
    if image_size is None and "width" in cam_sd and "height" in cam_sd:
        image_size = (int(cam_sd["width"]), int(cam_sd["height"]))

    return project_points_pinhole(pts_cam, K, image_size=image_size, min_depth=min_depth)


__all__ = [
    "transform_points",
    "invert_rt",
    "project_points_pinhole",
    "project_lidar_to_image",
    "project_nuscenes_lidar_to_cam",
]
