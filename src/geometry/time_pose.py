"""Time alignment and pose sanity-check hooks (Stage C stubs for RoboSense).

nuScenes provides per-sample_data ego poses; RS will need explicit sync between
hs64 LiDAR packets and the 4 OV fisheye streams. Fill TODOs when RS subset +
calibration metadata land.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class TimeSyncReport:
    """Summary of timestamp / pose sanity for one frame pair."""

    lidar_ts_us: Optional[int]
    cam_ts_us: Optional[int]
    delta_ms: Optional[float]
    ok: bool
    messages: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "lidar_ts_us": self.lidar_ts_us,
            "cam_ts_us": self.cam_ts_us,
            "delta_ms": self.delta_ms,
            "ok": self.ok,
            "messages": list(self.messages),
        }


def timestamp_delta_ms(ts_a_us: int, ts_b_us: int) -> float:
    """Absolute delta in milliseconds between two microsecond timestamps."""
    return abs(int(ts_a_us) - int(ts_b_us)) / 1000.0


def check_nuscenes_lidar_cam_sync(
    nusc: Any,
    sample_token: str,
    cam_channel: str = "CAM_FRONT",
    lidar_channel: str = "LIDAR_TOP",
    max_delta_ms: float = 50.0,
) -> TimeSyncReport:
    """nuScenes: compare LIDAR vs camera sample_data timestamps on a keyframe."""
    msgs: List[str] = []
    sample = nusc.get("sample", sample_token)
    lidar_sd = nusc.get("sample_data", sample["data"][lidar_channel])
    cam_sd = nusc.get("sample_data", sample["data"][cam_channel])
    l_ts = int(lidar_sd["timestamp"])
    c_ts = int(cam_sd["timestamp"])
    delta = timestamp_delta_ms(l_ts, c_ts)
    ok = delta <= float(max_delta_ms)
    if not ok:
        msgs.append(
            f"lidar/cam timestamp delta {delta:.2f} ms exceeds max_delta_ms={max_delta_ms}"
        )
    else:
        msgs.append(f"lidar/cam timestamp delta {delta:.2f} ms within {max_delta_ms} ms")
    return TimeSyncReport(
        lidar_ts_us=l_ts,
        cam_ts_us=c_ts,
        delta_ms=delta,
        ok=ok,
        messages=msgs,
    )


def pose_translation_norm(pose: Dict[str, Any]) -> float:
    """L2 norm of translation from a nuScenes-style ego_pose / calibrated_sensor dict."""
    t = np.asarray(pose["translation"], dtype=np.float64).reshape(3)
    return float(np.linalg.norm(t))


def sanity_check_pose_chain(
    translations: Sequence[Sequence[float]],
    max_step_m: float = 5.0,
) -> Tuple[bool, List[str]]:
    """Crude consecutive-pose jump check (ego motion between keyframes).

    Returns (ok, messages). Large jumps may indicate bad sync or wrong sequence.
    """
    msgs: List[str] = []
    arr = [np.asarray(t, dtype=np.float64).reshape(3) for t in translations]
    ok = True
    for i in range(1, len(arr)):
        step = float(np.linalg.norm(arr[i] - arr[i - 1]))
        if step > max_step_m:
            ok = False
            msgs.append(f"pose jump {step:.2f} m between idx {i-1}->{i} > {max_step_m} m")
    if ok and arr:
        msgs.append(f"pose chain OK ({len(arr)} poses, max_step_m={max_step_m})")
    if not arr:
        msgs.append("empty pose list")
    return ok, msgs


def check_rs_time_alignment(
    lidar_timestamps_us: Optional[Sequence[int]] = None,
    cam_timestamps_us: Optional[Sequence[int]] = None,
    max_delta_ms: float = 20.0,
) -> TimeSyncReport:
    """Stub for RoboSense multi-sensor sync.

    TODO(RS):
      - Load hs64 packet / frame timestamps from extracted subset metadata.
      - Load CAM_*_OV frame timestamps; account for exposure mid-point if available.
      - Nearest-neighbor or interpolate ego/extrinsic pose to each camera time.
      - Reject frames with |Δt| > threshold; log into autolabel4d/logs.
      - Wire calibration from RS meta (not from nuScenes calibrated_sensor).
    """
    msgs = [
        "TODO(RS): RoboSense time alignment not implemented; passthrough stub.",
        "Fill when RS_4F_1L subset + calib land under robosense_subset.",
    ]
    if lidar_timestamps_us is None or cam_timestamps_us is None:
        msgs.append("no RS timestamps provided")
        return TimeSyncReport(
            lidar_ts_us=None,
            cam_ts_us=None,
            delta_ms=None,
            ok=False,
            messages=msgs,
        )

    # Minimal nearest-pair check as a placeholder API surface
    l_ts = int(lidar_timestamps_us[0])
    # pick closest cam
    cams = [int(x) for x in cam_timestamps_us]
    c_ts = min(cams, key=lambda c: abs(c - l_ts))
    delta = timestamp_delta_ms(l_ts, c_ts)
    ok = delta <= float(max_delta_ms)
    msgs.append(f"placeholder nearest Δt={delta:.2f} ms (max={max_delta_ms})")
    return TimeSyncReport(
        lidar_ts_us=l_ts,
        cam_ts_us=c_ts,
        delta_ms=delta,
        ok=ok,
        messages=msgs,
    )


__all__ = [
    "TimeSyncReport",
    "timestamp_delta_ms",
    "check_nuscenes_lidar_cam_sync",
    "pose_translation_norm",
    "sanity_check_pose_chain",
    "check_rs_time_alignment",
]
