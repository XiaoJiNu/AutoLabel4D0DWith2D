"""LiDAR teacher stub: fake 3D boxes from nuScenes sample (is_fake). No OpenPCDet yet."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from teachers.base import SerialTeacherGuard

# Default smoke cache root (paths.yaml pseudo_labels / teacher_cache_smoke)
DEFAULT_CACHE_ROOT = Path(
    "/data/data/automomous/autolabel4d/pseudo_labels/teacher_cache_smoke"
)


class LidarTeacherStub:
    """Fake single-sweep 3D detector for Stage D interface smoke only."""

    name = "lidar_stub"

    def __init__(self, cache_root: str | Path | None = None):
        self.cache_root = Path(cache_root) if cache_root else DEFAULT_CACHE_ROOT
        self._loaded = False
        self._nusc = None  # optional NuScenes handle passed via frame or set_nusc

    def set_nusc(self, nusc: Any) -> None:
        self._nusc = nusc

    def load(self) -> None:
        SerialTeacherGuard.acquire(self)
        self._loaded = True
        # No torch / no OpenPCDet — interface smoke only.

    def unload(self) -> None:
        self._loaded = False
        self._nusc = None
        SerialTeacherGuard.release(self)

    def infer_frame(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        if not self._loaded:
            raise RuntimeError(f"{self.name}: call load() before infer_frame()")
        sample_token = frame["sample_token"]
        scene_name = frame.get("scene_name", "unknown")
        sample_idx = int(frame.get("sample_idx", 0))
        nusc = frame.get("nusc", self._nusc)

        objects = self._fake_boxes_from_nuscenes(nusc, sample_token, sample_idx)
        meta = {
            "generator": "teachers.lidar_stub.LidarTeacherStub",
            "is_fake": True,
            "is_stub": True,
            "notes": (
                "Stage D interface smoke only. Real OpenPCDet single-sweep teacher "
                "comes after RoboSense audit; do not use for training."
            ),
        }
        result: Dict[str, Any] = {
            "teacher": self.name,
            "sample_token": sample_token,
            "scene_name": scene_name,
            "objects": objects,
            "meta": meta,
        }
        cache_path = self._write_cache(result, frame.get("lidar_sd_token"))
        result["cache_path"] = str(cache_path)
        return result

    def _fake_boxes_from_nuscenes(
        self, nusc: Any, sample_token: str, sample_idx: int
    ) -> List[Dict[str, Any]]:
        """Build a few synthetic boxes; if nusc given, place them in global via GT annos or lidar pose."""
        catalog = [
            (8.0, 2.0, 0.0, 1.8, 4.5, 1.6, 0.1, "car"),
            (12.0, -3.5, 0.0, 1.9, 4.8, 1.7, -0.2, "car"),
            (5.0, 6.0, 0.0, 0.7, 0.8, 1.7, 1.2, "pedestrian"),
        ]
        n = 2 + (sample_idx % 2)  # 2..3
        objs: List[Dict[str, Any]] = []

        # Prefer transforming lidar-local offsets to global when NuScenes is available
        if nusc is not None:
            try:
                from pyquaternion import Quaternion
                import numpy as np

                sample = nusc.get("sample", sample_token)
                sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
                cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
                ep = nusc.get("ego_pose", sd["ego_pose_token"])
                q_cs = Quaternion(cs["rotation"])
                t_cs = np.array(cs["translation"], dtype=np.float64)
                q_ep = Quaternion(ep["rotation"])
                t_ep = np.array(ep["translation"], dtype=np.float64)

                for i, row in enumerate(catalog[:n]):
                    dx, dy, dz, w, l, h, yaw, cat = row
                    xyz_l = np.array([[dx, dy, dz]], dtype=np.float64)
                    xyz_ego = (q_cs.rotation_matrix @ xyz_l.T).T + t_cs
                    xyz_g = (q_ep.rotation_matrix @ xyz_ego.T).T + t_ep
                    fwd_l = np.array([[np.cos(yaw), np.sin(yaw), 0.0]], dtype=np.float64)
                    origin = np.zeros((1, 3), dtype=np.float64)
                    fwd_ego = (q_cs.rotation_matrix @ fwd_l.T).T
                    origin_ego = (q_cs.rotation_matrix @ origin.T).T
                    fwd_g = (q_ep.rotation_matrix @ fwd_ego.T).T - (
                        q_ep.rotation_matrix @ origin_ego.T
                    ).T
                    yaw_g = float(np.arctan2(fwd_g[0, 1], fwd_g[0, 0]))
                    q = Quaternion(axis=[0, 0, 1], radians=yaw_g)
                    objs.append(
                        {
                            "track_id": f"lidar_stub_{i:03d}",
                            "category": cat,
                            "translation": [
                                float(xyz_g[0, 0]),
                                float(xyz_g[0, 1]),
                                float(xyz_g[0, 2]),
                            ],
                            "size": [w, l, h],
                            "rotation": [float(q.w), float(q.x), float(q.y), float(q.z)],
                            "score": float(0.9 - 0.05 * i),
                            "quality": "C",
                            "valid_fields": ["translation", "size", "rotation", "category"],
                            "ignore": False,
                            "source": "lidar_stub_fake",
                        }
                    )
                return objs
            except Exception:
                pass  # fall through to lidar-local fake

        for i, row in enumerate(catalog[:n]):
            dx, dy, dz, w, l, h, yaw, cat = row
            objs.append(
                {
                    "track_id": f"lidar_stub_{i:03d}",
                    "category": cat,
                    "translation": [dx, dy, dz],
                    "size": [w, l, h],
                    "rotation": [1.0, 0.0, 0.0, 0.0],  # identity; lidar-local if no nusc
                    "score": float(0.9 - 0.05 * i),
                    "quality": "C",
                    "valid_fields": ["translation", "size", "rotation", "category"],
                    "ignore": False,
                    "source": "lidar_stub_fake",
                }
            )
        return objs

    def _write_cache(
        self, result: Dict[str, Any], lidar_sd_token: Optional[str] = None
    ) -> Path:
        self.cache_root.mkdir(parents=True, exist_ok=True)
        token = result["sample_token"]
        out = self.cache_root / f"lidar_stub_{token}.json"
        payload = {
            "version": "1.0",
            "sample_token": result["sample_token"],
            "scene_name": result["scene_name"],
            "lidar_sd_token": lidar_sd_token or "",
            "objects": result["objects"],
            "meta": result["meta"],
        }
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        return out
