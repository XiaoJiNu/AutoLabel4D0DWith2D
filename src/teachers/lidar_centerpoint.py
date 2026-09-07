"""LiDAR CenterPoint teacher (OpenPCDet single-sweep) for RoboSense hs64.

Uses pointFuse OpenPCDet runtime site-packages (8cacccec) + nuScenes voxel0075 ckpt.
HS64 bins: float64 XYZ. Boxes in LiDAR frame. SerialTeacherGuard + unload().
"""
from __future__ import annotations

import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from teachers.base import SerialTeacherGuard

DEFAULT_CACHE_ROOT = Path("/data/data/automomous/autolabel4d/pseudo_labels/rs_repr_lidar")
DEFAULT_CKPT = Path(
    "/data/models/pointfuse/centerpoint/"
    "openpcdet-nuscenes-voxel0075-issue1704-reshare/"
    "cbgs_voxel0075_centerpoint_nds_6648.pth"
)
DEFAULT_OPENPCDET = Path(
    "/data/code/location/pointFuse/output/vendor/"
    "OpenPCDet-8cacccec11db6f59bf6934600c9a175dae254806"
)
DEFAULT_CFG = DEFAULT_OPENPCDET / "tools/cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml"
DEFAULT_VENDOR_TOOLS = DEFAULT_OPENPCDET / "tools"
DEFAULT_RUNTIME_SITE = Path(
    "/data/code/location/pointFuse/output/runtime/"
    "openpcdet-centerpoint-8cacccec/site-packages"
)

NUSCENES_CLASS_NAMES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]


def load_hs64_bin(path: str | Path) -> np.ndarray:
    path = Path(path)
    raw = np.fromfile(str(path), dtype=np.float64)
    if raw.size % 3 != 0:
        raise ValueError(f"hs64 bin not float64 xyz: {path} n={raw.size}")
    xyz = raw.reshape(-1, 3).astype(np.float32)
    finite = np.isfinite(xyz).all(1)
    m = finite & (np.abs(xyz[:, 0]) < 200) & (np.abs(xyz[:, 1]) < 200) & (np.abs(xyz[:, 2]) < 50)
    m &= np.linalg.norm(xyz, axis=1) > 1e-3
    return xyz[m]


def xyz_to_nuscenes_features(xyz: np.ndarray) -> np.ndarray:
    n = xyz.shape[0]
    out = np.zeros((n, 5), dtype=np.float32)
    out[:, :3] = xyz
    return out


def yaw_to_quat_wxyz(yaw: float) -> List[float]:
    half = 0.5 * float(yaw)
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def quality_from_score(score: float) -> str:
    if score >= 0.5:
        return "A"
    if score >= 0.25:
        return "B"
    return "C"


def render_bev_png(
    points_xyz: np.ndarray,
    objects: Sequence[Dict[str, Any]],
    out_png: str | Path,
    title: str = "",
    pc_range: Sequence[float] = (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=120)
    if points_xyz is not None and len(points_xyz) > 0:
        pts = np.asarray(points_xyz)[:, :3]
        if len(pts) > 80000:
            idx = np.linspace(0, len(pts) - 1, 80000).astype(int)
            pts = pts[idx]
        ax.scatter(pts[:, 0], pts[:, 1], s=0.15, c="steelblue", alpha=0.35, linewidths=0)
    for obj in objects:
        x, y, _z = obj["translation"]
        w, l, _h = obj["size"]
        yaw = obj.get("yaw_lidar")
        if yaw is None:
            qw, qx, qy, qz = obj["rotation"]
            yaw = math.atan2(2.0 * (qw * qz), 1.0 - 2.0 * (qz * qz))
        yaw = float(yaw)
        hx, hy = l / 2.0, w / 2.0
        corners = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy], [hx, hy]], dtype=np.float64)
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        xy = corners @ R.T + np.array([x, y], dtype=np.float64)
        ax.plot(xy[:, 0], xy[:, 1], color="crimson", linewidth=1.2)
        ax.plot([x], [y], marker="+", color="orange", markersize=6)
    ax.set_xlim(pc_range[0], pc_range[3])
    ax.set_ylim(pc_range[1], pc_range[4])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, lidar)")
    ax.set_ylabel("y (m, lidar)")
    ax.set_title(title or f"BEV n_obj={len(objects)}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def points_to_bev_png(
    points,
    out_path: str | Path,
    boxes: Optional[Sequence[Dict[str, Any]]] = None,
    pc_range: Tuple[float, float, float, float] = (-54.0, -54.0, 54.0, 54.0),
    resolution: float = 0.2,
) -> Path:
    del resolution
    full = (pc_range[0], pc_range[1], -5.0, pc_range[2], pc_range[3], 3.0)
    return render_bev_png(points, list(boxes or []), out_path, title=Path(out_path).stem, pc_range=full)


class LidarCenterPointTeacher:
    name = "lidar_centerpoint"

    def __init__(
        self,
        ckpt: str | Path | None = None,
        cfg_file: str | Path | None = None,
        cache_root: str | Path | None = None,
        score_thresh: float = 0.1,
        runtime_site: str | Path | None = None,
        vendor_tools: str | Path | None = None,
        openpcdet_root: str | Path | None = None,
        device: str = "cuda:0",
    ) -> None:
        self.ckpt = Path(ckpt) if ckpt else DEFAULT_CKPT
        self.cfg_file = Path(cfg_file) if cfg_file else DEFAULT_CFG
        self.cache_root = Path(cache_root) if cache_root else DEFAULT_CACHE_ROOT
        self.score_thresh = float(score_thresh)
        self.runtime_site = Path(runtime_site) if runtime_site else DEFAULT_RUNTIME_SITE
        if vendor_tools:
            self.vendor_tools = Path(vendor_tools)
        elif openpcdet_root:
            self.vendor_tools = Path(openpcdet_root) / "tools"
        else:
            self.vendor_tools = DEFAULT_VENDOR_TOOLS
        self.device = device
        self._loaded = False
        self._model = None
        self._cfg = None
        self._class_names = list(NUSCENES_CLASS_NAMES)
        self._torch = None
        self._DatasetTemplate = None
        self._build_network = None
        self._load_data_to_gpu = None
        self._common_utils = None
        self._last_error: Optional[str] = None

    def _prep_path(self) -> None:
        site = str(self.runtime_site)
        if site not in sys.path:
            sys.path.insert(0, site)
        vendor = str(DEFAULT_OPENPCDET)
        while vendor in sys.path:
            sys.path.remove(vendor)


    def load(self) -> None:
        SerialTeacherGuard.acquire(self)
        try:
            self._prep_path()
            os.chdir(str(self.vendor_tools))
            import torch
            from pcdet.config import cfg, cfg_from_yaml_file
            from pcdet.datasets import DatasetTemplate
            from pcdet.models import build_network, load_data_to_gpu
            from pcdet.utils import common_utils

            self._torch = torch
            self._DatasetTemplate = DatasetTemplate
            self._build_network = build_network
            self._load_data_to_gpu = load_data_to_gpu
            self._common_utils = common_utils

            if not self.ckpt.is_file():
                raise FileNotFoundError(f"ckpt missing: {self.ckpt}")
            if not self.cfg_file.is_file():
                raise FileNotFoundError(f"cfg missing: {self.cfg_file}")

            cfg_from_yaml_file(str(self.cfg_file), cfg)
            self._cfg = cfg
            self._class_names = list(cfg.CLASS_NAMES)

            class _EmptyDS(DatasetTemplate):
                def __init__(self, dataset_cfg, class_names):
                    super().__init__(
                        dataset_cfg=dataset_cfg,
                        class_names=class_names,
                        training=False,
                        root_path=None,
                        logger=common_utils.create_logger(),
                    )
                    self._points = np.zeros((1, 5), dtype=np.float32)

                def __len__(self):
                    return 1

                def __getitem__(self, index):
                    return self.prepare_data({"points": self._points, "frame_id": 0})

            ds = _EmptyDS(cfg.DATA_CONFIG, cfg.CLASS_NAMES)
            logger = common_utils.create_logger()
            model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=ds)
            model.load_params_from_file(filename=str(self.ckpt), logger=logger, to_cpu=True)
            model.to(self.device)
            model.eval()
            self._model = model
            self._loaded = True
            self._last_error = None
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._loaded = False
            self._model = None
            SerialTeacherGuard.release(self)
            raise

    def unload(self) -> None:
        torch = self._torch
        self._model = None
        self._cfg = None
        self._loaded = False
        gc.collect()
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass
        SerialTeacherGuard.release(self)

    def vram_allocated_mb(self) -> Optional[float]:
        torch = self._torch
        if torch is None or not torch.cuda.is_available():
            return None
        return round(torch.cuda.memory_allocated() / (1024 ** 2), 2)

    def infer_frame(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        if not self._loaded or self._model is None or self._cfg is None:
            raise RuntimeError(f"{self.name}: call load() before infer_frame()")
        torch = self._torch
        assert torch is not None

        lidar_path = frame.get("lidar_path") or frame.get("hs64_path")
        if not lidar_path:
            raise ValueError("frame requires lidar_path")
        sample_token = str(frame.get("sample_token") or Path(lidar_path).stem)
        scene_name = str(frame.get("scene_name") or "SWEEPER-001")
        frame_id = frame.get("frame_id", sample_token)

        xyz = load_hs64_bin(lidar_path)
        n_raw = int(xyz.shape[0])
        pc_range = np.array(self._cfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float32)
        mask = (
            (xyz[:, 0] >= pc_range[0]) & (xyz[:, 0] <= pc_range[3])
            & (xyz[:, 1] >= pc_range[1]) & (xyz[:, 1] <= pc_range[4])
            & (xyz[:, 2] >= pc_range[2]) & (xyz[:, 2] <= pc_range[5])
        )
        xyz_in = xyz[mask]
        n_in = int(xyz_in.shape[0])
        points = xyz_to_nuscenes_features(xyz_in)

        DatasetTemplate = self._DatasetTemplate
        cfg = self._cfg
        common_utils = self._common_utils

        class _OneShot(DatasetTemplate):  # type: ignore[misc,valid-type]
            def __init__(self, dataset_cfg, class_names, pts):
                super().__init__(
                    dataset_cfg=dataset_cfg,
                    class_names=class_names,
                    training=False,
                    root_path=None,
                    logger=common_utils.create_logger(),
                )
                self._points = pts

            def __len__(self):
                return 1

            def __getitem__(self, index):
                return self.prepare_data({"points": self._points, "frame_id": frame_id})

        ds = _OneShot(cfg.DATA_CONFIG, cfg.CLASS_NAMES, points)
        data = ds.collate_batch([ds[0]])
        self._load_data_to_gpu(data)
        with torch.no_grad():
            pred_dicts, _ = self._model.forward(data)

        pred = pred_dicts[0]
        boxes = pred["pred_boxes"].detach().cpu().numpy()
        scores = pred["pred_scores"].detach().cpu().numpy()
        labels = pred["pred_labels"].detach().cpu().numpy().astype(int)
        objects = self._boxes_to_objects(boxes, scores, labels)

        meta = {
            "generator": "teachers.lidar_centerpoint.LidarCenterPointTeacher",
            "is_fake": False,
            "is_stub": False,
            "success": True,
            "notes": (
                "OpenPCDet CenterPoint nuScenes voxel0075 on RS hs64 float64-XYZ; "
                "intensity/timestamp=0; boxes in LiDAR frame."
            ),
            "ckpt": str(self.ckpt),
            "cfg_file": str(self.cfg_file),
            "score_thresh": self.score_thresh,
            "n_points_raw": n_raw,
            "n_points_in_range": n_in,
            "point_cloud_range": [float(x) for x in pc_range.tolist()],
            "n_pred_raw": int(len(scores)),
            "n_objects": len(objects),
            "vram_allocated_mb": self.vram_allocated_mb(),
            "runtime_site": str(self.runtime_site),
        }
        result: Dict[str, Any] = {
            "teacher": self.name,
            "status": "OK",
            "sample_token": sample_token,
            "scene_name": scene_name,
            "lidar_path": str(lidar_path),
            "objects": objects,
            "meta": meta,
            "points_xyz_in_range": xyz_in,
            "num_points": n_in,
        }
        result["cache_path"] = str(self._write_cache(result, frame))
        return result

    def _boxes_to_objects(self, boxes, scores, labels) -> List[Dict[str, Any]]:
        objs: List[Dict[str, Any]] = []
        for i in range(len(scores)):
            score = float(scores[i])
            if score < self.score_thresh:
                continue
            box = boxes[i]
            x, y, z = float(box[0]), float(box[1]), float(box[2])
            dx, dy, dz = float(box[3]), float(box[4]), float(box[5])
            yaw = float(box[6])
            w, l, h = dy, dx, dz
            lab = int(labels[i])
            cat = self._class_names[lab - 1] if 1 <= lab <= len(self._class_names) else f"class_{lab}"
            objs.append({
                "track_id": f"cp_{i:04d}",
                "category": cat,
                "translation": [x, y, z],
                "size": [w, l, h],
                "rotation": yaw_to_quat_wxyz(yaw),
                "score": score,
                "quality": quality_from_score(score),
                "valid_fields": ["translation", "size", "rotation", "category"],
                "ignore": False,
                "source": "lidar_centerpoint",
                "yaw_lidar": yaw,
                "openpcdet_label": lab,
            })
        return objs

    def _write_cache(self, result: Dict[str, Any], frame: Optional[Dict[str, Any]] = None) -> Path:
        frame = frame or {}
        self.cache_root.mkdir(parents=True, exist_ok=True)
        token = result["sample_token"]
        out = self.cache_root / f"lidar_centerpoint_{token}.json"
        payload = {
            "version": "1.0",
            "partial_subset": True,
            "is_rs_partial": True,
            "sample_token": result["sample_token"],
            "scene_name": result["scene_name"],
            "timestamp": frame.get("timestamp", ""),
            "lidar_sd_token": frame.get("lidar_sd_token", ""),
            "lidar_path": result.get("lidar_path", ""),
            "objects": result["objects"],
            "meta": result["meta"],
            "status": result.get("status", "OK"),
        }
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        return out
