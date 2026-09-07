#!/usr/bin/env python3
"""Stage F short train: OpenPCDet PointPillars on RoboSense hs64 + thr>=0.3 pseudo labels.

Prefer hednet-gpu / OpenPCDet PointPillar-MultiHead (nuScenes cbgs_pp_multihead).
Runs <=50 optimizer steps, microbatch=1. Writes:
  - loss.json + loss_curve.png under outputs/student_rs_short/
  - ckpt under /data/data/automomous/autolabel4d/checkpoints/student/
  - status JSON under out-dir

If OpenPCDet build/train fails, falls back to an honest PointPillar-like mini
student (still real hs64 points + labels) and documents how to plug full
OpenPCDet later. No long full train, no KITTI, no multipart, no push.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
for p in (_SRC, _REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CST = timezone(timedelta(hours=8))

DEFAULT_HS64 = Path(
    "/data/data/automomous/robosense/subset/lidar_occ_trainval/"
    "processed_data_20230906/SWEEPER-001/hs64"
)
DEFAULT_PSEUDO = Path(
    "/data/data/automomous/autolabel4d/pseudo_labels/rs_lidar_only_export"
)
DEFAULT_PSEUDO_ALT = Path(
    "/data/data/automomous/autolabel4d/pseudo_labels/rs_repr_lidar_thr03"
)
DEFAULT_CKPT_DIR = Path("/data/data/automomous/autolabel4d/checkpoints/student")
DEFAULT_OUT = _REPO_ROOT / "outputs" / "student_rs_short"
DEFAULT_CFG = Path(
    "/data/code/cv/AutoLabel/BEV-OD/HEDNet/tools/cfgs/nuscenes_models/cbgs_pp_multihead.yaml"
)
DEFAULT_HEDNET = Path("/data/code/cv/AutoLabel/BEV-OD/HEDNet-gpt")
DEFAULT_VENDOR_TOOLS = Path("/data/code/cv/AutoLabel/BEV-OD/HEDNet/tools")

# Map our pseudo-label categories → OpenPCDet nuScenes class names
_CAT_MAP = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "trailer": "trailer",
    "construction_vehicle": "construction_vehicle",
    "pedestrian": "pedestrian",
    "bicycle": "bicycle",
    "motorcycle": "motorcycle",
    "cyclist": "bicycle",
    "vehicle": "car",
    "barrier": "barrier",
    "traffic_cone": "traffic_cone",
}


def _now_cst() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S %Z")


def _nvidia_smi() -> str:
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return (r.stdout or r.stderr or "").strip() or "(empty)"
    except Exception as exc:  # noqa: BLE001
        return f"nvidia-smi error: {exc}"


def load_hs64_bin(path: str | Path) -> np.ndarray:
    """RoboSense hs64 bins are float64 XYZ (no intensity)."""
    raw = np.fromfile(str(path), dtype=np.float64)
    if raw.size % 3 != 0:
        raise ValueError(f"hs64 bin not float64 xyz: {path} n={raw.size}")
    return raw.reshape(-1, 3).astype(np.float32)


def xyz_to_nuscenes_features(xyz: np.ndarray) -> np.ndarray:
    n = int(xyz.shape[0])
    out = np.zeros((n, 5), dtype=np.float32)
    out[:, :3] = xyz
    return out


def discover_pairs(
    hs64_dir: Path,
    pseudo_dirs: Sequence[Path],
    score_thresh: float,
) -> List[Dict[str, Any]]:
    """Match label JSONs to hs64 bins by sample token / stem."""
    bins = {p.stem: p for p in sorted(hs64_dir.glob("*.bin"))}
    pairs: List[Dict[str, Any]] = []
    seen = set()
    for pd in pseudo_dirs:
        if not pd.is_dir():
            continue
        for jp in sorted(pd.glob("*.json")):
            if jp.name.startswith("nuscenes_results") or jp.name.startswith("stage_"):
                continue
            if jp.name.startswith("lidar_centerpoint_"):
                token = jp.name[len("lidar_centerpoint_") : -len(".json")]
            else:
                token = jp.stem
            if token in seen:
                continue
            bin_path = bins.get(token)
            if bin_path is None:
                continue
            try:
                label = json.loads(jp.read_text(encoding="utf-8"))
            except Exception:
                continue
            objs = label.get("objects") or label.get("boxes") or []
            if isinstance(label.get("results"), dict) and not objs:
                # thr03 centerpoint format may nest differently
                continue
            keep = []
            for o in objs:
                if not isinstance(o, dict):
                    continue
                if o.get("ignore"):
                    continue
                sc = float(o.get("detection_score", o.get("score", 0.0)))
                if sc < score_thresh:
                    continue
                keep.append(o)
            if not keep:
                continue
            pairs.append(
                {
                    "token": token,
                    "bin_path": bin_path,
                    "label_path": jp,
                    "n_objects": len(keep),
                    "objects": keep,
                    "scene_name": label.get("scene_name", "SWEEPER-001"),
                    "sample_token": label.get("sample_token", token),
                }
            )
            seen.add(token)
    return pairs


def objects_to_gt(
    objects: Sequence[Dict[str, Any]],
    class_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Pseudo objects → OpenPCDet gt_boxes (N,9) + gt_names.

    Label size is treated as [w, l, h] (width, length, height) as produced by
    the CenterPoint teacher export; OpenPCDet expects dx=length, dy=width, dz=h.
    """
    name_set = set(class_names)
    boxes: List[List[float]] = []
    names: List[str] = []
    for o in objects:
        cat = str(o.get("category", "")).strip().lower()
        mapped = _CAT_MAP.get(cat, cat)
        if mapped not in name_set:
            continue
        x, y, z = [float(v) for v in o["translation"]]
        size = o.get("size") or [1.0, 1.0, 1.0]
        w, l, h = float(size[0]), float(size[1]), float(size[2])
        dx, dy, dz = l, w, h
        if "yaw_lidar" in o:
            yaw = float(o["yaw_lidar"])
        else:
            rot = o.get("rotation") or [1, 0, 0, 0]
            qw, qx, qy, qz = [float(v) for v in rot]
            yaw = math.atan2(2.0 * (qw * qz), 1.0 - 2.0 * (qz * qz))
        boxes.append([x, y, z, dx, dy, dz, 0.0, 0.0, yaw])
        names.append(mapped)
    if not boxes:
        return np.zeros((0, 9), dtype=np.float32), np.array([], dtype=object)
    return np.asarray(boxes, dtype=np.float32), np.asarray(names, dtype=object)


def plot_loss_curve(losses: List[float], out_png: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(losses) + 1), losses, marker="o", markersize=3, linewidth=1.2)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_bev_sample(
    xyz: np.ndarray,
    objects: Sequence[Dict[str, Any]],
    out_png: Path,
    title: str,
    pc_range: Tuple[float, float, float, float, float, float] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    m = (
        (xyz[:, 0] >= pc_range[0])
        & (xyz[:, 0] <= pc_range[3])
        & (xyz[:, 1] >= pc_range[1])
        & (xyz[:, 1] <= pc_range[4])
    )
    pts = xyz[m]
    if len(pts) > 80000:
        idx = np.random.default_rng(0).choice(len(pts), 80000, replace=False)
        pts = pts[idx]
    ax.scatter(pts[:, 0], pts[:, 1], s=0.1, c="gray", alpha=0.35, linewidths=0)
    for o in objects:
        x, y, z = o["translation"]
        w, l, h = o["size"]
        yaw = float(o.get("yaw_lidar", 0.0))
        hx, hy = l / 2.0, w / 2.0
        corners = np.array(
            [[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy], [hx, hy]], dtype=np.float64
        )
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        xy = corners @ R.T + np.array([x, y], dtype=np.float64)
        ax.plot(xy[:, 0], xy[:, 1], color="crimson", linewidth=1.0)
    ax.set_xlim(pc_range[0], pc_range[3])
    ax.set_ylim(pc_range[1], pc_range[4])
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def try_openpcdet_short_train(
    pairs: List[Dict[str, Any]],
    *,
    cfg_file: Path,
    hednet_root: Path,
    vendor_tools: Path,
    max_steps: int,
    lr: float,
    device: str,
    ckpt_path: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    """Real OpenPCDet PointPillars ≤max_steps train. Raises on hard failure."""
    # Strip shadowing runtime sites; prefer editable HEDNet-gpt pcdet
    sys.path = [p for p in sys.path if "runtime" not in p.lower()]
    for bad in (
        "/data/code/cv/BEV/Lidar_AI_Solution/CUDA-PointPillars/OpenPCDet",
    ):
        while bad in sys.path:
            sys.path.remove(bad)
    if str(hednet_root) not in sys.path:
        sys.path.insert(0, str(hednet_root))

    os.chdir(str(vendor_tools))

    import torch
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import DatasetTemplate
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils

    cfg_from_yaml_file(str(cfg_file), cfg)
    # No DB / heavy augs for short train on RS pseudo labels
    cfg.DATA_CONFIG.DATA_AUGMENTOR = {
        "DISABLE_AUG_LIST": [
            "gt_sampling",
            "random_world_flip",
            "random_world_rotation",
            "random_world_scaling",
        ],
        "AUG_CONFIG_LIST": [],
    }

    class_names = list(cfg.CLASS_NAMES)
    pc_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float32)
    logger = common_utils.create_logger()

    # Preload samples
    samples: List[Dict[str, Any]] = []
    for pair in pairs:
        xyz = load_hs64_bin(pair["bin_path"])
        m = (
            (xyz[:, 0] >= pc_range[0])
            & (xyz[:, 0] <= pc_range[3])
            & (xyz[:, 1] >= pc_range[1])
            & (xyz[:, 1] <= pc_range[4])
            & (xyz[:, 2] >= pc_range[2])
            & (xyz[:, 2] <= pc_range[5])
        )
        pts = xyz_to_nuscenes_features(xyz[m])
        gt_boxes, gt_names = objects_to_gt(pair["objects"], class_names)
        if len(gt_boxes) == 0:
            continue
        samples.append(
            {
                "token": pair["token"],
                "points": pts,
                "gt_boxes": gt_boxes,
                "gt_names": gt_names,
                "xyz_full": xyz,
                "objects": pair["objects"],
            }
        )
    if not samples:
        raise RuntimeError("no samples with in-range GT after filtering")

    class RSPillarDS(DatasetTemplate):
        def __init__(self, dataset_cfg, class_names_, samples_):
            super().__init__(
                dataset_cfg=dataset_cfg,
                class_names=class_names_,
                training=True,
                root_path=None,
                logger=logger,
            )
            self._samples = samples_

        def __len__(self):
            return len(self._samples)

        def __getitem__(self, index):
            s = self._samples[index % len(self._samples)]
            data_dict = {
                "points": s["points"].copy(),
                "frame_id": s["token"],
                "gt_boxes": s["gt_boxes"].copy(),
                "gt_names": s["gt_names"].copy(),
            }
            return self.prepare_data(data_dict)

    ds = RSPillarDS(cfg.DATA_CONFIG, class_names, samples)
    model = build_network(
        model_cfg=cfg.MODEL, num_class=len(class_names), dataset=ds
    )
    model.to(device)
    model.train()
    # Freeze BN for short train stability (F2/F3 note)
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.BatchNorm2d):
            m.eval()

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    losses: List[float] = []
    tb_last: Dict[str, Any] = {}
    t0 = time.time()
    smi_before = _nvidia_smi()
    n_samp = len(samples)

    for step in range(1, max_steps + 1):
        sidx = (step - 1) % n_samp
        data = ds.collate_batch([ds[sidx]])
        load_data_to_gpu(data)
        opt.zero_grad(set_to_none=True)
        ret = model(data)
        if isinstance(ret, tuple) and len(ret) >= 1:
            loss_obj = ret[0]
            if isinstance(loss_obj, dict):
                loss = loss_obj["loss"]
            else:
                loss = loss_obj
            if len(ret) > 1 and isinstance(ret[1], dict):
                tb_last = {k: float(v) for k, v in ret[1].items()}
        elif isinstance(ret, dict):
            loss = ret["loss"]
        else:
            raise RuntimeError(f"unexpected model return: {type(ret)}")
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu().item()))
        if step == 1 or step == max_steps or step % 10 == 0:
            print(
                f"[openpcdet] step={step}/{max_steps} loss={losses[-1]:.4f} "
                f"token={samples[sidx]['token']} vram_mb="
                f"{torch.cuda.memory_allocated()/1024**2:.0f}",
                flush=True,
            )

    elapsed = time.time() - t0
    smi_after = _nvidia_smi()
    vram_peak = (
        round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        if torch.cuda.is_available()
        else None
    )

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "losses": losses,
            "steps": len(losses),
            "cfg_file": str(cfg_file),
            "class_names": class_names,
            "backend": "openpcdet_pointpillar_multihead",
            "tokens": [s["token"] for s in samples],
            "lr": lr,
            "microbatch": 1,
            "created_at_cst": _now_cst(),
        },
        ckpt_path,
    )

    loss_json = {
        "backend": "openpcdet_pointpillar_multihead",
        "cfg_file": str(cfg_file),
        "steps": len(losses),
        "losses": losses,
        "tb_last": tb_last,
        "n_samples": n_samp,
        "tokens": [s["token"] for s in samples],
        "elapsed_sec": round(elapsed, 3),
        "vram_peak_mb": vram_peak,
        "lr": lr,
        "microbatch": 1,
        "ckpt": str(ckpt_path),
    }
    (out_dir / "loss.json").write_text(
        json.dumps(loss_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    plot_loss_curve(
        losses,
        out_dir / "loss_curve.png",
        title=f"OpenPCDet PointPillars short train ({len(losses)} steps)",
    )
    # BEV viz of first sample + GT
    s0 = samples[0]
    plot_bev_sample(
        s0["xyz_full"],
        s0["objects"],
        out_dir / f"bev_{s0['token']}.png",
        title=f"GT thr>=0.3 | {s0['token']} | n={len(s0['objects'])}",
    )

    # free
    del model, opt, ds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "ok": True,
        "backend": "openpcdet_pointpillar_multihead",
        "steps": len(losses),
        "losses": losses,
        "ckpt": str(ckpt_path),
        "elapsed_sec": round(elapsed, 3),
        "vram_peak_mb": vram_peak,
        "smi_before": smi_before,
        "smi_after": smi_after,
        "n_samples": n_samp,
        "cfg_file": str(cfg_file),
        "tb_last": tb_last,
        "is_fake": False,
    }


def fallback_pillarlike_short_train(
    pairs: List[Dict[str, Any]],
    *,
    max_steps: int,
    lr: float,
    device: str,
    ckpt_path: Path,
    out_dir: Path,
    reason: str,
) -> Dict[str, Any]:
    """Honest mini PointPillar-like student: pillar scatter + tiny BEV CNN.

    Still loads real hs64 + labels. Documents OpenPCDet plug-in path.
    """
    import torch
    from torch import nn

    class PillarLikeStudent(nn.Module):
        def __init__(self, grid: int = 128, n_classes: int = 3):
            super().__init__()
            self.grid = grid
            self.enc = nn.Sequential(
                nn.Conv2d(4, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.head = nn.Conv2d(64, n_classes + 2, 1)  # cls logits + (dx,dy) proxy

        def pillars_from_xyz(self, xyz: np.ndarray, pc_range=(-51.2, -51.2, 51.2, 51.2)):
            x0, y0, x1, y1 = pc_range
            g = self.grid
            sx = (x1 - x0) / g
            sy = (y1 - y0) / g
            m = (
                (xyz[:, 0] >= x0)
                & (xyz[:, 0] < x1)
                & (xyz[:, 1] >= y0)
                & (xyz[:, 1] < y1)
            )
            pts = xyz[m]
            if len(pts) == 0:
                return torch.zeros(1, 4, g, g)
            ix = np.clip(((pts[:, 0] - x0) / sx).astype(np.int64), 0, g - 1)
            iy = np.clip(((pts[:, 1] - y0) / sy).astype(np.int64), 0, g - 1)
            feat = np.zeros((4, g, g), dtype=np.float32)
            # density / mean x / mean y / mean z (simple pillar stats)
            counts = np.zeros((g, g), dtype=np.float32)
            for i in range(len(pts)):
                xi, yi = ix[i], iy[i]
                counts[yi, xi] += 1.0
                feat[1, yi, xi] += pts[i, 0]
                feat[2, yi, xi] += pts[i, 1]
                feat[3, yi, xi] += pts[i, 2]
            nz = counts > 0
            feat[0] = np.log1p(counts)
            feat[1][nz] /= counts[nz]
            feat[2][nz] /= counts[nz]
            feat[3][nz] /= counts[nz]
            return torch.from_numpy(feat).unsqueeze(0)

        def forward(self, bev, target_cls, target_off):
            h = self.enc(bev)
            out = self.head(h)
            # downsample target to match
            th, tw = out.shape[-2:]
            tcls = torch.nn.functional.interpolate(
                target_cls, size=(th, tw), mode="nearest"
            )
            toff = torch.nn.functional.interpolate(
                target_off, size=(th, tw), mode="bilinear", align_corners=False
            )
            cls_logits = out[:, :3]
            off_pred = out[:, 3:]
            # sparse CE on occupied cells
            mask = (tcls.sum(dim=1, keepdim=True) > 0).float()
            # soft target from one-hot-ish map
            target_idx = tcls.argmax(dim=1)
            loss_cls = torch.nn.functional.cross_entropy(cls_logits, target_idx, reduction="none")
            loss_cls = (loss_cls * mask.squeeze(1)).sum() / mask.sum().clamp_min(1.0)
            loss_off = ((off_pred - toff).abs() * mask).sum() / mask.sum().clamp_min(1.0)
            return loss_cls + 0.1 * loss_off

    def targets_from_objects(objects, grid=128, pc_range=(-51.2, -51.2, 51.2, 51.2)):
        x0, y0, x1, y1 = pc_range
        sx = (x1 - x0) / grid
        sy = (y1 - y0) / grid
        cls = np.zeros((3, grid, grid), dtype=np.float32)
        off = np.zeros((2, grid, grid), dtype=np.float32)
        coarse = {"vehicle": 0, "car": 0, "truck": 0, "bus": 0, "trailer": 0,
                  "construction_vehicle": 0, "pedestrian": 1, "bicycle": 2,
                  "motorcycle": 2, "cyclist": 2}
        for o in objects:
            cat = str(o.get("category", "")).lower()
            ci = coarse.get(cat)
            if ci is None:
                continue
            x, y, z = o["translation"]
            if not (x0 <= x < x1 and y0 <= y < y1):
                continue
            ix = int(np.clip((x - x0) / sx, 0, grid - 1))
            iy = int(np.clip((y - y0) / sy, 0, grid - 1))
            cls[ci, iy, ix] = 1.0
            off[0, iy, ix] = ((x - x0) / sx) - ix
            off[1, iy, ix] = ((y - y0) / sy) - iy
        return torch.from_numpy(cls).unsqueeze(0), torch.from_numpy(off).unsqueeze(0)

    model = PillarLikeStudent().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses: List[float] = []
    t0 = time.time()
    smi_before = _nvidia_smi()

    # preload
    loaded = []
    for pair in pairs:
        xyz = load_hs64_bin(pair["bin_path"])
        loaded.append((pair, xyz))
    if not loaded:
        raise RuntimeError("fallback: no pairs")

    for step in range(1, max_steps + 1):
        pair, xyz = loaded[(step - 1) % len(loaded)]
        bev = model.pillars_from_xyz(xyz).to(device)
        tcls, toff = targets_from_objects(pair["objects"])
        tcls, toff = tcls.to(device), toff.to(device)
        opt.zero_grad(set_to_none=True)
        loss = model(bev, tcls, toff)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu().item()))
        if step == 1 or step == max_steps or step % 10 == 0:
            print(f"[pillarlike] step={step}/{max_steps} loss={losses[-1]:.4f}", flush=True)

    elapsed = time.time() - t0
    smi_after = _nvidia_smi()
    vram_peak = (
        round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        if torch.cuda.is_available()
        else None
    )
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "losses": losses,
            "steps": len(losses),
            "backend": "pillarlike_fallback",
            "fallback_reason": reason,
            "openpcdet_plug_in": {
                "env": "hednet-gpu",
                "cfg": str(DEFAULT_CFG),
                "script": "scripts/run_student_rs_short.py --backend openpcdet",
                "notes": (
                    "Real OpenPCDet PointPillar-MultiHead path is preferred; "
                    "this fallback only runs when OpenPCDet short train fails."
                ),
            },
            "created_at_cst": _now_cst(),
        },
        ckpt_path,
    )
    loss_json = {
        "backend": "pillarlike_fallback",
        "fallback_reason": reason,
        "steps": len(losses),
        "losses": losses,
        "elapsed_sec": round(elapsed, 3),
        "vram_peak_mb": vram_peak,
        "ckpt": str(ckpt_path),
    }
    (out_dir / "loss.json").write_text(
        json.dumps(loss_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    plot_loss_curve(
        losses,
        out_dir / "loss_curve.png",
        title=f"PillarLike fallback short train ({len(losses)} steps)",
    )
    pair0, xyz0 = loaded[0]
    plot_bev_sample(
        xyz0,
        pair0["objects"],
        out_dir / f"bev_{pair0['token']}.png",
        title=f"GT thr>=0.3 | {pair0['token']}",
    )
    del model, opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "ok": True,
        "backend": "pillarlike_fallback",
        "fallback_reason": reason,
        "steps": len(losses),
        "losses": losses,
        "ckpt": str(ckpt_path),
        "elapsed_sec": round(elapsed, 3),
        "vram_peak_mb": vram_peak,
        "smi_before": smi_before,
        "smi_after": smi_after,
        "n_samples": len(loaded),
        "is_fake": False,
        "notes": "Honest pillar-like mini net on real hs64+labels; not FakeStudent Linear.",
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stage F RS PointPillars short train")
    ap.add_argument("--hs64-dir", type=Path, default=DEFAULT_HS64)
    ap.add_argument("--pseudo-dir", type=Path, default=DEFAULT_PSEUDO)
    ap.add_argument("--pseudo-dir-alt", type=Path, default=DEFAULT_PSEUDO_ALT)
    ap.add_argument("--score-thresh", type=float, default=0.3)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    ap.add_argument("--hednet-root", type=Path, default=DEFAULT_HEDNET)
    ap.add_argument("--vendor-tools", type=Path, default=DEFAULT_VENDOR_TOOLS)
    ap.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--backend",
        choices=["auto", "openpcdet", "pillarlike"],
        default="auto",
        help="auto: try OpenPCDet then fallback",
    )
    ap.add_argument("--dry-run", action="store_true", help="Validate only, no train")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"[rs_short] start {_now_cst()}", flush=True)
    print(f"[rs_short] hs64={args.hs64_dir}", flush=True)
    print(f"[rs_short] pseudo={args.pseudo_dir} alt={args.pseudo_dir_alt}", flush=True)
    print(f"[rs_short] score_thresh={args.score_thresh} max_steps={args.max_steps}", flush=True)
    print(f"[rs_short] nvidia-smi:\n{_nvidia_smi()}", flush=True)

    pairs = discover_pairs(
        args.hs64_dir,
        [args.pseudo_dir, args.pseudo_dir_alt],
        args.score_thresh,
    )
    print(f"[rs_short] paired frames with labels: {len(pairs)}", flush=True)
    for p in pairs:
        print(f"  - {p['token']} n_obj={p['n_objects']} label={p['label_path'].name}", flush=True)

    status: Dict[str, Any] = {
        "task": "stage_f_pointpillars_short",
        "started_at_cst": _now_cst(),
        "hs64_dir": str(args.hs64_dir),
        "pseudo_dirs": [str(args.pseudo_dir), str(args.pseudo_dir_alt)],
        "score_thresh": args.score_thresh,
        "n_pairs": len(pairs),
        "tokens": [p["token"] for p in pairs],
        "max_steps": args.max_steps,
        "microbatch": 1,
        "command": " ".join(sys.argv),
    }

    if len(pairs) == 0:
        status["status"] = "BLOCKED"
        status["reason"] = "no hs64+label pairs with score>=thresh"
        (out_dir / "status.json").write_text(
            json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print("[rs_short] BLOCKED: no pairs", flush=True)
        return 2

    if args.dry_run:
        status["status"] = "DRY_READY"
        status["reason"] = f"{len(pairs)} pairs ready; dry-run only"
        (out_dir / "status.json").write_text(
            json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print("[rs_short] DRY_READY", flush=True)
        return 0

    ckpt_path = args.ckpt_dir / "pointpillars_rs_short.pth"
    result: Optional[Dict[str, Any]] = None
    openpcdet_error: Optional[str] = None

    if args.backend in ("auto", "openpcdet"):
        try:
            print("[rs_short] trying OpenPCDet PointPillars …", flush=True)
            result = try_openpcdet_short_train(
                pairs,
                cfg_file=args.cfg,
                hednet_root=args.hednet_root,
                vendor_tools=args.vendor_tools,
                max_steps=args.max_steps,
                lr=args.lr,
                device=args.device,
                ckpt_path=ckpt_path,
                out_dir=out_dir,
            )
        except Exception as exc:  # noqa: BLE001
            openpcdet_error = f"{type(exc).__name__}: {exc}"
            print(f"[rs_short] OpenPCDet failed: {openpcdet_error}", flush=True)
            traceback.print_exc()
            if args.backend == "openpcdet":
                status["status"] = "BLOCKED"
                status["reason"] = openpcdet_error
                status["traceback"] = traceback.format_exc()[-1500:]
                (out_dir / "status.json").write_text(
                    json.dumps(status, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                return 3
            ckpt_path = args.ckpt_dir / "pillarlike_rs_short.pth"
            result = fallback_pillarlike_short_train(
                pairs,
                max_steps=args.max_steps,
                lr=args.lr,
                device=args.device,
                ckpt_path=ckpt_path,
                out_dir=out_dir,
                reason=openpcdet_error,
            )
    else:
        ckpt_path = args.ckpt_dir / "pillarlike_rs_short.pth"
        result = fallback_pillarlike_short_train(
            pairs,
            max_steps=args.max_steps,
            lr=args.lr,
            device=args.device,
            ckpt_path=ckpt_path,
            out_dir=out_dir,
            reason="backend=pillarlike requested",
        )

    assert result is not None
    status["status"] = "TRAINED_SHORT"
    status["result"] = result
    status["ckpt"] = result.get("ckpt")
    status["finished_at_cst"] = _now_cst()
    if openpcdet_error:
        status["openpcdet_error"] = openpcdet_error
    (out_dir / "status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"[rs_short] TRAINED_SHORT backend={result.get('backend')} "
        f"steps={result.get('steps')} ckpt={result.get('ckpt')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
