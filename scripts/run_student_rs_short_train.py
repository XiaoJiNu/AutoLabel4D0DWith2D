#!/usr/bin/env python3
"""Stage F SHORT PointPillar-MultiHead train on RS LiDAR-only pseudo labels.

Constraints (幕僚长):
  - microbatch=1, few steps (default 30), stoppable
  - NO long full training / multipart / kitti / git push
  - Do NOT wait for GDINO/SAM
  - Unload GPU after
  - Prefer quality A/B; if too few frames, optionally include C

If real OpenPCDet PointPillars cannot start, falls back to FakeStudent 1-iter
and writes a documented real-train command + BLOCKED/DRY_READY status.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import pickle
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

CST = timezone(timedelta(hours=8))

REPO = Path("/data/code/cv/AutoLabel/AutoLabel4D0DWith2D")
PSEUDO_ROOT = Path("/data/data/automomous/autolabel4d/pseudo_labels/rs_lidar_only_export")
HS64_ROOT = Path(
    "/data/data/automomous/robosense/subset/lidar_occ_trainval/"
    "processed_data_20230906/SWEEPER-001/hs64"
)
OUT_DIR = REPO / "outputs" / "student_rs_short"
CKPT_DIR = Path("/data/data/automomous/autolabel4d/checkpoints/student")
STATUS_MD = Path("/home/yr/grok-bot-work/autolabel4d_stageB/RS_STUDENT_SHORT_STATUS.md")
CFG_YAML = OUT_DIR / "pp_multihead_rs_short.yaml"
HEDNET = Path("/data/code/cv/AutoLabel/BEV-OD/HEDNet-gpt")
VENDOR_TOOLS = HEDNET / "tools"

# Keep nuScenes fine names that exist in CLASS_NAMES; map coarse aliases.
CAT_MAP = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "trailer": "trailer",
    "construction_vehicle": "construction_vehicle",
    "barrier": "barrier",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "pedestrian": "pedestrian",
    "traffic_cone": "traffic_cone",
    "vehicle": "car",
    "cyclist": "bicycle",
}


def now_cst() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S CST")


def nvidia_smi() -> str:
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
        return (r.stdout or "").strip() or f"rc={r.returncode}"
    except Exception as exc:  # noqa: BLE001
        return f"nvidia-smi error: {exc}"


def load_hs64_bin(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.float64)
    if raw.size % 3 != 0:
        raise ValueError(f"hs64 bin not float64 xyz: {path} n={raw.size}")
    xyz = raw.reshape(-1, 3).astype(np.float32)
    finite = np.isfinite(xyz).all(1)
    m = (
        finite
        & (np.abs(xyz[:, 0]) < 200)
        & (np.abs(xyz[:, 1]) < 200)
        & (np.abs(xyz[:, 2]) < 50)
        & (np.linalg.norm(xyz, axis=1) > 1e-3)
    )
    return xyz[m]


def xyz_to_nuscenes_features(xyz: np.ndarray) -> np.ndarray:
    n = xyz.shape[0]
    out = np.zeros((n, 5), dtype=np.float32)
    out[:, :3] = xyz
    return out


def objects_to_gt(
    objects: Sequence[Dict[str, Any]],
    quality_keep: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Return gt_boxes (N,9)=[x,y,z,dx,dy,dz,yaw,vx,vy], gt_names (N,), counts."""
    boxes: List[List[float]] = []
    names: List[str] = []
    qcounts = {q: 0 for q in ("A", "B", "C", "other")}
    for obj in objects:
        if not isinstance(obj, dict) or obj.get("ignore"):
            continue
        q = str(obj.get("quality") or "other")
        if q not in qcounts:
            qcounts["other"] += 1
        else:
            qcounts[q] += 1
        if q not in quality_keep:
            continue
        cat = CAT_MAP.get(str(obj.get("category") or "").lower())
        if cat is None:
            continue
        x, y, z = [float(v) for v in obj["translation"][:3]]
        # label size is nuScenes wlh; OpenPCDet wants lwh (dx,dy,dz)
        w, l, h = [float(v) for v in obj["size"][:3]]
        yaw = obj.get("yaw_lidar")
        if yaw is None:
            qw, qx, qy, qz = [float(v) for v in obj["rotation"][:4]]
            yaw = math.atan2(2.0 * (qw * qz), 1.0 - 2.0 * (qz * qz))
        boxes.append([x, y, z, l, w, h, float(yaw), 0.0, 0.0])
        names.append(cat)
    if not boxes:
        return (
            np.zeros((0, 9), dtype=np.float32),
            np.array([], dtype=object),
            qcounts,
        )
    return np.asarray(boxes, dtype=np.float32), np.asarray(names, dtype=object), qcounts


def build_infos(
    pseudo_root: Path,
    hs64_root: Path,
    quality_keep: Sequence[str],
) -> Dict[str, Any]:
    samples: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for jp in sorted(pseudo_root.glob("*.json")):
        if jp.name.startswith("nuscenes") or jp.name.startswith("stage"):
            continue
        label = json.loads(jp.read_text())
        token = str(label.get("sample_token") or jp.stem)
        bin_path = hs64_root / f"{token}.bin"
        if not bin_path.is_file():
            skipped.append({"token": token, "reason": f"missing bin {bin_path}"})
            continue
        gt_boxes, gt_names, qcounts = objects_to_gt(
            label.get("objects") or [], quality_keep
        )
        if len(gt_boxes) == 0:
            skipped.append(
                {
                    "token": token,
                    "reason": "no boxes after quality filter",
                    "qcounts": qcounts,
                }
            )
            continue
        xyz = load_hs64_bin(bin_path)
        samples.append(
            {
                "token": token,
                "lidar_path": str(bin_path),
                "label_path": str(jp),
                "n_points": int(xyz.shape[0]),
                "n_gt": int(gt_boxes.shape[0]),
                "gt_names": gt_names.tolist(),
                "gt_boxes": gt_boxes.tolist(),
                "qcounts_all": qcounts,
                "scene_name": label.get("scene_name"),
                "timestamp": label.get("timestamp"),
            }
        )
    return {
        "created_at": now_cst(),
        "pseudo_root": str(pseudo_root),
        "hs64_root": str(hs64_root),
        "quality_keep": list(quality_keep),
        "n_samples": len(samples),
        "samples": samples,
        "skipped": skipped,
    }


def write_infos_pkl(infos_doc: Dict[str, Any], out_pkl: Path) -> Path:
    """Write OpenPCDet-ish list of sample infos (custom loader uses this)."""
    rows = []
    for s in infos_doc["samples"]:
        rows.append(
            {
                "token": s["token"],
                "lidar_path": s["lidar_path"],
                "gt_boxes": np.asarray(s["gt_boxes"], dtype=np.float32),
                "gt_names": np.asarray(s["gt_names"], dtype=object),
                "n_points": s["n_points"],
            }
        )
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(rows, f)
    return out_pkl


def prep_pcdet_path() -> None:
    # Prefer HEDNet editable pcdet (same as lidar teacher).
    for bad in list(sys.path):
        if "OpenPCDet-" in bad or "openpcdet-centerpoint" in bad:
            try:
                sys.path.remove(bad)
            except ValueError:
                pass
    hed = str(HEDNET)
    while hed in sys.path:
        sys.path.remove(hed)
    sys.path.insert(0, hed)


def run_real_pp_short(
    infos_pkl: Path,
    cfg_yaml: Path,
    ckpt_out: Path,
    *,
    max_steps: int = 30,
    lr: float = 1e-3,
    freeze_backbone: bool = True,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    prep_pcdet_path()
    os.chdir(str(VENDOR_TOOLS))

    import torch
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import DatasetTemplate
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils

    cfg_from_yaml_file(str(cfg_yaml), cfg)
    # Ensure no GT-DB sampling path is required
    if hasattr(cfg.DATA_CONFIG, "DATA_AUGMENTOR"):
        cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST = [
            "gt_sampling",
            "random_world_flip",
            "random_world_rotation",
            "random_world_scaling",
        ]
        cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST = []

    with open(infos_pkl, "rb") as f:
        infos = pickle.load(f)
    if not infos:
        raise RuntimeError("empty infos; cannot train")

    logger = common_utils.create_logger()

    class RSShortDS(DatasetTemplate):
        def __init__(self):
            super().__init__(
                dataset_cfg=cfg.DATA_CONFIG,
                class_names=list(cfg.CLASS_NAMES),
                training=True,
                root_path=None,
                logger=logger,
            )
            self.infos = infos

        def __len__(self):
            return len(self.infos)

        def __getitem__(self, index):
            info = self.infos[index % len(self.infos)]
            xyz = load_hs64_bin(Path(info["lidar_path"]))
            points = xyz_to_nuscenes_features(xyz)
            data_dict = {
                "points": points,
                "frame_id": info["token"],
                "gt_boxes": np.asarray(info["gt_boxes"], dtype=np.float32).copy(),
                "gt_names": np.asarray(info["gt_names"], dtype=object).copy(),
            }
            return self.prepare_data(data_dict=data_dict)

    ds = RSShortDS()
    model = build_network(
        model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=ds
    )
    model.to(device)
    model.train()

    # Freeze backbone (+ BN) per student_pointpillar_v3 F2/F3 note
    frozen = []
    if freeze_backbone:
        for name, mod in model.named_modules():
            # keep dense_head trainable; freeze VFE / map_to_bev / backbone_2d
            top = name.split(".")[0] if name else ""
            if top in ("vfe", "map_to_bev_module", "backbone_2d"):
                for p in mod.parameters(recurse=False):
                    p.requires_grad = False
                if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                    mod.eval()
                    frozen.append(name)
        # also freeze nested BN under those tops
        for name, mod in model.named_modules():
            top = name.split(".")[0] if name else ""
            if top in ("vfe", "map_to_bev_module", "backbone_2d") and isinstance(
                mod, torch.nn.modules.batchnorm._BatchNorm
            ):
                mod.eval()
                for p in mod.parameters():
                    p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_all = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=0.01)

    smi_before = nvidia_smi()
    losses: List[float] = []
    step_logs: List[Dict[str, Any]] = []
    t0 = time.time()
    model.train()

    for step in range(max_steps):
        idx = step % len(ds)
        # prepare_data may re-sample if empty after range filter; retry a few times
        data = None
        for _try in range(5):
            try:
                data = ds[idx]
                break
            except Exception:
                idx = (idx + 1) % len(ds)
        if data is None:
            raise RuntimeError(f"failed to prepare sample at step {step}")

        batch = ds.collate_batch([data])
        load_data_to_gpu(batch)
        opt.zero_grad(set_to_none=True)
        ret, tb_dict, _disp = model(batch)
        loss = ret["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        lv = float(loss.detach().item())
        losses.append(lv)
        step_logs.append(
            {
                "step": step,
                "idx": idx,
                "token": str(batch.get("frame_id", ["?"])[0]),
                "loss": lv,
                "tb": {k: float(v) for k, v in (tb_dict or {}).items() if np.isscalar(v)},
            }
        )
        if step % 5 == 0 or step == max_steps - 1:
            print(
                f"[step {step+1}/{max_steps}] loss={lv:.4f} token={step_logs[-1]['token']}",
                flush=True,
            )

    elapsed = time.time() - t0
    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "step": max_steps,
        "losses": losses,
        "cfg": str(cfg_yaml),
        "infos_pkl": str(infos_pkl),
        "freeze_backbone": freeze_backbone,
        "n_trainable": n_train,
        "n_params": n_all,
        "created_at": now_cst(),
        "note": "SHORT from-scratch PointPillar-MultiHead; no official PP pretrained loaded",
    }
    torch.save(state, str(ckpt_out))
    smi_peak = nvidia_smi()

    # unload
    del opt
    model.cpu()
    del model
    del ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    smi_after = nvidia_smi()

    return {
        "ok": True,
        "mode": "real_pointpillar_multihead",
        "steps": max_steps,
        "losses": losses,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "ckpt": str(ckpt_out),
        "elapsed_sec": round(elapsed, 3),
        "n_trainable": n_train,
        "n_params": n_all,
        "freeze_backbone": freeze_backbone,
        "nvidia_smi_before": smi_before,
        "nvidia_smi_peak": smi_peak,
        "nvidia_smi_after": smi_after,
        "step_logs_tail": step_logs[-5:],
        "pretrained": None,
        "pretrained_note": "official OpenPCDet PointPillars pretrained MISSING; trained from random init",
    }


def run_fake_student(pseudo_root: Path, out_dir: Path, device: str = "cuda") -> Dict[str, Any]:
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO))
    from student.pointpillar_smoke import (  # type: ignore
        FakeStudent,
        discover_fusion_pseudo,
        validate_dataloader_inputs,
    )

    summary = validate_dataloader_inputs(pseudo_root, require_n=1)
    labels = []
    from export.pseudo_label_io import read_label  # type: ignore

    for p in discover_fusion_pseudo(pseudo_root)[:8]:
        labels.append(read_label(p))
    fs = FakeStudent()
    use_cuda = device.startswith("cuda")
    import torch

    dev = "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
    smi_b = nvidia_smi()
    fs.build(device=dev)
    t0 = time.time()
    result = fs.train_one_step(labels, device=dev)
    elapsed = time.time() - t0
    # unload
    fs._model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    smi_a = nvidia_smi()
    result.update(
        {
            "ok": True,
            "mode": "fake_student",
            "elapsed_sec": round(elapsed, 3),
            "n_valid_pseudo": summary["n_valid"],
            "nvidia_smi_before": smi_b,
            "nvidia_smi_after": smi_a,
        }
    )
    (out_dir / "fake_student_result.json").write_text(json.dumps(result, indent=2))
    return result


def documented_real_train_cmd(infos_pkl: Path, cfg_yaml: Path, ckpt: Path) -> str:
    return f"""# Documented real OpenPCDet PointPillar-MultiHead SHORT train (Legion / hednet-gpu)
# Pretrained official PP weights are MISSING on this machine — this runs from-scratch SHORT.
PY=/data/software/conda/anaconda3/envs/hednet-gpu/bin/python
cd /data/code/cv/AutoLabel/AutoLabel4D0DWith2D
$PY scripts/run_student_rs_short_train.py \\
  --steps 30 --lr 1e-3 --freeze-backbone \\
  --pseudo-root {PSEUDO_ROOT} \\
  --quality A,B \\
  --cfg {cfg_yaml} \\
  --ckpt {ckpt}
# Infos pickle: {infos_pkl}
# Env: hednet-gpu (torch+spconv+pcdet via HEDNet-gpt)
# Do NOT start long full training; unload GPU after.
"""


def write_status_md(doc: Dict[str, Any]) -> None:
    status = doc["status"]
    train = doc.get("train") or {}
    lines = [
        f"# RS Student PointPillars SHORT — Stage F",
        "",
        f"**Time:** {doc.get('created_at')}  ",
        f"**Host:** Legion (`252f7d1f-95fc-44c6-94d1-8f76153ee6af`)  ",
        f"**Status:** `{status}`  ",
        f"**Labels:** `{PSEUDO_ROOT}` (LiDAR-only; no GDINO/SAM wait)  ",
        f"**Config base:** `configs/student_pointpillar_v3.yaml` + `{CFG_YAML.name}`  ",
        "",
        "## Summary",
        "",
        f"- mode: `{train.get('mode')}`",
        f"- steps: `{train.get('steps')}`",
        f"- loss_first / loss_last: `{train.get('loss_first')}` / `{train.get('loss_last')}`",
        f"- ckpt: `{train.get('ckpt')}`",
        f"- elapsed_sec: `{train.get('elapsed_sec')}`",
        f"- freeze_backbone: `{train.get('freeze_backbone')}`",
        f"- n_trainable / n_params: `{train.get('n_trainable')}` / `{train.get('n_params')}`",
        f"- pretrained: `{train.get('pretrained')}` — {train.get('pretrained_note')}",
        "",
        "## Dataset infos",
        "",
        f"- infos_json: `{doc.get('infos_json')}`",
        f"- infos_pkl: `{doc.get('infos_pkl')}`",
        f"- n_train_samples (quality filter): `{doc.get('n_samples')}`",
        f"- quality_keep: `{doc.get('quality_keep')}`",
        "",
        "### Samples",
        "",
    ]
    for s in (doc.get("samples_brief") or []):
        lines.append(
            f"- `{s['token']}` n_gt={s['n_gt']} n_pts={s['n_points']} names={s.get('gt_names_set')}"
        )
    lines += [
        "",
        "## VRAM / nvidia-smi",
        "",
        f"- before: `{train.get('nvidia_smi_before')}`",
        f"- peak/after-train: `{train.get('nvidia_smi_peak') or train.get('nvidia_smi_after')}`",
        f"- after unload: `{train.get('nvidia_smi_after')}`",
        "",
        "## Blockers / notes",
        "",
    ]
    for b in doc.get("blockers") or []:
        lines.append(f"- {b}")
    if not doc.get("blockers"):
        lines.append("- (none)")
    lines += [
        "",
        "## Documented real-train command",
        "",
        "```bash",
        doc.get("real_train_cmd", "").rstrip(),
        "```",
        "",
        "## FakeStudent fallback",
        "",
        f"```json",
        json.dumps(doc.get("fake_student") or {}, indent=2),
        "```",
        "",
    ]
    STATUS_MD.parent.mkdir(parents=True, exist_ok=True)
    STATUS_MD.write_text("\n".join(lines) + "\n")
    (OUT_DIR / "RS_STUDENT_SHORT_STATUS.json").write_text(json.dumps(doc, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--freeze-backbone", action="store_true", default=True)
    ap.add_argument("--no-freeze-backbone", action="store_true")
    ap.add_argument("--pseudo-root", default=str(PSEUDO_ROOT))
    ap.add_argument("--hs64-root", default=str(HS64_ROOT))
    ap.add_argument("--quality", default="A,B", help="comma qualities to keep; try A,B,C if empty")
    ap.add_argument("--cfg", default=str(CFG_YAML))
    ap.add_argument(
        "--ckpt",
        default=str(CKPT_DIR / "pointpillar_multihead_rs_short.pth"),
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-real", action="store_true")
    args = ap.parse_args()
    freeze = not args.no_freeze_backbone

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    quality = [q.strip() for q in args.quality.split(",") if q.strip()]
    infos_doc = build_infos(Path(args.pseudo_root), Path(args.hs64_root), quality)
    # If A/B empty, expand to include C (still short; honest about it)
    quality_note = None
    if infos_doc["n_samples"] == 0 and quality != ["A", "B", "C"]:
        quality_note = f"quality {quality} yielded 0 samples; expanding to A,B,C"
        quality = ["A", "B", "C"]
        infos_doc = build_infos(Path(args.pseudo_root), Path(args.hs64_root), quality)

    infos_json = OUT_DIR / "rs_lidar_only_train_infos.json"
    infos_pkl = OUT_DIR / "rs_lidar_only_train_infos.pkl"
    infos_json.write_text(json.dumps(infos_doc, indent=2))
    write_infos_pkl(infos_doc, infos_pkl)
    train_list = OUT_DIR / "train_list.txt"
    train_list.write_text(
        "\n".join(s["token"] for s in infos_doc["samples"]) + ("\n" if infos_doc["samples"] else "")
    )

    blockers: List[str] = []
    if quality_note:
        blockers.append(quality_note)
    blockers.append(
        "Official OpenPCDet PointPillars pretrained .pth MISSING "
        "(teacher_weights_v3: openpcdet_pointpillar_pretrained). "
        "SHORT train uses random init + freeze backbone/head-train."
    )

    fake: Dict[str, Any] = {}
    train: Dict[str, Any] = {}
    status = "BLOCKED"

    # Always attempt FakeStudent as auxiliary smoke on same labels
    try:
        fake = run_fake_student(Path(args.pseudo_root), OUT_DIR, device=args.device)
    except Exception as exc:  # noqa: BLE001
        fake = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        blockers.append(f"FakeStudent failed: {fake['error']}")

    if infos_doc["n_samples"] == 0:
        status = "BLOCKED"
        blockers.append("No usable train samples after quality filter + hs64 join")
        train = {"mode": "none", "steps": 0, "ckpt": None}
    elif args.skip_real:
        status = "DRY_READY"
        train = {
            "mode": "dry_infos_only",
            "steps": 0,
            "ckpt": None,
            "nvidia_smi_before": nvidia_smi(),
            "nvidia_smi_after": nvidia_smi(),
        }
    else:
        try:
            train = run_real_pp_short(
                infos_pkl,
                Path(args.cfg),
                Path(args.ckpt),
                max_steps=int(args.steps),
                lr=float(args.lr),
                freeze_backbone=freeze,
                device=args.device,
            )
            status = "TRAINED_SHORT"
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            blockers.append(f"Real PointPillar short train failed: {err}")
            blockers.append(traceback.format_exc()[-2000:])
            train = {
                "ok": False,
                "mode": "real_failed",
                "error": err,
                "steps": 0,
                "ckpt": None,
                "nvidia_smi_before": nvidia_smi(),
                "nvidia_smi_after": nvidia_smi(),
            }
            # If infos built + fake ok → DRY_READY; else BLOCKED
            status = "DRY_READY" if infos_doc["n_samples"] > 0 else "BLOCKED"

    doc = {
        "created_at": now_cst(),
        "status": status,
        "quality_keep": quality,
        "n_samples": infos_doc["n_samples"],
        "infos_json": str(infos_json),
        "infos_pkl": str(infos_pkl),
        "train_list": str(train_list),
        "samples_brief": [
            {
                "token": s["token"],
                "n_gt": s["n_gt"],
                "n_points": s["n_points"],
                "gt_names_set": sorted(set(s["gt_names"])),
            }
            for s in infos_doc["samples"]
        ],
        "skipped": infos_doc.get("skipped"),
        "train": train,
        "fake_student": fake,
        "blockers": blockers,
        "real_train_cmd": documented_real_train_cmd(
            infos_pkl, Path(args.cfg), Path(args.ckpt)
        ),
    }
    write_status_md(doc)
    print(json.dumps({"status": status, "n_samples": doc["n_samples"], "train": {
        k: train.get(k) for k in (
            "mode", "steps", "loss_first", "loss_last", "ckpt", "elapsed_sec",
            "nvidia_smi_after", "error"
        ) if k in train or True
    }}, indent=2))
    return 0 if status in ("TRAINED_SHORT", "DRY_READY") else 2


if __name__ == "__main__":
    raise SystemExit(main())
