"""Stage F interface smoke: PointPillar-MultiHead student skeleton.

Documents how a real OpenPCDet PointPillar-MultiHead + SimpleTrack pipeline will
plug in after RoboSense subset is ready. This module does **not** load OpenPCDet
weights or run long GPU training.

Plug-in plan (later, not this smoke):
  1. Teachers unload (Stage F1) — SerialTeacherGuard idle.
  2. Build nuScenes-style dataset from fusion_export / real pseudo labels
     (3 coarse classes: vehicle / pedestrian / cyclist).
  3. OpenPCDet PointPillar-MultiHead:
       - freeze backbone + BN (F2/F3)
       - microbatch=1 + gradient accumulation
       - train new heads only on quality A/B boxes
  4. Inference → SimpleTrack hook (see simpletrack_hook.py) for track_id eval.
  5. Optional later: Sparse4D-R50 local fine-tune (F5).

Reference OpenPCDet trees on this machine (recorded in configs, do not train yet):
  - /data/code/cv/BEV/Lidar_AI_Solution/CUDA-PointPillars/OpenPCDet
  - /data/code/location/pointFuse/output/vendor/OpenPCDet-8cacccec11db6f59bf6934600c9a175dae254806
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from export.pseudo_label_io import read_label, validate_label

# Coarse class map used by PointPillar-MultiHead (F2).
COARSE_CLASSES: Tuple[str, ...] = ("vehicle", "pedestrian", "cyclist")

_CATEGORY_TO_COARSE: Dict[str, str] = {
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "trailer": "vehicle",
    "construction_vehicle": "vehicle",
    "vehicle": "vehicle",
    "pedestrian": "pedestrian",
    "bicycle": "cyclist",
    "motorcycle": "cyclist",
    "cyclist": "cyclist",
}


def map_category_to_coarse(category: str) -> Optional[str]:
    """Map fine nuScenes-style category → 3 coarse classes; None if unknown."""
    key = (category or "").strip().lower()
    return _CATEGORY_TO_COARSE.get(key)


def discover_fusion_pseudo(root: str | Path) -> List[Path]:
    """List per-sample JSON labels under fusion_export_smoke (skip results dict)."""
    root_p = Path(root)
    if not root_p.is_dir():
        return []
    out: List[Path] = []
    for p in sorted(root_p.glob("*.json")):
        if p.name.startswith("nuscenes_results"):
            continue
        out.append(p)
    return out


def validate_dataloader_inputs(
    pseudo_root: str | Path,
    *,
    require_n: int = 1,
) -> Dict[str, Any]:
    """Validate fusion_export_smoke / smoke pseudo labels are readable.

    Returns a summary dict suitable for train_plan.json. Raises on hard failure
    when require_n > 0 and not enough valid labels exist.
    """
    paths = discover_fusion_pseudo(pseudo_root)
    samples: List[Dict[str, Any]] = []
    errors: List[str] = []

    for p in paths:
        try:
            label = read_label(p)
            validate_label(label)
            objs = label.get("objects") or []
            coarse_counts = {c: 0 for c in COARSE_CLASSES}
            for obj in objs:
                if not isinstance(obj, dict):
                    continue
                coarse = map_category_to_coarse(str(obj.get("category", "")))
                if coarse:
                    coarse_counts[coarse] += 1
            samples.append(
                {
                    "path": str(p),
                    "sample_token": label.get("sample_token"),
                    "scene_name": label.get("scene_name"),
                    "n_objects": len(objs),
                    "coarse_counts": coarse_counts,
                    "is_fake": bool((label.get("meta") or {}).get("is_fake", False)),
                }
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{p.name}: {exc}")

    summary: Dict[str, Any] = {
        "pseudo_root": str(pseudo_root),
        "n_files": len(paths),
        "n_valid": len(samples),
        "n_errors": len(errors),
        "errors": errors,
        "samples": samples,
        "coarse_classes": list(COARSE_CLASSES),
    }
    if require_n > 0 and len(samples) < require_n:
        raise FileNotFoundError(
            f"need >= {require_n} valid fusion pseudo label(s) under {pseudo_root}; "
            f"found {len(samples)} (files={len(paths)}, errors={errors})"
        )
    return summary


class FakeStudent:
    """Tiny stand-in for PointPillar-MultiHead: optional 1-step nn.Linear train.

    Never loads OpenPCDet. Used only for --one-iter smoke when torch is available.
    """

    name = "fake_pointpillar_student"

    def __init__(self, in_features: int = 8, n_classes: int = 3):
        self.in_features = in_features
        self.n_classes = n_classes
        self._model = None
        self._device = "cpu"

    def build(self, device: str = "cpu") -> None:
        import torch
        from torch import nn

        self._device = device
        self._model = nn.Linear(self.in_features, self.n_classes).to(device)
        self._model.train()

    def train_one_step(
        self,
        batch_labels: Optional[Sequence[Dict[str, Any]]] = None,
        *,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """One optimizer step on fake tensors (optionally sized from label count)."""
        import torch
        from torch import nn

        if self._model is None:
            self.build(device or self._device)
        assert self._model is not None

        dev = torch.device(device or self._device)
        self._model.to(dev)

        n = max(1, len(batch_labels) if batch_labels else 1)
        # Fake pillar features — no real point cloud encoding.
        x = torch.randn(n, self.in_features, device=dev)
        # Fake coarse-class targets from label objects when present.
        targets = []
        if batch_labels:
            for lb in batch_labels:
                objs = lb.get("objects") or []
                coarse = None
                for obj in objs:
                    coarse = map_category_to_coarse(str(obj.get("category", "")))
                    if coarse:
                        break
                idx = COARSE_CLASSES.index(coarse) if coarse in COARSE_CLASSES else 0
                targets.append(idx)
        while len(targets) < n:
            targets.append(0)
        y = torch.tensor(targets[:n], dtype=torch.long, device=dev)

        opt = torch.optim.SGD(self._model.parameters(), lr=1e-2)
        opt.zero_grad(set_to_none=True)
        logits = self._model(x)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        opt.step()

        return {
            "loss": float(loss.detach().cpu().item()),
            "n": n,
            "device": str(dev),
            "is_fake": True,
            "notes": "FakeStudent 1-step only; not OpenPCDet PointPillars.",
        }

    def unload(self) -> None:
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def build_train_plan(
    *,
    config_path: str | Path,
    pseudo_summary: Dict[str, Any],
    openpcdet_paths: Sequence[str],
    tracker_config: Optional[str] = None,
    dry_run: bool = True,
    one_iter: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble train_plan.json payload (no real training)."""
    plan: Dict[str, Any] = {
        "stage": "F",
        "stage_id": "P6_student_pointpillars",
        "dry_run": dry_run,
        "one_iter": one_iter,
        "config": str(config_path),
        "student": "pointpillar_multihead",
        "tracker": "simpletrack",
        "coarse_classes": list(COARSE_CLASSES),
        "microbatch": 1,
        "grad_accum_note": (
            "Use gradient accumulation to emulate larger batch on 24GB; "
            "freeze backbone + BN for first short train (F2/F3)."
        ),
        "freeze_backbone": True,
        "openpcdet_paths_found": list(openpcdet_paths),
        "openpcdet_train_now": False,
        "pseudo_summary": {
            "pseudo_root": pseudo_summary.get("pseudo_root"),
            "n_valid": pseudo_summary.get("n_valid"),
            "n_files": pseudo_summary.get("n_files"),
            "sample_tokens": [s.get("sample_token") for s in pseudo_summary.get("samples", [])],
        },
        "tracker_config_path": tracker_config,
        "warnings": [
            "Do not download huge OpenPCDet pretrained weights in this smoke.",
            "Do not start long GPU training; --one-iter max 1 fake Linear step.",
        ],
    }
    if extra:
        plan["extra"] = extra
    return plan


def write_train_plan(path: str | Path, plan: Dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return out
