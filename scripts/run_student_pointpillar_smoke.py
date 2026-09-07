#!/usr/bin/env python3
"""Stage F smoke: PointPillar student dry-run / optional 1-iter fake step.

Default --dry-run: check fusion_export_smoke inputs, write train_plan.json.
Optional --one-iter: if torch available, run FakeStudent 1-step (CPU by default;
  --cuda only if torch.cuda.is_available). NEVER occupy GPU >1 min.
No OpenPCDet weight download, no long training.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
for p in (_SRC, _REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data.paths import load_paths, repo_root  # noqa: E402
from export.pseudo_label_io import read_label  # noqa: E402
from student.pointpillar_smoke import (  # noqa: E402
    FakeStudent,
    build_train_plan,
    discover_fusion_pseudo,
    validate_dataloader_inputs,
    write_train_plan,
)
from student.simpletrack_hook import SimpleTrackHook  # noqa: E402

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


def _parse_args() -> argparse.Namespace:
    paths = load_paths()
    pseudo_root = Path(
        paths.get("pseudo_labels", "/data/data/automomous/autolabel4d/pseudo_labels")
    )
    default_pseudo = pseudo_root / "fusion_export_smoke"
    default_out = repo_root() / "outputs" / "student_smoke"
    default_cfg = repo_root() / "configs" / "student_pointpillar_v3.yaml"

    ap = argparse.ArgumentParser(description="Stage F PointPillar student smoke")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Validate inputs + write train_plan.json (default)",
    )
    ap.add_argument(
        "--one-iter",
        action="store_true",
        help="Optional: FakeStudent 1-step if torch available (never >1 min GPU)",
    )
    ap.add_argument(
        "--cuda",
        action="store_true",
        help="With --one-iter, prefer CUDA if available (still 1 step only)",
    )
    ap.add_argument("--config", default=str(default_cfg))
    ap.add_argument("--pseudo-root", default=str(default_pseudo))
    ap.add_argument("--out-dir", default=str(default_out))
    ap.add_argument(
        "--skip-if-no-torch",
        action="store_true",
        default=True,
        help="If --one-iter but torch missing, skip step and still PASS dry-run",
    )
    return ap.parse_args()


def _nvidia_smi_snapshot() -> str:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if r.returncode == 0:
            return (r.stdout or "").strip() or "(empty)"
        return f"nvidia-smi failed rc={r.returncode}: {(r.stderr or '')[:200]}"
    except FileNotFoundError:
        return "nvidia-smi not found"
    except Exception as exc:  # noqa: BLE001
        return f"nvidia-smi error: {exc}"


def _load_cfg(path: Path) -> Dict[str, Any]:
    if yaml is None:
        return {}
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _openpcdet_from_cfg(cfg: Dict[str, Any]) -> List[str]:
    block = cfg.get("openpcdet") or {}
    found = list(block.get("paths_found") or [])
    # Keep only existing paths
    return [p for p in found if Path(p).exists()]


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(args.config)
    cfg = _load_cfg(cfg_path)

    print("[smoke] Stage F PointPillar student — dry-run / optional 1-iter")
    print(f"[smoke] config={cfg_path}")
    print(f"[smoke] pseudo_root={args.pseudo_root}")
    print(f"[smoke] out_dir={out_dir}")

    # 1) Validate fusion_export_smoke inputs
    try:
        pseudo_summary = validate_dataloader_inputs(args.pseudo_root, require_n=1)
    except FileNotFoundError as exc:
        print(f"[smoke] FAIL inputs: {exc}")
        return 1

    print(
        f"[smoke] dataloader OK: n_valid={pseudo_summary['n_valid']} "
        f"n_files={pseudo_summary['n_files']}"
    )

    # 2) SimpleTrack hook stub — record config path only
    hook = SimpleTrackHook()
    tracker_cfg = hook.ensure_placeholder_config(out_dir)
    tracker_rec = hook.record_config(tracker_cfg)
    print(f"[smoke] simpletrack config recorded: {tracker_rec['config_path']}")

    openpcdet_paths = _openpcdet_from_cfg(cfg)
    if not openpcdet_paths:
        # Fallback discover (record only)
        candidates = [
            "/data/code/cv/BEV/Lidar_AI_Solution/CUDA-PointPillars/OpenPCDet",
            "/data/code/location/pointFuse/output/vendor/"
            "OpenPCDet-8cacccec11db6f59bf6934600c9a175dae254806",
        ]
        openpcdet_paths = [p for p in candidates if Path(p).exists()]
    print(f"[smoke] OpenPCDet found ({len(openpcdet_paths)}): {openpcdet_paths}")

    one_iter_result: Optional[Dict[str, Any]] = None
    smi_before = smi_after = None

    # 3) Optional 1-iter fake step
    if args.one_iter:
        try:
            import torch
        except ImportError:
            print("[smoke] torch not available — skip --one-iter (dry-run still OK)")
            one_iter_result = {"skipped": True, "reason": "no_torch"}
        else:
            use_cuda = bool(args.cuda and torch.cuda.is_available())
            device = "cuda" if use_cuda else "cpu"
            print(f"[smoke] --one-iter device={device} cuda_available={torch.cuda.is_available()}")
            if use_cuda:
                smi_before = _nvidia_smi_snapshot()
                print(f"[smoke] nvidia-smi BEFORE:\n{smi_before}")

            t0 = time.time()
            labels = []
            for p in discover_fusion_pseudo(args.pseudo_root)[:2]:
                labels.append(read_label(p))
            student = FakeStudent(in_features=8, n_classes=3)
            student.build(device=device)
            one_iter_result = student.train_one_step(labels, device=device)
            student.unload()
            elapsed = time.time() - t0
            one_iter_result["elapsed_sec"] = round(elapsed, 4)
            if elapsed > 60:
                print(f"[smoke] WARN: one-iter took {elapsed:.1f}s (>60s budget)")
            print(f"[smoke] one-iter result: {one_iter_result}")

            if use_cuda:
                smi_after = _nvidia_smi_snapshot()
                print(f"[smoke] nvidia-smi AFTER:\n{smi_after}")
                try:
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
    else:
        print("[smoke] dry-run only (pass --one-iter for FakeStudent 1-step)")

    plan = build_train_plan(
        config_path=cfg_path,
        pseudo_summary=pseudo_summary,
        openpcdet_paths=openpcdet_paths,
        tracker_config=str(tracker_cfg),
        dry_run=True,
        one_iter=bool(args.one_iter),
        extra={
            "tracker_record": tracker_rec,
            "one_iter_result": one_iter_result,
            "nvidia_smi_before": smi_before,
            "nvidia_smi_after": smi_after,
            "config_microbatch": (cfg.get("train") or {}).get("microbatch", 1),
            "freeze_backbone": (cfg.get("train") or {}).get("freeze_backbone", True),
        },
    )
    plan_path = write_train_plan(out_dir / "train_plan.json", plan)
    print(f"[smoke] wrote {plan_path}")

    # Also dump a tiny status sidecar
    status = {
        "pass": True,
        "stage": "F",
        "plan": str(plan_path),
        "n_valid_pseudo": pseudo_summary["n_valid"],
        "openpcdet_found": openpcdet_paths,
        "one_iter": one_iter_result,
    }
    status_path = out_dir / "smoke_status.json"
    with status_path.open("w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
