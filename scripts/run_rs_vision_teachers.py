#!/usr/bin/env python3
"""Serial vision teachers (GDINO → SAM) on RoboSense aligned frames.

- CAM_FRONT / folder 0 first (pinhole).
- Skip OV fisheye / folder 4 (missing) with notes.
- Cache under pseudo_labels/rs_vision_dino_sam/
- Unload between stages. No multipart / kitti / push.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
for p in (_SRC, _REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from teachers.base import SerialTeacherGuard, assert_serial_idle  # noqa: E402
from teachers.dino_real import DEFAULT_PROMPTS, DinoTeacherReal  # noqa: E402
from teachers.sam_real import SamTeacherReal  # noqa: E402

CST = timezone(timedelta(hours=8))
DEFAULT_MANIFEST = Path(
    "/data/data/automomous/autolabel4d/manifests/rs_align_frames_v0.json"
)
DEFAULT_CAM_MAP = Path(
    "/data/data/automomous/autolabel4d/manifests/rs_camera_map_v0.yaml"
)
DEFAULT_CACHE = Path(
    "/data/data/automomous/autolabel4d/pseudo_labels/rs_vision_dino_sam"
)
DEFAULT_DINO = Path(
    "/data/data/automomous/autolabel4d/checkpoints/dino/grounding-dino-tiny"
)
DEFAULT_SAM = Path(
    "/data/data/automomous/autolabel4d/checkpoints/sam/sam2.1-hiera-small/sam2.1_hiera_small.pt"
)


def cst_now() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S %Z")


def query_vram() -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
        )
        return out.strip()
    except Exception as exc:  # noqa: BLE001
        return f"(nvidia-smi unavailable: {exc})"


def load_camera_map(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def select_cameras(
    cam_map: Dict[str, Any],
    *,
    prefer_folders: Sequence[str] = ("0",),
    skip_ov: bool = True,
    skip_missing_folder4: bool = True,
) -> Dict[str, Any]:
    """Decide which folder_ids to run; return selection + skip notes."""
    folder_to_camera = cam_map.get("folder_to_camera") or {}
    selected: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for fid, info in sorted(folder_to_camera.items(), key=lambda kv: str(kv[0])):
        cam = str(info.get("camera", ""))
        model = str(info.get("model", ""))
        on_disk = bool(info.get("on_disk", True))
        reason = None
        if skip_missing_folder4 and str(fid) == "4":
            reason = "folder 4 CAM_FRONT_OV missing on disk (n_images_on_disk=0)"
        elif skip_ov and model == "fisheye":
            reason = "OV fisheye skipped (unwrap not implemented this round)"
        elif str(fid) not in set(str(x) for x in prefer_folders):
            # Still allow explicit prefer list only for this short run
            reason = f"not in prefer_folders={list(prefer_folders)} (pinhole others deferred)"
        if reason:
            skipped.append(
                {
                    "folder": str(fid),
                    "camera": cam,
                    "model": model,
                    "on_disk": on_disk,
                    "reason": reason,
                }
            )
            continue
        if not on_disk:
            skipped.append(
                {
                    "folder": str(fid),
                    "camera": cam,
                    "model": model,
                    "on_disk": on_disk,
                    "reason": "on_disk=false",
                }
            )
            continue
        selected.append(
            {
                "folder": str(fid),
                "camera": cam,
                "model": model,
                "on_disk": on_disk,
            }
        )
    return {"selected": selected, "skipped": skipped}


def main() -> int:
    ap = argparse.ArgumentParser(description="RS vision teachers serial GDINO→SAM")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--camera-map", type=Path, default=DEFAULT_CAM_MAP)
    ap.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--dino-dir", type=Path, default=DEFAULT_DINO)
    ap.add_argument("--sam-ckpt", type=Path, default=DEFAULT_SAM)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-frames", type=int, default=12)
    ap.add_argument("--box-threshold", type=float, default=0.25)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument(
        "--folders",
        nargs="*",
        default=["0"],
        help="camera folder ids to run (default: 0=CAM_FRONT)",
    )
    ap.add_argument("--skip-dino", action="store_true")
    ap.add_argument("--skip-sam", action="store_true")
    args = ap.parse_args()

    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    dino_dir = cache_root / "dino"
    sam_dir = cache_root / "sam"
    dino_dir.mkdir(parents=True, exist_ok=True)
    sam_dir.mkdir(parents=True, exist_ok=True)

    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    frames = list(man.get("frames") or [])[: max(1, int(args.max_frames))]
    cam_map = load_camera_map(args.camera_map)
    cam_sel = select_cameras(cam_map, prefer_folders=args.folders)

    report: Dict[str, Any] = {
        "generated_cst": cst_now(),
        "manifest": str(args.manifest),
        "n_frames": len(frames),
        "camera_selection": cam_sel,
        "vram_before": query_vram(),
        "dino": {"frames": [], "success": False, "blocker": None},
        "sam": {"frames": [], "success": False, "blocker": None},
        "notes": [
            "Skip OV fisheye (folders 5-7) — unwrap not in this round.",
            "Skip folder 4 CAM_FRONT_OV — missing on disk.",
            "Prefer CAM_FRONT folder 0 pinhole for vision teachers.",
        ],
    }
    print(f"[vision] frames={len(frames)} cams={cam_sel['selected']}")
    print(f"[vision] skip={cam_sel['skipped']}")
    print(f"[vision] VRAM before: {report['vram_before']}")

    SerialTeacherGuard.reset()

    # ---------- DINO ----------
    if not args.skip_dino and cam_sel["selected"]:
        teacher = DinoTeacherReal(
            model_dir=args.dino_dir,
            device=args.device,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            prompts=DEFAULT_PROMPTS,
        )
        try:
            teacher.load()
            print(f"[dino] loaded; VRAM={query_vram()}")
            for i, fr in enumerate(frames):
                stem = str(fr.get("stem"))
                for cam_info in cam_sel["selected"]:
                    fid = cam_info["folder"]
                    img_path = (fr.get("images") or {}).get(fid) or (fr.get("images") or {}).get(
                        int(fid) if str(fid).isdigit() else fid
                    )
                    row: Dict[str, Any] = {
                        "stem": stem,
                        "folder": fid,
                        "camera": cam_info["camera"],
                        "success": False,
                        "n_dets": 0,
                        "path": None,
                        "error": None,
                    }
                    if not img_path or not Path(img_path).is_file():
                        row["error"] = f"missing image for folder {fid}"
                        report["dino"]["frames"].append(row)
                        print(f"  [dino {i}] {stem} {fid}: MISSING")
                        continue
                    try:
                        result = teacher.infer_frame(
                            {
                                "sample_token": stem,
                                "image_path": str(img_path),
                                "camera": cam_info["camera"],
                                "folder": fid,
                            }
                        )
                        out_path = dino_dir / f"dino_{stem}_{cam_info['camera']}.json"
                        payload = dict(result)
                        payload["stem"] = stem
                        payload["folder"] = fid
                        payload["hs64_path"] = fr.get("hs64_path")
                        with out_path.open("w", encoding="utf-8") as f:
                            json.dump(payload, f, indent=2, ensure_ascii=False)
                            f.write("\n")
                        row["success"] = bool((result.get("meta") or {}).get("success", True))
                        row["n_dets"] = len(result.get("detections_2d") or [])
                        row["path"] = str(out_path)
                        print(
                            f"  [dino {i}] {stem} {cam_info['camera']}: "
                            f"n_dets={row['n_dets']} → {out_path.name}"
                        )
                    except Exception as exc:  # noqa: BLE001
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        traceback.print_exc()
                        print(f"  [dino {i}] {stem} FAIL: {row['error']}")
                    report["dino"]["frames"].append(row)
            report["dino"]["success"] = bool(report["dino"]["frames"]) and all(
                r.get("success") for r in report["dino"]["frames"]
            )
        except Exception as exc:  # noqa: BLE001
            report["dino"]["blocker"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            print(f"[dino] BLOCKED: {report['dino']['blocker']}")
        finally:
            try:
                teacher.unload()
            except Exception:
                SerialTeacherGuard.reset()
            print(f"[dino] unloaded; VRAM={query_vram()}")
    else:
        report["dino"]["blocker"] = "skipped by flag or no cameras"
        print("[dino] skipped")

    assert_serial_idle()

    # ---------- SAM ----------
    if not args.skip_sam and cam_sel["selected"]:
        teacher = SamTeacherReal(checkpoint=args.sam_ckpt, device=args.device)
        try:
            teacher.load()
            print(f"[sam] loaded; VRAM={query_vram()}")
            for i, fr in enumerate(frames):
                stem = str(fr.get("stem"))
                for cam_info in cam_sel["selected"]:
                    fid = cam_info["folder"]
                    cam = cam_info["camera"]
                    dino_path = dino_dir / f"dino_{stem}_{cam}.json"
                    img_path = (fr.get("images") or {}).get(fid)
                    row = {
                        "stem": stem,
                        "folder": fid,
                        "camera": cam,
                        "success": False,
                        "n_masks": 0,
                        "path": None,
                        "error": None,
                    }
                    if not dino_path.is_file():
                        row["error"] = f"missing dino cache {dino_path.name}"
                        report["sam"]["frames"].append(row)
                        continue
                    if not img_path or not Path(img_path).is_file():
                        row["error"] = "missing image"
                        report["sam"]["frames"].append(row)
                        continue
                    try:
                        dino = json.loads(dino_path.read_text(encoding="utf-8"))
                        result = teacher.infer_frame(
                            {
                                "sample_token": stem,
                                "image_path": str(img_path),
                                "camera": cam,
                                "folder": fid,
                                "detections_2d": dino.get("detections_2d") or [],
                            }
                        )
                        out_path = sam_dir / f"sam_{stem}_{cam}.json"
                        payload = dict(result)
                        payload["stem"] = stem
                        payload["folder"] = fid
                        payload["dino_cache"] = str(dino_path)
                        # drop huge packbits in summary? keep for fusion viz
                        with out_path.open("w", encoding="utf-8") as f:
                            json.dump(payload, f, indent=2, ensure_ascii=False)
                            f.write("\n")
                        row["success"] = bool((result.get("meta") or {}).get("success", True))
                        row["n_masks"] = len(result.get("masks") or [])
                        row["path"] = str(out_path)
                        print(
                            f"  [sam {i}] {stem} {cam}: n_masks={row['n_masks']} → {out_path.name}"
                        )
                    except Exception as exc:  # noqa: BLE001
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        traceback.print_exc()
                    report["sam"]["frames"].append(row)
            report["sam"]["success"] = bool(report["sam"]["frames"]) and all(
                r.get("success") for r in report["sam"]["frames"]
            )
        except Exception as exc:  # noqa: BLE001
            report["sam"]["blocker"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            try:
                teacher.unload()
            except Exception:
                SerialTeacherGuard.reset()
            print(f"[sam] unloaded; VRAM={query_vram()}")
    else:
        report["sam"]["blocker"] = "skipped by flag or no cameras"

    try:
        assert_serial_idle()
    except Exception as exc:  # noqa: BLE001
        report["serial_warn"] = str(exc)

    report["vram_after"] = query_vram()
    report["cache_root"] = str(cache_root)
    report_path = cache_root / "vision_teachers_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[vision] report → {report_path}")
    print(f"[vision] VRAM after: {report['vram_after']}")
    ok = (args.skip_dino or report["dino"]["success"]) and (
        args.skip_sam or report["sam"]["success"]
    )
    print("PASS" if ok else "PARTIAL/FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
