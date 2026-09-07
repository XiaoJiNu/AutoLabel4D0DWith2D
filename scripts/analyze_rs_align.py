#!/usr/bin/env python3
"""Build RoboSense align manifests and camera map from PKL + disk scan."""
from __future__ import annotations
import argparse, json, pickle, re, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import yaml
except ImportError:
    yaml = None

CST = timezone(timedelta(hours=8))
FOLDER_CAM = {
    "0": "CAM_FRONT", "1": "CAM_LEFT", "2": "CAM_RIGHT", "3": "CAM_BACK",
    "4": "CAM_FRONT_OV", "5": "CAM_LEFT_OV", "6": "CAM_RIGHT_OV", "7": "CAM_BACK_OV",
}
OV = {"CAM_FRONT_OV", "CAM_LEFT_OV", "CAM_RIGHT_OV", "CAM_BACK_OV"}
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d+)")

def now_cst():
    return datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S CST")

def day_of(p):
    m = TS_RE.match(Path(str(p)).stem)
    return m.group(1) if m else "unknown"

def pack_of(p):
    for seg in Path(str(p)).parts:
        if str(seg).startswith("processed_data_"):
            return str(seg)
    return None

def load_list(path: Path):
    with path.open("rb") as f:
        data = pickle.load(f)
    assert isinstance(data, list)
    return data

def analyze(split, samples):
    phs, pimg, hs_day, img_day = Counter(), Counter(), Counter(), Counter()
    seq, maps = Counter(), Counter()
    cam_folder = defaultdict(Counter)
    pairs = Counter()
    day_match = 0
    cams_u = set()
    calib0 = {}
    for i, s in enumerate(samples):
        seq[s.get("seq_token")] += 1
        maps[s.get("map_token")] += 1
        hs = s.get("hs64_path", "")
        hs_day[day_of(hs)] += 1
        hp = pack_of(hs)
        if hp: phs[hp] += 1
        cams = s.get("images", {}).get("cams", {})
        cams_u |= set(cams)
        front = cams.get("CAM_FRONT") or (next(iter(cams.values())) if cams else {})
        dp = front.get("data_path", "") if isinstance(front, dict) else ""
        img_day[day_of(dp)] += 1
        ip = pack_of(dp) if dp else None
        if ip: pimg[ip] += 1
        pairs[(ip, hp)] += 1
        if day_of(dp) == day_of(hs) and day_of(dp) != "unknown":
            day_match += 1
        for ck, cv in cams.items():
            if not isinstance(cv, dict):
                continue
            folder = Path(cv.get("data_path", "x")).parent.name
            cam_folder[ck][folder] += 1
            if i == 0 and ck not in calib0:
                calib0[ck] = {
                    "type": cv.get("type"),
                    "data_path": cv.get("data_path"),
                    "folder_id": folder,
                    "img_h": cv.get("img_height"),
                    "img_w": cv.get("img_width"),
                    "cam_intrinsic": cv.get("cam_intrinsic"),
                    "has_dist": cv.get("cam_dist") is not None,
                    "has_s2e": "sensor2ego_translation" in cv,
                }
    return {
        "split": split,
        "n_samples": len(samples),
        "n_seq": len(seq),
        "n_map": len(maps),
        "map_top": maps.most_common(20),
        "seq_top": [(int(k) if k is not None else None, v) for k, v in seq.most_common(10)],
        "processed_hs": dict(phs),
        "processed_img": dict(pimg),
        "pack_pairs": {f"{a}|{b}": c for (a, b), c in pairs.most_common()},
        "sample_days_match": day_match,
        "hs_day_top": hs_day.most_common(15),
        "img_day_top": img_day.most_common(15),
        "day_intersection": sorted(set(hs_day) & set(img_day)),
        "cam_keys": sorted(cams_u),
        "cam_folder": {k: dict(v) for k, v in sorted(cam_folder.items())},
        "calib_sample0": calib0,
        "sample0_keys": sorted(samples[0].keys()) if samples else [],
        "has_annos": bool(samples and "annos" in samples[0]),
        "gt_note": "annos in PKL for eval only; never feed to teachers",
    }

def scan(subset: Path):
    img_dirs = {}
    for images_dir in (subset / "image_trainval").glob("**/images"):
        for child in images_dir.iterdir():
            if child.is_dir() and child.name.isdigit():
                img_dirs[child.name] = child
    img_stems = {}
    img_days, img_packs = Counter(), Counter()
    for cid, d in sorted(img_dirs.items(), key=lambda x: int(x[0])):
        stems = set()
        for f in d.iterdir():
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            stems.add(f.stem)
            img_days[day_of(f.name)] += 1
            pk = pack_of(f)
            if pk: img_packs[pk] += 1
        img_stems[cid] = stems
    hs_roots = list(subset.glob("lidar_occ_trainval/**/hs64"))
    hs_stems, hs_days, hs_packs = set(), Counter(), Counter()
    hs_by = {}
    for root in hs_roots:
        for f in root.iterdir():
            if f.suffix.lower() != ".bin":
                continue
            hs_stems.add(f.stem)
            hs_days[day_of(f.name)] += 1
            pk = pack_of(f)
            if pk: hs_packs[pk] += 1
            hs_by[f.stem] = f
    primary = "0" if "0" in img_stems else (sorted(img_stems, key=int)[0] if img_stems else None)
    pstem = img_stems.get(primary, set()) if primary else set()
    exact = sorted(pstem & hs_stems)
    return {
        "image_counts": {k: len(v) for k, v in img_stems.items()},
        "image_days": dict(img_days),
        "image_packs": dict(img_packs),
        "hs64_count": len(hs_stems),
        "hs64_days": dict(hs_days),
        "hs64_packs": dict(hs_packs),
        "primary": primary,
        "exact_n": len(exact),
        "day_overlap": sorted(set(img_days) & set(hs_days)),
        "_pstem": pstem, "_hs": hs_stems, "_dirs": img_dirs, "_hs_by": hs_by,
    }

def frames_from(disk):
    frames = []
    pstem, hs, dirs, hs_by = disk["_pstem"], disk["_hs"], disk["_dirs"], disk["_hs_by"]
    for stem in sorted(pstem & hs):
        cams = {}
        for cid, d in dirs.items():
            for ext in (".jpg", ".jpeg", ".png"):
                p = d / f"{stem}{ext}"
                if p.is_file():
                    cams[cid] = str(p)
                    break
        frames.append({"stem": stem, "day": day_of(stem + ".bin"), "hs64_path": str(hs_by[stem]), "images": cams, "match": "exact_stem"})
    reasons = []
    if not disk["day_overlap"]:
        reasons.append(
            "NO_DAY_OVERLAP images=%s hs64=%s (independent modality part_01 truncates)"
            % (sorted(disk["image_days"]), sorted(disk["hs64_days"]))
        )
    reasons.append(
        "PACK_MISMATCH image_packs=%s hs_packs=%s; processed_data_* is packaging batch name; "
        "within one pack PKL pairs same-day image+hs64"
        % (list(disk["image_packs"]), list(disk["hs64_packs"]))
    )
    if not frames:
        reasons.append("ZERO_PAIRABLE_FRAMES exact_stem=0")
    stats = {
        "n_aligned_frames": len(frames),
        "unpaired_primary_images": len(pstem - hs),
        "unpaired_hs64": len(hs - pstem),
        "reasons": reasons,
    }
    return frames, stats

def cam_map(analyses, disk):
    votes = defaultdict(Counter)
    for a in analyses:
        for cam, folders in a.get("cam_folder", {}).items():
            for folder, n in folders.items():
                votes[folder][cam] += n
    mapping = {}
    for folder in sorted(set(list(FOLDER_CAM) + list(votes) + list(disk.get("image_counts", {}))), key=lambda x: int(x) if str(x).isdigit() else 999):
        folder = str(folder)
        if votes.get(folder):
            cam = votes[folder].most_common(1)[0][0]
            conf, ev = "pkl_data_path", dict(votes[folder])
        else:
            cam = FOLDER_CAM.get(folder, f"CAM_UNKNOWN_{folder}")
            conf, ev = ("canonical_fallback" if folder in FOLDER_CAM else "unknown"), {}
        mapping[folder] = {
            "camera": cam,
            "model": "fisheye" if cam in OV else "pinhole",
            "on_disk": folder in disk.get("image_counts", {}),
            "n_images_on_disk": disk.get("image_counts", {}).get(folder, 0),
            "confidence": conf,
            "pkl_folder_votes": ev,
            "in_rs_4f_1l_preset": cam in OV,
        }
    return {
        "version": "rs_camera_map_v0",
        "generated_at_cst": now_cst(),
        "source": "PKL images.cams[*].data_path parent folder id",
        "sensors_preset": "configs/sensors_rs_4f_1l.yaml",
        "note": "0-3 pinhole, 4-7 OV fisheye; disk missing folder 4; use CAM_FRONT for pinhole proj smoke",
        "folder_to_camera": mapping,
        "camera_to_folder": {v["camera"]: k for k, v in mapping.items()},
    }

def patch_estimate(disk, analyses):
    part_gb = 10.7374
    need_hs = sum(a.get("processed_hs", {}).get("processed_data_20231011", 0) for a in analyses)
    need_img = sum(a.get("processed_img", {}).get("processed_data_20230906", 0) for a in analyses)
    return {
        "did_fetch": False,
        "fetched_gb": 0.0,
        "stop": "next useful multipart volume is >=10.74GB (over 5GB gate)",
        "lower_bound_gb": part_gb,
        "options": [
            {"id": "A", "goal": "hs64 for image days in pack 20231011", "lb_gb": part_gb, "pkl_n": need_hs},
            {"id": "B", "goal": "images for hs64 days in pack 20230906", "lb_gb": part_gb, "pkl_n": need_img},
        ],
        "advice": "Ask chief to authorize >=10.74GB next volume or keep modality-separate tracks",
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="/data/data/automomous/robosense/subset")
    ap.add_argument("--manifests", default="/data/data/automomous/autolabel4d/manifests")
    args = ap.parse_args()
    subset, man = Path(args.subset), Path(args.manifests)
    man.mkdir(parents=True, exist_ok=True)
    analyses = []
    for split, name in (("val", "robosense_local_val.pkl"), ("train", "robosense_local_train.pkl")):
        path = subset / "splits" / name
        print(f"[{now_cst()}] load {split} {path.stat().st_size/1e6:.1f}MB")
        a = analyze(split, load_list(path))
        analyses.append(a)
        print(f"  n={a['n_samples']} seq={a['n_seq']} day_match={a['sample_days_match']} packs={a['processed_hs']}")
    disk = scan(subset)
    print(f"disk cams={disk['image_counts']} hs={disk['hs64_count']} overlap={disk['day_overlap']} exact={disk['exact_n']}")
    frames, stats = frames_from(disk)
    cmap = cam_map(analyses, disk)
    est = patch_estimate(disk, analyses)
    disk_pub = {k: v for k, v in disk.items() if not k.startswith("_")}
    payload = {
        "version": "rs_align_frames_v0",
        "generated_at_cst": now_cst(),
        "n_aligned_frames": stats["n_aligned_frames"],
        "coverage": stats,
        "disk": disk_pub,
        "pkl": analyses,
        "packaging_note": (
            "processed_data_YYYYMMDD is packaging/batch folder name. "
            "image 20231011 vs lidar 20230906 on disk is from truncated independent part_01 streams."
        ),
        "patch": est,
        "frames": frames[:5000],
    }
    jp = man / "rs_align_frames_v0.json"
    jp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tp = man / "rs_align_frames_v0.txt"
    lines = [f"# rs_align_frames_v0 {payload['generated_at_cst']}", f"# n_aligned_frames={stats['n_aligned_frames']}"]
    for r in stats["reasons"]:
        lines.append(f"# - {r}")
    lines.append("# stem\tday\tmatch\ths64\tcam0")
    for fr in frames:
        lines.append(f"{fr['stem']}\t{fr['day']}\t{fr['match']}\t{fr['hs64_path']}\t{fr['images'].get('0','')}")
    if not frames:
        lines.append("# (no pairable frames)")
    tp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    yp = man / "rs_camera_map_v0.yaml"
    if yaml:
        with yp.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cmap, f, sort_keys=False, allow_unicode=True)
    else:
        yp.write_text(json.dumps(cmap, indent=2), encoding="utf-8")
    summary = {
        "n_aligned_frames": stats["n_aligned_frames"],
        "camera_map": str(yp),
        "manifest_json": str(jp),
        "manifest_txt": str(tp),
        "fetched_gb": 0.0,
        "did_fetch": False,
        "lower_bound_gb": est["lower_bound_gb"],
        "reasons": stats["reasons"],
        "advice": est["advice"],
    }
    (man / "rs_align_summary_v0.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
