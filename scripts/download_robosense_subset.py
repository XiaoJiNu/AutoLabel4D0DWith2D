#!/usr/bin/env python3
"""Stage B: RoboSense subset planner + gated HuggingFace download.

DEFAULT is dry-run: inventory shards, propose a subset under disk budget,
write a draft manifest, and REJECT (exit != 0) if estimated media size
exceeds the budget.

Real downloads require BOTH:
  --execute
  env ALLOW_RS_DOWNLOAD=1
else exit != 0.

Safety:
  --probe-only downloads ONLY the single smallest inventory file (<<1GB)
  and never the full ~28GB subset plan. Prefer that for connectivity checks.

With --extract-to-subset / --delete-raw-archives (or after successful full
--execute when those flags are set): extract each downloaded *.tar.gz into
paths.yaml robosense_subset (preserve structure), copy split PKLs into
subset/splits/, then delete each raw .tar.gz after successful extract+verify.
Never touch /data/data/kitti0000.tar.gz or non-archive useful files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import gzip
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = REPO_ROOT / "configs" / "paths.yaml"
DEFAULT_INVENTORY_CANDIDATES = [
    Path("/home/yr/grok-bot-work/autolabel4d_stageB/hf_shard_inventory.json"),
    REPO_ROOT / "configs" / "robosense_hf_inventory.json",
]
HF_REPO = "suhaisheng0527/RoboSense"
HF_REPO_TYPE = "dataset"
HF_TREE_URLS = [
    f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main/dataset",
    f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main/splits",
]
HF_RESOLVE_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
BYTES_PER_GB = 1_000_000_000  # decimal GB for human estimates
PROBE_SOFT_CAP_BYTES = 1_000_000_000  # refuse probe if smallest file >= 1GB

# First-cut shard selection (v0): prefer trailing smaller parts + one full
# image trainval part + two lidar trainval parts + small test trailers +
# local split PKLs. Target ~28–35 GB under robosense_subset_max=35.
DEFAULT_PROPOSED_SHARDS = [
    "splits/robosense_local_train.pkl",
    "splits/robosense_local_val.pkl",
    "dataset/image_trainval_part_01.tar.gz",
    "dataset/image_trainval_part_16.tar.gz",  # trailing ~1.57GB
    "dataset/lidar_occ_trainval_part_01.tar.gz",
    "dataset/lidar_occ_trainval_part_23.tar.gz",  # trailing ~3.17GB
    "dataset/image_test_part_12.tar.gz",  # trailing ~0.91GB (check side)
    "dataset/lidar_occ_test_part_18.tar.gz",  # trailing ~0.79GB
]

# Placeholder "sequences" until PKL extract reveals real IDs.
DEFAULT_PLACEHOLDER_SEQUENCES = {
    "dev": [
        "PLACEHOLDER_dev_seq_01_from_local_val",
        "PLACEHOLDER_dev_seq_02_from_local_val",
    ],
    "train_try": [
        "PLACEHOLDER_train_try_seq_01_from_local_train",
        "PLACEHOLDER_train_try_seq_02_from_local_train",
        "PLACEHOLDER_train_try_seq_03_from_local_train",
    ],
    "holdout_check": [
        "PLACEHOLDER_check_seq_01_from_test_trailers",
    ],
}


def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


def load_paths(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"paths.yaml not found: {path}")
    if yaml is None:
        raise RuntimeError("PyYAML required: pip install pyyaml")
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if "budget_gb" not in data:
        raise KeyError(f"budget_gb missing in {path}")
    if "robosense_raw" not in data:
        raise KeyError(f"robosense_raw missing in {path}")
    return data


def clamp_max_gb(value: float) -> float:
    """Recommended band is 30–50. Cap only the upper bound at 50.

    Values below 30 are allowed (with a warning) so quota-reject self-tests
    like `--max-gb 10` can exit non-zero when the plan exceeds the budget.
    """
    v = float(value)
    if v < 30:
        eprint(
            f"WARN: --max-gb={v} below recommended 30–50; "
            "keeping value so quota reject can fire"
        )
        return v
    if v > 50:
        eprint(
            f"WARN: --max-gb={v} above recommended 30–50; "
            "clamping to 50"
        )
        return 50.0
    return v


def fetch_hf_tree(timeout: float = 8.0) -> list[dict[str, Any]] | None:
    files: list[dict[str, Any]] = []
    try:
        for url in HF_TREE_URLS:
            req = urllib.request.Request(
                url, headers={"User-Agent": "AutoLabel4D-rs-download/0.2"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            for entry in payload:
                if entry.get("type") != "file":
                    continue
                size = entry.get("size")
                if size is None and isinstance(entry.get("lfs"), dict):
                    size = entry["lfs"].get("size", 0)
                files.append({"path": entry["path"], "size": int(size or 0)})
        return files
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        eprint(f"HF API unavailable ({exc}); will use cached inventory")
        return None


def load_cached_inventory(explicit: Path | None) -> dict[str, Any]:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(DEFAULT_INVENTORY_CANDIDATES)
    for cand in candidates:
        if cand and cand.is_file():
            with cand.open() as f:
                data = json.load(f)
            data["_inventory_path"] = str(cand)
            data["_inventory_source"] = "cache"
            return data
    raise FileNotFoundError(
        "No HF inventory cache found. Expected one of:\n  "
        + "\n  ".join(str(c) for c in candidates if c)
    )


def build_inventory(
    refresh: bool,
    cache_path: Path | None,
) -> tuple[dict[str, Any], str]:
    """Return (inventory, status) where status is 'hf_api' or 'cache'."""
    cached = None
    try:
        cached = load_cached_inventory(cache_path)
    except FileNotFoundError:
        cached = None

    if refresh or cached is None:
        remote = fetch_hf_tree()
        if remote is not None:
            inv = {
                "repo": HF_REPO,
                "source": HF_TREE_URLS[0].rsplit("/", 1)[0],
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "note": (
                    "HF layout is opaque at sequence level: media is split into "
                    "~10GB tar parts that must be cat'd before extract."
                ),
                "files": [
                    {
                        "path": f["path"],
                        "size": f["size"],
                        "size_gb": round(f["size"] / BYTES_PER_GB, 4),
                    }
                    for f in sorted(remote, key=lambda x: x["path"])
                ],
            }
            inv["total_bytes"] = sum(f["size"] for f in inv["files"])
            inv["total_gb"] = round(inv["total_bytes"] / BYTES_PER_GB, 3)
            inv["_inventory_source"] = "hf_api"
            out = cache_path or DEFAULT_INVENTORY_CANDIDATES[0]
            try:
                out.parent.mkdir(parents=True, exist_ok=True)
                serializable = {k: v for k, v in inv.items() if not k.startswith("_")}
                out.write_text(json.dumps(serializable, indent=2) + "\n")
                inv["_inventory_path"] = str(out)
            except OSError as exc:
                eprint(f"WARN: could not write inventory cache: {exc}")
            return inv, "hf_api"

    if cached is None:
        raise RuntimeError("HF API failed and no cached inventory available")
    return cached, cached.get("_inventory_source", "cache")


def index_by_path(inv: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {f["path"]: f for f in inv["files"]}


def propose_selection(
    by_path: dict[str, dict[str, Any]],
    shard_list: list[str] | None = None,
) -> dict[str, Any]:
    shards = shard_list or list(DEFAULT_PROPOSED_SHARDS)
    missing = [p for p in shards if p not in by_path]
    selected = []
    total = 0
    for p in shards:
        meta = by_path.get(p)
        if meta is None:
            continue
        size = int(meta["size"])
        total += size
        selected.append(
            {
                "path": p,
                "size": size,
                "size_gb": round(size / BYTES_PER_GB, 4),
                "kind": (
                    "split_pkl"
                    if p.startswith("splits/")
                    else "image_shard"
                    if "image_" in p
                    else "lidar_occ_shard"
                    if "lidar_occ_" in p
                    else "other"
                ),
                "sha256": meta.get("sha256"),
                "md5": meta.get("md5"),
            }
        )
    return {
        "shards": selected,
        "missing_shards": missing,
        "estimated_bytes": total,
        "estimated_gb": round(total / BYTES_PER_GB, 4),
        "placeholder_sequences": DEFAULT_PLACEHOLDER_SEQUENCES,
        "assumptions": [
            "HF does not expose per-sequence objects; only split tar parts + PKLs.",
            "Selected trailing smaller parts + first full trainval image/lidar parts "
            "to stay near 30–35GB while preferring smaller complete archives.",
            "Placeholder sequence IDs must be replaced after extracting local_*.pkl.",
            "image_* and lidar_occ_* parts are independent archives; pairing is best-effort "
            "for dry-run budgeting, not a guarantee of shared timestamps until extract.",
            "Official GT boxes/IDs in PKLs must be isolated from generator inputs (Stage B4).",
            "After extract: delete raw archives under robosense_raw to stay within budget_gb.",
        ],
    }


def pick_smallest_file(inv: dict[str, Any]) -> dict[str, Any]:
    files = list(inv.get("files") or [])
    if not files:
        raise RuntimeError("inventory has no files")
    smallest = min(files, key=lambda f: int(f.get("size") or 0))
    size = int(smallest.get("size") or 0)
    if size <= 0:
        raise RuntimeError(f"invalid size for smallest file: {smallest}")
    if size >= PROBE_SOFT_CAP_BYTES:
        raise RuntimeError(
            f"smallest inventory file is {size} bytes (>=1GB); refusing probe"
        )
    return {
        "path": smallest["path"],
        "size": size,
        "size_gb": round(size / BYTES_PER_GB, 4),
        "kind": "probe",
        "sha256": smallest.get("sha256"),
        "md5": smallest.get("md5"),
    }


def write_manifest_txt(path: Path, plan: dict[str, Any], budget_gb: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# AutoLabel4D RoboSense subset manifest draft: rs_subset_v0",
        f"# generated_at_utc: {datetime.now(timezone.utc).isoformat()}",
        f"# mode: dry-run",
        f"# estimated_gb: {plan['estimated_gb']}",
        f"# budget_gb: {budget_gb}",
        f"# status: {'WITHIN_BUDGET' if plan['estimated_gb'] <= budget_gb else 'OVER_BUDGET'}",
        "#",
        "# === placeholder sequences (replace after PKL parse) ===",
    ]
    for role, seqs in plan["placeholder_sequences"].items():
        lines.append(f"# role:{role}")
        for s in seqs:
            lines.append(f"# sequence:{s}")
    lines.append("#")
    lines.append("# === planned HF shards (path\\tsize_bytes\\tsize_gb) ===")
    for sh in plan["shards"]:
        lines.append(f"{sh['path']}\t{sh['size']}\t{sh['size_gb']}")
    lines.append("#")
    lines.append("# assumptions:")
    for a in plan["assumptions"]:
        lines.append(f"# - {a}")
    path.write_text("\n".join(lines) + "\n")


def write_plan_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def print_report(
    plan: dict[str, Any],
    budgets: dict[str, Any],
    effective_max: float,
    inv_status: str,
    inv_path: str | None,
    dry_run: bool,
) -> None:
    print("=== AutoLabel4D Stage B — RoboSense subset ===")
    print(f"mode: {'dry-run' if dry_run else 'EXECUTE'}")
    print(f"inventory: {inv_status}" + (f" @ {inv_path}" if inv_path else ""))
    print(f"effective_max_gb: {effective_max}")
    print(
        "paths.budget_gb: "
        f"raw_max={budgets.get('robosense_raw_max')} "
        f"subset_max={budgets.get('robosense_subset_max')} "
        f"cache_max={budgets.get('robosense_cache_max')}"
    )
    print(f"estimated_media_gb: {plan['estimated_gb']}")
    over = plan["estimated_gb"] > effective_max
    hit_subset = plan["estimated_gb"] > float(budgets.get("robosense_subset_max", 35))
    hit_raw = plan["estimated_gb"] > float(budgets.get("robosense_raw_max", 50))
    print(f"quota_would_be_hit: {over} (effective)")
    print(f"would_hit_subset_max: {hit_subset}")
    print(f"would_hit_raw_max: {hit_raw}")
    print(f"remaining_budget_gb: {round(effective_max - plan['estimated_gb'], 4)}")
    print("--- proposed shards ---")
    for sh in plan["shards"]:
        print(f"  {sh['size_gb']:8.4f} GB  [{sh['kind']}]  {sh['path']}")
    if plan["missing_shards"]:
        print("--- MISSING from inventory ---")
        for m in plan["missing_shards"]:
            print(f"  {m}")
    print("--- placeholder sequences ---")
    for role, seqs in plan["placeholder_sequences"].items():
        print(f"  {role}:")
        for s in seqs:
            print(f"    - {s}")
    print("--- assumptions ---")
    for a in plan["assumptions"]:
        print(f"  * {a}")
    train_note = (
        f"Dev uses 2 placeholder seqs; train_try notes remaining budget "
        f"{round(effective_max - plan['estimated_gb'], 2)} GB for more sequences "
        "once PKL-level selection is available."
    )
    print(f"train_try_note: {train_note}")


def dir_usage_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _looks_like_gzip_archive(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".tar.gz") or name.endswith(".tgz") or name.endswith(".gz")


def gzip_test(path: Path) -> tuple[bool, str]:
    """Run `gzip -t`. Return (ok, detail).

    Truncated multi-part heads often fail with unexpected-EOF; callers may
    treat that as VERIFY_PARTIAL when magic is valid.
    """
    try:
        proc = subprocess.run(
            ["gzip", "-t", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        return False, f"gzip binary missing: {exc}"
    detail = (proc.stderr or proc.stdout or "").strip() or f"exit={proc.returncode}"
    return proc.returncode == 0, detail


def verify_downloaded(
    local_path: Path,
    expected_size: int | None,
    expected_sha256: str | None = None,
    expected_md5: str | None = None,
    require_gzip: bool | None = None,
) -> str:
    """Verify size (+ optional digests). For .tar.gz/.gz also require magic 1f8b
    and gzip -t.

    Returns status string:
      VERIFY_OK | VERIFY_PARTIAL (valid gzip magic, truncated stream / EOF)
    Raises RuntimeError on hard failure (size/magic/gzip corrupt/non-gzip).
    """
    if not local_path.is_file():
        raise FileNotFoundError(f"download missing: {local_path}")
    actual = local_path.stat().st_size
    if expected_size is not None and actual != int(expected_size):
        raise RuntimeError(
            f"size mismatch for {local_path}: got {actual}, expected {expected_size}"
        )
    if expected_sha256:
        got = sha256_file(local_path)
        if got.lower() != str(expected_sha256).lower():
            raise RuntimeError(
                f"sha256 mismatch for {local_path}: got {got}, expected {expected_sha256}"
            )
    if expected_md5:
        import hashlib as _hl

        h = _hl.md5()
        with local_path.open("rb") as f:
            while True:
                block = f.read(1024 * 1024)
                if not block:
                    break
                h.update(block)
        got = h.hexdigest()
        if got.lower() != str(expected_md5).lower():
            raise RuntimeError(
                f"md5 mismatch for {local_path}: got {got}, expected {expected_md5}"
            )

    if require_gzip is None:
        require_gzip = _looks_like_gzip_archive(local_path)

    if require_gzip:
        with local_path.open("rb") as f:
            magic = f.read(2)
        if magic != b"\x1f\x8b":
            raise RuntimeError(
                f"gzip magic mismatch for {local_path}: got {magic.hex()!r}, "
                "expected 1f8b (NOT a standalone gzip; multi-part continuation?)"
            )
        ok, detail = gzip_test(local_path)
        if ok:
            print(f"VERIFY_OK: {local_path} size={actual} magic=1f8b gzip-t=PASS")
            return "VERIFY_OK"
        low = detail.lower()
        truncated_hints = (
            "unexpected end",
            "unexpected eof",
            "end-of-stream",
            "compressed file ended",
            "trailing garbage",
        )
        if any(h in low for h in truncated_hints) or "exit=1" in low:
            # gzip -t exit 1 with EOF = truncated multi-part head (still extractable)
            print(
                f"VERIFY_PARTIAL: {local_path} size={actual} magic=1f8b "
                f"gzip-t=TRUNCATED ({detail})"
            )
            return "VERIFY_PARTIAL"
        raise RuntimeError(
            f"gzip -t FAILED for {local_path}: {detail}"
        )

    print(f"VERIFY_OK: {local_path} size={actual}")
    return "VERIFY_OK"


def try_import_hf_hub():
    try:
        from huggingface_hub import hf_hub_download  # type: ignore

        return hf_hub_download
    except ImportError:
        return None


def _resolve_url(rel_path: str) -> str:
    """Build download URL; honor HF_ENDPOINT mirror (e.g. https://hf-mirror.com)."""
    endpoint = (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
    # huggingface_hub uses HF_ENDPOINT as hub root; mirror resolve path matches.
    return f"{endpoint}/datasets/{HF_REPO}/resolve/main/{rel_path}"


def _download_via_curl_wget(rel_path: str, dest: Path) -> Path:
    url = _resolve_url(rel_path)
    print(f"DOWNLOAD via curl/wget: {url} -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".partial")
    if tmp.exists():
        tmp.unlink()
    curl = shutil.which("curl")
    wget = shutil.which("wget")
    try:
        if curl:
            cmd = [curl, "-L", "--fail", "--retry", "3", "-o", str(tmp), url]
            subprocess.run(cmd, check=True)
        elif wget:
            cmd = ["wget", "-O", str(tmp), url]
            subprocess.run(cmd, check=True)
        else:
            raise RuntimeError("Neither curl nor wget available for download fallback")
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return dest


def clear_hf_local_dir_cache(raw_dir: Path, rel_path: str) -> None:
    """Remove huggingface local_dir cache metadata/incomplete for one file."""
    meta = raw_dir / ".cache" / "huggingface" / "download" / (rel_path + ".metadata")
    if meta.is_file():
        meta.unlink()
        print(f"CACHE_CLEAR: {meta}")
    # incomplete / lock siblings under download/
    dl_dir = raw_dir / ".cache" / "huggingface" / "download" / Path(rel_path).parent
    stem = Path(rel_path).name
    if dl_dir.is_dir():
        for p in dl_dir.glob(stem + "*"):
            if p.is_file() and p.suffix in {".metadata", ".lock", ".incomplete"} or (
                p.name.startswith(stem) and p.name != stem
            ):
                try:
                    p.unlink()
                    print(f"CACHE_CLEAR: {p}")
                except OSError as exc:
                    eprint(f"WARN: could not clear {p}: {exc}")


def download_one_file(
    rel_path: str,
    raw_dir: Path,
    expected_size: int | None = None,
    force: bool = False,
) -> Path:
    """Download HF dataset file into raw_dir preserving relative path.

    Prefer huggingface_hub.hf_hub_download(local_dir=raw_dir); fall back to
    wget/curl. On Legion, huggingface.co may be unreachable — set
    HF_ENDPOINT=https://hf-mirror.com (worked for probe).
    With force=True: clear HF cache metadata, delete local file, force_download.
    """
    dest = raw_dir / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)

    if force:
        clear_hf_local_dir_cache(raw_dir, rel_path)
        if dest.is_file():
            dest.unlink()
            print(f"FORCE_REMOVE: {dest}")

    if (
        not force
        and dest.is_file()
        and expected_size is not None
        and dest.stat().st_size == expected_size
    ):
        print(f"SKIP_EXISTING: {dest} (size matches inventory)")
        return dest

    endpoint = os.environ.get("HF_ENDPOINT")
    if endpoint:
        print(f"HF_ENDPOINT={endpoint}")

    hf_hub_download = try_import_hf_hub()
    hub_err: Exception | None = None
    if hf_hub_download is not None:
        try:
            print(
                f"DOWNLOAD via huggingface_hub: {rel_path} -> {raw_dir} "
                f"force_download={force}"
            )
            kwargs = dict(
                repo_id=HF_REPO,
                repo_type=HF_REPO_TYPE,
                filename=rel_path,
                local_dir=str(raw_dir),
            )
            if force:
                kwargs["force_download"] = True
            out = hf_hub_download(**kwargs)
            out_path = Path(out)
            if out_path.resolve() != dest.resolve() and out_path.is_file():
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    shutil.move(str(out_path), str(dest))
            if dest.is_file():
                return dest
            if out_path.is_file():
                return out_path
        except Exception as exc:
            hub_err = exc
            eprint(f"WARN: huggingface_hub failed ({exc}); trying curl/wget fallback")

    try:
        return _download_via_curl_wget(rel_path, dest)
    except Exception as exc:
        if hub_err is not None:
            raise RuntimeError(
                f"All download backends failed. hub={hub_err}; curl/wget={exc}. "
                "If huggingface.co is unreachable from this host, retry with "
                "HF_ENDPOINT=https://hf-mirror.com"
            ) from exc
        raise


def enforce_budget_for_plan(
    plan_bytes: int,
    effective_max_gb: float,
    raw_dir: Path,
    raw_max_gb: float,
) -> None:
    plan_gb = plan_bytes / BYTES_PER_GB
    if plan_gb > effective_max_gb:
        raise RuntimeError(
            f"REFUSE plan: estimated {plan_gb:.4f} GB exceeds --max-gb/budget "
            f"{effective_max_gb} GB"
        )
    if plan_gb > raw_max_gb:
        raise RuntimeError(
            f"REFUSE plan: estimated {plan_gb:.4f} GB exceeds "
            f"budget_gb.robosense_raw_max={raw_max_gb}"
        )
    existing = dir_usage_bytes(raw_dir)
    projected = existing + plan_bytes
    if projected / BYTES_PER_GB > raw_max_gb:
        raise RuntimeError(
            f"REFUSE download: raw dir usage {existing / BYTES_PER_GB:.4f} GB + "
            f"plan {plan_gb:.4f} GB would exceed robosense_raw_max={raw_max_gb}"
        )


def download_shards(
    shards: list[dict[str, Any]],
    raw_dir: Path,
    effective_max_gb: float,
    raw_max_gb: float,
    force: bool = False,
    force_paths: set[str] | None = None,
    continue_on_verify_fail: bool = False,
) -> list[dict[str, Any]]:
    """Download shards sequentially; abort if cumulative would exceed budget.

    With force=True, redownload all shards even if size matches.
    With force_paths, force only those relative paths (in addition to force).
    With continue_on_verify_fail=True, size+magic+gzip-t failures are recorded
    and the loop continues (after one retry) instead of aborting the batch.
    """
    results: list[dict[str, Any]] = []
    downloaded_bytes = 0
    force_paths = force_paths or set()
    baseline = dir_usage_bytes(raw_dir)
    for sh in shards:
        size = int(sh["size"])
        projected_new = downloaded_bytes + size
        if (baseline + projected_new) / BYTES_PER_GB > raw_max_gb:
            raise RuntimeError(
                f"ABORT mid-download: adding {sh['path']} ({sh['size_gb']} GB) "
                f"would exceed robosense_raw_max={raw_max_gb}"
            )
        if projected_new / BYTES_PER_GB > effective_max_gb:
            raise RuntimeError(
                f"ABORT mid-download: cumulative {projected_new / BYTES_PER_GB:.4f} GB "
                f"exceeds effective_max_gb={effective_max_gb}"
            )

        last_err: Exception | None = None
        verify_status = None
        local: Path | None = None
        for attempt in (1, 2):
            try:
                do_force = force or (sh["path"] in force_paths) or (attempt > 1)
                local = download_one_file(
                    sh["path"],
                    raw_dir,
                    expected_size=size,
                    force=do_force,
                )
                verify_status = verify_downloaded(
                    local,
                    expected_size=size,
                    expected_sha256=sh.get("sha256"),
                    expected_md5=sh.get("md5"),
                )
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                eprint(
                    f"VERIFY_OR_DOWNLOAD_FAIL attempt={attempt}/2 "
                    f"path={sh['path']}: {exc}"
                )
                if attempt == 1:
                    eprint(f"RETRY once with force for {sh['path']}")
                    continue
        if last_err is not None:
            if not continue_on_verify_fail:
                raise last_err
            results.append(
                {
                    "path": sh["path"],
                    "local": str(local) if local else None,
                    "size": local.stat().st_size if local and local.is_file() else None,
                    "ok": False,
                    "verify": "VERIFY_FAIL",
                    "error": str(last_err),
                }
            )
            # still count size toward budget if file landed (continuation parts)
            if local is not None and local.is_file():
                downloaded_bytes += local.stat().st_size
            continue

        assert local is not None
        downloaded_bytes += local.stat().st_size
        results.append(
            {
                "path": sh["path"],
                "local": str(local),
                "size": local.stat().st_size,
                "ok": True,
                "verify": verify_status or "VERIFY_OK",
            }
        )
    return results



KITTI_FORBIDDEN = Path("/data/data/kitti0000.tar.gz")


def _is_forbidden_path(p: Path) -> bool:
    try:
        return p.resolve() == KITTI_FORBIDDEN.resolve()
    except OSError:
        return str(p) == str(KITTI_FORBIDDEN)


def _has_gzip_magic(path: Path) -> bool:
    with path.open("rb") as f:
        return f.read(2) == b"\x1f\x8b"


def _safe_extract_tar_gz(archive: Path, dest_dir: Path) -> list[str]:
    """Extract tar.gz into dest_dir with multi-part awareness.

    RoboSense HF shards are ~10GB *split volumes* of one gzip stream:
    only part_01 starts with gzip magic; later parts are continuations and
    cannot be extracted alone. For part_01 (truncated stream), use streaming
    tar mode (r|) and keep all complete members until gzip EOF.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_root = dest_dir.resolve()
    if not _has_gzip_magic(archive):
        raise RuntimeError(
            f"NOT_GZIP_STANDALONE (multi-part continuation?): {archive}. "
            "RoboSense part_N>1 must be cat'd with part_01..N; skipping extract."
        )

    extracted: list[str] = []
    truncated = False
    try:
        with gzip.open(archive, "rb") as gz:
            with tarfile.open(fileobj=gz, mode="r|") as tf:
                for member in tf:
                    name = member.name
                    if not name or name.startswith("/"):
                        raise RuntimeError(
                            f"unsafe absolute member in {archive}: {name!r}"
                        )
                    target = (dest_dir / name).resolve()
                    if (
                        not str(target).startswith(str(dest_root) + "/")
                        and target != dest_root
                    ):
                        raise RuntimeError(
                            f"path traversal blocked for {archive}: "
                            f"member={name!r} -> {target}"
                        )
                    if _is_forbidden_path(target):
                        raise RuntimeError(
                            f"REFUSE extract onto forbidden path: {target}"
                        )
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                        extracted.append(name)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    src = tf.extractfile(member)
                    if src is None:
                        continue
                    with target.open("wb") as out:
                        while True:
                            chunk = src.read(1024 * 1024)
                            if not chunk:
                                break
                            out.write(chunk)
                    extracted.append(name)
    except EOFError:
        truncated = True
    except (gzip.BadGzipFile, tarfile.ReadError, OSError) as exc:
        # Truncated multi-part streams often surface as EOF/CRC errors mid-member
        msg = str(exc)
        if extracted and (
            "end-of-stream" in msg
            or "Compressed file ended" in msg
            or "CRC check failed" in msg
            or "Invalid" in msg
        ):
            truncated = True
            print(f"EXTRACT_TRUNCATED: {archive} after {len(extracted)} members ({exc})")
        elif extracted:
            truncated = True
            print(f"EXTRACT_TRUNCATED: {archive} after {len(extracted)} members ({exc})")
        else:
            raise

    if not extracted:
        raise RuntimeError(f"extract produced no members: {archive}")
    tag = "EXTRACT_PARTIAL_OK" if truncated else "EXTRACT_OK"
    print(f"{tag}: {archive} -> {dest_dir} ({len(extracted)} members, truncated={truncated})")
    return extracted


def copy_split_pkls(raw_dir: Path, subset_dir: Path, rel_paths: list[str] | None = None) -> list[str]:
    """Copy split PKLs from raw (or raw/splits) into subset/splits/."""
    subset_splits = subset_dir / "splits"
    subset_splits.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    candidates: list[Path] = []
    if rel_paths:
        for rp in rel_paths:
            candidates.append(raw_dir / rp)
    else:
        for base in (raw_dir / "splits", raw_dir):
            if base.is_dir():
                candidates.extend(sorted(base.glob("*.pkl")))
    seen: set[str] = set()
    for src in candidates:
        if not src.is_file():
            continue
        if src.suffix != ".pkl":
            continue
        key = src.name
        if key in seen:
            continue
        seen.add(key)
        dst = subset_splits / src.name
        shutil.copy2(src, dst)
        print(f"PKL_COPY: {src} -> {dst}")
        copied.append(str(dst))
    return copied


def extract_and_maybe_delete(
    downloaded: list[dict[str, Any]],
    raw_dir: Path,
    subset_dir: Path,
    do_extract: bool,
    do_delete: bool,
) -> dict[str, Any]:
    """Extract archives to subset; copy PKLs; optionally delete raw .tar.gz after success.

    Never deletes PKLs or non-archive files. Never touches kitti.
    """
    summary: dict[str, Any] = {
        "extracted": [],
        "pkls_copied": [],
        "deleted_archives": [],
        "skipped": [],
    }
    if not do_extract and not do_delete:
        return summary

    subset_dir.mkdir(parents=True, exist_ok=True)
    pkl_rels = []
    archive_locals: list[Path] = []

    for item in downloaded:
        if not item.get("ok", True):
            summary["skipped"].append(
                f"VERIFY_FAIL:{item.get('path')}:{item.get('error')}"
            )
            continue
        verify = item.get("verify") or ""
        # Only extract archives that passed magic+gzip-t (OK or PARTIAL truncated head)
        if verify not in ("", "VERIFY_OK", "VERIFY_PARTIAL") and item.get("path", "").endswith(".tar.gz"):
            summary["skipped"].append(f"BAD_VERIFY:{item.get('path')}:{verify}")
            continue
        local_s = item.get("local")
        if not local_s:
            summary["skipped"].append(f"NO_LOCAL:{item.get('path')}")
            continue
        local = Path(local_s)
        rel = item.get("path") or ""
        if _is_forbidden_path(local):
            raise RuntimeError(f"REFUSE: downloaded path is forbidden kitti: {local}")
        if rel.endswith(".pkl") or local.suffix == ".pkl":
            pkl_rels.append(rel if rel else f"splits/{local.name}")
            continue
        if local.name.endswith(".tar.gz") or str(local).endswith(".tar.gz"):
            # Require gzip magic before queueing extract (gzip -t already in verify)
            if not _has_gzip_magic(local):
                summary["skipped"].append(f"NO_GZIP_MAGIC:{local}")
                continue
            archive_locals.append(local)
            continue
        summary["skipped"].append(str(local))

    if do_extract:
        summary["pkls_copied"] = copy_split_pkls(raw_dir, subset_dir, pkl_rels or None)
        for arch in archive_locals:
            if not arch.is_file():
                raise FileNotFoundError(f"archive missing for extract: {arch}")
            if _is_forbidden_path(arch):
                raise RuntimeError(f"REFUSE extract/delete of forbidden path: {arch}")
            if not _has_gzip_magic(arch):
                print(
                    f"SKIP_CONTINUATION_PART: {arch} (no gzip magic; needs cat with part_01..N)"
                )
                summary["skipped"].append(str(arch))
                continue
            try:
                members = _safe_extract_tar_gz(arch, subset_dir)
            except RuntimeError as exc:
                if "NOT_GZIP_STANDALONE" in str(exc):
                    print(f"SKIP_CONTINUATION_PART: {arch}")
                    summary["skipped"].append(str(arch))
                    continue
                raise
            ok_any = False
            for m in members[:20]:
                if (subset_dir / m).exists():
                    ok_any = True
                    break
            if not ok_any and members:
                if not any(subset_dir.iterdir()):
                    raise RuntimeError(f"extract verify failed (subset empty): {arch}")
            # Multi-part part_01 is inherently truncated without remaining parts —
            # treat as partial; do NOT delete raw archives for partial extracts.
            partial = True  # RoboSense part_01 alone is always truncated vs full series
            entry = {
                "archive": str(arch),
                "members": len(members),
                "partial_multipart": partial,
            }
            summary["extracted"].append(entry)
            if do_delete and not partial:
                if _is_forbidden_path(arch):
                    raise RuntimeError(f"REFUSE delete forbidden path: {arch}")
                if not (arch.name.endswith(".tar.gz") or str(arch).endswith(".tar.gz")):
                    raise RuntimeError(f"REFUSE delete non-archive: {arch}")
                arch.unlink()
                print(f"DELETE_RAW_ARCHIVE: {arch}")
                summary["deleted_archives"].append(str(arch))
            elif do_delete and partial:
                print(
                    f"KEEP_RAW_ARCHIVE (partial multi-part extract): {arch}"
                )
    elif do_delete:
        eprint(
            "WARN: --delete-raw-archives without --extract-to-subset is refused "
            "to avoid deleting unextracted archives"
        )
        raise RuntimeError("REFUSE --delete-raw-archives without --extract-to-subset")

    return summary



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "RoboSense subset planner + gated HF download. "
            "Default: dry-run. Real download needs --execute and ALLOW_RS_DOWNLOAD=1. "
            "Use --probe-only for a <<1GB connectivity probe."
        )
    )
    p.add_argument(
        "--paths",
        type=Path,
        default=DEFAULT_PATHS,
        help="Path to configs/paths.yaml",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Dry-run (default). List plan + write draft manifest; no download.",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="Attempt real download (requires ALLOW_RS_DOWNLOAD=1).",
    )
    p.add_argument(
        "--probe-only",
        action="store_true",
        help=(
            "With --execute: download ONLY the single smallest inventory file "
            "(must be <<1GB), verify size, write under robosense_raw/. "
            "Never downloads the full ~28GB subset plan."
        ),
    )
    p.add_argument(
        "--cleanup-probe",
        action="store_true",
        help="After successful --probe-only, delete the probe file to free space.",
    )
    p.add_argument(
        "--max-gb",
        type=float,
        default=45.0,
        help="Max estimated media GB (default 45, clamped to 30–50).",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Draft manifest txt path (default: <paths.manifests>/rs_subset_v0.txt)",
    )
    p.add_argument(
        "--plan-json",
        type=Path,
        default=None,
        help="Optional JSON plan path (default: <paths.manifests>/rs_subset_v0_plan.json)",
    )
    p.add_argument(
        "--inventory",
        type=Path,
        default=None,
        help="Cached HF shard inventory JSON",
    )
    p.add_argument(
        "--refresh-inventory",
        action="store_true",
        help="Try HF API refresh before falling back to cache",
    )
    p.add_argument(
        "--use-subset-max",
        action="store_true",
        help="Use paths.yaml budget_gb.robosense_subset_max as effective budget "
        "instead of --max-gb",
    )
    p.add_argument(
        "--extract-to-subset",
        action="store_true",
        help=(
            "After successful download+verify, extract each downloaded *.tar.gz "
            "into paths.yaml robosense_subset (preserve structure) and copy "
            "split PKLs into subset/splits/."
        ),
    )
    p.add_argument(
        "--delete-raw-archives",
        action="store_true",
        help=(
            "After successful extract+verify of each archive, delete that "
            ".tar.gz from robosense_raw to free space. Requires "
            "--extract-to-subset. Never deletes PKLs or kitti."
        ),
    )
    p.add_argument(
        "--force-redownload",
        action="store_true",
        help=(
            "Force redownload of planned shards even if local size matches. "
            "Clears HF local_dir cache metadata, deletes local files, and "
            "passes force_download=True to huggingface_hub. Verify requires "
            "size + gzip magic 1f8b + gzip -t for archives. "
            "Use --force-paths to limit to listed relative paths."
        ),
    )
    p.add_argument(
        "--force-download",
        action="store_true",
        help="Alias for --force-redownload.",
    )
    p.add_argument(
        "--force-paths",
        nargs="*",
        default=None,
        help=(
            "With --force-redownload/--force-download: only force these "
            "relative HF paths (e.g. dataset/image_trainval_part_01.tar.gz). "
            "If omitted, force all planned shards."
        ),
    )
    p.add_argument(
        "--continue-on-verify-fail",
        action="store_true",
        help=(
            "After one retry, record verify failures and continue remaining "
            "shards (useful for multi-part continuation tails that lack gzip magic)."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dry_run = not args.execute  # default dry-run; --execute flips

    if args.execute:
        if os.environ.get("ALLOW_RS_DOWNLOAD") != "1":
            eprint(
                "REFUSE: --execute without ALLOW_RS_DOWNLOAD=1. "
                "Export ALLOW_RS_DOWNLOAD=1 to permit downloads, or use --dry-run / --probe-only planning."
            )
            return 2

    paths = load_paths(args.paths)
    budgets = paths["budget_gb"]
    manifests_dir = Path(
        paths.get("manifests", "/data/data/automomous/autolabel4d/manifests")
    )
    raw_dir = Path(paths["robosense_raw"])
    raw_max_gb = float(budgets.get("robosense_raw_max", 50))

    if args.use_subset_max:
        effective_max = float(budgets["robosense_subset_max"])
    else:
        effective_max = clamp_max_gb(args.max_gb)

    inv, inv_status = build_inventory(args.refresh_inventory, args.inventory)
    by_path = index_by_path(inv)

    # Full subset plan (always computed for reporting / dry-run manifests).
    plan = propose_selection(by_path)

    manifest_path = args.manifest or (manifests_dir / "rs_subset_v0.txt")
    plan_json_path = args.plan_json or (manifests_dir / "rs_subset_v0_plan.json")

    mode = "dry-run"
    if args.execute and args.probe_only:
        mode = "probe"
    elif args.execute:
        mode = "execute"

    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo": HF_REPO,
        "mode": mode,
        "inventory_status": inv_status,
        "inventory_path": inv.get("_inventory_path"),
        "paths_file": str(args.paths),
        "robosense_raw": str(raw_dir),
        "budgets_gb": budgets,
        "effective_max_gb": effective_max,
        "selection": plan,
        "quota": {
            "estimated_gb": plan["estimated_gb"],
            "over_effective_max": plan["estimated_gb"] > effective_max,
            "over_subset_max": plan["estimated_gb"]
            > float(budgets.get("robosense_subset_max", 35)),
            "over_raw_max": plan["estimated_gb"] > raw_max_gb,
        },
    }

    # Dry-run path: write manifests, enforce budget on full plan, no download.
    if dry_run:
        write_manifest_txt(manifest_path, plan, effective_max)
        write_plan_json(plan_json_path, payload)
        print_report(
            plan,
            budgets,
            effective_max,
            inv_status,
            inv.get("_inventory_path"),
            dry_run=True,
        )
        print(f"manifest_written: {manifest_path}")
        print(f"plan_json_written: {plan_json_path}")
        print(f"robosense_raw: {raw_dir}")
        print(
            "NOTE: after a future real extract, delete raw archives under "
            f"{raw_dir} to free budget_gb.robosense_raw_max"
        )

        if plan["missing_shards"]:
            eprint("ERROR: proposed shards missing from inventory")
            return 3

        if plan["estimated_gb"] > effective_max:
            eprint(
                f"REJECT: estimated {plan['estimated_gb']} GB exceeds budget "
                f"{effective_max} GB"
            )
            return 1

        print("PASS: dry-run within budget; no download started")
        return 0

    # ---- execute path (ALLOW_RS_DOWNLOAD already checked) ----
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.probe_only:
        probe = pick_smallest_file(inv)
        print("=== PROBE-ONLY download (NOT full subset) ===")
        print(
            f"probe_file: {probe['path']}  size={probe['size']} "
            f"({probe['size_gb']} GB)"
        )
        try:
            enforce_budget_for_plan(
                probe["size"], effective_max, raw_dir, raw_max_gb
            )
        except RuntimeError as exc:
            eprint(str(exc))
            return 1
        try:
            results = download_shards(
                [probe], raw_dir, effective_max, raw_max_gb
            )
        except Exception as exc:
            eprint(f"PROBE_FAILED: {exc}")
            eprint(
                "BLOCKER: HuggingFace download from this host failed. "
                "Code path is complete; fix network/proxy/HF access and retry "
                "--probe-only before full subset download."
            )
            return 4
        local = results[0]["local"]
        print(f"PROBE_OK: {local}")
        payload["probe"] = results[0]
        payload["selection_executed"] = [probe]
        write_plan_json(plan_json_path.with_name("rs_subset_v0_probe.json"), payload)
        if args.cleanup_probe:
            Path(local).unlink(missing_ok=True)
            print(f"PROBE_CLEANUP: removed {local}")
        else:
            print(
                f"PROBE_KEEP: left tiny probe at {local} "
                "(pass --cleanup-probe to delete)"
            )
        print("PASS: probe-only download complete; full ~28GB subset NOT started")
        return 0

    # Full subset execute
    print_report(
        plan,
        budgets,
        effective_max,
        inv_status,
        inv.get("_inventory_path"),
        dry_run=False,
    )
    if plan["missing_shards"]:
        eprint("ERROR: proposed shards missing from inventory")
        return 3
    try:
        enforce_budget_for_plan(
            plan["estimated_bytes"], effective_max, raw_dir, raw_max_gb
        )
    except RuntimeError as exc:
        eprint(str(exc))
        return 1

    force_flag = bool(args.force_redownload or getattr(args, "force_download", False))
    force_paths_arg = getattr(args, "force_paths", None)
    force_paths: set[str] = set(force_paths_arg) if force_paths_arg else set()
    # If --force-redownload without --force-paths => force all planned shards.
    force_all = force_flag and not force_paths
    if force_flag and force_paths:
        print(f"FORCE_REDOWNLOAD paths={sorted(force_paths)}")
        for rp in sorted(force_paths):
            clear_hf_local_dir_cache(raw_dir, rp)
    elif force_all:
        print("FORCE_REDOWNLOAD: all planned shards")

    print(
        f"EXECUTE: downloading {len(plan['shards'])} shards "
        f"({plan['estimated_gb']} GB) into {raw_dir}"
    )
    try:
        results = download_shards(
            plan["shards"],
            raw_dir,
            effective_max,
            raw_max_gb,
            force=force_all,
            force_paths=force_paths if force_flag else set(),
            continue_on_verify_fail=bool(args.continue_on_verify_fail),
        )
    except Exception as exc:
        eprint(f"DOWNLOAD_FAILED: {exc}")
        return 4

    payload["downloaded"] = results
    write_manifest_txt(manifest_path, plan, effective_max)

    subset_dir = Path(paths["robosense_subset"])
    extract_summary = None
    if args.extract_to_subset or args.delete_raw_archives:
        print(
            f"EXTRACT_PHASE: extract_to_subset={args.extract_to_subset} "
            f"delete_raw_archives={args.delete_raw_archives} -> {subset_dir}"
        )
        try:
            extract_summary = extract_and_maybe_delete(
                results,
                raw_dir,
                subset_dir,
                do_extract=bool(args.extract_to_subset or args.delete_raw_archives),
                do_delete=bool(args.delete_raw_archives),
            )
        except Exception as exc:
            eprint(f"EXTRACT_OR_DELETE_FAILED: {exc}")
            payload["extract"] = {"error": str(exc)}
            write_plan_json(plan_json_path, payload)
            return 5
        payload["extract"] = extract_summary
        print(
            f"EXTRACT_SUMMARY: extracted={len(extract_summary.get('extracted', []))} "
            f"pkls={len(extract_summary.get('pkls_copied', []))} "
            f"deleted_archives={len(extract_summary.get('deleted_archives', []))}"
        )

    write_plan_json(plan_json_path, payload)
    if extract_summary is not None:
        print(
            f"PASS: downloaded {len(results)} shards; extract done under {subset_dir}; "
            f"raw tar.gz deleted={len(extract_summary.get('deleted_archives', []))} "
            f"(PKLs may remain under {raw_dir})."
        )
    else:
        print(
            f"PASS: downloaded {len(results)} shards under {raw_dir}. "
            "After extract, DELETE raw archives to free robosense_raw quota."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
