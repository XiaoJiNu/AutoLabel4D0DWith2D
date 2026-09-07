# RS_ALIGN_PATCH_STATUS — same-day image↔hs64 restore

**Updated:** 2026-09-06 19:26:26 CST  
**Machine:** Legion `252f7d1f-95fc-44c6-94d1-8f76153ee6af`  
**Repo:** `/data/code/cv/AutoLabel/AutoLabel4D0DWith2D`  
**Subset:** `/data/data/automomous/robosense/subset`  
**Auth:** `ALLOW_RS_DOWNLOAD=1` `HF_ENDPOINT=https://hf-mirror.com` (幕僚长)

## Report card

| Field | Value |
|-------|-------|
| n_aligned_frames | **12** (was 0) |
| overlap days | 2023-09-07, 2023-09-08, 2023-09-09 |
| unlock pack | `processed_data_20230906` (images patched onto existing hs64) |
| viz | 4× LiDAR→CAM_FRONT overlays (png+jpg) |
| raw now | **477M** (archives deleted after extract) |
| subset now | **28G** (≤35) |
| `/data` free | **231G** (reserve OK) |

## Decision (step 1)

On-disk before patch:

| modality | pack | days |
|----------|------|------|
| images | 20231011 | 09-28, 10-08 |
| hs64 | 20230906 | 09-07..09 |

Zero day overlap. TOC of multipart streams:

- **lidar** part_01+02: `20230906` (complete hs64/livox/occ) → start `20231123` hs64 (560 bins)
- **image** part_01..04: `20231011` → long `20230710`; `20230906` begins mid **part_09**

Cheapest unlock with existing complete hs64_20230906: stream **image** series until `processed_data_20230906` folder-0 frames matching hs64 stems (Option B).

Did **not** join full 16/23-part series into one file.

## Files downloaded (network)

All sizes match HF inventory (10 737 418 240 B = 10.74 GB) unless noted.

### Lidar (probe / side extract; later deleted from raw)

| file | size_gb | verify | fate |
|------|---------|--------|------|
| `dataset/lidar_occ_trainval_part_01.tar.gz` | 10.7374 | VERIFY_PARTIAL (1f8b, gzip EOF) | extracted 560× `20231123` hs64; deleted |
| `dataset/lidar_occ_trainval_part_02.tar.gz` | 10.7374 | size OK (continuation) | same; deleted |

### Image (align path)

| file | size_gb | where | fate |
|------|---------|-------|------|
| `image_trainval_part_01.tar.gz` | 10.7374 | `/data/.../raw` | cat’d; deleted after success |
| `image_trainval_part_02.tar.gz` | 10.7374 | raw | deleted |
| `image_trainval_part_03.tar.gz` | 10.7374 | raw | deleted |
| `image_trainval_part_04.tar.gz` | 10.7374 | raw | deleted |
| `image_trainval_part_05.tar.gz` | 10.7374 | `/home/.../tmp_shards` | streamed+deleted |
| `image_trainval_part_06.tar.gz` | 10.7374 | tmp_shards | streamed+deleted |
| `image_trainval_part_07.tar.gz` | 10.7374 | tmp_shards | streamed+deleted |
| `image_trainval_part_08.tar.gz` | 10.7374 | tmp_shards | streamed+deleted |
| `image_trainval_part_09.tar.gz` | 10.7374 | tmp_shards | hit `20230906` matches; stop; cleaned |

**Approx total fetched:** 2×lidar + 9×image ≈ **118.1 GB** network.  
**Peak robosense_raw:** ~41 GB (parts 01–04) ≤50. Later parts used `/home` tmp to avoid raw>50.  
**Retained subset delta:** 12× CAM_FRONT jpgs (~3.8 MB pack folder) + prior 560× `20231123` hs64 (~3.1 GB).

Also deleted unused `livox/` under 20230906 (~3.2 GB) early for subset headroom.

## Extract / verify

- Leaders: magic `1f8b` + `gzip -t` → `VERIFY_PARTIAL` (expected truncated multipart).
- Continuations: size-only (no gzip magic).
- Join method: `cat part_01..N` / ConcatReader → stream tar `r|`; filter `processed_data_20230906/**/images/0/{stem}.jpg` for stems present in hs64_20230906.
- Stopped at **12** matches (`MAX_MATCHED=12`).

## Analyze + viz (steps 3–4)

```text
n_aligned_frames=12
exact_stem pairs on days 2023-09-07/08/09
```

Manifests rebuilt:

- `/data/data/automomous/autolabel4d/manifests/rs_align_frames_v0.json`
- `/data/data/automomous/autolabel4d/manifests/rs_align_frames_v0.txt`
- `/data/data/automomous/autolabel4d/manifests/rs_align_summary_v0.json`
- camera map unchanged: `rs_camera_map_v0.yaml`

Viz (`scripts/viz_rs_align_proj.py --n 4 --cam-folder 0`):

| # | absolute path |
|---|---------------|
| 0 | `/data/code/cv/AutoLabel/AutoLabel4D0DWith2D/outputs/rs_align_proj/proj_00_2023-09-07-11-24-38-051_CAM_FRONT.png` |
| 1 | `/data/code/cv/AutoLabel/AutoLabel4D0DWith2D/outputs/rs_align_proj/proj_01_2023-09-07-11-35-30-049_CAM_FRONT.png` |
| 2 | `/data/code/cv/AutoLabel/AutoLabel4D0DWith2D/outputs/rs_align_proj/proj_02_2023-09-07-17-36-01-549_CAM_FRONT.png` |
| 3 | `/data/code/cv/AutoLabel/AutoLabel4D0DWith2D/outputs/rs_align_proj/proj_03_2023-09-07-17-49-21-550_CAM_FRONT.png` |

(Also `.jpg` siblings from the script.)  
Projection frac≈0.06–0.10 of 80k pts on CAM_FRONT (pinhole); plausible smoke.

## Disk (after cleanup)

```text
raw     477M   (PKLs only)
subset   28G   (≤35)
/data   231G free
```

## Blockers / residual

1. Only **12** folder-0 images for 20230906 (smoke set). More cams/frames need continuing image stream past part_09 (still not a full series join).
2. `processed_data_20231123` hs64 (560 bins, ~3.1G) has **no** matching images yet (image stream had not reached that pack when stopped).
3. Folder **4 / CAM_FRONT_OV** still missing; fisheye unwrap still stub — OV cams not used for viz.
4. Full trainval multipart join still ≫ budget; do not assemble 119GB+ series.
5. No push; no kitti touch; annos not fed to teachers.

## Scripts added (stageB workdir)

- `scripts/stream_extract_images_v2.py` — budget-safe concat + HF tmp shards
- `scripts/extract_pack_hs64.py`, `toc_*.py`, `download_one_shard.py`, …

## Logs

- `/data/data/automomous/autolabel4d/logs/rs_align_patch_*.log`
- `/data/data/automomous/autolabel4d/logs/rs_stream_img_v2_*.log`
- `/data/data/automomous/autolabel4d/logs/rs_align_analyze_patch.log`
- `/data/data/automomous/autolabel4d/logs/rs_align_viz_patch.log`
