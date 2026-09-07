# AutoLabel4D0DWith2D

本地 **RTX 5090（24GB）** 上的 4D OD 自动标注实验仓（方案依据：v3，暂不使用 A100）。

## 路径

| 用途 | 路径 |
|---|---|
| 代码 | `/data/code/cv/AutoLabel/AutoLabel4D0DWith2D` |
| 数据根 | `/data/data/automomous` |
| 伪标签 | `/data/data/automomous/autolabel4d/pseudo_labels` |
| 实现计划 | [`docs/IMPLEMENTATION_PLAN_v3_5090.md`](docs/IMPLEMENTATION_PLAN_v3_5090.md) |
| 标签 schema | [`configs/label_schema.yaml`](configs/label_schema.yaml) |

## 当前策略

1. RoboSense **只下载 30–50GB 子集**（完整序列若干条），禁止全仓 705GB。
2. **边写代码、边下载**；先用已有 `nuscenes/v1.0-mini` 冒烟。
3. 教师全部冻结、**串行**跑；学生首轮为 PointPillar-MultiHead + SimpleTrack。

## 快速入口

```bash
# GPU 探针（日志写入 autolabel4d/logs）
bash scripts/probe_gpu.sh

# Stage A4：nuScenes mini 假伪标签 I/O + BEV/相机可视化
# 推荐 env：mv2d（已装 nuscenes / cv2 / numpy / matplotlib / pyquaternion）
export PYTHONPATH=src
/data/software/conda/anaconda3/envs/mv2d/bin/python scripts/smoke_nuscenes_mini.py \
  --scene scene-0103 --max-samples 8

# 可选覆盖路径
# --dataroot /data/data/automomous/nuscenes/v1.0-mini
# --out-pseudo /data/data/automomous/autolabel4d/pseudo_labels/smoke_nuscenes_mini
# --out-viz outputs/smoke_nuscenes_mini
```

冒烟通过标准：控制台打印 `PASS`；`outputs/smoke_nuscenes_mini/*.png` 存在；伪标签 JSON 可读回且字段齐全（`meta.is_fake=true`）。

## Stage B — RoboSense subset download (gated)

HF 全仓约 **705GB**；本机只规划 **~28–50GB** 子集。默认 **dry-run**（不下载）。

写入目录：`configs/paths.yaml` → `robosense_raw`（`/data/data/automomous/robosense/raw`）。  
配额：`budget_gb.robosense_raw_max` / `--max-gb`；计划超预算则 **exit != 0**，下载过程中也会中止以免超限。

```bash
# A) 无 ALLOW：--execute 必须失败（exit != 0）
python3 scripts/download_robosense_subset.py --execute

# B) dry-run（默认）：列 shard、估体积、写 draft manifest
ALLOW_RS_DOWNLOAD=1 python3 scripts/download_robosense_subset.py --dry-run
python3 scripts/download_robosense_subset.py --max-gb 45
python3 scripts/download_robosense_subset.py --max-gb 10   # 应 REJECT（拟选 ~28GB）

# C) 连通性探针：只下 inventory 中最小文件（当前为 splits/robosense_local_val.pkl ~84MB）
#    推荐用 mv2d（已装 huggingface_hub）；否则回退 curl/wget
#    Legion 上 huggingface.co 常不可达：请设 HF_ENDPOINT=https://hf-mirror.com
ALLOW_RS_DOWNLOAD=1 HF_ENDPOINT=https://hf-mirror.com \
  /data/software/conda/anaconda3/envs/mv2d/bin/python \
  scripts/download_robosense_subset.py --execute --probe-only
# 可选：探针后删文件腾空间
#   ... --execute --probe-only --cleanup-probe

# D) 全量子集（~28GB）——仅在探针通过且明确需要时：
# ALLOW_RS_DOWNLOAD=1 python3 scripts/download_robosense_subset.py --execute
```


**网络注意（Legion）**：直连 `huggingface.co` 可能 `Network is unreachable` / timeout；探针已验证 `HF_ENDPOINT=https://hf-mirror.com` 可用。全量子集下载前请先 `--probe-only` 通过。

**解压后请删除 raw 归档**：extract 到 `robosense_subset` 后，删掉 `robosense_raw/` 下对应 tar/pkl 归档，以免打满 `budget_gb.robosense_raw_max`。不要删无关文件（例如 `kitti0000.tar.gz`）。

相关配置：[`configs/paths.yaml`](configs/paths.yaml)、[`configs/rs_subset_v0.yaml`](configs/rs_subset_v0.yaml)、[`configs/robosense_hf_inventory.json`](configs/robosense_hf_inventory.json)。  
Draft manifest：`/data/data/automomous/autolabel4d/manifests/rs_subset_v0*`。

详细阶段、目录配额、验收闸门见实现计划正文。

## Stage B4 — Subset static audit (skeleton)

After download+extract fills `paths.yaml` → `robosense_subset`, run:

```bash
/data/software/conda/anaconda3/envs/mv2d/bin/python scripts/audit_robosense_subset.py
# or: python3 scripts/audit_robosense_subset.py
```

Writes `/data/data/automomous/autolabel4d/manifests/dataset_audit.json`.

- If subset is still empty: **exit 0**, `status=WAITING_FOR_SUBSET` (safe to poll while download runs).
- If subset has data: checks `hs64_path` (forbid `livox` / `velodyne_path` as top LiDAR per [`configs/sensors_rs_4f_1l.yaml`](configs/sensors_rs_4f_1l.yaml)), four fisheye `CAM_*_OV` heuristics, and GT isolation note (**generator must not read `annos.id`**).

## Stage C — Geometry / LiDAR→camera projection (thin)

CPU-side skeleton under `src/geometry/`:

| Module | Role |
|---|---|
| `sensors.py` | Load `configs/sensors_rs_4f_1l.yaml` (RS_4F_1L) + `NuScenesCamLidar` preset |
| `project.py` | LiDAR→image projection (K, extrinsics); returns `uv, mask, depth` |
| `time_pose.py` | Timestamp / pose sanity hooks (nuScenes OK; RS TODOs) |
| `fisheye.py` | Fisheye→virtual pinhole stub (`NotImplemented` or `passthrough=True`) |

Prior data: **nuScenes v1.0-mini** until RoboSense subset is ready. Viz outputs are small PNGs only.

```bash
export PYTHONPATH=src
/data/software/conda/anaconda3/envs/mv2d/bin/python scripts/viz_lidar_cam_project_mini.py \
  --scene scene-0103 --n 4 --cam CAM_FRONT --extra-cam CAM_FRONT_LEFT

# outputs: outputs/geometry_proj_mini/*_proj.png
# log:     /data/data/automomous/autolabel4d/logs/geometry_proj_mini_*.log
```


## Stage D — Teacher serial interfaces (smoke stubs)

**Interface smoke only.** Real OpenPCDet / Grounding-DINO / SAM / OMNI-DC come **after** RoboSense subset audit. No heavy teacher weights, no long GPU occupancy in this stage.

| Order | Module | Stub | Later (real) |
|---|---|---|---|
| D1 | LiDAR 3D det | `src/teachers/lidar_stub.py` (fake boxes, `is_fake`) | OpenPCDet single-sweep |
| D2 | 2D det | `src/teachers/dino_stub.py` (`is_stub` / NotImplemented) | Grounding DINO Tiny |
| D3 | Seg | `src/teachers/sam_stub.py` | SAM 2.1 Small |
| D4 | Depth | `src/teachers/depth_stub.py` | OMNI-DC v1 / E3-lite |

Protocol: `TeacherStage` (`load` / `infer_frame` / `unload`) + `SerialTeacherGuard` — **one resident teacher** at a time (`src/teachers/base.py`).

```bash
export PYTHONPATH=src
/data/software/conda/anaconda3/envs/mv2d/bin/python scripts/run_teacher_serial_smoke.py \
  --scene scene-0103 --max-samples 2

# cache: /data/data/automomous/autolabel4d/pseudo_labels/teacher_cache_smoke/
# orchestrator Stage D ids: StageRunner().list_stage_d()  → P2_lidar_teacher, P3_dino, P3b_sam, P4_depth
```

Pass criteria: console `PASS`; cache JSON under `teacher_cache_smoke/`; `SerialTeacherGuard` idle after unload; nvidia-smi queried without loading torch models.


## Stage E — Fusion / track / export (thin skeleton)

LiDAR-primary association + quality A/B/C + `track_id` + nuScenes-style results dict. Interface smoke only (uses `teacher_cache_smoke` / `lidar_stub`).

| Module | Role |
|---|---|
| `src/fusion/associate.py` | LiDAR-primary match API; BEV IoU / center-distance placeholders |
| `src/fusion/quality.py` | Assign `quality`∈{A,B,C}, `valid_fields`, `ignore` (schema-aligned) |
| `src/fusion/track.py` | Thin `track_id` assignment (identity or greedy cross-frame) |
| `src/export/nuscenes_tables.py` | Pseudo labels → `{meta, results}` + `track_id` / `tracking_id` |

```bash
export PYTHONPATH=src
/data/software/conda/anaconda3/envs/mv2d/bin/python scripts/run_fusion_export_smoke.py \
  --scene scene-0103 --max-samples 2

# pseudo: /data/data/automomous/autolabel4d/pseudo_labels/fusion_export_smoke/
# results JSON: .../fusion_export_smoke/nuscenes_results_smoke.json
# viz (optional): outputs/fusion_export_smoke/*.png
# orchestrator: StageRunner().list_stage_e() → P5_fusion_track_export
```

Pass criteria: console `PASS`; fused pseudo labels validate against schema; results dict boxes carry `track_id`.

## Stage F — Student PointPillar short-train (smoke / skeleton)

**Interface smoke only** while RoboSense download continues. No long GPU training, no OpenPCDet weight download.

| Module | Role |
|---|---|
| `src/student/pointpillar_smoke.py` | Docs for PointPillar-MultiHead + SimpleTrack plug-in; `FakeStudent` 1-step OR validate fusion pseudo dataloader |
| `src/student/simpletrack_hook.py` | Stub: record tracker config path (ref [tusen-ai/SimpleTrack](https://github.com/tusen-ai/SimpleTrack)); no full track |
| `configs/student_pointpillar_v3.yaml` | microbatch=1, grad-accum note, freeze backbone/BN, 3 coarse classes; records local OpenPCDet paths |
| `scripts/run_student_pointpillar_smoke.py` | Default `--dry-run` → `outputs/student_smoke/train_plan.json`; optional `--one-iter` FakeStudent |

```bash
export PYTHONPATH=src
# dry-run (default): check fusion_export_smoke, write train_plan.json
/data/software/conda/anaconda3/envs/mv2d/bin/python scripts/run_student_pointpillar_smoke.py

# optional 1-iter fake Linear step on CPU (or --cuda if free; never >1 min GPU)
# /data/software/conda/anaconda3/envs/mv2d/bin/python scripts/run_student_pointpillar_smoke.py --one-iter
# /data/software/conda/anaconda3/envs/mv2d/bin/python scripts/run_student_pointpillar_smoke.py --one-iter --cuda

# orchestrator: StageRunner().list_stage_f() → P6_student_pointpillars
```

Outputs: `outputs/student_smoke/train_plan.json`, `smoke_status.json`, placeholder SimpleTrack YAML.  
OpenPCDet trees (record only, do not train yet) are listed under `openpcdet.paths_found` in the YAML.

## RoboSense data note (PARTIAL)

Current local extract under `robosense/subset` is **partial `part_01` only** (`is_rs_partial=true`).  
Do **not** join multi-part archives for teacher smoke. LiDAR `hs64` (SWEEPER-001, ~2023-09-07..09) and images (~2023-09-28..10-08) are **not time-aligned** in this extract.

Representative-frame LiDAR CenterPoint smoke:

```bash
export LD_LIBRARY_PATH=/data/software/conda/anaconda3/envs/hednet-gpu/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTHONPATH=src
/data/software/conda/anaconda3/envs/hednet-gpu/bin/python scripts/run_rs_repr_lidar_centerpoint.py --n 3
# JSON: /data/data/automomous/autolabel4d/pseudo_labels/rs_repr_lidar/
# BEV:  outputs/rs_repr_lidar/
# Runbook: docs/RS_REPR_LIDAR_CENTERPOINT_RUNBOOK.md
```

