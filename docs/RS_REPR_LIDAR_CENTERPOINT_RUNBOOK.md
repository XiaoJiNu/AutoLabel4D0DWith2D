# RoboSense representative-frame LiDAR CenterPoint smoke (PARTIAL subset)

## Status flags
- `is_rs_partial=true` / `partial_subset=true`
- RS extract is **part_01 only** (lidar_occ + image shards); do **not** join multi-part archives in this task.
- LiDAR (`hs64`, ~2023-09-07..09) and images (~2023-09-28..10-08) are **not time-aligned** in this partial extract.

## Paths
| Item | Path |
|---|---|
| Repo | `/data/code/cv/AutoLabel/AutoLabel4D0DWith2D` |
| hs64 | `/data/data/automomous/robosense/subset/lidar_occ_trainval/processed_data_20230906/SWEEPER-001/hs64` |
| CenterPoint ckpt | `/data/models/pointfuse/centerpoint/openpcdet-nuscenes-voxel0075-issue1704-reshare/cbgs_voxel0075_centerpoint_nds_6648.pth` (34MB) |
| Config (prefer HEDNet tree w/ hednet-gpu) | `.../HEDNet-gpt/tools/cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml` |
| Vendor OpenPCDet | `/data/code/location/pointFuse/output/vendor/OpenPCDet-8cacccec11db6f59bf6934600c9a175dae254806` |
| Pseudo JSON | `/data/data/automomous/autolabel4d/pseudo_labels/rs_repr_lidar/` |
| BEV viz | `outputs/rs_repr_lidar/` |

## Point format
RoboSense `hs64` `.bin` = **float64 xyz** (`pointcloud_num_features=3` in split pkl).  
Teacher pads to nuScenes 5-feat `(x,y,z,intensity=0,time=0)` for CenterPoint.

## Env (RTX 5090 / sm_120)
Recommended: `hednet-gpu` (`torch 2.13+cu130`, `pcdet` from HEDNet-gpt, `spconv 2.3.8`).

```bash
export LD_LIBRARY_PATH=/data/software/conda/anaconda3/envs/hednet-gpu/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/data/code/cv/AutoLabel/AutoLabel4D0DWith2D/src
cd /data/code/cv/AutoLabel/AutoLabel4D0DWith2D
nvidia-smi
/data/software/conda/anaconda3/envs/hednet-gpu/bin/python scripts/run_rs_repr_lidar_smoke.py --n 4
nvidia-smi
```

## Blockers / fallback
If OpenPCDet CUDA ops / spconv / ckpt key mismatch / Blackwell incompatibility:
1. Script catches the exception and writes **structural stub** JSON with **real point counts** + empty boxes + BEV of points.
2. Document the exception in `rs_repr_lidar_smoke_report.json`.
3. Do **not** download GDINO/SAM here.

## Constraints honored
- No GDINO/SAM download
- No multi-part archive join
- Short GPU only; unload after
- No git push; no kitti touch
- Outputs marked partial

## Smoke result (2026-09-06 CST)

- **Real inference: YES** via `hednet-gpu` + HEDNet pcdet
- Frames: 2023-09-07-11-22-49-051 (220 boxes), 2023-09-08-10-13-45-949 (108), 2023-09-08-17-25-56-850 (244)
- Peak torch alloc ~44 MB; nvidia-smi after unload ~15 MiB / 24463 MiB
- Blocker resolved: do not prepend `openpcdet-centerpoint-8cacccec` runtime (hard-imports Argo2/av2); prefer HEDNet
- Outputs marked `is_rs_partial=true`
