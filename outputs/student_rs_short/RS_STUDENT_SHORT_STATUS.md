# RS Student PointPillars SHORT — Stage F status

**Status:** `TRAINED_SHORT`  
**Time:** 2026-09-06 14:42–14:43 CST (UTC+8)  
**Host:** Legion (`252f7d1f-95fc-44c6-94d1-8f76153ee6af`) — RTX 5090 Laptop 24GB  
**Repo:** `/data/code/cv/AutoLabel/AutoLabel4D0DWith2D`  
**Script:** `scripts/run_student_rs_short.py`  
**Env:** `hednet-gpu` (OpenPCDet / pcdet 0.6.0 editable from HEDNet-gpt)

## Result summary

| field | value |
|---|---|
| status | **TRAINED_SHORT** |
| backend | **real OpenPCDet** `PointPillar` + `AnchorHeadMulti` (`cbgs_pp_multihead`) |
| optimizer steps | **50** (≤50 budget) |
| microbatch | **1** |
| loss first → last | **1.539 → 1.271** |
| VRAM peak (torch) | **~1318 MiB** |
| wall time | **~3.8 s** |
| ckpt | `/data/data/automomous/autolabel4d/checkpoints/student/pointpillars_rs_short.pth` (~70 MB) |

Not FakeStudent. Real OpenPCDet forward/backward on RoboSense hs64 + thr≥0.3 pseudo labels. BN frozen in eval mode for short-train stability; no long full train / no push / no KITTI / no multipart. GDINO download left untouched.

## Steps taken

1. Confirmed Legion GPU free (~15 MiB used) and `hednet-gpu` has working `pcdet` + CUDA.
2. Paired HS64 bins with Stage-E export labels (`score≥0.3`): **5 frames** with boxes (2 empty-at-0.3 frames skipped).
3. Built OpenPCDet PointPillars from  
   `/data/code/cv/AutoLabel/BEV-OD/HEDNet/tools/cfgs/nuscenes_models/cbgs_pp_multihead.yaml`  
   (disabled gt_sampling / world augs — no nuScenes DB).
4. Ran **50** Adam steps, lr=1e-3, microbatch=1, cycling the 5 frames.
5. Saved ckpt + `loss.json` + `loss_curve.png` + BEV GT viz.
6. Fallback path (`PillarLikeStudent`) is implemented in the script but **was not needed**.

## Data

- **HS64:** `/data/data/automomous/robosense/subset/lidar_occ_trainval/processed_data_20230906/SWEEPER-001/hs64/` (float64 XYZ)
- **Pseudo labels (primary):** `/data/data/automomous/autolabel4d/pseudo_labels/rs_lidar_only_export/`
- **Alt thr03:** `/data/data/automomous/autolabel4d/pseudo_labels/rs_repr_lidar_thr03/`
- **Score keep:** ≥0.3
- **Paired tokens (n_obj):**
  - `2023-09-07-11-22-49-051` (34)
  - `2023-09-08-10-13-45-949` (3)
  - `2023-09-08-16-09-46-950` (41)
  - `2023-09-08-17-25-56-850` (28)
  - `2023-09-09-17-19-59-151` (37)

## Loss curve

Logged every step in:

`/data/code/cv/AutoLabel/AutoLabel4D0DWith2D/outputs/student_rs_short/loss.json`

Plot:

`/data/code/cv/AutoLabel/AutoLabel4D0DWith2D/outputs/student_rs_short/loss_curve.png`

Selected points:

| step | loss |
|---:|---:|
| 1 | 1.539 |
| 10 | 1.555 |
| 20 | 1.454 |
| 30 | 1.277 |
| 40 | 1.291 |
| 50 | 1.271 |

Last tb_dict: `rpn_loss_cls≈0.978`, `rpn_loss_loc≈0.293`.

## Artifacts

| artifact | path |
|---|---|
| train script | `scripts/run_student_rs_short.py` |
| ckpt | `/data/data/automomous/autolabel4d/checkpoints/student/pointpillars_rs_short.pth` |
| loss JSON | `outputs/student_rs_short/loss.json` |
| loss plot | `outputs/student_rs_short/loss_curve.png` |
| BEV viz | `outputs/student_rs_short/bev_2023-09-07-11-22-49-051.png` |
| run log | `outputs/student_rs_short/train_run.log` |
| status JSON | `outputs/student_rs_short/status.json` |

## Command

```bash
source /data/software/conda/anaconda3/etc/profile.d/conda.sh
conda activate hednet-gpu
cd /data/code/cv/AutoLabel/AutoLabel4D0DWith2D
python scripts/run_student_rs_short.py --max-steps 50 --backend auto
```

Useful flags: `--dry-run`, `--backend openpcdet|pillarlike`, `--score-thresh 0.3`, `--ckpt-dir`, `--out-dir`.

## VRAM

- Before/after train: GPU essentially idle (~15 MiB) — model unloaded after save.
- During train peak (torch allocated): **~1.3 GB** (plenty of headroom on 24GB).
- `nvidia-smi` during step logs reported ~308–310 MiB process footprint for the short loop.

## How to plug longer OpenPCDet train later

1. Keep `hednet-gpu` + HEDNet-gpt editable `pcdet`.
2. Reuse cfg `cbgs_pp_multihead.yaml`; optionally enable controlled augs once a GT database exists.
3. Convert denser SWEEPER sequences to OpenPCDet infos (pkl) instead of on-the-fly OneShot DS.
4. Unfreeze BN / backbone gradually; raise grad accum; load this short ckpt as init:
   `pointpillars_rs_short.pth` → `model_state`.
5. Do **not** use KITTI PointPillars range (`x≥0` only) for RS 360° — stay on nuScenes-range PP-MultiHead.

## Notes / honesty

- Previous agent left empty `outputs/student_rs_short/`; this run fills required student outputs.
- Older sibling artifacts may also exist under the same dirs (`pp_rs_short_final.pth`, `fake_student_result.json` from parallel attempts). **Canonical short-train deliverable for this task is `pointpillars_rs_short.pth` + `loss_curve.png` + this status.**
- Size convention: pseudo `size=[w,l,h]` → OpenPCDet `dx,dy,dz = l,w,h`.
