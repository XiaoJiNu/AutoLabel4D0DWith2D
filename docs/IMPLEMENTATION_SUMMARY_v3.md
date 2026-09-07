# AutoLabel4D v3（RTX 5090）实现总结报告

**日期：** 2026-09-06～09-07  
**代码仓：** `/data/code/cv/AutoLabel/AutoLabel4D0DWith2D`  
**数据根：** `/data/data/automomous`  
**算力：** 联想拯救者 RTX 5090 Laptop（约 24GB），经 Tailscale SSH 由 Grok Bot「算法复现」执行  
**依据：** `docs/IMPLEMENTATION_PLAN_v3_5090.md`（单卡 24GB，暂不用 A100）

---

## 1. 目标与范围

### 1.1 本轮目标

在单卡 5090 上跑通 **离线 4D OD 自动标注最小闭环**：

1. 数据清单 / 审计 / 路径配额  
2. 几何（LiDAR↔相机投影，针孔优先）  
3. 教师串行：LiDAR（CenterPoint）+ 视觉（GDINO→SAM）  
4. 融合 / 质量 / `track_id` → nuScenes 风格伪标  
5. 学生 PointPillars 短训，验证「伪标能否训动学生」

### 1.2 明确未做 / 刻意限制

| 项 | 说明 |
|----|------|
| 全量 RoboSense（~705GB） | 禁止；媒体目标 30–50GB |
| 全 multipart 拼接（约 119–239GB+） | 超配额，未做 |
| A100 / 多教师常驻 | 未用；教师串行、阶段间只交换磁盘缓存 |
| 微调 SAM / GDINO / OMNI-DC | 首轮全部冻结推理 |
| 鱼眼 unwrap | `src/geometry/fisheye.py` 仍为 stub |
| cam4 / `CAM_FRONT_OV` | partial 子集中缺失 |
| git push | 全程未推送 |

---

## 2. 仓库与模块结构

```text
AutoLabel4D0DWith2D/
  docs/IMPLEMENTATION_PLAN_v3_5090.md
  README.md
  configs/
    paths.yaml                 # 路径 + budget_gb 硬上限
    pipeline_v3_5090.yaml
    sensors_rs_4f_1l.yaml
    label_schema.yaml
    student_pointpillar_v3.yaml
    teacher_weights_v3.yaml
    rs_camera_map（见 manifests）
  scripts/
    probe_gpu.sh
    smoke_nuscenes_mini.py
    download_robosense_subset.py   # ALLOW_RS_DOWNLOAD + hf-mirror + 校验
    audit_robosense_subset.py
    analyze_rs_align.py / viz_rs_align_proj.py
    run_teacher_serial_smoke.py
    run_fusion_export_smoke.py
    run_student_pointpillar_smoke.py / run_student_rs_short.py
    （以及 RS 代表帧 / 融合 / 视觉教师等相关入口）
  src/
    data/       # 路径、nuScenes / RS 读取
    geometry/   # sensors, project, time_pose, fisheye(stub)
    teachers/   # lidar / dino / sam / depth 独立入口 + 串行调度
    fusion/     # associate, quality, track
    export/     # 伪标 IO、nuScenes 表
    student/    # PointPillars 短训封装、SimpleTrack hook
    eval/ pipeline/
  outputs/      # 冒烟 / 对齐投影 / 融合可视化 / 短训曲线
```

工作目录约定：实验大产物在 `/data/data/automomous/autolabel4d/`；仓内 `outputs/` 放调试可视化。

---

## 3. 分阶段实现与验收

### 阶段 A — 骨架与 nuScenes mini 冒烟

- 仓库骨架、路径常量、`probe_gpu.sh`、label schema、编排雏形  
- **验收：** mini 读 sample → 写假伪标 → 读回字段齐全；BEV / CAM 可视化通过  

### 阶段 B — RoboSense 子集下载

- 脚本支持 dry-run、配额拒绝、`ALLOW_RS_DOWNLOAD` 硬开关  
- HF 直连超时，实际使用 `HF_ENDPOINT=https://hf-mirror.com`  
- 关键发现：`*_part_XX.tar.gz` 为 **同一 gzip 流分卷**，不能当独立包解；仅 `part_01` 有 gzip magic  
- 最终盘上 **subset ≈ 28GB partial**；raw 解压后清至约 **477MB**（主要留 splits PKL）  
- audit：`PASS_WITH_WARNINGS`（livox 存在但禁止作顶雷；GT 在 PKL 中隔离策略；早期相机映射未知）

### 阶段 C — 几何

- mini 上 LiDAR→相机投影可视化通过  
- RS：建立 `rs_camera_map_v0.yaml`（数字目录 0–7 → CAM_* / *_OV）  
- **对齐补丁 PATCH_OK：** 对齐帧 **0→12**（`processed_data_20230906`，采集日 2023-09-07/08/09）  
- 流式补齐 image part_01–09 + lidar part_01–02（网络流量约 118GB 量级，未拼全系列）  
- 投影可视化：`outputs/rs_align_proj/`

### 阶段 D — 教师

| 教师 | 状态 |
|------|------|
| LiDAR CenterPoint | SWEEPER-001 hs64 代表帧真推理 PASS（约 1.2GB 显存） |
| GDINO Tiny | 权重已下；对齐帧推理产出 67 检 |
| SAM 2.1 Small | 权重已下；对 DINO 框出掩码 |
| 深度 OMNI-DC | 未作为主路径；DA-V2 权重存在，跨日阶段曾跳过 |

权重合计约 **0.83–1.1GB**（远低于 checkpoints 15GB 配额）。

### 阶段 E — 融合 / 导出

- **LiDAR-only：** `rs_lidar_only_export/`（例：score≥0.3 时约 143 框 / 7 帧，含 `track_id`/质量）  
- **融合：** `rs_fused_export/` → 再锁定 nuScenes 10 类到 `rs_fused_export_nuscenes/`  
  - 12 帧，约 **473** 框（thr≥0.3），约 **21** 图雷匹配  
  - 类别已是官方 10 类，重映射 **0 丢 0 改名**  
- 可视化：`outputs/rs_fused_viz/`、`outputs/rs_nuscenes_pc_viz/`

### 阶段 F — 学生短训

| ckpt | 说明 |
|------|------|
| `pointpillars_rs_short.pth`（≈70MB） | LiDAR-only 伪标，50 Adam steps，loss ≈1.54→1.27 |
| `pointpillars_rs_fused_short.pth` | **融合伪标短训（建议主对照）**，50 steps，loss ≈1.46→1.18 |
| `pointpillar_multihead_rs_short.pth` / `pp_rs_short_final.pth` | 次要 / FakeStudent 对照，勿混用 |

环境：`hednet-gpu` 等本机 conda；OpenPCDet PointPillar-MultiHead；freeze backbone；microbatch=1；峰值显存约 1.3GB。

---

## 4. 关键数据结论（对齐问题）

1. **不是「官方数据天生全乱」**：PKL 内同一样本的图、hs64 本应同 pack、同天（day_match 100%）。  
2. **早期 0 对齐帧** 来自：只截断解压两条独立 multipart 流的 `part_01`，盘上图像 pack 与雷达 pack 错位。  
3. **`processed_data_YYYYMMDD` 是打包批次名**，不等于「模态各采一天」。  
4. 补丁后同 pack `processed_data_20230906` 下取得 **12** 个 exact_stem 对齐帧，针孔 `CAM_FRONT` 投影可用。

---

## 5. 磁盘与配额（收工快照量级）

| 路径 | 约占用 |
|------|--------|
| `robosense/subset` | ~28G |
| `robosense/raw` | ~0.5G |
| `autolabel4d/checkpoints` | ~1.1G |
| `/data` 剩余 | ~231G |

未动无关的 `kitti0000.tar.gz`。

---

## 6. 主产物索引

| 类型 | 路径 |
|------|------|
| 实现计划 | `docs/IMPLEMENTATION_PLAN_v3_5090.md` |
| 相机映射 | `/data/data/automomous/autolabel4d/manifests/rs_camera_map_v0.yaml` |
| 对齐清单 | `.../manifests/rs_align_frames_v0.json` |
| 融合伪标（10 类） | `.../pseudo_labels/rs_fused_export_nuscenes/` |
| 视觉缓存 | `.../pseudo_labels/rs_vision_dino_sam/` |
| 主学生 ckpt（融合） | `.../checkpoints/student/pointpillars_rs_fused_short.pth` |
| 全流程状态 | `~/grok-bot-work/autolabel4d_stageB/RS_FULL_PIPELINE_STATUS.md` |
| 10 类可视化状态 | `~/grok-bot-work/autolabel4d_stageB/NUSCENES_CLASSES_VIZ_STATUS.md` |

---

## 7. 结论与后续建议

**结论：** 在 **partial RoboSense + 12 对齐帧** 尺度上，v3 主链路已打通：  
数据审计 → 几何对齐 → LiDAR/视觉教师 → 融合伪标（nuScenes 10 类）→ PointPillars 短训可加载。  

这是 **工程可行性 MVP**，不是全量数据上的 SOTA 指标报告；框噪声、图雷匹配偏少（21/多检）、类别混淆（如树被检成 construction_vehicle）仍存在。

**建议后续（按优先级）：**

1. 提高融合阈值 / 匹配门控，做 E1/E2/E3 消融与简单指标  
2. 实现鱼眼 unwrap，补齐 cam4，扩展到 OV 四路  
3. 在配额内按 PKL **定点扩同 pack 对齐帧**（避免再盲下全分卷）  
4. 正式短训加长 + SimpleTrack 评测协议固化  
5. （可选）Sparse4Dv3-R50 局部微调  

---

## 8. 协作说明

- 调度：幕僚长；执行：算法复现；本机通道：Tailscale（`yr@100.104.167.76`）  
- 云端优先文档/调度；GPU 与大数据仅在本机经批准后使用  

*报告由幕僚长根据 2026-09-06～07 交付记录整理。*
