# 4D OD 自动标注 · v3 本地实现计划（RTX 5090 24GB）

**版本：** 2026-09-06  
**依据：** `4D_OD自动标注方案_v2与v3合订本` 中的 **v3（单卡 24GB，暂不使用 A100）**  
**代码仓：** `/data/code/cv/AutoLabel/AutoLabel4D0DWith2D`  
**数据根：** `/data/data/automomous`  
**原则：** 先写可运行骨架与下载脚本；RoboSense **只下 30–50GB 子集**；边下边写代码；数据就绪后再做端到端实验。

> 本文件是**实现计划**，不是已验收的实验结果。显存、吞吐、标签精度均需在本机实测。

---

## 0. 目标与非目标

### 0.1 本阶段目标

1. 在 **单卡 RTX 5090（按 24GB 预算）** 上跑通离线自动标注最小闭环。  
2. 输出 **nuScenes 风格关系表 + 显式 `track_id` + 质量字段**。  
3. 用伪标签微调 **PointPillar-MultiHead + SimpleTrack**，验证「标签能否训动学生」。  
4. （可选后续）再尝试 **Sparse4Dv3-R50** 局部微调，验证视觉补盲可学性。

### 0.2 明确不做（本阶段）

- 不下完整 RoboSense（HF 全仓约 705GB）。  
- 不上 A100；不在本机并行常驻多教师。  
- 不微调 SAM / Grounding DINO / OMNI-DC（首轮全部冻结）。  
- 不把 PointPillars+跟踪宣称成端到端 4D；不把无 LiDAR 证据的目标强塞给纯 LiDAR 学生。

---

## 1. 磁盘与路径约定（必须先守住）

### 1.1 当前机器快照（2026-09-06）

| 路径 | 现状 |
|---|---|
| `/data` 可用 | 约 **259GB** |
| `/data/data/automomous/nuscenes` | 已占用约 **134GB**（含 mini 软链） |
| `/data/data/kitti0000.tar.gz` | 约 **78GB**（与本任务无关，占空间） |
| 代码仓 | `/data/code/cv/AutoLabel/AutoLabel4D0DWith2D`（空仓起步） |

**结论：** 250G 级余量 **只够 RoboSense 精选子集 + 缓存**，不够全量。建议 RoboSense 原始+解压峰值控制在 **≤45GB**，工作缓存/伪标签 **≤40GB**，永久留 **≥20GB** 余量。

### 1.2 目录布局

```text
/data/code/cv/AutoLabel/AutoLabel4D0DWith2D/     # 本仓库
  docs/IMPLEMENTATION_PLAN_v3_5090.md            # 本计划
  configs/                                       # 传感器预设、阶段 YAML
  scripts/                                       # 下载、验收、跑阶段
  src/                                           # 业务代码
  envs/                                          # 环境锁文件说明
  third_party/                                   # 可选 submodule / 稀疏依赖说明
  outputs/                                       # 本地调试小输出（大产物不放这里）

/data/data/automomous/
  nuscenes/                                      # 已有；格式闭环用 v1.0-mini
  robosense/
    raw/                                         # 下载分片/压缩包（可删）
    subset/                                      # 解压后的选中序列（只读主数据）
    cache/                                       # 虚拟视图、中间索引（可清）
  autolabel4d/
    manifests/                                   # 不可变样本清单
    pseudo_labels/                               # 伪标签与扩展 JSON
    checkpoints/                                 # 教师权重缓存 + 学生 ckpt
    logs/                                        # 运行与 nvidia-smi 日志
```

### 1.3 空间配额（建议硬上限）

| 用途 | 路径 | 预算 |
|---|---|---|
| RoboSense 下载暂存 | `robosense/raw` | ≤ 50GB（解压后删包） |
| 选中序列媒体 | `robosense/subset` | ≤ 35GB |
| 几何/掩码/深度缓存 | `robosense/cache` | ≤ 25GB |
| 伪标签与导出 | `autolabel4d/pseudo_labels` | ≤ 10GB |
| 模型权重 | `autolabel4d/checkpoints` | ≤ 15GB |
| **合计（本任务新增）** | | **约 30–50GB 媒体 + ≤50GB 工作区** |

若空间继续吃紧：优先删 `robosense/raw` 与过期 `cache`；评估是否挪走或删除无关的 `kitti0000.tar.gz`。

---

## 2. 总体技术路线（v3 落地版）

```text
传感器只读输入
  四鱼眼(+可选前视) + 顶部 hs64 + 标定/位姿
        │
        ▼
[P0] 数据清单与审计 ──► manifest.json / dataset_audit.json
        │
        ▼
[P1] 几何层：鱼眼→虚拟针孔、投影、时间对齐（CPU 为主）
        │
        ▼
[P2] 三维教师（串行，batch=1）→ 原始 3D 框缓存
        │
        ▼
[P3] 图像：Grounding DINO Tiny → 落盘 → 退出
           SAM 2.1 Small 按框分割 → 掩码缓存
        │
        ▼
[P4] 深度：OMNI-DC v1.0（batch=1）或 E3-lite（DA V2 Small+LiDAR 对齐）
        │
        ▼
[P5] 融合 / 跟踪 / 质量分级 / ignore → nuScenes 表 + track_id
        │
        ▼
[P6] 学生：PointPillar-MultiHead 微调 + 固定 SimpleTrack 评价
        │
        └─(可选) Sparse4Dv3-R50 局部微调
```

**同一时刻只驻留一个大模型阶段**；阶段间只交换磁盘缓存。

---

## 3. 分阶段实现计划（可并行）

### 阶段 A（立即开始，不依赖 RoboSense 下载）— 预计 2–4 天

| 编号 | 任务 | 产出 |
|---|---|---|
| A1 | 仓库骨架、README、配置模板、路径常量 | 可导航仓 |
| A2 | 环境矩阵：三维 / 图像 / 深度 **三套隔离** | `envs/README.md` |
| A3 | GPU 与显存探针 | `scripts/probe_gpu.sh` |
| A4 | 基于已有 **nuScenes mini** 的读入与空跑导出 | `scripts/smoke_nuscenes_mini.py` |
| A5 | 伪标签 schema 与 JSON 样例 | `configs/label_schema.yaml` |
| A6 | 阶段编排器接口（只调度） | `src/pipeline/` 雏形 |

**验收：** mini 上能读 sample → 写假伪标签 → 再读回，字段齐全。

---

### 阶段 B（与 A 并行：数据下载）— 预计 1–3 天（视网速）

| 编号 | 任务 | 产出 |
|---|---|---|
| B1 | HF 分片清单 + 只下指定 shard 脚本 | `scripts/download_robosense_subset.py` |
| B2 | 选定完整序列列表 | `autolabel4d/manifests/rs_subset_v0.txt` |
| B3 | 下载→校验→解压到 subset→**删除 raw 包** | 占用回落预算内 |
| B4 | 静态审计：强制 `hs64_path`、四鱼眼、GT 隔离 | `dataset_audit.json` |

**子集规模建议（第一刀）：**

- 训练试跑：完整序列 **6–10** 条  
- 开发：完整序列 **2–3** 条  
- 独立检查：完整序列 **2–3** 条（官方 val 侧，固定）  
- 媒体体积目标：**30–50GB**；超 50GB 停止扩下

**验收：** audit 通过；生成器输入清单不含官方框/ID。

---

### 阶段 C（有一条序列即可开始）— 几何与 I/O

| 编号 | 任务 |
|---|---|
| C1 | RoboSense global 索引解析；GT 隔离 |
| C2 | 传感器预设 `RS_4F_1L` / `RS_4F_1P_1L` |
| C3 | 鱼眼→虚拟针孔与映射缓存 |
| C4 | LiDAR→图像投影、时间对齐、位姿抽检 |

**验收：** 随机 N 帧投影可视化合理。

---

### 阶段 D — 教师串行推理（5090）

| 顺序 | 模块 | v3 默认 | 失败回退 |
|---|---|---|---|
| D1 | 点云检测教师 | 可跑的 OpenPCDet 单 sweep | 掩码选点+几何拟合（弱基线） |
| D2 | 2D 检测 | Grounding DINO Tiny | 暂缓图像支路 |
| D3 | 分割 | SAM 2.1 Small | Tiny；暂不用 SAM3 |
| D4 | 深度 | OMNI-DC v1.0；否则 E3-lite | 可关闭深度，保留 E2 |

**验收：** 代表帧有限输出；整卡占用目标 ≤ **20–21GB**。

---

### 阶段 E — 融合、跟踪、导出

| 编号 | 任务 |
|---|---|
| E1 | LiDAR 主证据；图像补漏；深度不得数量压过真实点 |
| E2 | 三维跟踪 + 真实 `dt`（1Hz 按 ~1s） |
| E3 | 质量 A/B/C、`valid_fields`、`ignore` |
| E4 | 导出 nuScenes 表 + `track_id` |
| E5 | 消融 E1 / E2 / E3 |

---

### 阶段 F — 学生微调（5090）

| 编号 | 任务 |
|---|---|
| F1 | 退出全部教师；只驻留学生 |
| F2 | PointPillar-MultiHead：3 粗类；先冻骨干训新头 |
| F3 | microbatch=1 + 梯度累积；冻结 BN |
| F4 | 固定 SimpleTrack 比较不同伪标签 |
| F5 | （可选）Sparse4D-R50 局部微调 |

---

## 4. 代码模块划分

```text
src/
  data/       # 读取、GT 隔离、manifest
  geometry/   # 鱼眼、虚拟针孔、投影
  teachers/   # dino / sam / lidar / depth 独立入口
  fusion/     # 关联、质量、跟踪接口
  export/     # nuScenes 导出
  student/    # OpenPCDet / Sparse4D 补丁
  eval/       # 标签评测与学生评测
  pipeline/   # 编排、缓存、显存策略
```

重模型用独立 conda/venv；本仓只保留薄封装。

---

## 5. 边下边写时间线

```text
[A 骨架/mini冒烟] ████████
[B 下载 30–50G]     ░░████████
[C 几何]                ████（有1条序列即可）
[D 教师]                   ████
[E 融合导出]                  ███
[F 学生]                        ███
```

**闸门：** 未过 GPU 冒烟不下大批量；未过 audit 不跑教师；教师未验收不批量产标；标签未评测不扩大学生训练。

---

## 6. 风险与应对

| 风险 | 应对 |
|---|---|
| Blackwell + 旧 spconv | 先 PointPillars；不通则弱几何基线 |
| OMNI-DC 旧环境 | 独立 env；失败用 E3-lite |
| 1Hz 伪装高频 | 真实 dt；禁伪多 sweep |
| 误用 livox | audit 强制 `hs64_path` |
| 磁盘写满 | 配额；解压删 raw；禁止全仓下载 |

---

## 7. MVP 完成定义

1. RoboSense 子集媒体约 30–50GB 量级。  
2. 检查序列产出带 `track_id`/质量字段的伪标签，生成器未读 GT 框。  
3. 至少完成 E1 与 E2 两档标签对比。  
4. PointPillars 短训可加载推理。  
5. 单卡串行，关键步骤有显存与配置日志。

---

## 8. 参考

- 原文：`/home/yr/Downloads/4D_OD自动标注方案_v2与v3合订本_gpt6-Astra-Ultra-20260906.md`（以 v3 为准）  
- RoboSense：https://github.com/suhaisheng/RoboSense  
- HF：https://huggingface.co/datasets/suhaisheng0527/RoboSense  
