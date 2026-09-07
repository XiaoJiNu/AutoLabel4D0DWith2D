# 环境隔离（v3）

建议至少三套环境，禁止把旧 MMCV/Apex 与全部新模型塞进同一环境：

1. `al4d-lidar`：OpenPCDet / 点云教师 / 学生 PointPillars  
2. `al4d-vision`：Grounding DINO Tiny + SAM 2.1  
3. `al4d-depth`：OMNI-DC 或 Depth Anything V2（E3-lite）

公共编排脚本用系统 Python 或第四个轻量 env，只负责任务调度与读写缓存。

RTX 5090 = Blackwell：优先 PyTorch 官方 cu128 构建；三维扩展需单独验收，不假设旧 wheel 可用。

## 现阶段如何激活（冒烟）

Stage A 冒烟**暂用**已有 `mv2d`（含 nuscenes / opencv / numpy / matplotlib）：

```bash
# 若已 conda init：
conda activate mv2d

# 或直接调用解释器：
PY=/data/software/conda/anaconda3/envs/mv2d/bin/python
cd /data/code/cv/AutoLabel/AutoLabel4D0DWith2D
PYTHONPATH=src $PY scripts/smoke_nuscenes_mini.py
```

后续再建 `al4d-lidar` / `al4d-vision` / `al4d-depth`；本目录将放各 env 的锁文件说明（不把重模型依赖塞进编排 env）。
