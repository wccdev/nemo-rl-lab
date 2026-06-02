# <method>_<model>_<dataset>_<tag>

> 复制本模板新建实验：`bash scripts/new_experiment.sh experiments <新实验名>`
> 实验名遵循 `docs/naming-convention.md`。

## 目标

一句话说明这个实验要验证 / 达成什么。

## 配置（NeMo-RL 0.6.0）

- 基础模型：`Qwen/Qwen3.5-?B`
- 数据集：`<dataset>`
- 方法：`SFT | GRPO | 多轮 Agent`
- 入口 / 基底：在 `run.sh` 顶部设 `ENTRY` 与 `BASE_CONFIG`（见 `configs/README.md` 的方法对照表）
- 硬件 profile：`gb10-spark | h200`
- 关键 override 写在本目录 `overrides.conf`

## SwanLab

- project：`<实验名>`
- run：`<超参组合>`
- 链接：<贴上 SwanLab 链接>

## 运行

```bash
# 先准备好 NeMo-RL 0.6.0 源码，并（多节点时）拉起 Ray 集群
NEMO_RL_DIR=/path/to/NeMo-RL CLUSTER_PROFILE=gb10-spark bash run.sh
```

产物（checkpoint / 日志）会落到本目录 `outputs/`（已 .gitignore）。

## 结果与结论

- 关键指标：
- 结论 / 下一步：
