# <method>_<model>_<dataset>_<tag>

> 复制本模板新建实验：`cp -r templates/experiment-template experiments/<新实验名>`
> 实验名遵循 `docs/naming-convention.md`。

## 目标

一句话说明这个实验要验证 / 达成什么。

## 配置

- 基础模型：`qwen3.5-?b`
- 数据集：`<dataset>`
- 方法：`sft | grpo | agent-grpo | ...`
- 硬件 profile：`gb10-spark | h200`
- 关键超参：lr / batch / kl / group size ...

## SwanLab

- project：`<实验名>`
- run：`<超参组合>`
- 链接：<贴上 SwanLab 链接>

## 运行

```bash
CLUSTER_PROFILE=gb10-spark bash run.sh
```

## 结果与结论

- 关键指标：
- 结论 / 下一步：
