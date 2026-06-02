# common/ — 跨实验复用代码

避免每个实验各写一份。包含：

- `data/`         — 数据集下载 / 清洗 / 转换（转成 NeMo-RL 需要的格式），及自定义 data processor
- `environments/` — 自定义 Environment（NeMo-RL 里 GRPO 的奖励来源；多轮 Agent 用）
- `utils/`        — 通用工具

> 说明（NeMo-RL 0.6.0）：
> - **日志不用自己写**——SwanLab 是框架原生 logger，配置 `logger.swanlab_enabled=true` 即可。
> - **奖励来自 Environment**，不是独立 reward 函数；自定义环境见 `environments/`。
> - 自定义数据集通过 `data` 配置 + data processor 接入，见 `data/`。
