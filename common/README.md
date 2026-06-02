# common/ — 跨实验复用代码

避免每个实验各写一份。包含：

- `data/`      — 数据集下载 / 清洗 / 格式转换（转成 NeMo-RL 需要的 jsonl）
- `rewards/`   — GRPO 奖励函数库（数学、代码、Agent 任务等）
- `callbacks/` — SwanLab logger 等训练回调
- `utils/`     — 通用工具（seed、路径、配置合并等）

实验里通过 import 引用，例如配置中 `reward.fn: "common.rewards.math_reward:compute_reward"`。
