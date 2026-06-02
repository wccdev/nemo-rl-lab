# configs/ — 公共训练配置模板

按方法分类的 NeMo-RL 配置模板。新实验从这里拷贝再改，避免每次从零写。

- `sft/`   — 监督微调模板
- `grpo/`  — GRPO 强化学习模板
- `agent/` — 多轮 Agent / 工具调用训练模板

> 字段名以你所用的 NeMo-RL 版本为准，这里给出的是结构骨架与必填项注释。
> 硬件相关的并行度 / 节点数不要写死在这里——放 `cluster/<profile>/profile.yaml`。
