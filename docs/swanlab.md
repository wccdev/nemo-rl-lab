# SwanLab 接入

所有训练日志统一上传云端 SwanLab，便于跨实验、跨硬件对比。

## 1. 登录

```bash
pip install swanlab
swanlab login        # 粘贴 API Key；或设置环境变量
export SWANLAB_API_KEY=xxxxxxxx   # 建议放到本地 .env（已 .gitignore）
```

## 2. NeMo-RL 中启用 SwanLab

NeMo-RL 配置里把 logger 指向 SwanLab（字段以所用 NeMo-RL 版本为准），核心是三项：

```yaml
logger:
  swanlab:
    project: grpo_qwen3.5-9b_gsm8k_v2   # 对齐实验目录名
    experiment_name: lr1e6-bs64-kl0.001 # 对齐关键超参
    # workspace / tags 可选
```

若所用版本暂不支持原生 SwanLab，可用 SwanLab 对 wandb 的兼容模式，或在 `common/callbacks/swanlab_logger.py` 里接管日志回调。

## 3. 命名对齐（重要）

| SwanLab 概念 | 取值 |
| --- | --- |
| project | 实验目录名（或按模型聚合，如 `qwen3.5-9b`） |
| experiment / run | 关键超参组合，如 `lr1e6-bs64-kl0.001-gb10` |
| tags | `sft`/`grpo`/`agent`、硬件 `gb10`/`h200`、数据集名 |

## 4. 回填链接

每个实验的 `README.md` 必须贴上对应的 SwanLab 链接，做到「代码 ↔ 看板」一一对应。
