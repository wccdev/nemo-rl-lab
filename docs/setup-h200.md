# H200 / Hopper 训练环境（历史文档）

> **当前仓库**：x86 单机 Hopper 训练请用 **`h100` profile**（`cluster/h100/`），远程提交见 [`docs/remote-submit.md`](remote-submit.md)。
> 本文档保留 H200 规划说明；B300 见 `cluster/b300/`。

后续可能在 NVIDIA H200（x86_64 + Hopper）上做 SFT 与 GRPO。与 GB10 的主要差异：

| 维度 | GB10 Spark | H200 / H100 |
| --- | --- | --- |
| 架构 | aarch64 + Blackwell | x86_64 + Hopper |
| 内存 | 统一内存 | 独立 HBM（H100 80GB / H200 141GB） |
| 容器 / wheel | aarch64 版 | x86_64 版 |
| 组网 | 2 节点 Ray | 视实际节点数（当前 h100 为单机单卡） |

## 配置位置

- **当前可用**：`cluster/h100/overrides.conf` + `cluster/h100/submit.env`（单机 1× H100）
- **规划中**：`cluster/b300/`（B300）

实验侧切换 profile 即可，无需改训练脚本：

```bash
CLUSTER_PROFILE=h100 bash run.sh
# 或：lab submit <exp> --profile h100
```

## 注意

- Hopper 单卡显存充足时可适当增大 batch / 上下文 / rollout 并发（按具体卡型调）。
- 同一实验在不同硬件上跑，建议用 `tag` 或 SwanLab `name` 后缀区分（如 `-h100` / `-gb10`），便于对比。
- 依赖（CUDA / 容器镜像）与 GB10 不同，按架构分别维护，见 [`cluster/README.md`](../cluster/README.md)（§依赖与环境）。
