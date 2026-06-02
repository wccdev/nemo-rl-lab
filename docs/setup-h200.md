# H200 训练环境

后续会在 NVIDIA H200（x86_64 + Hopper 架构）上做 SFT 与 GRPO 训练。与 GB10 的主要差异：

| 维度 | GB10 Spark | H200 |
| --- | --- | --- |
| 架构 | aarch64 + Blackwell | x86_64 + Hopper |
| 内存 | 统一内存 | 独立 HBM3e（单卡 141GB） |
| 容器 / wheel | aarch64 版 | x86_64 版 |
| 组网 | 2 节点 Ray | 视实际节点数 |

## 配置位置

H200 的资源与并行度写在 `cluster/h200/profile.yaml`，启动脚本在 `cluster/h200/`。实验侧无需改动，只切换 profile：

```bash
CLUSTER_PROFILE=h200 bash run.sh
```

## 注意

- H200 单卡 141GB HBM3e，显存充足，可适当增大 batch size / 上下文长度 / rollout 并发。
- 同一实验在不同硬件上跑，建议用 `tag` 区分 run（如 `..._h200` / `..._gb10`），SwanLab 上便于对比吞吐与收敛。
- 依赖（CUDA / 容器镜像）与 GB10 不同，按架构分别维护，见 `env/README.md`。
