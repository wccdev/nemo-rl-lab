# 单机 1× H100 (80GB) 环境。被实验 run.sh（及可选的 start_ray_head.sh）统一 source。
# 单机单卡没有跨节点通信，故不配 RoCE/IB/多网卡那一套——
# 那是 gb10-spark/env.sh（2 节点）才需要的；在单机上指定网卡名只会误绑不存在的接口。

# --- PyTorch 显存分配：H100 80GB 开 expandable_segments 抗碎片 ---
# RL 的 rollout 长度可变、生成与训练交替占显存，碎片化是 OOM 主因；
# expandable_segments 比固定 max_split_size 更省，长序列/多轮更稳。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Megatron 推荐：固定单条 CUDA 流连接，保证 kernel 顺序与数值可复现 ---
# 单卡 TP=1 时无 TP 通信，留着无害；NeMo/Megatron 容器默认即此值。
export CUDA_DEVICE_MAX_CONNECTIONS=1

# --- Ray 本地实例内存监控（单机 host RAM 足够时放宽，避免训练进程被 OOM killer 误杀）---
export RAY_memory_usage_threshold=0.95
export RAY_memory_monitor_refresh_ms=2000

# --- NCCL：单卡通信平凡，只留日志级别 ---
# 不要设 NCCL_SOCKET_IFNAME / NCCL_IB_* —— 单机会因网卡名不匹配而初始化失败。
export NCCL_DEBUG=WARN
