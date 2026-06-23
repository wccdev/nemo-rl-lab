# scripts/ — 通用脚本

- `new_experiment.sh` — 从模板快速新建实验
  ```bash
  bash scripts/new_experiment.sh experiments grpo_qwen3.5-4b_gsm8k_v1
  ```
- `submit_job.sh` — 从开发机（Mac）把作业提交到远程 Ray 集群（执行在集群容器内）
  ```bash
  # 一次性准备（Ray CLI 由 uv 管理，版本对齐集群）
  uv sync --extra submit
  cp cluster/submit.env.example cluster/submit.env   # 填 Ray 地址 / 容器内 NeMo-RL 路径 / 密钥
  # 提交
  bash scripts/submit_job.sh experiments/agent-grpo_qwen3.5-9b_multitool_v1
  ```
- `prefetch_hf_model.sh` — **在集群容器内**预下载 HF 模型到 `HF_HOME`（避免训练时连不上 hf-mirror.com）
  ```bash
  # 在 Spark 容器里（与 submit.env 同目录约定）
  source cluster/submit.env
  bash scripts/prefetch_hf_model.sh Qwen/Qwen3.5-4B
  ```
- `sync_base_configs.sh` — 升级 NeMo-RL 版本时同步官方基底配置到 `configs/base/`
- `post_train.sh` — **训练后闭环**（在集群容器内执行，由 `lab export` / `lab eval` 经 ray job submit 调起）：
  把 checkpoint 转 HF（按后端自适应 `convert_dcp_to_hf.py` / `convert_megatron_to_hf.py`，可推 HF Hub），
  或对 checkpoint 跑 `examples/run_eval.py` 评测。带 `LAB_DRY_RUN=1` 只打印命令不执行。
  ```bash
  # 通常用 CLI（从 Mac 提交，执行在集群）：
  uv run lab export grpo_qwen3.5-9b_gsm8k_v1 [--step N] [--push-repo user/name]
  uv run lab eval   grpo_qwen3.5-9b_gsm8k_v1 [--step N] [-- generation.temperature=0.6]
  # 也可在 head 容器内直跑：
  NEMO_RL_DIR=/opt/nemo-rl OUTPUT_ROOT=/data/runs bash scripts/post_train.sh export experiments/grpo_qwen3.5-9b_gsm8k_v1
  ```
