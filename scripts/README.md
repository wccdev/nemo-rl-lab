# scripts/ — 通用脚本

- `new_experiment.sh` — 从模板快速新建实验
  ```bash
  bash scripts/new_experiment.sh experiments grpo_qwen3.5-4b_gsm8k_v1
  ```
- `submit_job.sh` — 从开发机（Mac）把作业提交到远程 Ray 集群（执行在集群容器内）
  ```bash
  # 一次性准备
  pip install "ray[default]"
  cp cluster/submit.env.example cluster/submit.env   # 填 Ray 地址 / 容器内 NeMo-RL 路径 / 密钥
  # 提交
  bash scripts/submit_job.sh experiments/agent-grpo_qwen3.5-9b_multitool_v1
  ```
- `sync_base_configs.sh` — 升级 NeMo-RL 版本时同步官方基底配置到 `configs/base/`

后续可加：批量评测、checkpoint 导出（转 HF）等。
