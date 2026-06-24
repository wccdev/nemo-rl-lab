# scripts/ — 通用脚本

作业提交统一走中心化服务：`lab login` 接入后 `lab submit <exp>`，由服务端打包上传并在集群代理执行。
本目录下的脚本只负责「在集群侧执行」或「本地工具」，不再有从本机直连 Ray 的提交脚本。

- `new_experiment.sh` — 从模板快速新建实验
  ```bash
  bash scripts/new_experiment.sh experiments grpo_qwen3.5-4b_gsm8k_v1
  ```
- `_run_experiment.sh` — **实验启动通用逻辑**（集群侧执行）：各实验 `run.sh` 收口于此，叠加
  `cluster/<profile>/overrides.conf` + `env.sh`，落盘到服务端注入的 `OUTPUT_ROOT`。
- `prefetch_hf_model.sh` — **在集群容器内**预下载 HF 模型到 `HF_HOME`（避免训练时连不上 hf-mirror.com）
  ```bash
  # 在容器里先导出所需环境变量
  export HF_TOKEN=... HF_HOME=/data/hf_cache
  bash scripts/prefetch_hf_model.sh Qwen/Qwen3.5-4B
  ```
- `sync_base_configs.sh` — 升级 NeMo-RL 版本时同步官方基底配置到 `configs/base/`
- `post_train.sh` — **训练后闭环**（集群侧执行，由 `lab export` / `lab eval` 经服务端代理调起）：
  把 checkpoint 转 HF（按后端自适应 `convert_dcp_to_hf.py` / `convert_megatron_to_hf.py`，可推 HF Hub），
  或对 checkpoint 跑 `examples/run_eval.py` 评测。带 `LAB_DRY_RUN=1` 只打印命令不执行。
  ```bash
  # 通常用 CLI（经服务端提交，执行在集群）：
  uv run lab export grpo_qwen3.5-9b_gsm8k_v1 [--step N] [--push-repo user/name]
  uv run lab eval   grpo_qwen3.5-9b_gsm8k_v1 [--step N] [-- generation.temperature=0.6]
  ```
