# grpo_qwen3.5-9b_qa-rl_v1

用 **GRPO** 在自有**技术培训考题题库**上强化训练 **Qwen 3.5 9B**。单轮：模型答一道题，环境判分即结束。

## 目标

- 提升模型在公司技术培训考题（单选/多选/判断/填空/简答）上的作答准确率。
- 客观题用规则判分；**简答题用 LLM-as-judge**（裁判 LLM 打 0~1 分，端点连不上自动回退关键词覆盖率）。

## 组成

| 部分 | 位置 |
| --- | --- |
| 判分逻辑 | `common/rewards/`（`qa_reward.py` 规则 + `qa_judge_reward.py` 裁判，`synonyms.json` 同义词） |
| 奖励环境 | `common/environments/qa_env.py` 的 `QARewardEnv`（单轮，包装上面的判分） |
| 数据 | `datasets/qa_rl/`（由 `common/data/prepare_qa_rl.py` 从 `raw/` 生成） |
| 启动 | 本目录 `run.py`（自建数据集 + 实例化环境 + grpo_train）；`config.yaml` 写差异 |

数据答案格式（`expected_answer` 带 `[type]` 前缀）见 `common/rewards/README.md`。

## 跑起来

```bash
# 1) 预处理数据（把 raw/ 的 train/val/shortanswer 整理进 datasets/qa_rl/）
lab prepare qa_rl
export QA_RL_DATA_DIR="$(pwd)/datasets/qa_rl"

# 2)（可选）简答裁判：在集群起一个本地 vLLM 裁判端点，并设环境变量
#    vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001
#    export JUDGE_BASE_URL=http://127.0.0.1:8001/v1 JUDGE_MODEL=Qwen/Qwen2.5-7B-Instruct JUDGE_API_KEY=EMPTY
#    不配也能跑：简答会回退到关键词覆盖率（config 里 env.qa.cfg.use_judge 控制是否启用裁判）

# 3) 提交到集群（推荐）或在集群容器内直接跑
lab submit grpo_qwen3.5-9b_qa-rl_v1
#   或： NEMO_RL_DIR=/opt/NeMo-RL CLUSTER_PROFILE=gb10-spark lab run grpo_qwen3.5-9b_qa-rl_v1
```

> 注意：环境（Ray actor）需要能 `import common.*`，所以要让本仓库根目录在集群作业的工作目录/PYTHONPATH 里（`lab submit` 的 `ray job submit --working-dir` 会带上仓库代码）。

## 关键超参 / 调参入口（`config.yaml`）

- 后端：Megatron-Core + **LoRA**（继承 `grpo_megatron` + `grpo_lora`；GB10 实测起点 lr 1e-4/dim8）。回全参数：删 `defaults` 里 `grpo_lora.yaml`。
- batch：`num_prompts_per_step=4` / `num_generations_per_prompt=8` / `train_global_batch_size=32`（须整除 prompts×gen）/ `micro=1` / `seq=1536`。
- `loss_fn.reference_policy_kl_penalty`：KL 约束强度。
- `policy.max_total_sequence_length`：多选题带选项较长，按显存调；OOM 就降。
- `env.qa.cfg.use_judge`：简答是否走裁判 LLM。
- `logger.swanlab.*`：云端日志项目/run 名。

## 想纯规则、零成本？

把 `env.qa.cfg.use_judge` 设为 `false`，全部走规则判分（简答=关键词覆盖率），不需要裁判端点。

## 结论 / 记录

（训练后补：最佳 step、val 准确率、SwanLab 链接、踩坑。）
