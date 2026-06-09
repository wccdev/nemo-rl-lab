# grpo_qwen3.5-9b_qa-rl-agent_v1（对比实验 · 实验二 / treatment）

用 **GRPO** 在自有**技术培训考题题库**上强化训练 **Qwen 3.5 9B**。**多轮**：模型回答前可多次调用 `<search>` **检索外部知识库**，拿到资料再作答。

> **这是 A/B 对比的处理组**：多轮 + **知识库检索工具**。
> 基线组（实验一 / baseline）= [`grpo_qwen3.5-9b_qa-rl_v1`](../grpo_qwen3.5-9b_qa-rl_v1)：单轮、无工具。
> 两个实验共用同一数据集 / 模型 / LoRA / batch / 裁判奖励，**唯一变量**是「能否多轮检索知识库」。
> 对比目标：检索外部知识库**能否提升**模型在公司技术考题上的作答准确率。

> ⚠️ **知识库搭建中。** 现在 `<search>` 在未配 `KB_BASE_URL` 时返回占位提示（"知识库未接入"），
> 训练流水线可跑通但模型拿不到真实资料。**等知识库就绪、配好 `KB_BASE_URL` 后再正式跑本实验。**

## 与基线的唯一差异

| 维度 | 实验一 baseline | 实验二 treatment（本实验） |
| --- | --- | --- |
| 轮数 | 单轮（答一次即结束） | 多轮（`max_rollout_turns=4`） |
| 工具 | 无 | `<search>查询词</search>` 检索知识库 |
| 环境 | `common/environments/qa_env.py` `QARewardEnv` | `common/environments/qa_kb_agent_env.py` `QAKBAgentEnv` |
| 奖励 | qa 规则 + 简答裁判 | **同源**（最终答案复用同一套 qa 奖励） |
| 数据 / 模型 / LoRA / batch | —— 完全一致 —— | —— 完全一致 —— |
| seq | 1536 | 2048（多轮 + 检索片段更长） |

模型作答协议：检索用 `<search>查询词</search>`；作答把要点放入 `\boxed{...}`（与基线同一答案格式，保证判分一致）。

## 知识库检索接入（`common/environments/qa_kb_agent_env.py` 的 `kb_search()`）

通过环境变量配置（`lab submit` 会从 `cluster/submit.env` 转发到集群作业）：

```bash
KB_BASE_URL=http://<你的知识库检索服务>/   # 空 = 未接入（返回占位提示）
KB_API_KEY=...                              # 可选鉴权
KB_TOP_K=3                                  # 取前 K 条片段
KB_TIMEOUT=15                               # 单次检索超时（秒）
```

`kb_search()` 默认假设检索服务是 `POST {KB_BASE_URL}/search`，请求 `{"query","top_k"}`、响应 `{"results":[{"text"...}]}`。
**你的知识库 API 不同就改 `kb_search()` 里请求/响应解析两处**（已注释标好）。

## 跑起来

前置与基线相同（题库在集群 + `QA_RL_DATA_DIR` + 简答裁判 `JUDGE_*`），详见基线 README。本实验额外需要 `KB_*`：

```bash
# 1) 确保题库在集群、submit.env 里 QA_RL_DATA_DIR / JUDGE_* / KB_* 已配
# 2) 提交
lab submit grpo_qwen3.5-9b_qa-rl-agent_v1
```

## GB10 显存提醒

多轮 + 检索片段回灌会显著拉长上下文。2×GB10 统一内存(~122GB)对 9B 偏紧，
agent 实验实测 `seq=2048` 长跑可能 host RAM OOM。若 OOM，按顺序降：
`env.qa_kb.cfg.max_turns→3` → `qa_kb_agent_env._KB_MAX_CHARS→500` → `seq→1792/1536` →
`train_global_batch_size→16` 且 `num_generations_per_prompt→4`。

## 看多轮检索轨迹

```bash
uv run lab job samples <JOB_ID> -n 1      # 最近一次验证的完整多轮对话（含 <search> 与检索结果）
```

## 结论 / 记录

（训练后补：最佳 step、val 准确率、与 baseline 的对比、检索是否带来提升、SwanLab 链接、踩坑。）
