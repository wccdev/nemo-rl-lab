# grpo_qwen3.5-9b_qa-rl-agent_v1（对比实验 · 实验二 / treatment）

用 **GRPO** 在自有**技术培训考题题库**上强化训练 **Qwen 3.5 9B**。**多轮**：模型回答前可多次调用 `<search>` **检索外部知识库**，拿到资料再作答。

> **这是 A/B 对比的处理组**：多轮 + **知识库检索工具**。
> 基线组（实验一 / baseline）= [`grpo_qwen3.5-9b_qa-rl_v1`](../grpo_qwen3.5-9b_qa-rl_v1)：单轮、无工具。
> 两个实验共用同一数据集 / 模型 / LoRA / batch / 裁判奖励，**唯一变量**是「能否多轮检索知识库」。
> 对比目标：检索外部知识库**能否提升**模型在公司技术考题上的作答准确率。

> ✅ **知识库已接入 RAGFlow。** `<search>` 调 RAGFlow `POST {KB_BASE_URL}/api/v1/retrieval`。
> 未配 `KB_BASE_URL` / `KB_DATASET_IDS` 时返回占位提示（"知识库未接入"），流水线仍可跑通但拿不到真实资料。
> **正式跑前先在集群容器里 curl 自测检索能通（见下）。**

## 与基线的唯一差异

| 维度 | 实验一 baseline | 实验二 treatment（本实验） |
| --- | --- | --- |
| 轮数 | 单轮（答一次即结束） | 多轮（`max_rollout_turns=4`） |
| 工具 | 无 | `<search>查询词</search>` 检索知识库 |
| 环境 | `common/environments/qa_env.py` `QARewardEnv` | `common/environments/qa_kb_agent_env.py` `QAKBAgentEnv` |
| 奖励 | qa 规则 + 简答裁判 | **同源**（最终答案复用同一套 qa 奖励） |
| 数据 / 模型 / LoRA / batch | —— 完全一致 —— | —— 完全一致 —— |
| seq | 1536 | 1536（与 baseline 对齐；baseline 1536 都会 ~step280 OOM，agent 多轮更吃内存，故不敢上 2048） |

模型作答协议：检索用 `<search>查询词</search>`；作答把要点放入 `\boxed{...}`（与基线同一答案格式，保证判分一致）。

## 知识库检索接入 · RAGFlow（`common/environments/qa_kb_agent_env.py` 的 `kb_search()`）

`kb_search()` 调 RAGFlow 检索接口 `POST {KB_BASE_URL}/api/v1/retrieval`
（请求 `{"question","dataset_ids","page_size","similarity_threshold"}`，响应 `{"code":0,"data":{"chunks":[{"content",...}]}}`）。
通过环境变量配置（`lab submit` 会从 `cluster/submit.env` 转发到集群作业）：

```bash
KB_BASE_URL=http://192.168.1.x:9380   # RAGFlow 服务地址（不含 /api/...），docker 默认端口 9380。空=未接入
KB_API_KEY=ragflow-xxxxx              # 页面右上「API」生成
KB_DATASET_IDS=id1,id2                # 要检索的知识库 dataset id，逗号分隔（必填）
KB_TOP_K=3                            # 返回片段数（映射 RAGFlow page_size）
KB_TIMEOUT=15                         # 单次检索超时（秒）
KB_SIMILARITY_THRESHOLD=0.2           # 相似度下限，过滤弱命中
```

> ⚠️ 检索发生在【集群训练进程】里 → `KB_BASE_URL` 必须从【集群容器】可达（同 `JUDGE_*`）。
> dataset id 在 RAGFlow 页面知识库设置里看，或 `GET /api/v1/datasets`。

**正式跑前，在集群容器里自测检索连通（务必）：**

```bash
curl -s -X POST http://192.168.1.x:9380/api/v1/retrieval \
  -H "Authorization: Bearer ragflow-xxxxx" -H "Content-Type: application/json" \
  -d '{"question":"随便挑一道题里的关键词","dataset_ids":["<dataset_id>"],"page_size":3}'
# 期望: {"code":0,"data":{"chunks":[{"content":"..."}...]}}；code!=0 看 message 排错（多半 api_key / dataset_id / 未解析完成）
```

> 换别的知识库 API，只改 `kb_search()` 里「请求体」「响应解析」两处（已注释标好）。

## 跑起来

前置与基线相同（题库在集群 + `QA_RL_DATA_DIR` + 简答裁判 `JUDGE_*`），详见基线 README。本实验额外需要 `KB_*`：

```bash
# 1) 确保题库在集群、submit.env 里 QA_RL_DATA_DIR / JUDGE_* / KB_* 已配
# 2) 提交
lab submit grpo_qwen3.5-9b_qa-rl-agent_v1
```

## GB10 显存提醒（重要）

baseline 实测 **seq=1536 都会在 ~step280 被 Ray(host RAM) OOM**；agent 多轮每步堆更多 rollout 进内存，
seq=2048 会必崩且崩更早。所以本实验把**所有可调项都对齐到 baseline 的最省档**，且**唯一变量仍是「多轮+检索」**：

- `num_prompts_per_step=4` / `num_generations_per_prompt=4` / `train_global_batch_size=16`（与 baseline 严格一致，勿动，动了破坏对比）
- `max_total_sequence_length=1536`（与 baseline 一致）
- `max_rollout_turns=3` + `env.qa_kb.cfg.max_turns=3`（收紧轮数省内存）
- `KB_MAX_CHARS=500`（单次检索回灌字符上限，env 可调；见 submit.env）

任务 ~100-200 步即收敛，`val_period=50` 在 50/100/150/200/250 都有验证点 → 即便 ~step280 崩，250 步前的对比曲线已可用（和 baseline 同样窗口）。
**仍 OOM 再按序降**：`max_turns→2` → `KB_MAX_CHARS→400` → `seq→1280`。**batch 任何时候都不要动。**

## 看多轮检索轨迹

```bash
uv run lab job samples <JOB_ID> -n 1      # 最近一次验证的完整多轮对话（含 <search> 与检索结果）
```

## 结论 / 记录

（训练后补：最佳 step、val 准确率、与 baseline 的对比、检索是否带来提升、SwanLab 链接、踩坑。）
