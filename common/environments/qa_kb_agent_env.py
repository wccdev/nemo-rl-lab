"""多轮「知识库检索」Agent 奖励环境（NeMo-RL 0.6.0）。

定位：与 `qa_env.QARewardEnv`（单轮、无工具）做 A/B 对比的**对照组**。
区别只有一个——这里模型可以**多轮调用 `search` 工具检索外部知识库**，再作答；
**最终判分复用同一套 qa 奖励**（客观题规则 / 简答题裁判 LLM），保证两实验唯一变量是「能否检索」。

协议（写进数据集 prompt，见实验 run.py）：
  - 检索知识库： <search>查询词</search>           # 环境执行检索，把命中片段作为 observation 回灌
  - 给出答案  ： 正常作答，并把关键要点放入 \\boxed{...}（与单轮实验完全一致的答案格式）
  每一轮模型输出要么是一次 <search>，要么是带 \\boxed{} 的最终作答：
    - 含 \\boxed{}            → 视为最终答案，复用 qa 奖励判分并结束（不强制必须先检索）
    - 含 <search>…</search>  → 调知识库检索，返回片段，继续下一轮
    - 都没有                 → 提示格式，继续（计一轮）
    - 超过 max_turns 仍无答案 → 判 0 结束

外部知识库（KB）检索：见 `kb_search()`。知识库**搭建中**——
  - 未配置 `KB_BASE_URL` 时返回占位提示（"知识库未接入"），环境与训练流程仍可跑通（便于先搭流水线）。
  - 配好后改 `kb_search()` 适配你的检索 API 的真实请求/响应格式即可。
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Optional, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

# 确保 Ray actor 进程里能 import 到本仓库的 common 包
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ============================ 知识库检索工具（RAGFlow）============================
# 接入自建 RAGFlow 知识库的检索接口 `POST {KB_BASE_URL}/api/v1/retrieval`。
#   KB_BASE_URL      RAGFlow 服务地址（不含 /api/...），如 http://192.168.1.x:9380。空=未接入（返回占位提示）。
#   KB_API_KEY       RAGFlow 的 API Key（页面右上「API」里生成，形如 ragflow-xxx）。
#   KB_DATASET_IDS   要检索的知识库(dataset)ID，逗号分隔（页面知识库列表 / List datasets API 可取）。必填。
#   KB_TOP_K         返回命中片段条数（映射到 RAGFlow 的 page_size）。
#   KB_TIMEOUT       单次检索超时（秒）。
#   KB_SIMILARITY_THRESHOLD  相似度下限，过滤弱命中（默认 0.2，与 RAGFlow 默认一致）。
# ⚠️ 检索发生在【集群训练进程】（Ray actor）里，所以 KB_BASE_URL 要从【集群容器】能访问到（Mac 不一定通）。
KB_BASE_URL = os.environ.get("KB_BASE_URL", "")          # 空 = 知识库未接入（返回占位提示）
KB_API_KEY = os.environ.get("KB_API_KEY", "EMPTY")
KB_DATASET_IDS = [s.strip() for s in os.environ.get("KB_DATASET_IDS", "").split(",") if s.strip()]
KB_TOP_K = int(os.environ.get("KB_TOP_K", "3"))
KB_TIMEOUT = float(os.environ.get("KB_TIMEOUT", "15"))
KB_SIMILARITY_THRESHOLD = float(os.environ.get("KB_SIMILARITY_THRESHOLD", "0.2"))
# 回灌片段总长上限（字符），避免撑爆上下文 / 拉高 host RAM。GB10 上 seq=1536 多轮时建议 ~500。
_KB_MAX_CHARS = int(os.environ.get("KB_MAX_CHARS", "500"))


def kb_search(query: str) -> str:
    """检索 RAGFlow 知识库，返回拼好的命中片段文本（失败/未接入时返回提示，不抛异常）。

    RAGFlow 检索 API（POST {KB_BASE_URL}/api/v1/retrieval）：
      请求体 {"question","dataset_ids","page_size","similarity_threshold"}
      响应   {"code":0,"data":{"chunks":[{"content","similarity",...}], ...}}
    换别的知识库就改下面「请求体」和「响应解析」两处。
    """
    query = (query or "").strip()
    if not query:
        return "search 错误: 查询为空"
    if not KB_BASE_URL:
        # 知识库搭建中：给个明确占位，训练流水线照常跑（模型只是拿不到真实资料）。
        return "（知识库未接入：KB_BASE_URL 未配置，无法检索。请联系管理员配置后重试。）"
    if not KB_DATASET_IDS:
        return "（知识库未接入：KB_DATASET_IDS 未配置，不知道检索哪个知识库。）"

    # —— 请求体（RAGFlow 格式）——
    body = json.dumps({
        "question": query,
        "dataset_ids": KB_DATASET_IDS,
        "page_size": KB_TOP_K,                # RAGFlow 用 page_size 控制返回片段数
        "similarity_threshold": KB_SIMILARITY_THRESHOLD,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{KB_BASE_URL.rstrip('/')}/api/v1/retrieval",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {KB_API_KEY}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=KB_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError) as e:
        return f"search 错误: 检索失败（{type(e).__name__}）"

    # —— 响应解析（RAGFlow 格式）——
    if data.get("code", 0) != 0:
        return f"search 错误: 知识库返回 {data.get('message', data.get('code'))}"
    chunks = (data.get("data") or {}).get("chunks") or []
    passages: list[str] = []
    for c in chunks[:KB_TOP_K]:
        text = c.get("content") or c.get("content_ltks") or ""
        if text:
            passages.append(str(text).strip())
    if not passages:
        return "未检索到相关资料"
    joined = "\n---\n".join(passages)
    return joined[:_KB_MAX_CHARS]


# ============================ 元数据 / 文本解析 ============================
class QAKBMetadata(TypedDict, total=False):
    expected_answer: str   # 带 [type] 前缀的金标准（与单轮实验一致）
    query: str             # 题面（裁判 LLM / 检索上下文用）
    num_turns: int         # 已交互轮数
    max_turns: int         # 最大轮数


def _extract_tag(text: str, tag: str) -> Optional[str]:
    """取最后一个 <tag>...</tag> 的内容；没有则 None。"""
    open_t, close_t = f"<{tag}>", f"</{tag}>"
    s = text.rfind(open_t)
    if s == -1:
        return None
    e = text.find(close_t, s + len(open_t))
    if e == -1:
        return None
    return text[s + len(open_t):e].strip()


def _last_assistant_text(message_log: LLMMessageLogType) -> str:
    for msg in reversed(message_log):
        if msg.get("role") == "assistant":
            return str(msg.get("content", "")).strip()
    return ""


# ============================ 环境 ============================
@ray.remote  # pragma: no cover
class QAKBAgentEnv(EnvironmentInterface[QAKBMetadata]):
    """多轮知识库检索 QA 环境（Ray Actor）。最终判分复用 common/rewards 的 qa 奖励。"""

    SEARCH_STOP_STRINGS = ["</search>"]

    def __init__(self, cfg: Optional[dict[str, Any]] = None):
        self.cfg = cfg or {}
        self.use_judge = bool(self.cfg.get("use_judge", True))
        # 与 QARewardEnv 同源：客观题走规则；简答 use_judge=true 走裁判、失败回退关键词覆盖率。
        if self.use_judge:
            from common.rewards.qa_judge_reward import qa_judge_reward_fn

            self._reward_fn = qa_judge_reward_fn
        else:
            from common.rewards.qa_reward import qa_rule_reward_fn

            self._reward_fn = qa_rule_reward_fn
        # boxed 检测复用 qa_reward 的实现（正确处理嵌套花括号）
        from common.rewards.qa_reward import extract_boxed

        self._extract_boxed = extract_boxed

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[QAKBMetadata],
    ) -> EnvironmentReturn[QAKBMetadata]:
        n = len(message_log_batch)
        observations: list[dict[str, str]] = [None] * n  # type: ignore[list-item]
        rewards: list[float] = [0.0] * n
        terminateds: list[bool] = [False] * n
        next_stops: list[Optional[list[str]]] = [None] * n
        next_meta: list[Optional[QAKBMetadata]] = [None] * n
        answers: list[Optional[list[str]]] = [None] * n

        # 收集"本轮给出最终答案"的样本，最后批量判分（简答裁判是并发批处理，批量更省）
        final_idx: list[int] = []
        final_q: list[str] = []
        final_comp: list[str] = []
        final_exp: list[str] = []

        for i, (log, meta) in enumerate(zip(message_log_batch, metadata)):
            content = _last_assistant_text(log)
            num_turns = int(meta.get("num_turns", 0))
            max_turns = int(meta.get("max_turns", 4))
            expected = str(meta.get("expected_answer", ""))
            query = str(meta.get("query", ""))

            boxed = self._extract_boxed(content)
            search_q = _extract_tag(content, "search")

            # 1) 最终答案（含 \boxed{}）：批量判分后结束。不强制必须先检索。
            if boxed is not None:
                final_idx.append(i)
                final_q.append(query)
                final_comp.append(content)
                final_exp.append(expected)
                terminateds[i] = True
                answers[i] = [expected]
                continue

            # 2) 超过最大轮数仍无答案：判 0 结束
            if num_turns >= max_turns:
                observations[i] = {"role": "environment", "content": f"已达最大轮数 {max_turns}，结束。"}
                terminateds[i] = True
                continue

            nm: QAKBMetadata = dict(meta)  # type: ignore[assignment]
            nm["num_turns"] = num_turns + 1

            # 3) 检索知识库：返回片段，继续
            if search_q is not None:
                obs = kb_search(search_q)
                observations[i] = {"role": "environment", "content": f"[检索结果]\n{obs}"}
                next_stops[i] = self.SEARCH_STOP_STRINGS
                next_meta[i] = nm
                continue

            # 4) 格式不对：提示并重试（计一轮）
            observations[i] = {
                "role": "environment",
                "content": (
                    "格式不对。检索知识库用 <search>查询词</search>；"
                    "作答把关键要点放入 \\boxed{...}（多个用 ; 分隔）。"
                ),
            }
            next_stops[i] = self.SEARCH_STOP_STRINGS
            next_meta[i] = nm

        # 批量判分给出最终答案的样本
        if final_idx:
            scores = self._reward_fn(final_q, final_comp, final_exp)
            for i, s in zip(final_idx, scores):
                rewards[i] = float(s)
                observations[i] = {"role": "environment", "content": f"得分: {float(s):.3f}"}

        return EnvironmentReturn(
            observations=observations,
            metadata=next_meta,
            next_stop_strings=next_stops,
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.tensor(terminateds, dtype=torch.bool),
            answers=answers,
        )

    def shutdown(self):
        pass

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        rewards = batch.get(
            "total_reward", torch.tensor([0.0] * len(batch["idx"]))
        ).float()
        if len(rewards) == 0:
            return batch, {}
        metrics = {
            "qa_kb_mean_reward": rewards.mean().item(),
            "qa_kb_perfect_rate": (rewards >= 1.0).float().mean().item(),
            "qa_kb_format_penalty_rate": (rewards < 0).float().mean().item(),
        }
        return batch, metrics
