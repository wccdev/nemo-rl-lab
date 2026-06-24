"""多轮「本地文档 grep 检索」Agent 奖励环境（NeMo-RL 0.6.0）。

定位：与 `qa_env.QARewardEnv`（单轮、无工具）做 A/B 对比的**对照组**。
区别只有一个——这里模型可以**多轮调用 `search` 工具检索集群容器内的本地资料**，再作答；
**最终判分复用同一套 qa 奖励**（客观题规则 / 简答题裁判 LLM），保证两实验唯一变量是「能否检索」。

检索方式：在【集群训练进程】所在容器里，对 `DOCS_DIR`（默认 /data/docs，含子目录）下的
**markdown 文件**跑 `grep`（递归、忽略大小写、带上下文行），把命中片段回灌给模型。
之所以用 grep 而不是向量检索：零依赖、零外部服务、结果可解释，且贴合「在容器里 grep 查资料」的真实工作流。

协议（写进数据集 prompt，见实验 run.py）：
  - 检索资料： <search>关键词</search>            # 环境对本地 markdown 跑 grep，把命中片段作为 observation 回灌
  - 给出答案： 正常作答，并把关键要点放入 \\boxed{...}（与单轮实验完全一致的答案格式）
  每一轮模型输出要么是一次 <search>，要么是带 \\boxed{} 的最终作答：
    - 含 \\boxed{}            → 视为最终答案，复用 qa 奖励判分并结束（不强制必须先检索）
    - 含 <search>…</search>  → 跑 grep 检索本地文档，返回命中片段，继续下一轮
    - 都没有                 → 提示格式，继续（计一轮）
    - 超过 max_turns 仍无答案 → 判 0 结束
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
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


# ============================ 本地文档检索工具（grep）============================
# 在集群容器内对本地资料目录跑 grep。全部通过环境变量配置（由中心化服务在集群侧注入到作业）：
#   DOCS_DIR             资料根目录（含子目录），默认 /data/docs。目录不存在 → 返回占位提示（不抛异常）。
#   DOCS_GLOB            只搜哪些文件（grep --include），默认 *.md（只搜 markdown）。
#   DOCS_TOP_K           最多回灌几个命中片段（按文件聚合），默认 3。
#   DOCS_CONTEXT_LINES   每个命中额外带几行上下文（grep -C），默认 2。
#   DOCS_MAX_CHARS       单次检索回灌进上下文的总字符上限，默认 500（GB10 seq=1536 多轮防 host RAM OOM）。
#   DOCS_MAX_PER_FILE    单个文件最多取几处命中（grep -m），默认 3，避免一个文件刷屏。
#   DOCS_TIMEOUT         单次 grep 子进程超时（秒），默认 15。
#   DOCS_OR_FALLBACK     整句精确匹配查不到时，是否再做「关键词分词 OR 召回」（默认 1 开；0 关）。
#   DOCS_MAX_TERMS       OR 回退时最多用几个关键词（防止碎词把所有行都召回），默认 12。
# ⚠️ 检索发生在【集群训练进程】（Ray actor）所在容器里，所以 DOCS_DIR 必须是【容器内】真实存在的路径。
DOCS_DIR = os.environ.get("DOCS_DIR", "/data/docs")
DOCS_GLOB = os.environ.get("DOCS_GLOB", "*.md")
DOCS_TOP_K = int(os.environ.get("DOCS_TOP_K", "3"))
DOCS_CONTEXT_LINES = int(os.environ.get("DOCS_CONTEXT_LINES", "2"))
DOCS_MAX_CHARS = int(os.environ.get("DOCS_MAX_CHARS", "500"))
DOCS_MAX_PER_FILE = int(os.environ.get("DOCS_MAX_PER_FILE", "3"))
DOCS_TIMEOUT = float(os.environ.get("DOCS_TIMEOUT", "15"))
DOCS_OR_FALLBACK = os.environ.get("DOCS_OR_FALLBACK", "1") not in ("0", "false", "False", "")
DOCS_MAX_TERMS = int(os.environ.get("DOCS_MAX_TERMS", "12"))

# 关键词分词用：英文/数字/缩写/型号（如 CMP、PVD、Qwen3.5）直接抽；中文按 2-gram 滑窗（无需 jieba 也能召回）。
_ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+#/-]+")
_ZH_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
# 极简中文停用字：跳过含这些字的 2-gram，避免「的X」「在X」之类虚词碎片刷屏。
_ZH_STOP = set("的了和与及或在是有为對对把被让从向到这那此其之也都很更最就还按如并且则等")


def _tokenize(query: str) -> list[str]:
    """把查询切成关键词（OR 召回用），零依赖、不引 jieba：
    - 英文/数字/缩写/型号：正则直接抽（高信息量，原样当关键词）。
    - 中文：≤4 字整体作一个词（精度更好）；更长的按 2-gram 滑窗切，跳过含停用字的 gram。
    返回去重后、最多 DOCS_MAX_TERMS 个关键词（保持出现顺序）。
    """
    terms: list[str] = []
    seen: set[str] = set()

    def _add(t: str) -> None:
        t = t.strip()
        if len(t) >= 2 and t.lower() not in seen:
            seen.add(t.lower())
            terms.append(t)

    for tok in _ASCII_TOKEN_RE.findall(query):
        _add(tok)
    for run in _ZH_RUN_RE.findall(query):
        if len(run) <= 4:
            _add(run)
        else:
            for i in range(len(run) - 1):
                bg = run[i:i + 2]
                if bg[0] in _ZH_STOP or bg[1] in _ZH_STOP:
                    continue
                _add(bg)
    return terms[:DOCS_MAX_TERMS]


def _run_grep(terms: list[str]) -> tuple[int, str, str]:
    """对 DOCS_DIR 下的 markdown 跑一次 grep；多个 term 用多个 -e（固定字符串、OR 语义、无需转义正则）。
    返回 (returncode, stdout, stderr)；returncode<0 表示子进程异常（超时/缺 grep）。
    """
    cmd = [
        "grep", "-rinI", "-F",
        f"-C{max(0, DOCS_CONTEXT_LINES)}",
        f"-m{max(1, DOCS_MAX_PER_FILE)}",
        f"--include={DOCS_GLOB}",
    ]
    for t in terms:
        cmd += ["-e", t]   # 每个 -e 是一个固定字符串模式，命中任一即算（OR）
    cmd += ["--", DOCS_DIR]  # -- 终止选项解析，防止 term/路径以 '-' 开头被当成参数
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=DOCS_TIMEOUT)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        return -1, "", type(e).__name__
    return proc.returncode, proc.stdout, proc.stderr


def docs_search(query: str) -> str:
    """对本地 markdown 文档跑 grep，返回拼好的命中片段文本（失败/未命中返回提示，不抛异常）。

    两段式：先整句精确匹配（高精度）；查不到再把查询分词后做 OR 召回（高召回，DOCS_OR_FALLBACK 开关）。
    换检索方式（如向量检索/全文索引），只改本函数即可，环境其余逻辑不变。
    """
    query = " ".join((query or "").split())  # 折叠空白/去换行：让一次 grep 匹配单行更稳
    if not query:
        return "search 错误: 查询为空"
    if not os.path.isdir(DOCS_DIR):
        return f"（本地资料目录未接入：DOCS_DIR={DOCS_DIR} 不存在或不可访问。请联系管理员确认容器内已挂载资料。）"

    # 第一段：整句精确匹配（固定字符串）。
    rc, out, err = _run_grep([query])
    if rc == 0:
        return _format_grep_output(out, [query])
    if rc < 0:
        return f"search 错误: 检索失败（{err}）"
    if rc > 1:
        msg = (err or "").strip().splitlines()
        return f"search 错误: grep 返回 {rc}{('：' + msg[0]) if msg else ''}"

    # 第二段：整句没命中（rc==1），分词后 OR 召回。
    if DOCS_OR_FALLBACK:
        terms = _tokenize(query)
        # 仅当分词结果跟整句不同（即确实拆出了多个/不同关键词）才值得再查一次。
        if terms and terms != [query]:
            rc2, out2, err2 = _run_grep(terms)
            if rc2 == 0:
                return _format_grep_output(out2, terms)
            if rc2 < 0:
                return f"search 错误: 检索失败（{err2}）"
            if rc2 > 1:
                msg = (err2 or "").strip().splitlines()
                return f"search 错误: grep 返回 {rc2}{('：' + msg[0]) if msg else ''}"

    return "未检索到相关资料（换个关键词再试）"


def _block_file_path(first_line: str) -> Optional[str]:
    """从块首行里切出文件绝对路径。

    grep -r 每行是 `<文件路径><分隔><行号><分隔><内容>`（命中用 ':'，上下文用 '-'）。
    优先用 DOCS_GLOB 的扩展名（如 .md）+ 紧跟的分隔符来定位路径结尾——
    这样即便文件名里含 '-数字-'（如 v1-2-foo.md）也不会切错；扩展名取不到时退回首个 `<分隔><数字><分隔>`。
    """
    ext = DOCS_GLOB.replace("*", "")  # "*.md" -> ".md"
    if ext:
        m = re.search(re.escape(ext) + r"[:-]", first_line)
        if m:
            return first_line[: m.start() + len(ext)]
    m = re.match(r"^(.+?)[:-]\d+[:-]", first_line)
    return m.group(1) if m else None


def _parse_grep_line(line: str, base: str) -> Optional[tuple[str, Optional[int], str]]:
    """解析 grep -r 的一行 → (相对路径, 行号或None, 正文)。无法解析返回 None。

    每行形如 `<文件路径>:<行号>:<命中行>`（命中）或 `<文件路径>-<行号>-<上下文行>`（上下文）。
    路径定位复用 _block_file_path（按扩展名，文件名含 '-数字-' 也不会切错）。
    """
    absfile = _block_file_path(line)
    if not absfile:
        return None
    rel = absfile[len(base):] if absfile.startswith(base) else absfile
    rest = line[len(absfile):] if line.startswith(absfile) else line
    mm = re.match(r"^[:-](\d+)[:-]?(.*)$", rest)
    if mm:
        return rel, int(mm.group(1)), mm.group(2)
    return rel, None, rest


def _format_grep_output(raw: str, terms: list[str]) -> str:
    """把 grep -r 的原始输出**按文件聚合**成片段，按命中关键词数排序后取前 TOP_K，再按字符上限截断。

    排序：命中块多于 DOCS_TOP_K 时，**优先保留命中关键词更多的文件块**
    （按该文件正文命中的不同 term 数降序；同分保持 grep 原始（即文件首次出现）顺序）。
    term 命中只在【正文】里数，不含文件路径前缀，避免文件名误计。
    按文件分组而非按 '--' 分块：grep 在 -C0 时不输出 '--'，分组才对 context=0 也健壮，也正好对应「文件块」。
    """
    base = DOCS_DIR.rstrip("/") + "/"
    lowered_terms = [t.lower() for t in terms if t]

    files: dict[str, list[tuple[Optional[int], str]]] = {}
    order: list[str] = []
    for line in raw.splitlines():
        if not line.strip() or line == "--":
            continue
        parsed = _parse_grep_line(line, base)
        if not parsed:
            continue
        rel, lno, text = parsed
        if rel not in files:
            files[rel] = []
            order.append(rel)
        files[rel].append((lno, text))

    if not order:
        return "未检索到相关资料（换个关键词再试）"

    scored: list[tuple[int, int, str]] = []  # (命中 term 数, 文件首次出现序号, 排版后文本)
    for idx, rel in enumerate(order):
        rows = files[rel]
        content = "\n".join(t for _, t in rows).lower()
        score = sum(1 for t in lowered_terms if t in content)
        body: list[str] = []
        prev: Optional[int] = None
        for lno, text in rows:
            if prev is not None and lno is not None and lno - prev > 1:
                body.append("  ⋯")  # 同文件内不连续的命中区域之间插省略号
            body.append(f"L{lno}: {text.strip()}" if lno is not None else text.strip())
            prev = lno
        scored.append((score, idx, f"【{rel}】\n" + "\n".join(body)))

    scored.sort(key=lambda x: (-x[0], x[1]))  # 命中多的文件优先；同分稳定（保持首次出现顺序）
    out_blocks = [b for _, _, b in scored[:DOCS_TOP_K]]
    return "\n---\n".join(out_blocks)[:DOCS_MAX_CHARS]


# ============================ 元数据 / 文本解析 ============================
class QADocsMetadata(TypedDict, total=False):
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
class QADocsAgentEnv(EnvironmentInterface[QADocsMetadata]):
    """多轮本地文档 grep 检索 QA 环境（Ray Actor）。最终判分复用 common/rewards 的 qa 奖励。"""

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
        metadata: list[QADocsMetadata],
    ) -> EnvironmentReturn[QADocsMetadata]:
        n = len(message_log_batch)
        observations: list[dict[str, str]] = [None] * n  # type: ignore[list-item]
        rewards: list[float] = [0.0] * n
        terminateds: list[bool] = [False] * n
        next_stops: list[Optional[list[str]]] = [None] * n
        next_meta: list[Optional[QADocsMetadata]] = [None] * n
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

            nm: QADocsMetadata = dict(meta)  # type: ignore[assignment]
            nm["num_turns"] = num_turns + 1

            # 3) 检索本地文档：grep 返回片段，继续
            if search_q is not None:
                obs = docs_search(search_q)
                observations[i] = {"role": "environment", "content": f"[检索结果]\n{obs}"}
                next_stops[i] = self.SEARCH_STOP_STRINGS
                next_meta[i] = nm
                continue

            # 4) 格式不对：提示并重试（计一轮）
            observations[i] = {
                "role": "environment",
                "content": (
                    "格式不对。检索本地资料用 <search>关键词</search>；"
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
            "qa_docs_mean_reward": rewards.mean().item(),
            "qa_docs_perfect_rate": (rewards >= 1.0).float().mean().item(),
            "qa_docs_format_penalty_rate": (rewards < 0).float().mean().item(),
        }
        return batch, metrics
