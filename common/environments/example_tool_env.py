"""可运行的多轮「工具调用」Environment（NeMo-RL 0.6.0）。

完全照官方 `nemo_rl/environments/games/sliding_puzzle.py` 的结构写成，可直接用于 GRPO
多轮 Agent 训练。任务是一个**计算器工具调用**示例：模型需要通过调用计算器工具、经过若干轮
交互后给出算式的最终答案。完全自包含、确定性奖励，适合作为你自定义工具环境的模板。

协议（写在数据集 prompt 里，见实验 run.py）：
  - 调用工具： <tool>calc: 2+3*4</tool>     —— env 返回计算结果作为下一轮 observation
  - 给出答案： <answer>14</answer>          —— env 校验对错并结束 episode

接入方式：
  - 环境是 Ray actor，通过自定义 run 脚本实例化并放进 task_to_env（见实验 run.py）。
  - 配置里 env.<task_name>.cfg 提供环境参数（max_turns 等）。
"""
from __future__ import annotations

import ast
import operator
from typing import Any, Optional, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn


# ============================ 工具实现 ============================
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expr: str) -> float:
    """安全地计算一个纯算术表达式（只允许数字与 + - * / // % ** 和括号）。"""

    def _ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](_ev(node.left), _ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_ev(node.operand))
        raise ValueError("不支持的表达式")

    return _ev(ast.parse(expr, mode="eval"))


def _tool_calc(arg: str) -> str:
    try:
        return f"{safe_eval(arg):g}"
    except Exception as e:  # noqa: BLE001
        return f"calc 错误: {e}"


# 工具注册表：要加工具就在这里加 name -> callable(arg:str)->str
TOOLS = {"calc": _tool_calc}


# ============================ 环境状态 / 元数据 ============================
class ToolAgentMetadata(TypedDict):
    target: float          # 正确答案
    num_turns: int         # 已交互轮数
    max_turns: int         # 最大轮数
    question: str          # 题面（用于生成提示）


def _extract_tag(text: str, tag: str) -> Optional[str]:
    """取最后一个 <tag>...</tag> 的内容；没有则返回 None。"""
    open_t, close_t = f"<{tag}>", f"</{tag}>"
    s = text.rfind(open_t)
    if s == -1:
        return None
    e = text.find(close_t, s + len(open_t))
    if e == -1:
        return None
    return text[s + len(open_t) : e].strip()


class ToolAgentRunner:
    """单条轨迹一轮的处理逻辑（与 SlidingPuzzleRunner 对应）。"""

    NEXT_STOP_STRINGS = ["</tool>", "</answer>"]

    def process_turn(
        self,
        message_log: LLMMessageLogType,
        metadata: ToolAgentMetadata,
    ) -> tuple[
        dict[str, str],
        float,
        bool,
        Optional[list[str]],
        Optional[ToolAgentMetadata],
        Optional[list[str]],
    ]:
        num_turns = metadata["num_turns"]
        max_turns = metadata["max_turns"]
        tol = float(metadata.get("answer_tolerance", 1e-6))

        # 超过最大轮数：判负并结束
        if num_turns >= max_turns:
            return (
                {"role": "environment", "content": f"已达最大轮数 {max_turns}，结束。"},
                0.0,
                True,
                None,
                None,
                None,
            )

        # 取最后一条 assistant 内容
        content = ""
        if message_log and message_log[-1]["role"] == "assistant":
            content = str(message_log[-1]["content"]).strip()

        next_meta: Optional[ToolAgentMetadata] = dict(metadata)  # type: ignore[assignment]
        next_meta["num_turns"] = num_turns + 1

        # 1) 先看是否给出最终答案
        answer_str = _extract_tag(content, "answer")
        if answer_str is not None:
            try:
                pred = safe_eval(answer_str)
                correct = abs(pred - metadata["target"]) <= tol
            except Exception:  # noqa: BLE001
                correct = False
            obs = "回答正确！" if correct else f"回答错误。正确答案是 {metadata['target']:g}。"
            return (
                {"role": "environment", "content": obs},
                1.0 if correct else 0.0,
                True,
                None,
                None,
                [answer_str],
            )

        # 2) 再看是否调用工具
        tool_call = _extract_tag(content, "tool")
        if tool_call is not None:
            name, _, arg = tool_call.partition(":")
            name = name.strip().lower()
            arg = arg.strip()
            if name in TOOLS:
                result = TOOLS[name](arg)
                obs = f"{name}({arg}) = {result}"
            else:
                obs = f"未知工具 '{name}'，可用工具：{', '.join(TOOLS)}"
            return (
                {"role": "environment", "content": obs},
                0.0,
                False,
                self.NEXT_STOP_STRINGS,
                next_meta,
                None,
            )

        # 3) 格式非法：提示并让其重试（不结束，计一轮）
        obs = (
            "格式不对。调用工具用 <tool>calc: 表达式</tool>，"
            "给答案用 <answer>数值</answer>。"
        )
        return (
            {"role": "environment", "content": obs},
            0.0,
            False,
            self.NEXT_STOP_STRINGS,
            next_meta,
            None,
        )


@ray.remote  # pragma: no cover
class ToolAgentEnv(EnvironmentInterface[ToolAgentMetadata]):
    """多轮工具调用环境（Ray Actor）。"""

    def __init__(self, cfg: Optional[dict[str, Any]] = None):
        self.cfg = cfg or {}
        self.runner = ToolAgentRunner()

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[ToolAgentMetadata],
    ) -> EnvironmentReturn[ToolAgentMetadata]:
        results = [
            self.runner.process_turn(log, meta)
            for log, meta in zip(message_log_batch, metadata)
        ]

        observations, rewards, terminateds = [], [], []
        all_stop_strings, all_next_metadata, all_answers = [], [], []
        for obs, rew, term, stops, meta, answ in results:
            observations.append(obs)
            rewards.append(rew)
            terminateds.append(term)
            all_stop_strings.append(stops)
            all_next_metadata.append(meta)
            all_answers.append(answ)

        return EnvironmentReturn(
            observations=observations,
            metadata=all_next_metadata,
            next_stop_strings=all_stop_strings,
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.tensor(terminateds, dtype=torch.bool),
            answers=all_answers,
        )

    def shutdown(self):
        pass

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        final_rewards = batch.get(
            "total_reward", torch.tensor([0.0] * len(batch["idx"]))
        )
        success_rate = (
            (final_rewards == 1.0).float().mean().item()
            if len(final_rewards) > 0
            else 0.0
        )
        return batch, {"calc_tool_success_rate": success_rate}
