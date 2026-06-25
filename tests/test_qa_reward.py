"""题库 GRPO 规则奖励的单元测试（守住 boxed 解析 / 各题型判分）。"""
from __future__ import annotations

import pytest

from common.rewards.qa_reward import (
    FORMAT_PENALTY,
    extract_boxed,
    qa_rule_reward_fn,
)


def grade(expected: str, completion: str) -> float:
    return qa_rule_reward_fn([""], [completion], [expected])[0]


def test_extract_boxed_simple():
    assert extract_boxed(r"答案是 \boxed{B}") == "B"


def test_extract_boxed_nested_braces():
    assert extract_boxed(r"\boxed{a_{1} + b}") == "a_{1} + b"


def test_extract_boxed_takes_last():
    assert extract_boxed(r"\boxed{A} 再想想 \boxed{C}") == "C"


def test_extract_boxed_none():
    assert extract_boxed("没有框") is None


@pytest.mark.parametrize(
    "expected,completion,want",
    [
        ("[single] B", r"答案 \boxed{B}", 1.0),
        ("[single] B", r"\boxed{C}", 0.0),
        ("[single] B", r"我觉得是 B", FORMAT_PENALTY),  # 没写 boxed → 格式罚分
        ("[bool] A", r"\boxed{A}", 1.0),
        ("[multiple] A,C,D", r"\boxed{D, A, C}", 1.0),
        ("[multiple] A,C,D", r"\boxed{A,C}", 2 / 3),         # 漏选: (2-0)/3
        ("[multiple] A,C,D", r"\boxed{A,B,C,D}", 2.5 / 3),   # 全选(多1错): (3-0.5·1)/3，w=0.5
        (
            "[fill] 拒收/reject ||| 特采/waive ||| 放行/Release",
            r"\boxed{reject; 特采; Release}",
            1.0,
        ),
        ("[fill] 正向 ||| 3V ||| 0V ||| 0.7V", r"\boxed{正向; 3V; 1V; 0.7V}", 0.75),
        (
            "[short] 低温/掺杂 ||| 纯度高 ||| 横向扩散小",
            r"离子注入是低温工艺，纯度高。\boxed{低温; 纯度高}",
            2 / 3,
        ),
    ],
)
def test_grade_cases(expected, completion, want):
    assert grade(expected, completion) == pytest.approx(want)


def test_reward_fn_returns_same_length():
    comps = [r"\boxed{B}", r"\boxed{C}", "无框"]
    exps = ["[single] B", "[single] B", "[single] B"]
    out = qa_rule_reward_fn(["", "", ""], comps, exps)
    assert len(out) == 3
    assert out[0] == 1.0 and out[1] == 0.0 and out[2] == FORMAT_PENALTY
