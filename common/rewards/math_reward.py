"""数学类 GRPO 奖励函数示例。

被配置引用：reward.fn: "common.rewards.math_reward:compute_reward"
真实接口签名以所用 NeMo-RL 版本为准，这里给出常见形态：根据生成答案与参考答案比对给分。
"""
from __future__ import annotations

import re


def _extract_answer(text: str) -> str | None:
    """从模型输出里抽取最终答案，约定写在 \\boxed{...} 或 'answer:' 后。"""
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"answer\s*[:：]\s*(.+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().splitlines()[0]
    return None


def compute_reward(response: str, reference: str, **kwargs) -> float:
    """答案正确给 1.0，否则 0.0；可叠加格式分等。"""
    pred = _extract_answer(response)
    if pred is None:
        return 0.0
    gold = reference.strip()
    return 1.0 if pred == gold else 0.0
