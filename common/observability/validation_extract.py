"""从 NeMo-RL message_logs 提取结构化验证样本（与 console log_parse 字段对齐）。"""
from __future__ import annotations


def _role_text(message_log: list, role: str) -> str:
    parts: list[str] = []
    want = role.upper()
    for msg in message_log:
        if not isinstance(msg, dict):
            continue
        r = str(msg.get("role", "")).upper()
        if r == want:
            content = msg.get("content")
            if content is not None:
                parts.append(str(content))
    return "\n".join(parts).strip()


def extract_message_log_samples(
    message_logs: list,
    rewards: list[float],
    *,
    num_samples: int = 5,
) -> tuple[list[dict], list[dict], float | None]:
    """返回 (samples, dist, avg_reward)。采样策略与 print_message_log_samples 一致。"""
    if not message_logs or not rewards or num_samples <= 0:
        return [], [], None
    n = len(message_logs)
    indices = list(range(n))
    num_to_show = min(num_samples, n)
    if len(indices) > num_to_show:
        sorted_indices = sorted(indices, key=lambda i: rewards[i], reverse=True)
        half = num_to_show // 2
        indices = sorted_indices[:half] + sorted_indices[-half:]
        if num_to_show % 2 == 1:
            indices.append(sorted_indices[len(sorted_indices) // 2])
        indices = indices[:num_to_show]

    samples: list[dict] = []
    for i, idx in enumerate(indices):
        ml = message_logs[idx]
        reward = float(rewards[idx])
        samples.append(
            {
                "idx": i + 1,
                "reward": reward,
                "user": _role_text(ml, "user"),
                "assistant": _role_text(ml, "assistant"),
                "env": _role_text(ml, "environment") or _role_text(ml, "system"),
            }
        )

    counts: dict[float, int] = {}
    for r in rewards:
        counts[r] = counts.get(r, 0) + 1
    dist = [{"reward": k, "count": v} for k, v in sorted(counts.items())]
    avg_reward = sum(rewards) / len(rewards) if rewards else None
    return samples, dist, avg_reward
