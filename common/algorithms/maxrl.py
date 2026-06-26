"""MaxRL 优势估计器 —— 论文《Maximum Likelihood Reinforcement Learning》(MaxRL, arXiv:2602.02710)。

核心论点：传统 RL（REINFORCE / GRPO）最大化的是「期望通过率 pass@1」，只是最大似然目标
    J_ML(x) = log p(x)  的一阶近似。把 J_ML 按失败事件做 Maclaurin 展开并截断到 T 阶，得到
    一族在「RL（T=1）↔ 精确最大似然（T→∞）」之间插值的目标；用 N 个 on-policy rollout 估计时
    其无偏梯度恰好对应截断阶 T=N。增加 rollout 数 N 不只是降方差，而是直接逼近更高保真的最大似然目标。

落到实现上，MaxRL 相对 GRPO 只改**优势计算这一行**（论文 Algorithm 1 / 附录 D.1）：

    GRPO :      Â_i = (r_i − μ) / σ        （按组内奖励【标准差】归一化）
    REINFORCE : Â_i = (r_i − μ)
    MaxRL:      Â_i = (r_i − μ) / μ        （按组内【平均奖励】=通过率 归一化）

其中 μ 是该 prompt 一组 rollout 的平均奖励（= 经验通过率 p̂，含样本自身，非 leave-one-out），
σ 是组内标准差。当 μ=0（整组无一答对）时该组优势全部置 0，与 GRPO「无成功样本不回传梯度」一致；
当 μ=1（全对）时优势也为 0。二元奖励下，难题（μ 小）一旦答对会得到 (1−μ)/μ ≈ 1/p̂ 的大正优势，
失败样本得 −1 —— 这正是 MaxRL「把学习信号集中到低通过率难题」的来源（GRPO 只 ~1/√p̂）。

KL 惩罚、ratio clip、token-level loss 等其余部分与 GRPO 完全一致，不受影响。
"""

from __future__ import annotations

from typing import Any

import torch


def maxrl_advantages(
    prompt_ids: torch.Tensor,
    rewards: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """计算 MaxRL 的 token-level 优势张量。

    Args:
        prompt_ids: [b] 或 [b, s]，标识每个样本属于哪个 prompt 组（同组的 prompt token 完全相同）。
        rewards:    [b]，每个样本的标量奖励（二元任务里是 0/1）。
        mask:       [b, seq_len]，response token 掩码，仅用于把样本级优势广播成 token 级形状。
        epsilon:    除法保护项（μ>0 时分母为 μ+epsilon，影响可忽略）。

    Returns:
        [b, seq_len] 的优势张量，Â_i = (r_i − μ_g) / μ_g（μ_g>0），否则 0。
    """
    rewards = rewards.float()
    mu = _group_mean_per_prompt(prompt_ids, rewards)

    advantages = torch.zeros_like(rewards)
    nonzero = mu > 0
    advantages[nonzero] = (rewards[nonzero] - mu[nonzero]) / (mu[nonzero] + epsilon)
    return advantages.unsqueeze(-1).expand(mask.shape)


def _group_mean_per_prompt(
    prompt_ids: torch.Tensor, rewards: torch.Tensor
) -> torch.Tensor:
    """对每个 prompt 组求平均奖励 μ（含样本自身），返回与 rewards 同形状的逐样本 μ。"""
    mu = torch.zeros_like(rewards)
    if prompt_ids.dim() == 1:
        uniques = torch.unique(prompt_ids)
        for u in uniques:
            m = prompt_ids == u
            mu[m] = rewards[m].mean()
    else:
        uniques = torch.unique(prompt_ids, dim=0)
        for i in range(len(uniques)):
            m = (prompt_ids == uniques[i]).all(dim=1)
            mu[m] = rewards[m].mean()
    return mu


class MaxRLAdvantageEstimator:
    """与 NeMo-RL 的 GRPOAdvantageEstimator 接口一致的 MaxRL 优势估计器。

    用法：通过 install_maxrl_estimator() 让 NeMo-RL 在 grpo.adv_estimator.name == "maxrl" 时选用本类。
    """

    def __init__(self, estimator_config: Any, loss_config: Any):
        # 组均值（通过率）归一化的除法保护项；可在 config 的 adv_estimator.epsilon 覆盖。
        if hasattr(estimator_config, "get"):
            eps = estimator_config.get("epsilon", 1e-6)
        else:
            eps = getattr(estimator_config, "epsilon", 1e-6)
        self.epsilon = float(eps)

    def compute_advantage(
        self,
        prompt_ids: torch.Tensor,
        rewards: torch.Tensor,
        mask: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        # 其余 kwargs（repeated_batch / logprobs_policy / logprobs_reference）MaxRL 不需要。
        return maxrl_advantages(prompt_ids, rewards, mask, epsilon=self.epsilon)


def install_maxrl_estimator() -> None:
    """给 NeMo-RL 的 _create_advantage_estimator 打补丁，新增 name=="maxrl" 分支。

    NeMo-RL 内置只识别 grpo / gdpo / reinforce_plus_plus，没有插件注册口；这里在运行时
    包一层：name=="maxrl" 返回 MaxRLAdvantageEstimator，其余原样委托给官方实现。
    grpo_train 在调用时按模块全局名查找该函数，故只要在 grpo_train() 之前安装即可生效。
    幂等：重复安装无副作用。
    """
    import nemo_rl.algorithms.grpo as grpo_mod

    if getattr(grpo_mod, "_maxrl_installed", False):
        return

    original = grpo_mod._create_advantage_estimator

    def _section(master_config, key):
        # 兼容不同 NeMo-RL 版本：main 分支 MasterConfig 是 pydantic（顶层属性访问 master_config.grpo），
        # v0.6.0 等老版本用下标（master_config["grpo"]）。两种都试。
        if hasattr(master_config, key):
            return getattr(master_config, key)
        return master_config[key]

    def _create_with_maxrl(master_config):
        grpo_cfg = _section(master_config, "grpo")
        # 嵌套 section 通常是 dict（.get 可用）；个别版本是对象，用 getattr 兜底。
        if hasattr(grpo_cfg, "get"):
            adv_cfg = grpo_cfg.get("adv_estimator", {}) or {}
        else:
            adv_cfg = getattr(grpo_cfg, "adv_estimator", {}) or {}
        adv_name = adv_cfg.get("name") if hasattr(adv_cfg, "get") else getattr(adv_cfg, "name", None)
        if adv_name == "maxrl":
            print(
                "  ✓ Using MaxRL advantage estimator "
                "(组均值/通过率归一化 (r-μ)/μ；逼近最大似然目标，arXiv:2602.02710)",
                flush=True,
            )
            return MaxRLAdvantageEstimator(adv_cfg, _section(master_config, "loss_fn"))
        return original(master_config)

    grpo_mod._create_advantage_estimator = _create_with_maxrl
    grpo_mod._maxrl_installed = True
