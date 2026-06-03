"""题库 QA 奖励函数（规则判分 + 简答 LLM 裁判）。

- qa_reward.qa_rule_reward_fn(queries, completions, expected_answers) -> list[float]
    纯规则、零成本：按 expected_answer 的 [type] 前缀分派（single/bool/multiple/fill/short）。
- qa_judge_reward.qa_judge_reward_fn(...) -> list[float]
    混合：非简答走规则；简答调裁判 LLM 打分，失败回退关键词覆盖率。

两者接口一致，被 common/environments/qa_env.py 的单轮 GRPO 环境调用。
"""
