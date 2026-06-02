#!/usr/bin/env python
# 多轮 Agent（工具调用）GRPO 训练脚本（NeMo-RL 0.6.0）。
# 改编自官方 examples/run_grpo_sliding_puzzle.py，把环境换成自定义计算器工具环境，
# 数据集换成随机算术题。由本实验 run.sh 通过 ENTRY 自动调用。
import argparse
import itertools
import os
import pprint
import random
import sys
from typing import Any, Iterator

from omegaconf import OmegaConf
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

# 让 `import common.*` 能找到仓库根目录
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir

from common.environments.example_tool_env import ToolAgentEnv, safe_eval

TASK_NAME = "calc_tool"
STOP_STRINGS = ["</tool>", "</answer>"]


def parse_args():
    parser = argparse.ArgumentParser(description="多轮工具调用 GRPO 训练")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置路径")
    args, overrides = parser.parse_known_args()
    return args, overrides


def _make_problem(max_number: int, num_operands: int) -> tuple[str, float]:
    """随机生成一道只含 + - * 的算术题，返回 (题面表达式, 正确答案)。"""
    ops = ["+", "-", "*"]
    nums = [str(random.randint(1, max_number)) for _ in range(num_operands)]
    expr_parts = [nums[0]]
    for n in nums[1:]:
        expr_parts.append(random.choice(ops))
        expr_parts.append(n)
    expr = " ".join(expr_parts)
    return expr, safe_eval(expr)


def _build_prompt(question: str) -> str:
    return (
        "你是一个会使用工具的智能体。请通过调用计算器工具求解算式，然后给出最终答案。\n"
        "可用工具：\n"
        "- calc：计算一个算术表达式。调用格式（单独一行）：<tool>calc: 2+3*4</tool>\n"
        "当你得到结果后，用如下格式给出最终答案：<answer>数值</answer>\n"
        f"算式：{question}\n"
        "请先调用计算器，再回答。"
    )


def generate_datum(tokenizer, env_cfg: dict[str, Any], idx: int) -> DatumSpec:
    question, target = _make_problem(
        max_number=int(env_cfg.get("max_number", 50)),
        num_operands=int(env_cfg.get("num_operands", 3)),
    )
    prompt = _build_prompt(question)
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    ).strip()
    token_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ][0]
    message_log: LLMMessageLogType = [
        {"role": "user", "content": prompt_text, "token_ids": token_ids}
    ]
    metadata = {
        "target": float(target),
        "num_turns": 0,
        "max_turns": int(env_cfg.get("max_turns", 6)),
        "question": question,
        "answer_tolerance": float(env_cfg.get("answer_tolerance", 1e-6)),
    }
    return {
        "message_log": message_log,
        "length": len(token_ids),
        "extra_env_info": metadata,
        "loss_multiplier": 1.0,
        "idx": idx,
        "task_name": TASK_NAME,
        "stop_strings": STOP_STRINGS,
    }


class IterableToolDataset(IterableDataset):
    def __init__(self, tokenizer, env_cfg, length):
        super().__init__()
        self.tokenizer = tokenizer
        self.env_cfg = env_cfg
        self.length = length

    def __iter__(self) -> Iterator[DatumSpec]:
        for i in itertools.count():
            yield generate_datum(self.tokenizer, self.env_cfg, i)

    def __len__(self):
        return self.length


def main():
    register_omegaconf_resolvers()
    args, overrides = parse_args()
    if not args.config:
        args.config = os.path.join(THIS_DIR, "config.yaml")

    config = load_config(args.config)
    print(f"已加载配置: {args.config}")
    if overrides:
        print(f"CLI overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)
    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("最终配置：")
    pprint.pprint(config)

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 日志目录: {config['logger']['log_dir']}")

    init_ray()
    set_seed(config["grpo"]["seed"])

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    env_cfg = config["env"][TASK_NAME]["cfg"]
    env = ToolAgentEnv.options(num_gpus=0).remote(cfg=dict(env_cfg))
    task_to_env = {TASK_NAME: env}

    ds_length = (
        config["grpo"]["num_prompts_per_step"]
        * config["grpo"]["num_generations_per_prompt"]
        * config["grpo"]["max_num_steps"]
    )
    dataset = IterableToolDataset(tokenizer, env_cfg, ds_length)
    val_dataset = IterableToolDataset(tokenizer, env_cfg, config["grpo"]["max_val_samples"])

    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )


if __name__ == "__main__":
    main()
