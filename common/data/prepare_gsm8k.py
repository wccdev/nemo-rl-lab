#!/usr/bin/env python
"""把 HuggingFace 上的 GSM8K 预处理成 NeMo-RL ResponseDataset 可用的干净 jsonl。

GSM8K 的 answer 字段是「推理过程 + #### 最终数字」，math 环境需要干净的金标准答案，
所以这里抽取 #### 后的数字作为 answer，写成 {"question": ..., "answer": "<数字>"}。

用法：
    python common/data/prepare_gsm8k.py            # 写到 <repo>/datasets/gsm8k/
    python common/data/prepare_gsm8k.py --out /abs/dir

之后让实验配置能找到数据：
    export GSM8K_DATA_DIR=<上面输出的目录>
"""
import argparse
import json
import os
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ANS_RE = re.compile(r"####\s*(-?[\d,]+)")


def extract_answer(answer_field: str) -> str:
    """从 GSM8K answer 里抽取 #### 后的最终数字（去掉千分位逗号）。"""
    m = _ANS_RE.search(answer_field)
    return m.group(1).replace(",", "").strip() if m else answer_field.strip()


def main():
    parser = argparse.ArgumentParser(description="预处理 GSM8K -> 干净 jsonl")
    parser.add_argument(
        "--out",
        default=os.path.join(REPO_ROOT, "datasets", "gsm8k"),
        help="输出目录（默认 <repo>/datasets/gsm8k）",
    )
    parser.add_argument("--hf_name", default="openai/gsm8k")
    parser.add_argument("--subset", default="main")
    args = parser.parse_args()

    from datasets import load_dataset

    os.makedirs(args.out, exist_ok=True)
    ds = load_dataset(args.hf_name, args.subset)

    for split, out_name in [("train", "train.jsonl"), ("test", "val.jsonl")]:
        path = os.path.join(args.out, out_name)
        n = 0
        with open(path, "w", encoding="utf-8") as f:
            for ex in ds[split]:
                rec = {
                    "question": ex["question"].strip(),
                    "answer": extract_answer(ex["answer"]),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        print(f"写入 {n} 条 -> {path}")

    print("\n完成。请设置环境变量供实验配置使用：")
    print(f"  export GSM8K_DATA_DIR={args.out}")


if __name__ == "__main__":
    main()
