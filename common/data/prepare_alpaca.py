#!/usr/bin/env python
"""把 Alpaca 指令数据预处理成 NeMo-RL ResponseDataset 可用的 jsonl（SFT 用）。

输出 {"input": <指令(可含输入上下文)>, "output": <目标回复>}，
配 ResponseDataset(input_key=input, output_key=output) + sft_processor。

用法：
    python common/data/prepare_alpaca.py            # 写到 <repo>/datasets/alpaca/
    python common/data/prepare_alpaca.py --val_size 1000

之后：
    export ALPACA_DATA_DIR=<上面输出的目录>
"""
import argparse
import json
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def build_prompt(instruction: str, context: str) -> str:
    instruction = (instruction or "").strip()
    context = (context or "").strip()
    return f"{instruction}\n\n{context}" if context else instruction


def main():
    parser = argparse.ArgumentParser(description="预处理 Alpaca -> SFT jsonl")
    parser.add_argument(
        "--out",
        default=os.path.join(REPO_ROOT, "datasets", "alpaca"),
        help="输出目录（默认 <repo>/datasets/alpaca）",
    )
    parser.add_argument("--hf_name", default="tatsu-lab/alpaca")
    parser.add_argument("--val_size", type=int, default=1000, help="留作验证的条数")
    args = parser.parse_args()

    from datasets import load_dataset

    os.makedirs(args.out, exist_ok=True)
    ds = load_dataset(args.hf_name, split="train")
    val_size = min(args.val_size, len(ds) // 10)
    splits = {"val.jsonl": range(val_size), "train.jsonl": range(val_size, len(ds))}

    for out_name, idx_range in splits.items():
        path = os.path.join(args.out, out_name)
        n = 0
        with open(path, "w", encoding="utf-8") as f:
            for i in idx_range:
                ex = ds[i]
                rec = {
                    "input": build_prompt(ex.get("instruction", ""), ex.get("input", "")),
                    "output": (ex.get("output", "") or "").strip(),
                }
                if not rec["input"] or not rec["output"]:
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        print(f"写入 {n} 条 -> {path}")

    print("\n完成。请设置环境变量供实验配置使用：")
    print(f"  export ALPACA_DATA_DIR={args.out}")


if __name__ == "__main__":
    main()
