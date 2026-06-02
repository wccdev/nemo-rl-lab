# common/data — 数据处理

放数据集的下载 / 清洗 / 转换脚本，以及自定义 data processor。

## 接入 NeMo-RL（0.6.0）

- 内置数据集：配置 `data.train.dataset_name=...`（如 `squad`、`OpenMathInstruct-2`）。
- 本地数据：
  - GRPO 用 `ResponseDataset`：`data.train.data_path=/abs/train.jsonl`、`data.train.input_key=question`、`data.default.dataset_name=ResponseDataset`。
  - SFT 用 OpenAI messages 格式：`data.train_data_path=/abs/train.jsonl`、`data.chat_key=messages`。
- 自定义处理逻辑：实现 data processor，在 `data.default.processor` 引用（参考官方 `math_hf_data_processor` / `sft_processor`）。

## 约定

- 脚本入库，**原始 / 大数据不入库**（见根 `.gitignore`）。
- 处理产物落到 `datasets/<name>/`（小元数据入库，大文件本地 / 对象存储）。
- 每个数据集一个转换脚本，如 `prepare_gsm8k.py`。
