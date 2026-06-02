# common/data — 数据处理

放数据集的下载 / 清洗 / 格式转换脚本，把原始数据转成 NeMo-RL 需要的 jsonl 格式。

约定：

- 脚本入库，**原始 / 大数据不入库**（见根 `.gitignore`）。
- 输出统一落到 `datasets/<name>/`（small 元数据入库，大文件本地/对象存储）。
- 每个数据集一个转换脚本，如 `prepare_gsm8k.py`。
