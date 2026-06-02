# datasets/ — 数据集元数据

**只放元数据，不放大文件。** 原始 / 处理后的大数据走本地磁盘或对象存储，由 `.gitignore` 排除。

每个数据集一个子目录：

```
datasets/
└── gsm8k/
    ├── README.md      # 来源、版本、license、字段说明、处理命令
    ├── train.jsonl    # 仅当文件足够小才入库；否则 .gitignore + 写明获取方式
    └── raw/           # 原始数据（.gitignore）
```

`README.md` 至少写清楚：

- 数据来源 / 下载方式 / 版本（保证可复现）
- 用 `common/data/` 下哪个脚本处理、命令是什么
- 字段格式（prompt / response / answer 等）
