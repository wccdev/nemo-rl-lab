# NeMo-RL Lab Console — Web 平台设计规格

**日期：** 2026-06-23  
**状态：** Phase 1 已批准  
**部署：** 本机轻量（`--no-auth`）+ 团队服务（JWT + SQLite）

## 目标

将 `lab web` 从单文件内嵌 HTML 升级为 **React SPA + FastAPI**，作为 NeMo-RL 微调平台的统一 Web 控制台，覆盖 CLI 大部分只读/监控能力，Phase 2 再补 submit/export/eval。

## 数据源

| 来源 | 用途 |
|------|------|
| Ray Dashboard | 作业列表/日志/停止/GPU |
| `.lab/runs.jsonl` | 提交台账 + run_id 关联 |
| 仓库 `experiments/` | config、实验列表 |
| `cluster/submit.env` | profile、地址（服务端读取，不返密钥明文） |

## Phase 1 范围

- [x] 设计系统（`design-system/nemo-rl-lab-console/`）
- [ ] 后端 `nemo_rl_lab/web/`
- [ ] 前端 `web/`（Vite + React + shadcn + ECharts）
- [ ] 页面：登录、概览、作业、对比、台账、实验列表
- [ ] 认证双模式、SSE 日志
- [ ] `lab web` 启动新栈

## Phase 2（后续）

Web submit / export / eval、实验新建、Settings 引导

## 目录

```
nemo_rl_lab/web/     # FastAPI
web/                 # React SPA
scripts/web_log_parse.py  # 保留，backend 复用
scripts/web_dashboard.py  # deprecated shim → 新入口
```
