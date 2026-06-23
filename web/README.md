# NeMo-RL Lab Console

React SPA + FastAPI backend for the fine-tuning platform.

## 一条命令启动

```bash
uv run lab web    # 缺 dist 或 src 有改动时自动 pnpm install + build，再启服务
```

## 开发联调（热更新）

```bash
uv run lab web --no-build --no-open --port 8080   # 只起 API
pnpm -C web dev                                  # Vite :5173 代理 /api
```

## 团队部署

```bash
uv run lab web --auth --serve --port 8080
# 首次：浏览器 /login →「创建管理员」
```

设计规范见 `design-system/nemo-rl-lab-console/MASTER.md`（IBM Plex、靛蓝主色、浅/深双主题，避免 AI 味）。
