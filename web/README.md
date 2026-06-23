# NeMo-RL Lab Console

React SPA + FastAPI backend for the fine-tuning platform.

## 开发联调

```bash
# 终端 1：API（本机免登）
uv run lab web --no-open --port 8080

# 终端 2：前端（Vite 代理 /api → :8080）
pnpm install
pnpm dev
# 打开 http://127.0.0.1:5173
```

## 生产构建

```bash
pnpm build          # 输出 web/dist/
uv run lab web      # 自动挂载 dist/
```

## 团队部署

```bash
pnpm build
uv run lab web --auth --serve --port 8080
# 首次：浏览器打开 /login →「创建管理员」
```

设计规范见 `design-system/nemo-rl-lab-console/MASTER.md`（IBM Plex、靛蓝主色、浅/深双主题，避免 AI 味）。
