# NeMo-RL Lab Console — Design System (Master)

> 页面级 override：`design-system/nemo-rl-lab-console/pages/[page].md`

**原则：工程控制台，不是 AI 聊天产品。** 禁止紫粉渐变、发光霓虹、大圆角卡片堆叠、Inter/Fira 默认组合。

---

## 双主题色板

### Light（默认）

| Token | Hex | 用途 |
|-------|-----|------|
| `--background` | `#f4f4f5` | 页面底 |
| `--card` | `#ffffff` | 卡片/侧栏 |
| `--foreground` | `#18181b` | 主文字 |
| `--muted-foreground` | `#52525b` | 次要文字（≥4.5:1 对比） |
| `--primary` | `#4338ca` | 主操作（靛蓝，非亮蓝） |
| `--border` | `#e4e4e7` | 分隔线 |
| `--success` | `#059669` | RUNNING / 进步 |
| `--destructive` | `#dc2626` | FAILED / 停止 |
| `--warning` | `#d97706` | PENDING |

### Dark

| Token | Hex | 用途 |
|-------|-----|------|
| `--background` | `#09090b` | 页面底（zinc-950，非纯黑 OLED 风） |
| `--card` | `#18181b` | 卡片 |
| `--foreground` | `#fafafa` | 主文字 |
| `--muted-foreground` | `#a1a1aa` | 次要 |
| `--primary` | `#818cf8` | 主操作 |
| `--border` | `#27272a` | 分隔 |
| `--success` | `#34d399` | |
| `--destructive` | `#f87171` | |
| `--warning` | `#fbbf24` | |

**图表序列（最多 6 条）：** `#4338ca` `#0891b2` `#059669` `#d97706` `#7c3aed` `#dc2626`（dark 模式各 +100 亮度）

---

## 字体

- **UI：** IBM Plex Sans（400/500/600）
- **数据/ID/日志：** IBM Plex Mono（400/500）
- **禁止：** Fira、Inter、system-ui 作为品牌字体

```css
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
```

---

## 布局

- **Sidebar 240px**（collapsible → 64px icon rail），顶栏 56px
- 内容区 `max-w-7xl`，padding `24px`
- 表格优先于大 KPI 卡片；数字用 mono

---

## 组件规则

- 图标：**Lucide**，24px，禁止 emoji
- 按钮圆角 `6px`（不是 12px+ pill）
- 卡片：`border` + 轻 shadow，hover 仅 `border-color` 变化，**禁止 translateY/scale**
- 过渡：`150–200ms` `transition-colors`
- 焦点：ring 2px primary/40

---

## 图表（ECharts）

- 训练 reward / 验证 accuracy：**折线 + 点 hover**，支持 zoom
- 多实验对比：同色系列 + 图例 toggle
- 实时作业：SSE 追加，曲线尾部更新（非闪烁）

---

## Anti-patterns（NeMo-RL 专属）

- ❌ 渐变 hero / 「AI 助手」式大标题
- ❌ 玻璃拟态半透明卡片（light 下看不清）
- ❌ 紫罗兰 `#8b5cf6` 作主色
- ❌ 全屏 loading  spinner 遮罩（用 skeleton + 局部刷新）
- ❌ 把 LLM 对话气泡 UI 套在训练监控上

---

## Pre-delivery

- [ ] Light + Dark 均测对比度
- [ ] `prefers-reduced-motion` 关闭动画
- [ ] 375 / 768 / 1024 / 1440  responsive
- [ ] 可点击元素 `cursor-pointer`
- [ ] Sidebar 移动端 Sheet 模式
