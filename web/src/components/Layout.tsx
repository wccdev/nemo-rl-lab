import { NavLink, Outlet } from 'react-router-dom'
import {
  Activity,
  FlaskConical,
  GitCompare,
  LayoutDashboard,
  ListOrdered,
  LogOut,
  Moon,
  Sun,
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import { useTheme } from '../context/ThemeContext'
import { cn } from '../lib/utils'

const nav = [
  { to: '/', icon: LayoutDashboard, label: '概览' },
  { to: '/jobs', icon: Activity, label: '作业' },
  { to: '/compare', icon: GitCompare, label: '对比' },
  { to: '/runs', icon: ListOrdered, label: '台账' },
  { to: '/experiments', icon: FlaskConical, label: '实验' },
]

export function Layout() {
  const { user, logout, noAuth } = useAuth()
  const { theme, toggle } = useTheme()

  return (
    <div className="flex min-h-screen bg-background">
      <aside className="flex w-60 shrink-0 flex-col border-r border-border bg-card">
        <div className="flex h-14 items-center gap-2 border-b border-border px-5">
          <div className="flex h-8 w-8 items-center justify-center rounded-md bg-primary text-sm font-semibold text-primary-fg">
            NL
          </div>
          <div>
            <div className="text-sm font-semibold leading-tight">NeMo-RL Lab</div>
            <div className="text-xs text-muted">微调控制台</div>
          </div>
        </div>
        <nav className="flex flex-1 flex-col gap-1 p-3">
          {nav.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                cn(
                  'flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors duration-200',
                  isActive
                    ? 'bg-primary/10 text-primary'
                    : 'text-muted hover:bg-background hover:text-foreground',
                )
              }
            >
              <Icon className="h-4 w-4 shrink-0" />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="border-t border-border p-3 space-y-2">
          <button
            type="button"
            onClick={toggle}
            className="flex w-full items-center gap-3 rounded-md px-3 py-2 text-sm text-muted transition-colors hover:bg-background hover:text-foreground"
          >
            {theme === 'light' ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
            {theme === 'light' ? '深色模式' : '浅色模式'}
          </button>
          {!noAuth && user && (
            <button
              type="button"
              onClick={logout}
              className="flex w-full items-center gap-3 rounded-md px-3 py-2 text-sm text-muted transition-colors hover:bg-background hover:text-foreground"
            >
              <LogOut className="h-4 w-4" />
              退出 ({user.username})
            </button>
          )}
        </div>
      </aside>
      <main className="flex-1 overflow-auto">
        <div className="mx-auto max-w-7xl p-6">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
