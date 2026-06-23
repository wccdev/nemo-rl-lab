import { type FormEvent, useState } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import { api, setToken } from '../lib/api'
import { useAuth } from '../context/AuthContext'

export function LoginPage() {
  const { user, login, noAuth } = useAuth()
  const nav = useNavigate()
  const [u, setU] = useState('')
  const [p, setP] = useState('')
  const [err, setErr] = useState('')
  const [setup, setSetup] = useState(false)

  if (noAuth || user) return <Navigate to="/" replace />

  async function onSubmit(e: FormEvent) {
    e.preventDefault()
    setErr('')
    try {
      if (setup) {
        const res = await api<{ token: string }>('/api/auth/setup', {
          method: 'POST',
          body: JSON.stringify({ username: u, password: p }),
        })
        setToken(res.token)
        nav('/')
        window.location.reload()
        return
      }
      await login(u, p)
      nav('/')
    } catch (ex) {
      setErr(String(ex))
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-4">
      <form
        onSubmit={onSubmit}
        className="w-full max-w-sm rounded-lg border border-border bg-card p-8 shadow-sm"
      >
        <div className="mb-6 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-md bg-primary text-sm font-semibold text-primary-fg">
            NL
          </div>
          <div>
            <h1 className="text-lg font-semibold">NeMo-RL Lab</h1>
            <p className="text-xs text-muted">{setup ? '创建管理员' : '登录控制台'}</p>
          </div>
        </div>
        <label className="block text-sm font-medium">
          用户名
          <input
            className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
            value={u}
            onChange={(e) => setU(e.target.value)}
            autoComplete="username"
          />
        </label>
        <label className="mt-4 block text-sm font-medium">
          密码
          <input
            type="password"
            className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
            value={p}
            onChange={(e) => setP(e.target.value)}
            autoComplete={setup ? 'new-password' : 'current-password'}
          />
        </label>
        {err && <p className="mt-3 text-sm text-destructive">{err}</p>}
        <button
          type="submit"
          className="mt-6 w-full rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-fg transition-opacity hover:opacity-90"
        >
          {setup ? '创建并进入' : '登录'}
        </button>
        <button
          type="button"
          onClick={() => setSetup(!setup)}
          className="mt-3 w-full text-center text-xs text-muted hover:text-primary"
        >
          {setup ? '已有账号？去登录' : '首次部署？创建管理员'}
        </button>
      </form>
    </div>
  )
}
