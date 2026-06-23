import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import { api, getToken, setToken } from '../lib/api'

type User = { username: string; role: string }

type AuthCtx = {
  user: User | null
  loading: boolean
  noAuth: boolean
  login: (u: string, p: string) => Promise<void>
  logout: () => void
}

const Ctx = createContext<AuthCtx | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)
  const [noAuth, setNoAuth] = useState(false)

  useEffect(() => {
    ;(async () => {
      try {
        const cfg = await fetch('/api/auth/config').then((r) => r.json())
        if (cfg.no_auth) {
          setNoAuth(true)
          setUser({ username: 'local', role: 'admin' })
          setLoading(false)
          return
        }
        if (!getToken()) {
          setLoading(false)
          return
        }
        const me = await api<User>('/api/auth/me')
        setUser(me)
      } catch {
        setToken(null)
      } finally {
        setLoading(false)
      }
    })()
  }, [])

  async function login(username: string, password: string) {
    const res = await api<{ token: string; user: User }>('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    })
    setToken(res.token)
    setUser(res.user)
  }

  function logout() {
    setToken(null)
    setUser(null)
    window.location.href = '/login'
  }

  return (
    <Ctx.Provider value={{ user, loading, noAuth, login, logout }}>{children}</Ctx.Provider>
  )
}

export function useAuth() {
  const v = useContext(Ctx)
  if (!v) throw new Error('AuthProvider missing')
  return v
}
