const TOKEN_KEY = 'nrlab_token'

export function getToken() {
  return localStorage.getItem(TOKEN_KEY)
}

export function setToken(t: string | null) {
  if (t) localStorage.setItem(TOKEN_KEY, t)
  else localStorage.removeItem(TOKEN_KEY)
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers)
  if (!headers.has('Content-Type') && init?.body) headers.set('Content-Type', 'application/json')
  const token = getToken()
  if (token && token !== 'local') headers.set('Authorization', `Bearer ${token}`)
  const res = await fetch(path, { ...init, headers })
  if (res.status === 401) {
    setToken(null)
    window.location.href = '/login'
    throw new Error('未登录')
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || res.statusText)
  }
  return res.json() as Promise<T>
}

export type JobRow = {
  id: string
  exp: string
  status: string
  entrypoint: string
  start: string
  dur: string
  running: boolean
  lab_run_id?: string | null
}

export type JobOverview = {
  job_id: string
  exp: string
  model?: string
  steps: { step: number; total: number; avg_reward: number | null; step_time: number | null }[]
  validations: {
    step: number
    avg_reward: number | null
    accuracy: number | null
    avg_len: number | null
    sample_count: number
  }[]
  summary: Record<string, number | null>
}
