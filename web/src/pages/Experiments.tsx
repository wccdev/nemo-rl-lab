import { useEffect, useState } from 'react'
import { api } from '../lib/api'

type Exp = { name: string; kind: string; path: string; profile: string | null; has_config: boolean }

export function ExperimentsPage() {
  const [exps, setExps] = useState<Exp[]>([])

  useEffect(() => {
    api<{ experiments: Exp[] }>('/api/experiments').then((d) => setExps(d.experiments))
  }, [])

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold">实验</h1>
        <p className="text-sm text-muted">仓库 experiments/ 与 projects/ 目录</p>
      </div>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {exps.map((e) => (
          <div key={e.path} className="rounded-lg border border-border bg-card p-4 transition-colors hover:border-primary/40">
            <div className="text-xs text-muted">{e.kind}</div>
            <div className="mt-1 font-medium">{e.name}</div>
            <div className="mt-2 flex gap-2 text-xs">
              {e.profile && (
                <span className="rounded border border-border px-2 py-0.5 font-mono">{e.profile}</span>
              )}
              {e.has_config && (
                <span className="rounded border border-border px-2 py-0.5 text-muted">config.yaml</span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
