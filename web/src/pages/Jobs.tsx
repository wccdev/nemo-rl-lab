import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, type JobRow } from '../lib/api'
import { statusColor } from '../lib/utils'

export function JobsPage() {
  const [jobs, setJobs] = useState<JobRow[]>([])
  const [filter, setFilter] = useState<'all' | 'active'>('all')

  useEffect(() => {
    api<JobRow[]>('/api/jobs').then(setJobs).catch(console.error)
  }, [])

  const rows = filter === 'active' ? jobs.filter((j) => j.running) : jobs

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">作业</h1>
          <p className="text-sm text-muted">Ray 集群上的训练 / 导出 / 评测作业</p>
        </div>
        <div className="flex gap-2">
          {(['all', 'active'] as const).map((f) => (
            <button
              key={f}
              type="button"
              onClick={() => setFilter(f)}
              className={`rounded-md border px-3 py-1.5 text-sm transition-colors ${
                filter === f ? 'border-primary bg-primary/10 text-primary' : 'border-border text-muted hover:text-foreground'
              }`}
            >
              {f === 'all' ? '全部' : '活跃'}
            </button>
          ))}
        </div>
      </div>
      <div className="overflow-hidden rounded-lg border border-border bg-card">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-left text-xs text-muted">
              <th className="px-4 py-3 font-medium">实验</th>
              <th className="px-4 py-3 font-medium">Job ID</th>
              <th className="px-4 py-3 font-medium">状态</th>
              <th className="px-4 py-3 font-medium">开始</th>
              <th className="px-4 py-3 font-medium">用时</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((j) => (
              <tr key={j.id} className="border-t border-border transition-colors hover:bg-background">
                <td className="px-4 py-3">
                  <Link to={`/jobs/${j.id}`} className="font-medium hover:text-primary">
                    {j.exp}
                  </Link>
                </td>
                <td className="px-4 py-3 font-mono text-xs text-muted">{j.id.slice(0, 22)}…</td>
                <td className={`px-4 py-3 font-mono text-xs ${statusColor(j.status)}`}>{j.status}</td>
                <td className="px-4 py-3 font-mono text-xs text-muted">{j.start}</td>
                <td className="px-4 py-3 font-mono text-xs text-muted">{j.dur}</td>
              </tr>
            ))}
            {!rows.length && (
              <tr>
                <td colSpan={5} className="px-4 py-8 text-center text-muted">
                  暂无作业
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
