import { useEffect, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { api, type JobRow } from '../lib/api'
import { statusColor } from '../lib/utils'

type ClusterStatus = {
  gpu: {
    accel: string[]
    gpu_used: number
    gpu_total: number
    gpu_free: number
  } | null
  active_jobs: JobRow[]
  active_count: number
}

export function DashboardPage() {
  const [cluster, setCluster] = useState<ClusterStatus | null>(null)
  const [runs, setRuns] = useState<Record<string, unknown>[]>([])
  const [err, setErr] = useState('')

  useEffect(() => {
    ;(async () => {
      try {
        const [c, r] = await Promise.all([
          api<ClusterStatus>('/api/cluster/status'),
          api<{ runs: Record<string, unknown>[] }>('/api/runs?limit=8'),
        ])
        setCluster(c)
        setRuns(r.runs)
      } catch (e) {
        setErr(String(e))
      }
    })()
  }, [])

  if (err) return <p className="text-destructive">{err}</p>

  const gpu = cluster?.gpu
  const accel = gpu?.accel?.join('/') || 'GPU'

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">概览</h1>
        <p className="mt-1 text-sm text-muted">集群资源与最近提交</p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label={`${accel} 占用`}
          value={gpu ? `${gpu.gpu_used}/${gpu.gpu_total}` : '—'}
          sub={gpu ? `空闲 ${gpu.gpu_free}` : '连集群中…'}
          accent={gpu && gpu.gpu_free > 0 ? 'success' : 'warning'}
        />
        <StatCard label="活跃作业" value={String(cluster?.active_count ?? '—')} sub="RUNNING / PENDING" />
        <StatCard label="最近台账" value={String(runs.length)} sub="本地 .lab/runs.jsonl" />
        <StatCard
          label="快捷入口"
          value="对比"
          sub={<Link to="/compare" className="text-primary hover:underline">多实验曲线 →</Link>}
        />
      </div>

      {cluster && cluster.active_jobs.length > 0 && (
        <section className="rounded-lg border border-border bg-card">
          <h2 className="border-b border-border px-4 py-3 text-sm font-semibold">活跃作业</h2>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-muted">
                <th className="px-4 py-2 font-medium">实验</th>
                <th className="px-4 py-2 font-medium">状态</th>
                <th className="px-4 py-2 font-medium">用时</th>
                <th className="px-4 py-2 font-medium"></th>
              </tr>
            </thead>
            <tbody>
              {cluster.active_jobs.map((j) => (
                <tr key={j.id} className="border-t border-border">
                  <td className="px-4 py-2 font-medium">{j.exp}</td>
                  <td className={`px-4 py-2 font-mono text-xs ${statusColor(j.status)}`}>{j.status}</td>
                  <td className="px-4 py-2 font-mono text-xs text-muted">{j.dur}</td>
                  <td className="px-4 py-2 text-right">
                    <Link to={`/jobs/${j.id}`} className="text-sm text-primary hover:underline">
                      详情
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {runs.length > 0 && (
        <section className="rounded-lg border border-border bg-card">
          <h2 className="border-b border-border px-4 py-3 text-sm font-semibold">最近提交</h2>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-muted">
                <th className="px-4 py-2">时间</th>
                <th className="px-4 py-2">动作</th>
                <th className="px-4 py-2">实验</th>
                <th className="px-4 py-2">状态</th>
                <th className="px-4 py-2">commit</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r, i) => (
                <tr key={i} className="border-t border-border">
                  <td className="px-4 py-2 font-mono text-xs">{String(r.time ?? '—')}</td>
                  <td className="px-4 py-2">{String(r.action ?? 'submit')}</td>
                  <td className="px-4 py-2">{String(r.exp ?? '').split('/').pop()}</td>
                  <td className={`px-4 py-2 font-mono text-xs ${statusColor(String(r.job_status ?? '-'))}`}>
                    {String(r.job_status ?? '-')}
                  </td>
                  <td className="px-4 py-2 font-mono text-xs text-muted">
                    {String(r.git_commit ?? '—')}
                    {r.git_dirty ? '*' : ''}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  )
}

function StatCard({
  label,
  value,
  sub,
  accent,
}: {
  label: string
  value: string
  sub: ReactNode
  accent?: 'success' | 'warning'
}) {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="text-xs font-medium text-muted">{label}</div>
      <div
        className={`mt-2 font-mono text-2xl font-semibold ${
          accent === 'success' ? 'text-success' : accent === 'warning' ? 'text-warning' : ''
        }`}
      >
        {value}
      </div>
      <div className="mt-1 text-xs text-muted">{sub}</div>
    </div>
  )
}
