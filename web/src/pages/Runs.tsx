import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import { statusColor } from '../lib/utils'

export function RunsPage() {
  const [runs, setRuns] = useState<Record<string, unknown>[]>([])

  useEffect(() => {
    api<{ runs: Record<string, unknown>[] }>('/api/runs?limit=100').then((d) => setRuns(d.runs))
  }, [])

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold">提交台账</h1>
        <p className="text-sm text-muted">本地 .lab/runs.jsonl，run_id 关联集群作业状态</p>
      </div>
      <div className="overflow-x-auto rounded-lg border border-border bg-card">
        <table className="w-full min-w-[800px] text-sm">
          <thead>
            <tr className="border-b border-border text-left text-xs text-muted">
              <th className="px-4 py-3">时间</th>
              <th className="px-4 py-3">动作</th>
              <th className="px-4 py-3">实验</th>
              <th className="px-4 py-3">Profile</th>
              <th className="px-4 py-3">状态</th>
              <th className="px-4 py-3">Commit</th>
              <th className="px-4 py-3">Run ID</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r, i) => (
              <tr key={i} className="border-t border-border">
                <td className="px-4 py-2 font-mono text-xs">{String(r.time ?? '—')}</td>
                <td className="px-4 py-2">{String(r.action ?? 'submit')}</td>
                <td className="px-4 py-2">{String(r.exp ?? '').split('/').pop()}</td>
                <td className="px-4 py-2 font-mono text-xs">{String(r.profile ?? '—')}</td>
                <td className={`px-4 py-2 font-mono text-xs ${statusColor(String(r.job_status ?? '-'))}`}>
                  {String(r.job_status ?? '-')}
                </td>
                <td className="px-4 py-2 font-mono text-xs">
                  {String(r.git_commit ?? '—')}
                  {r.git_dirty ? '*' : ''}
                </td>
                <td className="max-w-[200px] truncate px-4 py-2 font-mono text-xs text-muted" title={String(r.run_id)}>
                  {String(r.run_id ?? '—')}
                </td>
              </tr>
            ))}
            {!runs.length && (
              <tr>
                <td colSpan={7} className="px-4 py-8 text-center text-muted">
                  台账为空；lab submit / export / eval 后会写入
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
