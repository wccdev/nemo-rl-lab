import { useEffect, useMemo, useState } from 'react'
import { LineChart } from '../components/LineChart'
import { api, type JobOverview, type JobRow } from '../lib/api'
import { pct } from '../lib/utils'

const MAX = 4

export function ComparePage() {
  const [jobs, setJobs] = useState<JobRow[]>([])
  const [selected, setSelected] = useState<string[]>([])
  const [ovs, setOvs] = useState<Record<string, JobOverview>>({})

  useEffect(() => {
    api<JobRow[]>('/api/jobs').then((list) => {
      setJobs(list)
      const init: string[] = []
      const seen = new Set<string>()
      for (const j of list) {
        if (!seen.has(j.exp)) {
          seen.add(j.exp)
          init.push(j.id)
        }
        if (init.length >= 2) break
      }
      setSelected(init)
    })
  }, [])

  useEffect(() => {
    selected.forEach((id) => {
      if (ovs[id]) return
      api<JobOverview>(`/api/job?id=${encodeURIComponent(id)}`).then((o) =>
        setOvs((prev) => ({ ...prev, [id]: o })),
      )
    })
  }, [selected, ovs])

  function toggle(id: string) {
    setSelected((s) => {
      if (s.includes(id)) return s.filter((x) => x !== id)
      if (s.length >= MAX) return s
      return [...s, id]
    })
  }

  const accSeries = useMemo(
    () =>
      selected
        .map((id) => {
          const o = ovs[id]
          if (!o) return null
          return {
            name: o.exp,
            data: o.validations
              .filter((v) => v.accuracy != null)
              .map((v) => [v.step, v.accuracy!] as [number, number]),
          }
        })
        .filter(Boolean) as { name: string; data: [number, number][] }[],
    [selected, ovs],
  )

  const rewSeries = useMemo(
    () =>
      selected
        .map((id) => {
          const o = ovs[id]
          if (!o) return null
          return {
            name: o.exp,
            data: o.steps
              .filter((s) => s.avg_reward != null)
              .map((s) => [s.step, s.avg_reward!] as [number, number]),
          }
        })
        .filter(Boolean) as { name: string; data: [number, number][] }[],
    [selected, ovs],
  )

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">实验对比</h1>
        <p className="text-sm text-muted">选择 2–4 个作业，对齐验证准确率与训练 reward</p>
      </div>

      <div className="flex flex-wrap gap-2">
        {jobs.slice(0, 20).map((j) => (
          <button
            key={j.id}
            type="button"
            onClick={() => toggle(j.id)}
            className={`rounded-md border px-3 py-2 text-left text-sm transition-colors ${
              selected.includes(j.id)
                ? 'border-primary bg-primary/10'
                : 'border-border hover:border-primary/50'
            }`}
          >
            <div className="font-medium">{j.exp}</div>
            <div className="font-mono text-xs text-muted">{j.status}</div>
          </button>
        ))}
      </div>

      {selected.length > 0 && (
        <>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {selected.map((id) => {
              const o = ovs[id]
              const sum = o?.summary
              return (
                <div key={id} className="rounded-lg border border-border bg-card p-4">
                  <div className="font-medium">{o?.exp ?? '…'}</div>
                  <div className="mt-2 font-mono text-2xl font-semibold">
                    {pct(sum?.final_acc as number | null | undefined)}
                  </div>
                  <div className="mt-1 text-xs text-muted">
                    基线 {pct(sum?.base_acc as number | null | undefined)}
                  </div>
                </div>
              )
            })}
          </div>

          <section className="rounded-lg border border-border bg-card p-4">
            <h2 className="mb-3 text-sm font-semibold">验证准确率</h2>
            <LineChart series={accSeries} yLabel="acc" />
          </section>
          <section className="rounded-lg border border-border bg-card p-4">
            <h2 className="mb-3 text-sm font-semibold">训练 Reward</h2>
            <LineChart series={rewSeries} yLabel="reward" />
          </section>
        </>
      )}
    </div>
  )
}
