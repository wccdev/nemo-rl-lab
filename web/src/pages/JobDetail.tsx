import { useCallback, useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { LineChart } from '../components/LineChart'
import { api, type JobOverview } from '../lib/api'
import { pct } from '../lib/utils'

export function JobDetailPage() {
  const { id } = useParams<{ id: string }>()
  const [ov, setOv] = useState<JobOverview | null>(null)
  const [vidx, setVidx] = useState(-1)
  const [samples, setSamples] = useState<Record<string, unknown>[]>([])
  const [offset, setOffset] = useState(0)
  const [total, setTotal] = useState(0)

  const load = useCallback(() => {
    if (!id) return
    api<JobOverview>(`/api/job?id=${encodeURIComponent(id)}`).then(setOv)
  }, [id])

  useEffect(() => {
    load()
    const t = setInterval(load, 15000)
    return () => clearInterval(t)
  }, [load])

  useEffect(() => {
    if (!ov?.validations.length || !id) return
    const idx = vidx >= 0 ? vidx : ov.validations.length - 1
    setVidx(idx)
    api<{ samples: Record<string, unknown>[]; total: number; offset: number }>(
      `/api/samples?id=${encodeURIComponent(id)}&vidx=${idx}&offset=0&limit=6`,
    ).then((d) => {
      setSamples(d.samples)
      setTotal(d.total)
      setOffset(d.offset + d.samples.length)
    })
  }, [ov, id, vidx])

  async function loadMore() {
    if (!id || vidx < 0) return
    const d = await api<{ samples: Record<string, unknown>[]; total: number }>(
      `/api/samples?id=${encodeURIComponent(id)}&vidx=${vidx}&offset=${offset}&limit=6`,
    )
    setSamples((s) => [...s, ...d.samples])
    setOffset((o) => o + d.samples.length)
    setTotal(d.total)
  }

  if (!ov) return <p className="text-muted">加载中…</p>
  const s = ov.summary

  const rewardSeries = [
    {
      name: 'Avg Reward',
      data: ov.steps.filter((x) => x.avg_reward != null).map((x) => [x.step, x.avg_reward!] as [number, number]),
    },
  ]
  const accSeries = [
    {
      name: 'Accuracy',
      data: ov.validations
        .filter((x) => x.accuracy != null)
        .map((x) => [x.step, x.accuracy!] as [number, number]),
    },
  ]

  return (
    <div className="space-y-6">
      <div>
        <Link to="/jobs" className="text-sm text-muted hover:text-primary">
          ← 作业列表
        </Link>
        <h1 className="mt-2 text-xl font-semibold">{ov.exp}</h1>
        <p className="font-mono text-xs text-muted">{ov.job_id}</p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MiniStat label="最终 Acc" value={pct(s.final_acc as number | null)} />
        <MiniStat label="基线 Acc" value={pct(s.base_acc as number | null)} />
        <MiniStat label="步数" value={`${s.last_step ?? '—'}/${s.total ?? '—'}`} />
        <MiniStat label="模型" value={ov.model?.split('/').pop() ?? '—'} />
      </div>

      <section className="rounded-lg border border-border bg-card p-4">
        <h2 className="mb-3 text-sm font-semibold">训练 Reward</h2>
        <LineChart series={rewardSeries} yLabel="reward" />
      </section>

      <section className="rounded-lg border border-border bg-card p-4">
        <h2 className="mb-3 text-sm font-semibold">验证 Accuracy</h2>
        <LineChart series={accSeries} yLabel="acc" />
      </section>

      <section className="rounded-lg border border-border bg-card p-4">
        <h2 className="mb-3 text-sm font-semibold">验证对话样本</h2>
        <div className="mb-4 flex flex-wrap gap-2">
          {ov.validations.map((v, i) => (
            <button
              key={i}
              type="button"
              onClick={() => setVidx(i)}
              className={`rounded-md border px-3 py-1 text-xs font-mono transition-colors ${
                i === vidx ? 'border-primary bg-primary/10 text-primary' : 'border-border text-muted'
              }`}
            >
              step {v.step}
              {v.accuracy != null ? ` · ${(v.accuracy * 100).toFixed(0)}%` : ''}
            </button>
          ))}
        </div>
        <div className="space-y-3">
          {samples.map((sam, i) => (
            <div key={i} className="rounded-md border border-border bg-background p-3 text-sm">
              <div className="mb-2 font-mono text-xs text-muted">
                reward {(sam.reward as number)?.toFixed?.(3) ?? sam.reward}
              </div>
              <pre className="max-h-48 overflow-auto whitespace-pre-wrap font-mono text-xs leading-relaxed">
                {(sam.assistant as string) || (sam.user as string) || '—'}
              </pre>
            </div>
          ))}
        </div>
        {offset < total && (
          <button
            type="button"
            onClick={loadMore}
            className="mt-4 rounded-md border border-border px-4 py-2 text-sm text-muted transition-colors hover:border-primary hover:text-primary"
          >
            加载更多 ({offset}/{total})
          </button>
        )}
      </section>
    </div>
  )
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <div className="text-xs text-muted">{label}</div>
      <div className="mt-1 font-mono text-lg font-semibold">{value}</div>
    </div>
  )
}
