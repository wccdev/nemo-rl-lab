import ReactECharts from 'echarts-for-react'
import { useMemo } from 'react'

const COLORS = ['#4338ca', '#0891b2', '#059669', '#d97706', '#7c3aed', '#dc2626']

type Series = { name: string; data: [number, number][] }

export function LineChart({
  series,
  yLabel,
  height = 280,
}: {
  series: Series[]
  yLabel?: string
  height?: number
}) {
  const option = useMemo(
    () => ({
      color: COLORS,
      grid: { left: 48, right: 16, top: 24, bottom: 32 },
      tooltip: { trigger: 'axis' },
      legend: series.length > 1 ? { top: 0, textStyle: { color: 'var(--color-muted)' } } : undefined,
      xAxis: {
        type: 'value',
        name: 'step',
        nameTextStyle: { color: 'var(--color-muted)' },
        axisLine: { lineStyle: { color: 'var(--color-border)' } },
        splitLine: { lineStyle: { color: 'var(--color-border)', opacity: 0.5 } },
      },
      yAxis: {
        type: 'value',
        name: yLabel,
        nameTextStyle: { color: 'var(--color-muted)' },
        axisLine: { show: false },
        splitLine: { lineStyle: { color: 'var(--color-border)', opacity: 0.5 } },
      },
      series: series.map((s) => ({
        name: s.name,
        type: 'line',
        showSymbol: true,
        symbolSize: 6,
        data: s.data,
        smooth: false,
      })),
    }),
    [series, yLabel],
  )
  return <ReactECharts option={option} style={{ height }} notMerge lazyUpdate />
}
