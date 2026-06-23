import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function pct(x: number | null | undefined) {
  if (x == null) return '—'
  return `${(x * 100).toFixed(1)}%`
}

export function statusColor(st: string) {
  switch (st) {
    case 'RUNNING':
      return 'text-success'
    case 'SUCCEEDED':
      return 'text-primary'
    case 'FAILED':
      return 'text-destructive'
    case 'PENDING':
      return 'text-warning'
    default:
      return 'text-muted'
  }
}
