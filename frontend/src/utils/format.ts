export function fmtOR(v: number | undefined): string {
  if (v == null) return '—'
  return `${(v * 100).toFixed(2)} %`
}

export function fmtPct(v: number | undefined): string {
  if (v == null) return '—'
  return `${(v * 100).toFixed(1)} %`
}

export function fmtScore(v: number): string {
  return v.toFixed(1)
}

export function fmtDate(iso: string): string {
  const d = new Date(iso)
  return d.toLocaleDateString('de-DE', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
  })
}

export function fmtTime(iso: string): string {
  const d = new Date(iso)
  return d.toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit' })
}

export function fmtDateTime(iso: string): string {
  const d = new Date(iso)
  return d.toLocaleString('de-DE', {
    day: '2-digit',
    month: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function fmtNum(v: number, digits = 0): string {
  return v.toLocaleString('de-DE', { maximumFractionDigits: digits })
}

export function scoreColor(score: number): string {
  if (score >= 75) return 'var(--green)'
  if (score >= 55) return 'var(--amber)'
  return 'var(--red)'
}

export function scoreVariant(score: number): 'green' | 'amber' | 'red' {
  if (score >= 75) return 'green'
  if (score >= 55) return 'amber'
  return 'red'
}

export function orColor(or: number): string {
  if (or >= 0.06) return 'var(--green)'
  if (or >= 0.04) return 'var(--amber)'
  return 'var(--red)'
}
