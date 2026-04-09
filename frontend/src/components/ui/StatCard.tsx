import { ReactNode } from 'react'

interface StatCardProps {
  label: string
  value: ReactNode
  sub?: ReactNode
  accent?: boolean
  style?: React.CSSProperties
}

export function StatCard({ label, value, sub, accent, style }: StatCardProps) {
  return (
    <div
      style={{
        background: accent ? 'var(--accent-light)' : 'var(--white)',
        border: `1px solid ${accent ? '#c7d2fe' : 'var(--border)'}`,
        borderRadius: 'var(--radius)',
        padding: '16px 20px',
        boxShadow: 'var(--shadow-sm)',
        ...style,
      }}
    >
      <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '6px', fontWeight: 500 }}>
        {label}
      </div>
      <div
        style={{
          fontSize: '22px',
          fontWeight: 700,
          color: accent ? 'var(--accent)' : 'var(--text)',
          lineHeight: 1.2,
          fontVariantNumeric: 'tabular-nums',
        }}
      >
        {value}
      </div>
      {sub && (
        <div style={{ fontSize: '12px', color: 'var(--text-tertiary)', marginTop: '4px' }}>{sub}</div>
      )}
    </div>
  )
}
