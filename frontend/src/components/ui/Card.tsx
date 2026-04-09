import type { ReactNode } from 'react'

interface CardProps {
  children: ReactNode
  style?: React.CSSProperties
  className?: string
}

export function Card({ children, style }: CardProps) {
  return (
    <div
      style={{
        background: 'var(--white)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius)',
        boxShadow: 'var(--shadow-sm)',
        overflow: 'hidden',
        ...style,
      }}
    >
      {children}
    </div>
  )
}

export function CardHeader({ children, style }: CardProps) {
  return (
    <div
      style={{
        padding: '16px 20px',
        borderBottom: '1px solid var(--border-light)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '12px',
        ...style,
      }}
    >
      {children}
    </div>
  )
}

export function CardTitle({ children, style }: CardProps) {
  return (
    <h3
      style={{
        margin: 0,
        fontSize: '14px',
        fontWeight: 600,
        color: 'var(--text)',
        ...style,
      }}
    >
      {children}
    </h3>
  )
}

export function CardContent({ children, style }: CardProps) {
  return <div style={{ padding: '16px 20px', ...style }}>{children}</div>
}
