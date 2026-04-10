import type { ReactNode } from 'react'

type BadgeVariant = 'default' | 'green' | 'red' | 'amber' | 'blue' | 'purple'

interface BadgeProps {
  variant?: BadgeVariant
  children: ReactNode
  style?: React.CSSProperties
}

const variantMap: Record<BadgeVariant, React.CSSProperties> = {
  default: {
    background: 'var(--border-light)',
    color: 'var(--text-secondary)',
  },
  green: {
    background: 'var(--green-bg)',
    color: 'var(--green)',
    border: '1px solid var(--green-border)',
  },
  red: {
    background: 'var(--red-bg)',
    color: 'var(--red)',
    border: '1px solid var(--red-border)',
  },
  amber: {
    background: 'var(--amber-bg)',
    color: 'var(--amber)',
    border: '1px solid var(--amber-border)',
  },
  blue: {
    background: 'var(--blue-bg)',
    color: '#3b82f6',
    border: '1px solid var(--blue-border)',
  },
  purple: {
    background: 'var(--accent-light)',
    color: 'var(--accent)',
    border: '1px solid #c7d2fe',
  },
}

export function Badge({ variant = 'default', children, style }: BadgeProps) {
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        fontSize: '11px',
        fontWeight: 600,
        padding: '2px 7px',
        borderRadius: '4px',
        letterSpacing: '0.01em',
        ...variantMap[variant],
        ...style,
      }}
    >
      {children}
    </span>
  )
}
