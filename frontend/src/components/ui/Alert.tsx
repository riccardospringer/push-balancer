import { ReactNode } from 'react'

type AlertVariant = 'info' | 'success' | 'warning' | 'error'

interface AlertProps {
  variant?: AlertVariant
  title?: string
  children: ReactNode
  style?: React.CSSProperties
}

const variantMap: Record<AlertVariant, { bg: string; border: string; color: string; icon: string }> = {
  info: { bg: 'var(--blue-bg)', border: 'var(--blue-border)', color: '#3b82f6', icon: 'ℹ' },
  success: { bg: 'var(--green-bg)', border: 'var(--green-border)', color: 'var(--green)', icon: '✓' },
  warning: { bg: 'var(--amber-bg)', border: 'var(--amber-border)', color: 'var(--amber)', icon: '⚠' },
  error: { bg: 'var(--red-bg)', border: 'var(--red-border)', color: 'var(--red)', icon: '✕' },
}

export function Alert({ variant = 'info', title, children, style }: AlertProps) {
  const v = variantMap[variant]
  return (
    <div
      style={{
        background: v.bg,
        border: `1px solid ${v.border}`,
        borderRadius: 'var(--radius)',
        padding: '12px 16px',
        display: 'flex',
        gap: '10px',
        alignItems: 'flex-start',
        ...style,
      }}
    >
      <span style={{ color: v.color, fontWeight: 700, flexShrink: 0, marginTop: '1px' }}>{v.icon}</span>
      <div>
        {title && (
          <div style={{ fontWeight: 600, color: v.color, marginBottom: '2px', fontSize: '13px' }}>{title}</div>
        )}
        <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>{children}</div>
      </div>
    </div>
  )
}
