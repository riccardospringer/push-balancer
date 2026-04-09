import { ReactNode } from 'react'

interface FilterChipProps {
  active?: boolean
  onClick?: () => void
  children: ReactNode
  style?: React.CSSProperties
}

export function FilterChip({ active, onClick, children, style }: FilterChipProps) {
  return (
    <button
      onClick={onClick}
      style={{
        fontFamily: 'inherit',
        fontSize: '12px',
        fontWeight: active ? 600 : 500,
        padding: '4px 12px',
        borderRadius: '20px',
        border: `1px solid ${active ? 'var(--accent)' : 'var(--border)'}`,
        background: active ? 'var(--accent-light)' : 'var(--white)',
        color: active ? 'var(--accent)' : 'var(--text-secondary)',
        cursor: 'pointer',
        transition: 'all 0.15s',
        whiteSpace: 'nowrap',
        ...style,
      }}
    >
      {children}
    </button>
  )
}
