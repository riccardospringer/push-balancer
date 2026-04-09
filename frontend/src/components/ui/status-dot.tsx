type StatusDotStatus = 'live' | 'off' | 'warn' | 'research'

interface StatusDotProps {
  status?: StatusDotStatus
  size?: number
}

const colorMap: Record<StatusDotStatus, string> = {
  live: 'var(--green)',
  off: 'var(--text-tertiary)',
  warn: 'var(--amber)',
  research: 'var(--green)',
}

export function StatusDot({ status = 'live', size = 7 }: StatusDotProps) {
  const isAnimated = status === 'live' || status === 'research'
  return (
    <span
      style={{
        display: 'inline-block',
        width: size,
        height: size,
        borderRadius: '50%',
        background: colorMap[status],
        animation: isAnimated ? 'pulse 2s infinite' : undefined,
        flexShrink: 0,
      }}
    />
  )
}
