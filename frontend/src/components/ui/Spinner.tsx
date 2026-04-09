interface SpinnerProps {
  size?: number
  color?: string
}

export function Spinner({ size = 18, color = 'var(--accent)' }: SpinnerProps) {
  return (
    <span
      style={{
        display: 'inline-block',
        width: size,
        height: size,
        border: `2px solid ${color}22`,
        borderTopColor: color,
        borderRadius: '50%',
        animation: 'spin 0.7s linear infinite',
        flexShrink: 0,
      }}
      aria-label="Lädt…"
    />
  )
}
