import { ButtonHTMLAttributes, forwardRef, ReactNode } from 'react'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'primary' | 'ghost' | 'danger'
  size?: 'sm' | 'md'
  children: ReactNode
}

const styles: Record<string, React.CSSProperties> = {
  base: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
    fontFamily: 'inherit',
    fontWeight: 500,
    borderRadius: '6px',
    border: '1px solid var(--border)',
    cursor: 'pointer',
    transition: 'all 0.15s',
    whiteSpace: 'nowrap',
  },
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ variant = 'default', size = 'md', style, children, disabled, ...rest }, ref) => {
    const variantStyle: React.CSSProperties =
      variant === 'primary'
        ? { background: 'var(--accent)', color: '#fff', borderColor: 'var(--accent)' }
        : variant === 'ghost'
          ? { background: 'transparent', borderColor: 'transparent', color: 'var(--text-secondary)' }
          : variant === 'danger'
            ? { background: 'var(--red-bg)', color: 'var(--red)', borderColor: 'var(--red-border)' }
            : { background: 'var(--white)', color: 'var(--text)' }

    const sizeStyle: React.CSSProperties =
      size === 'sm'
        ? { fontSize: '12px', padding: '4px 10px' }
        : { fontSize: '13px', padding: '7px 14px' }

    return (
      <button
        ref={ref}
        style={{
          ...styles.base,
          ...variantStyle,
          ...sizeStyle,
          opacity: disabled ? 0.5 : 1,
          cursor: disabled ? 'not-allowed' : 'pointer',
          ...style,
        }}
        disabled={disabled}
        {...rest}
      >
        {children}
      </button>
    )
  },
)
Button.displayName = 'Button'
