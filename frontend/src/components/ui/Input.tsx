import { forwardRef } from 'react'
import type { InputHTMLAttributes } from 'react'

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ label, style, ...rest }, ref) => {
    const input = (
      <input
        ref={ref}
        style={{
          fontFamily: 'inherit',
          fontSize: '13px',
          padding: '7px 12px',
          borderRadius: '6px',
          border: '1px solid var(--border)',
          background: 'var(--white)',
          color: 'var(--text)',
          outline: 'none',
          width: '100%',
          transition: 'border-color 0.15s',
          ...style,
        }}
        onFocus={(e) => {
          e.currentTarget.style.borderColor = 'var(--accent)'
          rest.onFocus?.(e)
        }}
        onBlur={(e) => {
          e.currentTarget.style.borderColor = 'var(--border)'
          rest.onBlur?.(e)
        }}
        {...rest}
      />
    )

    if (!label) return input

    return (
      <label style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
        <span
          style={{
            fontSize: '12px',
            fontWeight: 500,
            color: 'var(--text-secondary)',
          }}
        >
          {label}
        </span>
        {input}
      </label>
    )
  },
)
Input.displayName = 'Input'
