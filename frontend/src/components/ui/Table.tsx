import type { ReactNode, TdHTMLAttributes, ThHTMLAttributes } from 'react'

interface TableProps {
  children: ReactNode
  style?: React.CSSProperties
}

export function Table({ children, style }: TableProps) {
  return (
    <div style={{ overflowX: 'auto', ...style }}>
      <table
        style={{
          width: '100%',
          borderCollapse: 'collapse',
          fontSize: '13px',
        }}
      >
        {children}
      </table>
    </div>
  )
}

export function TableHeader({
  children,
  style,
  ...rest
}: ThHTMLAttributes<HTMLTableCellElement> & { children?: ReactNode }) {
  return (
    <th
      style={{
        padding: '10px 14px',
        textAlign: 'left',
        fontSize: '11px',
        fontWeight: 600,
        color: 'var(--text-secondary)',
        textTransform: 'uppercase',
        letterSpacing: '0.04em',
        borderBottom: '1px solid var(--border)',
        whiteSpace: 'nowrap',
        background: 'var(--bg)',
        ...style,
      }}
      {...rest}
    >
      {children}
    </th>
  )
}

export function TableRow({
  children,
  style,
  onClick,
  title,
}: {
  children: ReactNode
  style?: React.CSSProperties
  onClick?: () => void
  title?: string
}) {
  return (
    <tr
      onClick={onClick}
      title={title}
      style={{
        borderBottom: '1px solid var(--border-light)',
        transition: 'background 0.1s',
        cursor: onClick ? 'pointer' : undefined,
        ...style,
      }}
      onMouseEnter={onClick ? (e) => ((e.currentTarget as HTMLElement).style.background = 'var(--bg)') : undefined}
      onMouseLeave={onClick ? (e) => ((e.currentTarget as HTMLElement).style.background = '') : undefined}
    >
      {children}
    </tr>
  )
}

export function TableCell({
  children,
  style,
  ...rest
}: TdHTMLAttributes<HTMLTableCellElement> & { children?: ReactNode }) {
  return (
    <td
      style={{
        padding: '10px 14px',
        color: 'var(--text)',
        verticalAlign: 'middle',
        ...style,
      }}
      {...rest}
    >
      {children}
    </td>
  )
}
