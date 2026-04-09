import { createContext, useContext, ReactNode } from 'react'

interface TabsContextValue {
  active: string
  setActive: (v: string) => void
}

const TabsCtx = createContext<TabsContextValue>({ active: '', setActive: () => {} })

interface TabsProps {
  value: string
  onValueChange: (v: string) => void
  children: ReactNode
  style?: React.CSSProperties
}

export function Tabs({ value, onValueChange, children, style }: TabsProps) {
  return (
    <TabsCtx.Provider value={{ active: value, setActive: onValueChange }}>
      <div style={style}>{children}</div>
    </TabsCtx.Provider>
  )
}

export function TabsList({ children, style }: { children: ReactNode; style?: React.CSSProperties }) {
  return (
    <div
      style={{
        display: 'flex',
        gap: '2px',
        borderBottom: '1px solid var(--border)',
        ...style,
      }}
    >
      {children}
    </div>
  )
}

export function TabsTrigger({ value, children }: { value: string; children: ReactNode }) {
  const { active, setActive } = useContext(TabsCtx)
  const isActive = active === value
  return (
    <button
      onClick={() => setActive(value)}
      style={{
        fontFamily: 'inherit',
        fontSize: '13px',
        fontWeight: isActive ? 600 : 500,
        padding: '10px 16px',
        border: 'none',
        borderBottom: `2px solid ${isActive ? 'var(--accent)' : 'transparent'}`,
        background: 'none',
        color: isActive ? 'var(--accent)' : 'var(--text-secondary)',
        cursor: 'pointer',
        transition: 'all 0.15s',
        marginBottom: '-1px',
      }}
    >
      {children}
    </button>
  )
}

export function TabsContent({ value, children }: { value: string; children: ReactNode }) {
  const { active } = useContext(TabsCtx)
  if (active !== value) return null
  return <div style={{ animation: 'fadeIn 0.2s ease' }}>{children}</div>
}
