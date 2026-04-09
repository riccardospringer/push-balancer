import { useNavigate, useLocation } from 'react-router-dom'
import { StatusDot } from '@/components/ui/StatusDot'
import { useHealth } from '@/hooks/useApi'

const TABS = [
  { path: '/kandidaten', label: 'Kandidaten' },
  { path: '/live', label: 'Live Pushes' },
  { path: '/analyse', label: 'Analyse' },
  { path: '/konkurrenz', label: 'Konkurrenz' },
  { path: '/forschung', label: 'Forschung' },
  { path: '/tagesplan', label: 'Tagesplan' },
]

export function TopNav() {
  const navigate = useNavigate()
  const location = useLocation()
  const { data: health } = useHealth()

  const isLive = health?.status === 'healthy'

  return (
    <nav
      style={{
        background: 'var(--white)',
        borderBottom: '1px solid var(--border)',
        padding: '0 24px',
        height: '56px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        position: 'sticky',
        top: 0,
        zIndex: 100,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '24px' }}>
        <div
          style={{
            fontSize: '16px',
            fontWeight: 700,
            color: 'var(--accent)',
            letterSpacing: '-0.3px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            cursor: 'pointer',
          }}
          onClick={() => navigate('/kandidaten')}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path
              d="M18 8A6 6 0 006 8c0 7-3 9-3 9h18s-3-2-3-9"
              stroke="#4f46e5"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <path
              d="M13.73 21a2 2 0 01-3.46 0"
              stroke="#4f46e5"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <circle cx="18" cy="4" r="3" fill="#dc2626" />
          </svg>
          Push Balancer
        </div>

        <div style={{ display: 'flex', height: '56px', alignItems: 'stretch' }}>
          {TABS.map((tab) => {
            const active = location.pathname === tab.path
            return (
              <button
                key={tab.path}
                onClick={() => navigate(tab.path)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  padding: '0 16px',
                  fontSize: '14px',
                  color: active ? 'var(--accent)' : 'var(--text-secondary)',
                  background: 'none',
                  border: 'none',
                  borderBottom: `2px solid ${active ? 'var(--accent)' : 'transparent'}`,
                  cursor: 'pointer',
                  fontWeight: active ? 600 : 400,
                  transition: 'color 0.15s',
                  fontFamily: 'inherit',
                  whiteSpace: 'nowrap',
                }}
              >
                {tab.label}
              </button>
            )
          })}
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <StatusDot status={isLive ? 'live' : 'off'} />
        <span style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>
          {health == null ? 'Verbinde…' : isLive ? 'Live' : 'Offline'}
        </span>
      </div>
    </nav>
  )
}
