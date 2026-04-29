import { Outlet } from 'react-router-dom'
import { TopNav } from '@/components/top-nav'

export function MainLayout() {
  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)' }}>
      <TopNav />
      <Outlet />
    </div>
  )
}
