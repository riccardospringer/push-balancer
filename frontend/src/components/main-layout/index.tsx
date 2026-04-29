import { Outlet } from 'react-router-dom'
import { TopNav } from '@/components/top-nav'
import { PushAlarmBanner } from '@/components/push-alarm-banner'

export function MainLayout() {
  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)' }}>
      <TopNav />
      <PushAlarmBanner />
      <Outlet />
    </div>
  )
}
