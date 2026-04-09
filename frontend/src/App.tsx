import { Routes, Route, Navigate } from 'react-router-dom'
import { TopNav } from '@/components/layout/TopNav'
import { KandidatenPage } from '@/pages/KandidatenPage'
import { LivePushesPage } from '@/pages/LivePushesPage'
import { AnalysePage } from '@/pages/AnalysePage'
import { KonkurrenzPage } from '@/pages/KonkurrenzPage'
import { ForschungPage } from '@/pages/ForschungPage'
import { TagesplanPage } from '@/pages/TagesplanPage'

export default function App() {
  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)' }}>
      <TopNav />
      <Routes>
        <Route path="/" element={<Navigate to="/kandidaten" replace />} />
        <Route path="/kandidaten" element={<KandidatenPage />} />
        <Route path="/live" element={<LivePushesPage />} />
        <Route path="/analyse" element={<AnalysePage />} />
        <Route path="/konkurrenz" element={<KonkurrenzPage />} />
        <Route path="/forschung" element={<ForschungPage />} />
        <Route path="/tagesplan" element={<TagesplanPage />} />
      </Routes>
    </div>
  )
}
