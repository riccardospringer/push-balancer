import { Navigate, createBrowserRouter } from 'react-router-dom'
import { MainLayout } from '@/components/main-layout'
import { AnalysePage } from '@/pages/AnalysePage'
import { ForschungPage } from '@/pages/ForschungPage'
import { KandidatenPage } from '@/pages/KandidatenPage'
import { KonkurrenzPage } from '@/pages/KonkurrenzPage'
import { LivePushesPage } from '@/pages/LivePushesPage'
import { TagesplanPage } from '@/pages/TagesplanPage'
import { NotFoundPage } from './not-found'

export const appRouter = createBrowserRouter(
  [
    {
      path: '/',
      element: <MainLayout />,
      children: [
        { index: true, element: <Navigate to="/kandidaten" replace /> },
        { path: 'kandidaten', element: <KandidatenPage /> },
        { path: 'live', element: <LivePushesPage /> },
        { path: 'analyse', element: <AnalysePage /> },
        { path: 'konkurrenz', element: <KonkurrenzPage /> },
        { path: 'forschung', element: <ForschungPage /> },
        { path: 'tagesplan', element: <TagesplanPage /> },
        { path: '*', element: <NotFoundPage /> },
      ],
    },
  ],
  {
    basename: import.meta.env.BASE_URL,
  },
)
