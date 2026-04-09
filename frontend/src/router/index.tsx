import { Navigate, createBrowserRouter } from 'react-router-dom'
import { MainLayout } from '@/components/main-layout'
import { AnalysePage } from '@/pages/analyse'
import { ForschungPage } from '@/pages/forschung'
import { KandidatenPage } from '@/pages/kandidaten'
import { KonkurrenzPage } from '@/pages/konkurrenz'
import { LivePushesPage } from '@/pages/live-pushes'
import { TagesplanPage } from '@/pages/tagesplan'
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
