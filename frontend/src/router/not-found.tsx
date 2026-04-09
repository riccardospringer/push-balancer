import { Link } from 'react-router-dom'
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@spring-media/editorial-one-ui'

export function NotFoundPage() {
  return (
    <main
      style={{
        minHeight: 'calc(100vh - 56px)',
        display: 'grid',
        placeItems: 'center',
        padding: '32px 20px',
      }}
    >
      <Card style={{ maxWidth: '520px', width: '100%' }}>
        <CardHeader>
          <CardTitle>Seite nicht gefunden</CardTitle>
        </CardHeader>
        <CardContent>
          <p style={{ margin: '0 0 16px', color: 'var(--text-secondary)' }}>
            Die angeforderte Ansicht existiert nicht oder wurde verschoben.
          </p>
          <Link
            to="/kandidaten"
            style={{
              color: 'var(--accent)',
              fontWeight: 600,
              textDecoration: 'none',
            }}
          >
            Zur Kandidaten-Ansicht
          </Link>
        </CardContent>
      </Card>
    </main>
  )
}
