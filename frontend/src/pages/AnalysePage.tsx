import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import { useAdobeTraffic } from '@/hooks/useApi'
import {
  Alert,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Spinner,
  Table,
  TableCell,
  TableHeader,
  TableRow,
} from '@spring-media/editorial-one-ui'
import { fmtNum, fmtDateTime } from '@/lib/format'

export function AnalysePage() {
  const { data, isLoading, error } = useAdobeTraffic()

  const chartData =
    data?.hourly?.map((h) => ({
      hour: `${String(h.hour).padStart(2, '0')}h`,
      Pageviews: h.pageviews,
      Visitors: h.visitors,
    })) ?? []

  const peakHour = data?.hourly?.reduce(
    (max, h) => (h.pageviews > max.pageviews ? h : max),
    { hour: 0, pageviews: 0, visitors: 0 },
  )

  return (
    <div
      style={{
        padding: '16px 24px',
        maxWidth: '1400px',
        margin: '0 auto',
        background: 'var(--bg)',
        animation: 'fadeIn 0.2s ease',
      }}
    >
      <div
        style={{
          marginBottom: '20px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <div>
          <h1 style={{ fontSize: '18px', fontWeight: 700, margin: 0 }}>
            Traffic-Analyse
          </h1>
          {data?.fetchedAt && (
            <p
              style={{
                fontSize: '12px',
                color: 'var(--text-secondary)',
                margin: '2px 0 0',
              }}
            >
              Adobe Analytics · Abgerufen: {fmtDateTime(data.fetchedAt)}
            </p>
          )}
        </div>
      </div>

      {isLoading && (
        <div
          style={{ padding: '60px', display: 'flex', justifyContent: 'center' }}
        >
          <Spinner size={28} />
        </div>
      )}

      {error && (
        <Alert variant="error">
          Adobe Analytics konnte nicht geladen werden. Sind die Credentials
          konfiguriert?
        </Alert>
      )}

      {data && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Peak summary */}
          {peakHour && peakHour.pageviews > 0 && (
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
                gap: '12px',
              }}
            >
              <div
                style={{
                  background: 'var(--white)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius)',
                  padding: '16px',
                  boxShadow: 'var(--shadow-sm)',
                }}
              >
                <div
                  style={{
                    fontSize: '12px',
                    color: 'var(--text-secondary)',
                    marginBottom: '4px',
                  }}
                >
                  Peak-Stunde
                </div>
                <div style={{ fontSize: '22px', fontWeight: 700 }}>
                  {String(peakHour.hour).padStart(2, '0')}:00
                </div>
              </div>
              <div
                style={{
                  background: 'var(--accent-light)',
                  border: '1px solid #c7d2fe',
                  borderRadius: 'var(--radius)',
                  padding: '16px',
                  boxShadow: 'var(--shadow-sm)',
                }}
              >
                <div
                  style={{
                    fontSize: '12px',
                    color: 'var(--text-secondary)',
                    marginBottom: '4px',
                  }}
                >
                  Peak Pageviews
                </div>
                <div
                  style={{
                    fontSize: '22px',
                    fontWeight: 700,
                    color: 'var(--accent)',
                  }}
                >
                  {fmtNum(peakHour.pageviews)}
                </div>
              </div>
              <div
                style={{
                  background: 'var(--white)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius)',
                  padding: '16px',
                  boxShadow: 'var(--shadow-sm)',
                }}
              >
                <div
                  style={{
                    fontSize: '12px',
                    color: 'var(--text-secondary)',
                    marginBottom: '4px',
                  }}
                >
                  Peak Visitors
                </div>
                <div style={{ fontSize: '22px', fontWeight: 700 }}>
                  {fmtNum(peakHour.visitors)}
                </div>
              </div>
              <div
                style={{
                  background: 'var(--white)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius)',
                  padding: '16px',
                  boxShadow: 'var(--shadow-sm)',
                }}
              >
                <div
                  style={{
                    fontSize: '12px',
                    color: 'var(--text-secondary)',
                    marginBottom: '4px',
                  }}
                >
                  Gesamt PV
                </div>
                <div style={{ fontSize: '22px', fontWeight: 700 }}>
                  {fmtNum(data.hourly.reduce((s, h) => s + h.pageviews, 0))}
                </div>
              </div>
            </div>
          )}

          {/* Hourly Chart */}
          <Card>
            <CardHeader>
              <CardTitle>Stündlicher Traffic</CardTitle>
            </CardHeader>
            <CardContent>
              {chartData.length > 0 ? (
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={chartData} barSize={14}>
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="var(--border)"
                      vertical={false}
                    />
                    <XAxis
                      dataKey="hour"
                      tick={{ fontSize: 11 }}
                      tickLine={false}
                      axisLine={false}
                    />
                    <YAxis
                      yAxisId="pv"
                      tick={{ fontSize: 11 }}
                      tickLine={false}
                      axisLine={false}
                      width={50}
                      tickFormatter={(v) => fmtNum(v)}
                    />
                    <Tooltip
                      formatter={(v: number) => fmtNum(v)}
                      contentStyle={{ fontSize: '12px', borderRadius: '6px' }}
                    />
                    <Bar
                      yAxisId="pv"
                      dataKey="Pageviews"
                      fill="var(--accent)"
                      radius={[3, 3, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div
                  style={{
                    height: '200px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'var(--text-tertiary)',
                    fontSize: '13px',
                  }}
                >
                  Keine Stundendaten vorhanden
                </div>
              )}
            </CardContent>
          </Card>

          {/* Top Articles */}
          {data.topArticles && data.topArticles.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Top-Artikel (nach Pageviews)</CardTitle>
              </CardHeader>
              <Table>
                <thead>
                  <tr>
                    <TableHeader>#</TableHeader>
                    <TableHeader>Artikel</TableHeader>
                    <TableHeader style={{ textAlign: 'right' }}>
                      Pageviews
                    </TableHeader>
                  </tr>
                </thead>
                <tbody>
                  {data.topArticles.map((a, i) => (
                    <TableRow
                      key={i}
                      onClick={() => window.open(a.url, '_blank')}
                    >
                      <TableCell
                        style={{ color: 'var(--text-tertiary)', width: '32px' }}
                      >
                        {i + 1}
                      </TableCell>
                      <TableCell>
                        <div
                          style={{
                            fontWeight: 500,
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                            maxWidth: '500px',
                          }}
                        >
                          {a.title}
                        </div>
                      </TableCell>
                      <TableCell
                        style={{
                          textAlign: 'right',
                          fontVariantNumeric: 'tabular-nums',
                          fontWeight: 600,
                        }}
                      >
                        {fmtNum(a.pageviews)}
                      </TableCell>
                    </TableRow>
                  ))}
                </tbody>
              </Table>
            </Card>
          )}
        </div>
      )}
    </div>
  )
}
