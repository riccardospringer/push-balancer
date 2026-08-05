import { useMemo } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import { usePushStats } from '@/hooks/use-api'
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
import { getApiErrorMessage } from '@/utils/api-errors'
import { fmtDateTime, fmtNum, fmtOR } from '@/utils/format'

const HOURS = Array.from({ length: 24 }, (_, hour) => hour)

export function AnalysePage() {
  const { data, isLoading, error } = usePushStats({
    limit: 2000,
    days: 30,
    sort: 'sentAt',
  })

  const analysis = useMemo(() => {
    const buckets = HOURS.map((hour) => ({
      hour,
      pushes: 0,
      recipients: 0,
      opened: 0,
      openRate: 0,
    }))

    for (const push of data?.pushes ?? []) {
      const sentAt = new Date(push.sentAt)
      if (Number.isNaN(sentAt.getTime())) continue
      const bucket = buckets[sentAt.getHours()]
      bucket.pushes += 1
      bucket.recipients += push.recipients
      bucket.opened += push.opened
    }

    for (const bucket of buckets) {
      bucket.openRate =
        bucket.recipients > 0 ? bucket.opened / bucket.recipients : 0
    }

    const populated = buckets.filter((bucket) => bucket.pushes > 0)
    const peakHour = populated.reduce<(typeof buckets)[number] | undefined>(
      (peak, bucket) => (!peak || bucket.opened > peak.opened ? bucket : peak),
      undefined,
    )
    const totalRecipients = populated.reduce(
      (sum, bucket) => sum + bucket.recipients,
      0,
    )
    const totalOpened = populated.reduce(
      (sum, bucket) => sum + bucket.opened,
      0,
    )
    const topPushes = [...(data?.pushes ?? [])]
      .sort((a, b) => b.opened - a.opened)
      .slice(0, 10)

    return {
      chartData: populated.map((bucket) => ({
        hour: `${String(bucket.hour).padStart(2, '0')}h`,
        Pushes: bucket.pushes,
        'Opening Rate': Number((bucket.openRate * 100).toFixed(2)),
      })),
      peakHour,
      totalRecipients,
      totalOpened,
      openRate: totalRecipients > 0 ? totalOpened / totalRecipients : 0,
      topPushes,
    }
  }, [data])

  const summaryCards = [
    {
      label: 'Peak-Stunde',
      value: analysis.peakHour
        ? `${String(analysis.peakHour.hour).padStart(2, '0')}:00`
        : '—',
    },
    { label: 'Pushes (30 Tage)', value: fmtNum(data?.total ?? 0) },
    { label: 'Empfänger', value: fmtNum(analysis.totalRecipients) },
    { label: 'Ø Opening Rate', value: fmtOR(analysis.openRate) },
  ]

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
      <div style={{ marginBottom: '20px' }}>
        <h1 style={{ fontSize: '18px', fontWeight: 700, margin: 0 }}>
          Push-Analyse
        </h1>
        <p
          style={{
            fontSize: '12px',
            color: 'var(--text-secondary)',
            margin: '2px 0 0',
          }}
        >
          Push-Performance · letzte 30 Tage
        </p>
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
          {getApiErrorMessage(
            error,
            'Push-Daten konnten nicht geladen werden.',
          )}
        </Alert>
      )}

      {data && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
              gap: '12px',
            }}
          >
            {summaryCards.map((card) => (
              <div
                key={card.label}
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
                  {card.label}
                </div>
                <div style={{ fontSize: '22px', fontWeight: 700 }}>
                  {card.value}
                </div>
              </div>
            ))}
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Stündliche Push-Performance</CardTitle>
            </CardHeader>
            <CardContent>
              {analysis.chartData.length > 0 ? (
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={analysis.chartData} barSize={12}>
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
                      yAxisId="pushes"
                      allowDecimals={false}
                      tick={{ fontSize: 11 }}
                      tickLine={false}
                      axisLine={false}
                      width={36}
                    />
                    <YAxis
                      yAxisId="or"
                      orientation="right"
                      tick={{ fontSize: 11 }}
                      tickLine={false}
                      axisLine={false}
                      width={42}
                      unit="%"
                    />
                    <Tooltip
                      contentStyle={{ fontSize: '12px', borderRadius: '6px' }}
                    />
                    <Bar
                      yAxisId="pushes"
                      dataKey="Pushes"
                      fill="var(--accent)"
                      radius={[3, 3, 0, 0]}
                    />
                    <Bar
                      yAxisId="or"
                      dataKey="Opening Rate"
                      fill="var(--green)"
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
                  Noch keine persistierten Push-Daten vorhanden
                </div>
              )}
            </CardContent>
          </Card>

          {analysis.topPushes.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Stärkste Pushes (nach Öffnungen)</CardTitle>
              </CardHeader>
              <Table>
                <thead>
                  <tr>
                    <TableHeader>Push</TableHeader>
                    <TableHeader>Versand</TableHeader>
                    <TableHeader style={{ textAlign: 'right' }}>
                      Empfänger
                    </TableHeader>
                    <TableHeader style={{ textAlign: 'right' }}>
                      Öffnungen
                    </TableHeader>
                    <TableHeader style={{ textAlign: 'right' }}>OR</TableHeader>
                  </tr>
                </thead>
                <tbody>
                  {analysis.topPushes.map((push) => (
                    <TableRow key={push.id}>
                      <TableCell>
                        <div
                          style={{
                            fontWeight: 500,
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                            maxWidth: '480px',
                          }}
                        >
                          {push.title}
                        </div>
                      </TableCell>
                      <TableCell>{fmtDateTime(push.sentAt)}</TableCell>
                      <TableCell style={{ textAlign: 'right' }}>
                        {fmtNum(push.recipients)}
                      </TableCell>
                      <TableCell style={{ textAlign: 'right' }}>
                        {fmtNum(push.opened)}
                      </TableCell>
                      <TableCell style={{ textAlign: 'right' }}>
                        {fmtOR(push.openRate)}
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
