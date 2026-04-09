import { useMemo } from 'react'
import { usePushStats, useSyncPush } from '@/hooks/useApi'
import {
  Alert,
  Badge,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  FilterChip,
  Spinner,
  StatCard,
  Table,
  TableCell,
  TableHeader,
  TableRow,
} from '@spring-media/editorial-one-ui'
import { useAppStore } from '@/stores/app-store'
import { fmtOR, fmtNum, fmtDateTime } from '@/lib/format'
import type { Push } from '@/types/api'

function orVariant(or: number): 'green' | 'amber' | 'red' {
  if (or >= 0.06) return 'green'
  if (or >= 0.04) return 'amber'
  return 'red'
}

function PushRow({ push }: { push: Push }) {
  const variant = orVariant(push.openRate)
  return (
    <TableRow
      onClick={push.url ? () => window.open(push.url, '_blank') : undefined}
    >
      <TableCell style={{ maxWidth: '360px' }}>
        <div
          style={{
            fontWeight: 500,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {push.title}
        </div>
        <div
          style={{
            fontSize: '12px',
            color: 'var(--text-tertiary)',
            marginTop: '2px',
          }}
        >
          {fmtDateTime(push.sentAt)}
        </div>
      </TableCell>
      <TableCell>
        <Badge variant="blue">{push.channel}</Badge>
      </TableCell>
      <TableCell style={{ fontVariantNumeric: 'tabular-nums' }}>
        {fmtNum(push.recipients)}
      </TableCell>
      <TableCell>
        <Badge variant={variant}>{fmtOR(push.openRate)}</Badge>
      </TableCell>
      <TableCell>
        {push.predictedOR != null ? (
          <span
            style={{
              fontSize: '12px',
              color: 'var(--text-secondary)',
              fontVariantNumeric: 'tabular-nums',
            }}
          >
            {fmtOR(push.predictedOR)}
          </span>
        ) : (
          <span style={{ color: 'var(--text-tertiary)' }}>—</span>
        )}
      </TableCell>
    </TableRow>
  )
}

export function LivePushesPage() {
  const { data, isLoading, error } = usePushStats()
  const { liveChannel, setLiveChannel } = useAppStore()
  const syncMutation = useSyncPush()

  const channels = useMemo(() => {
    return ['alle', ...(data?.channels ?? [])]
  }, [data])

  const filtered = useMemo(() => {
    if (!data?.pushes) return []
    if (liveChannel === 'alle') return data.pushes
    return data.pushes.filter((p) => p.channel === liveChannel)
  }, [data, liveChannel])

  const today = data?.today

  return (
    <div
      style={{
        padding: '16px 24px',
        maxWidth: '1400px',
        margin: '0 auto',
        animation: 'fadeIn 0.2s ease',
      }}
    >
      <div
        style={{
          marginBottom: '16px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <h1 style={{ fontSize: '18px', fontWeight: 700, margin: 0 }}>
          Live Pushes
        </h1>
        <button
          onClick={() => syncMutation.mutate()}
          disabled={syncMutation.isPending}
          style={{
            fontFamily: 'inherit',
            fontSize: '13px',
            padding: '7px 14px',
            borderRadius: '6px',
            border: '1px solid var(--border)',
            background: 'var(--white)',
            cursor: syncMutation.isPending ? 'not-allowed' : 'pointer',
            opacity: syncMutation.isPending ? 0.6 : 1,
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}
        >
          {syncMutation.isPending ? <Spinner size={14} /> : '↓'} Sync
        </button>
      </div>

      {/* KPI Cards */}
      {today && (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
            gap: '12px',
            marginBottom: '16px',
          }}
        >
          <StatCard label="Pushes heute" value={today.count} />
          <StatCard label="Ø OR heute" value={fmtOR(today.avgOR)} accent />
          <StatCard label="Top OR heute" value={fmtOR(today.topOR)} />
          <StatCard label="Empfänger" value={fmtNum(today.recipients)} />
        </div>
      )}

      {/* Channel Filter */}
      <Card style={{ marginBottom: '16px' }}>
        <CardContent
          style={{
            padding: '12px 16px',
            display: 'flex',
            gap: '6px',
            flexWrap: 'wrap',
          }}
        >
          {channels.map((ch) => (
            <FilterChip
              key={ch}
              active={liveChannel === ch}
              onClick={() => setLiveChannel(ch)}
            >
              {ch.charAt(0).toUpperCase() + ch.slice(1)}
            </FilterChip>
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Push-History</CardTitle>
          <span style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>
            {filtered.length} Pushes
          </span>
        </CardHeader>
        {isLoading && (
          <div
            style={{
              padding: '40px',
              display: 'flex',
              justifyContent: 'center',
            }}
          >
            <Spinner size={24} />
          </div>
        )}
        {error && (
          <CardContent>
            <Alert variant="error">
              Push-Daten konnten nicht geladen werden.
            </Alert>
          </CardContent>
        )}
        {!isLoading && !error && (
          <Table>
            <thead>
              <tr>
                <TableHeader>Push</TableHeader>
                <TableHeader>Kanal</TableHeader>
                <TableHeader>Empfänger</TableHeader>
                <TableHeader>Opening Rate</TableHeader>
                <TableHeader>XOR (Prognose)</TableHeader>
              </tr>
            </thead>
            <tbody>
              {filtered.length === 0 ? (
                <TableRow>
                  <TableCell
                    style={{
                      textAlign: 'center',
                      color: 'var(--text-tertiary)',
                      padding: '32px',
                    }}
                    colSpan={5}
                  >
                    Keine Push-Daten vorhanden
                  </TableCell>
                </TableRow>
              ) : (
                filtered.map((p) => <PushRow key={p.id} push={p} />)
              )}
            </tbody>
          </Table>
        )}
      </Card>
    </div>
  )
}
