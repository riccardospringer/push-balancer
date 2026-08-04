import { useMemo, useState } from 'react'
import { usePushStats, useSyncPush } from '@/hooks/use-api'
import {
  Alert,
  Badge,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  FilterChip,
  Select,
  Spinner,
  StatCard,
  Table,
  TableCell,
  TableHeader,
  TableRow,
} from '@spring-media/editorial-one-ui'
import { useLivePushFilterStore } from '@/stores/live-push-filter-store'
import { getApiErrorMessage } from '@/utils/api-errors'
import { fmtDateTime, fmtNum, fmtOR } from '@/utils/format'
import type { Push } from '@/types/api'

const PUSH_CATEGORIES = [
  'alle',
  'news',
  'politik',
  'sport',
  'wirtschaft',
  'unterhaltung',
  'regional',
  'digital',
]

function orVariant(or: number): 'green' | 'amber' | 'red' {
  if (or >= 0.06) return 'green'
  if (or >= 0.04) return 'amber'
  return 'red'
}

function PushRow({ push }: { push: Push }) {
  const variant = orVariant(push.openRate)
  const delta = push.performanceDelta
  const deltaVariant =
    delta == null ? 'default' : delta > 0.002 ? 'green' : delta < -0.002 ? 'red' : 'amber'
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
        <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
          <Badge variant="blue">{push.channel}</Badge>
          {push.category && <Badge variant="default">{push.category}</Badge>}
          {push.type === 'video' && <Badge variant="purple">Video</Badge>}
        </div>
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
      <TableCell>
        {delta != null ? (
          <Badge variant={deltaVariant}>{`${delta > 0 ? '+' : ''}${(delta * 100).toFixed(2)} pp`}</Badge>
        ) : (
          <span style={{ color: 'var(--text-tertiary)' }}>—</span>
        )}
      </TableCell>
    </TableRow>
  )
}

export function LivePushesPage() {
  const [days, setDays] = useState('7')
  const [sort, setSort] = useState<
    'sentAt' | 'openRate' | 'performanceDelta' | 'recipients'
  >('sentAt')
  const [category, setCategory] = useState('alle')
  const { data, isLoading, error } = usePushStats({
    days: Number(days),
    sort,
    category: category === 'alle' ? undefined : category,
    limit: 500,
  })
  const { liveChannel, setLiveChannel } = useLivePushFilterStore()
  const syncMutation = useSyncPush()

  const channels = useMemo(() => {
    return ['alle', ...(data?.channels ?? [])]
  }, [data])

  const filtered = useMemo(() => {
    if (!data?.pushes) return []
    if (liveChannel === 'alle') return data.pushes
    return data.pushes.filter((p) => p.channel === liveChannel)
  }, [data, liveChannel])

  const avgDelta = useMemo(() => {
    const deltas = filtered
      .map((push) => push.performanceDelta)
      .filter((value): value is number => value != null)
    if (!deltas.length) return null
    return deltas.reduce((sum, value) => sum + value, 0) / deltas.length
  }, [filtered])

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
            gap: '12px',
            flexWrap: 'wrap',
            alignItems: 'center',
          }}
        >
          <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
            {channels.map((ch) => (
              <FilterChip
                key={ch}
                active={liveChannel === ch}
                onClick={() => setLiveChannel(ch)}
              >
                {ch.charAt(0).toUpperCase() + ch.slice(1)}
              </FilterChip>
            ))}
          </div>
          <Select
            label="Zeitraum"
            value={days}
            onChange={(e) => setDays(e.target.value)}
            options={[
              { value: '7', label: 'Letzte 7 Tage' },
              { value: '30', label: 'Letzte 30 Tage' },
              { value: '90', label: 'Letzte 90 Tage' },
            ]}
            style={{ width: '170px' }}
          />
          <Select
            label="Sortierung"
            value={sort}
            onChange={(e) =>
              setSort(
                e.target.value as
                  | 'sentAt'
                  | 'openRate'
                  | 'performanceDelta'
                  | 'recipients',
              )
            }
            options={[
              { value: 'sentAt', label: 'Neueste zuerst' },
              { value: 'openRate', label: 'Beste OR' },
              { value: 'performanceDelta', label: 'Über Prognose' },
              { value: 'recipients', label: 'Größte Reichweite' },
            ]}
            style={{ width: '180px' }}
          />
          <Select
            label="Ressort"
            value={category}
            onChange={(e) => setCategory(e.target.value)}
            options={PUSH_CATEGORIES.map((value) => ({
              value,
              label: value === 'alle' ? 'Alle Ressorts' : value,
            }))}
            style={{ width: '180px' }}
          />
          {avgDelta != null && (
            <span
              style={{
                marginLeft: 'auto',
                fontSize: '12px',
                color: 'var(--text-secondary)',
              }}
            >
              Ø Delta vs. Prognose:{' '}
              <strong>{`${avgDelta > 0 ? '+' : ''}${(avgDelta * 100).toFixed(2)} pp`}</strong>
            </span>
          )}
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
              {getApiErrorMessage(
                error,
                'Push-Daten konnten nicht geladen werden.',
              )}
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
                <TableHeader>Delta</TableHeader>
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
                    colSpan={6}
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
