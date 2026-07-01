import { useMemo, useState } from 'react'
import {
  Alert,
  Badge,
  Button,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Input,
  Select,
  Spinner,
  StatCard,
  Table,
  TableCell,
  TableHeader,
  TableRow,
} from '@spring-media/editorial-one-ui'
import {
  usePersistTeamsDailyPlan,
  useTeamsRecommendations,
} from '@/hooks/use-api'
import { getApiErrorMessage } from '@/utils/api-errors'
import { fmtDateTime, fmtNum } from '@/utils/format'
import type { TeamsRecommendationRecord } from '@/types/api'

type RecommendationSource = 'teams_alert' | 'daily_plan'
type StatusFilter = 'alle' | 'sent' | 'failed' | 'blocked'

const STATUS_OPTIONS = [
  { value: 'alle', label: 'Alle Status' },
  { value: 'sent', label: 'Gesendet' },
  { value: 'failed', label: 'Fehler' },
  { value: 'blocked', label: 'Blockiert' },
]

const SOURCE_OPTIONS = [
  { value: 'teams_alert', label: 'Echte Teams-Historie' },
  { value: 'daily_plan', label: 'Tagesplan-Snapshots' },
]

function formatStoredOR(value: number, label?: string): string {
  if (label) return label
  if (!value) return 'keine Prognose'
  return `${value.toLocaleString('de-DE', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })} % OR`
}

function formatScore(value: number): string {
  if (!value) return '—'
  return value.toLocaleString('de-DE', {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  })
}

function statusVariant(status: string, sendStatus: string) {
  const combined = `${status} ${sendStatus}`.toLowerCase()
  if (combined.includes('sent')) return 'green'
  if (combined.includes('failed') || combined.includes('fehler')) return 'red'
  if (combined.includes('blocked')) return 'amber'
  if (combined.includes('fix')) return 'green'
  if (combined.includes('optional')) return 'blue'
  if (combined.includes('ruhiger')) return 'amber'
  return 'default'
}

function statusLabel(item: TeamsRecommendationRecord): string {
  if (item.sendStatus === 'planned') return item.status || 'geplant'
  if (item.sendStatus) return item.sendStatus
  return item.status || 'offen'
}

function typeLabel(type: string): string {
  if (type === 'teams_alert') return 'Teams Alert'
  if (type === 'daily_plan') return 'Tagesplan'
  return type || 'Vorschlag'
}

function primaryTime(item: TeamsRecommendationRecord): string {
  if (item.sentAt) return item.sentAt
  if (item.scheduledFor) return item.scheduledFor
  return item.decidedAt || ''
}

function timeLabel(item: TeamsRecommendationRecord): string {
  const value = primaryTime(item)
  return value ? fmtDateTime(value) : '—'
}

function matchesFilter(item: TeamsRecommendationRecord, filter: StatusFilter) {
  if (filter === 'alle') return true
  if (filter === 'sent') return item.sendStatus === 'sent' || item.status === 'sent'
  if (filter === 'failed') return item.sendStatus === 'failed' || item.status === 'failed'
  if (filter === 'blocked') {
    return (
      item.status.includes('blocked') ||
      item.sendStatus.includes('blocked') ||
      item.blockingReasons.length > 0
    )
  }
  return true
}

function strongestReason(item: TeamsRecommendationRecord): string {
  return item.summary || item.reasons[0] || item.blockingReasons[0] || ''
}

function RecommendationRow({
  item,
  expanded,
  onToggle,
}: {
  item: TeamsRecommendationRecord
  expanded: boolean
  onToggle: () => void
}) {
  const reason = strongestReason(item)
  const variant = statusVariant(item.status, item.sendStatus)

  return (
    <>
      <TableRow onClick={onToggle}>
        <TableCell style={{ minWidth: '360px', maxWidth: '520px' }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              marginBottom: '5px',
            }}
          >
            <Badge variant={item.type === 'teams_alert' ? 'purple' : 'blue'}>
              {typeLabel(item.type)}
            </Badge>
            {item.shouldNotify && <Badge variant="green">Pushwürdig</Badge>}
          </div>
          <div
            style={{
              fontWeight: 650,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {item.articleTitle || 'Ohne Titel'}
          </div>
          <div
            style={{
              fontSize: '12px',
              color: 'var(--text-tertiary)',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              marginTop: '2px',
            }}
          >
            {reason || item.articleUrl}
          </div>
        </TableCell>
        <TableCell>
          <Badge variant={variant}>{statusLabel(item)}</Badge>
        </TableCell>
        <TableCell>
          <div style={{ fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>
            {timeLabel(item)}
          </div>
          {item.scheduledFor && item.sentAt == null && (
            <div style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>
              geplant
            </div>
          )}
        </TableCell>
        <TableCell>
          <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
            <Badge variant="default">Push {formatScore(item.score)}</Badge>
            {item.teamsAlertScore > 0 && (
              <Badge variant="purple">
                Teams {formatScore(item.teamsAlertScore)}
              </Badge>
            )}
            {item.editorialScore > 0 && (
              <Badge variant="blue">CvD {formatScore(item.editorialScore)}</Badge>
            )}
          </div>
        </TableCell>
        <TableCell>
          <div style={{ fontWeight: 650, fontVariantNumeric: 'tabular-nums' }}>
            {formatStoredOR(item.predictedOR, item.predictedORLabel)}
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>
            {item.expectedVisits > 0
              ? `${fmtNum(item.expectedVisits)} Visits`
              : 'keine Visit-Schätzung'}
          </div>
        </TableCell>
        <TableCell>
          <div style={{ fontVariantNumeric: 'tabular-nums' }}>
            {item.dashboardRank > 0 ? `#${item.dashboardRank}` : '—'}
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>
            {item.section || 'News'}
          </div>
        </TableCell>
      </TableRow>
      {expanded && (
        <TableRow>
          <TableCell colSpan={6} style={{ background: 'var(--bg)' }}>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
                gap: '16px',
              }}
            >
              <ReasonBlock title="Warum vorgeschlagen?" items={item.reasons} />
              <ReasonBlock
                title="Blocker / Hinweise"
                items={
                  item.blockingReasons.length > 0
                    ? item.blockingReasons
                    : item.sendError
                      ? [item.sendError]
                      : ['Keine harten Blocker gespeichert.']
                }
              />
            </div>
            <div
              style={{
                display: 'flex',
                gap: '8px',
                flexWrap: 'wrap',
                marginTop: '12px',
              }}
            >
              {item.articleUrl && (
                <Button
                  size="sm"
                  onClick={(event) => {
                    event.stopPropagation()
                    window.open(item.articleUrl, '_blank')
                  }}
                >
                  Artikel öffnen
                </Button>
              )}
              <span
                style={{
                  fontSize: '12px',
                  color: 'var(--text-tertiary)',
                  alignSelf: 'center',
                }}
              >
                Entscheidung: {item.decidedAt ? fmtDateTime(item.decidedAt) : '—'}
              </span>
            </div>
          </TableCell>
        </TableRow>
      )}
    </>
  )
}

function ReasonBlock({ title, items }: { title: string; items: string[] }) {
  return (
    <div>
      <div
        style={{
          fontSize: '11px',
          fontWeight: 700,
          color: 'var(--text-secondary)',
          textTransform: 'uppercase',
          marginBottom: '6px',
        }}
      >
        {title}
      </div>
      <ul
        style={{
          margin: 0,
          paddingLeft: '18px',
          color: 'var(--text-secondary)',
          fontSize: '13px',
          lineHeight: 1.6,
        }}
      >
        {items.slice(0, 6).map((item, index) => (
          <li key={`${item}-${index}`}>{item}</li>
        ))}
      </ul>
    </div>
  )
}

export function TeamsRecommendationsPage() {
  const [query, setQuery] = useState('')
  const [source, setSource] = useState<RecommendationSource>('teams_alert')
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('alle')
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const { data, isLoading, error, refetch, isFetching } =
    useTeamsRecommendations(120, source)
  const persistPlan = usePersistTeamsDailyPlan()

  const items = data?.items ?? []
  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase()
    return items.filter((item) => {
      if (!matchesFilter(item, statusFilter)) return false
      if (!needle) return true
      return [
        item.articleTitle,
        item.articleUrl,
        item.section,
        item.summary,
        item.status,
        item.type,
        ...item.reasons,
        ...item.blockingReasons,
      ]
        .join(' ')
        .toLowerCase()
        .includes(needle)
    })
  }, [items, query, statusFilter])

  const stats = useMemo(() => {
    const sent = items.filter(
      (item) => item.status === 'sent' || item.sendStatus === 'sent',
    )
    const failed = items.filter(
      (item) => item.status === 'failed' || item.sendStatus === 'failed',
    )
    const blocked = items.filter(
      (item) =>
        item.status.includes('blocked') ||
        item.sendStatus.includes('blocked') ||
        item.blockingReasons.length > 0,
    )
    const visitTotal = items.reduce(
      (sum, item) => sum + Math.max(0, item.expectedVisits || 0),
      0,
    )
    return { sent: sent.length, failed: failed.length, blocked: blocked.length, visitTotal }
  }, [items])

  const isTeamsHistory = source === 'teams_alert'
  const title = isTeamsHistory ? 'Teams-Historie' : 'Tagesplan-Snapshots'
  const subtitle = isTeamsHistory
    ? 'Nur echte Teams-Alert-Entscheidungen aus der lokalen DB. Tagesplan-Vorschläge sind bewusst getrennt.'
    : 'Planungsvorschläge aus dem CvD-Tagesplan. Das ist keine Teams-Sendehistorie.'

  return (
    <div
      style={{
        padding: '16px 24px',
        maxWidth: '1600px',
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
          gap: '12px',
          flexWrap: 'wrap',
        }}
      >
        <div>
          <h1 style={{ fontSize: '18px', fontWeight: 700, margin: 0 }}>
            {title}
          </h1>
          <div style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>
            {subtitle}
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          <Button
            onClick={() => refetch()}
            disabled={isFetching}
            style={{ minWidth: '112px', justifyContent: 'center' }}
          >
            {isFetching ? <Spinner size={14} /> : 'Aktualisieren'}
          </Button>
          {!isTeamsHistory && (
            <Button
              variant="primary"
              onClick={() => persistPlan.mutate()}
              disabled={persistPlan.isPending}
              style={{ minWidth: '176px', justifyContent: 'center' }}
            >
              {persistPlan.isPending ? (
                <Spinner size={14} />
              ) : (
                'Tagesplan lokal speichern'
              )}
            </Button>
          )}
        </div>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))',
          gap: '12px',
          marginBottom: '16px',
        }}
      >
        <StatCard
          label={isTeamsHistory ? 'Echte Teams-Vorschläge' : 'Tagesplan-Slots'}
          value={items.length}
          accent
        />
        <StatCard label="Gesendet" value={stats.sent} />
        <StatCard label="Fehler" value={stats.failed} />
        <StatCard label="Blockiert / Hinweise" value={stats.blocked} />
        <StatCard label="Visit-Schätzung" value={fmtNum(stats.visitTotal)} />
      </div>

      <Card style={{ marginBottom: '16px' }}>
        <CardContent
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
            gap: '12px',
            alignItems: 'end',
          }}
        >
          <Input
            label="Suche"
            placeholder="Titel, Ressort, Status oder Grund"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
          <Select
            label="Quelle"
            value={source}
            options={SOURCE_OPTIONS}
            onChange={(event) => {
              setSource(event.target.value as RecommendationSource)
              setStatusFilter('alle')
              setExpandedId(null)
            }}
          />
          <Select
            label="Ansicht"
            value={statusFilter}
            options={STATUS_OPTIONS}
            onChange={(event) =>
              setStatusFilter(event.target.value as StatusFilter)
            }
          />
        </CardContent>
      </Card>

      {persistPlan.isSuccess && (
        <Alert variant="success" style={{ marginBottom: '16px' }}>
          Tagesplan wurde lokal berechnet und in der lokalen DB gespeichert.
        </Alert>
      )}
      {persistPlan.isError && (
        <Alert variant="error" style={{ marginBottom: '16px' }}>
          {getApiErrorMessage(
            persistPlan.error,
            'Tagesplan konnte nicht lokal gespeichert werden.',
          )}
        </Alert>
      )}

      <Card>
        <CardHeader>
          <CardTitle>{isTeamsHistory ? 'Echte Teams-Historie' : 'Tagesplan-Snapshots'}</CardTitle>
          <span style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>
            {filtered.length} von {items.length}
          </span>
        </CardHeader>
        {isLoading && (
          <div
            style={{
              padding: '48px',
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
                'Teams-Vorschläge konnten nicht geladen werden.',
              )}
            </Alert>
          </CardContent>
        )}
        {!isLoading && !error && (
          <Table>
            <thead>
              <tr>
                <TableHeader>Vorschlag</TableHeader>
                <TableHeader>Status</TableHeader>
                <TableHeader>Zeit</TableHeader>
                <TableHeader>Scores</TableHeader>
                <TableHeader>Prognose</TableHeader>
                <TableHeader>Rang</TableHeader>
              </tr>
            </thead>
            <tbody>
              {filtered.length === 0 ? (
                <TableRow>
                  <TableCell
                    colSpan={6}
                    style={{
                      padding: '36px',
                      textAlign: 'center',
                      color: 'var(--text-tertiary)',
                    }}
                  >
                    {isTeamsHistory
                      ? 'Keine echten Teams-Alert-Einträge in dieser lokalen DB. Tagesplan-Vorschläge werden hier absichtlich nicht angezeigt.'
                      : 'Keine Tagesplan-Snapshots gefunden. Nutze “Tagesplan lokal speichern”, um Planungsvorschläge getrennt abzulegen.'}
                  </TableCell>
                </TableRow>
              ) : (
                filtered.map((item) => (
                  <RecommendationRow
                    key={item.id}
                    item={item}
                    expanded={expandedId === item.id}
                    onToggle={() =>
                      setExpandedId(expandedId === item.id ? null : item.id)
                    }
                  />
                ))
              )}
            </tbody>
          </Table>
        )}
      </Card>
    </div>
  )
}
