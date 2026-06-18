import { useState, useMemo } from 'react'
import { useFeed, useTeamsAlerts } from '@/hooks/use-api'
import {
  Alert,
  Badge,
  Card,
  CardContent,
  FilterChip,
  Input,
  Spinner,
  Table,
  TableCell,
  TableHeader,
  TableRow,
} from '@spring-media/editorial-one-ui'
import { useKandidatenFilterStore } from '@/stores/kandidaten-filter-store'
import { PushPreviewModal } from '@/components/ui/push-preview-modal'
import { getApiErrorMessage } from '@/utils/api-errors'
import { fmtDateTime, fmtOR, fmtScore, scoreVariant } from '@/utils/format'
import type { Article, TeamsAlertRecord } from '@/types/api'

const CATEGORIES = [
  'alle',
  'politik',
  'sport',
  'unterhaltung',
  'wirtschaft',
  'regional',
]

function ScoreBar({ score }: { score: number }) {
  const pct = Math.min(100, Math.max(0, score))
  const color =
    scoreVariant(score) === 'green'
      ? 'var(--green)'
      : scoreVariant(score) === 'amber'
        ? 'var(--amber)'
        : 'var(--red)'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
      <div
        style={{
          height: '6px',
          width: '60px',
          background: 'var(--border)',
          borderRadius: '3px',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            height: '100%',
            width: `${pct}%`,
            background: color,
            borderRadius: '3px',
          }}
        />
      </div>
      <span
        style={{
          fontSize: '13px',
          fontWeight: 600,
          color,
          fontVariantNumeric: 'tabular-nums',
          minWidth: '28px',
        }}
      >
        {fmtScore(score)}
      </span>
    </div>
  )
}

function teamsAlertVariant(
  teamsAlert: Article['teamsAlert'],
): 'green' | 'amber' | 'default' | 'red' {
  if (!teamsAlert) return 'default'
  if (teamsAlert.shouldNotify || teamsAlert.status === 'sent') return 'green'
  if (teamsAlert.status === 'observe') return 'amber'
  if (teamsAlert.status === 'failed') return 'red'
  return 'default'
}

function teamsAlertLabel(teamsAlert: Article['teamsAlert']): string {
  if (!teamsAlert) return 'Teams offen'
  if (teamsAlert.status === 'sent') return 'Teams gesendet'
  if (teamsAlert.shouldNotify || teamsAlert.status === 'notify') return 'Teams empfohlen'
  if (teamsAlert.status === 'observe') return 'Teams beobachtet'
  if (teamsAlert.status === 'failed') return 'Teams Fehler'
  return 'Teams kein Alert'
}

function teamsAlertRecordVariant(
  alert: TeamsAlertRecord,
): 'green' | 'amber' | 'default' | 'red' {
  if (alert.status === 'sent') return 'green'
  if (alert.status === 'failed') return 'red'
  if (alert.status === 'observe') return 'amber'
  return 'default'
}

function teamsAlertRecordLabel(alert: TeamsAlertRecord): string {
  if (alert.status === 'sent') return 'Teams gesendet'
  if (alert.status === 'failed') return 'Teams Fehler'
  if (alert.status === 'observe') return 'Teams beobachtet'
  return alert.status || 'Teams Entscheidung'
}

function fmtTeamsAlertOR(value: number): string {
  if (!value || value <= 0) return 'keine belastbare OR'
  return `${value.toLocaleString('de-DE', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })} % OR`
}

function TeamsAlertHistory({
  alerts,
  isLoading,
}: {
  alerts: TeamsAlertRecord[]
  isLoading: boolean
}) {
  return (
    <Card style={{ marginBottom: '16px' }}>
      <CardContent style={{ padding: '14px 16px' }}>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            gap: '12px',
            marginBottom: alerts.length > 0 ? '10px' : 0,
          }}
        >
          <div>
            <h2 style={{ fontSize: '15px', fontWeight: 700, margin: 0 }}>
              Letzte Teams-Empfehlungen
            </h2>
            <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
              Persistenter Verlauf der redaktionellen Handlungsempfehlungen
            </div>
          </div>
          {isLoading && <Spinner size={16} />}
        </div>

        {!isLoading && alerts.length === 0 && (
          <div style={{ fontSize: '13px', color: 'var(--text-tertiary)' }}>
            Noch keine Teams-Empfehlung gespeichert.
          </div>
        )}

        {alerts.length > 0 && (
          <div style={{ display: 'grid', gap: '8px' }}>
            {alerts.slice(0, 5).map((alert) => {
              const title =
                alert.articleTitle || alert.articleUrl || alert.articleId || alert.articleKey
              const time = alert.lastAlertAt || alert.lastDecisionAt
              return (
                <div
                  key={`${alert.articleKey}-${alert.lastDecisionAt ?? ''}`}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'minmax(0, 1fr) auto',
                    gap: '10px',
                    alignItems: 'start',
                    padding: '10px 0',
                    borderTop: '1px solid var(--border-light)',
                  }}
                >
                  <div style={{ minWidth: 0 }}>
                    <div
                      style={{
                        display: 'flex',
                        gap: '6px',
                        alignItems: 'center',
                        flexWrap: 'wrap',
                        marginBottom: '4px',
                      }}
                    >
                      <Badge variant={teamsAlertRecordVariant(alert)}>
                        {teamsAlertRecordLabel(alert)}
                      </Badge>
                      {alert.isBreaking && <Badge variant="red">Breaking</Badge>}
                      {time && (
                        <span
                          style={{
                            fontSize: '12px',
                            color: 'var(--text-tertiary)',
                          }}
                        >
                          {fmtDateTime(time)}
                        </span>
                      )}
                    </div>
                    <div
                      style={{
                        fontSize: '13px',
                        fontWeight: 600,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                      title={title}
                    >
                      {alert.articleUrl ? (
                        <a
                          href={alert.articleUrl}
                          target="_blank"
                          rel="noreferrer"
                          style={{ color: 'inherit', textDecoration: 'none' }}
                        >
                          {title}
                        </a>
                      ) : (
                        title
                      )}
                    </div>
                    <div
                      style={{
                        fontSize: '12px',
                        color:
                          alert.status === 'failed'
                            ? 'var(--red)'
                            : 'var(--text-secondary)',
                        marginTop: '3px',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                      title={alert.status === 'failed' ? alert.lastError : alert.reason}
                    >
                      {alert.status === 'failed'
                        ? alert.lastError || alert.reason || 'Teams-Versand fehlgeschlagen'
                        : alert.reason || 'Push empfohlen'}
                    </div>
                  </div>
                  <div
                    style={{
                      textAlign: 'right',
                      fontSize: '12px',
                      color: 'var(--text-secondary)',
                      whiteSpace: 'nowrap',
                      fontVariantNumeric: 'tabular-nums',
                    }}
                  >
                    <div style={{ fontWeight: 700, color: 'var(--text-primary)' }}>
                      Score {fmtScore(alert.score)}
                    </div>
                    <div>{fmtTeamsAlertOR(alert.predictedOR)}</div>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function ArticleRow({
  article,
  onPreview,
}: {
  article: Article
  onPreview: (a: Article) => void
}) {
  const priorityVariant: 'green' | 'amber' | 'default' =
    article.mixPriority === 'hoch'
      ? 'green'
      : article.mixPriority === 'mittel'
        ? 'amber'
        : 'default'

  return (
    <TableRow onClick={() => onPreview(article)} title="Vorschau öffnen">
      <TableCell style={{ maxWidth: '340px' }}>
        <div
          style={{
            fontWeight: 500,
            marginBottom: '2px',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {article.title}
        </div>
        <div style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>
          {fmtDateTime(article.pubDate)} · {article.category}
        </div>
        {article.scoreReason && (
          <div
            style={{
              fontSize: '12px',
              color: 'var(--text-secondary)',
              marginTop: '4px',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {article.scoreReason}
          </div>
        )}
        {article.teamsAlert?.summary && (
          <div
            style={{
              fontSize: '12px',
              color: article.teamsAlert.shouldNotify
                ? 'var(--green)'
                : 'var(--text-secondary)',
              marginTop: '4px',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            Teams: {article.teamsAlert.summary}
            {article.teamsAlert.lastTeamsAlertAt
              ? ` · zuletzt ${fmtDateTime(article.teamsAlert.lastTeamsAlertAt)}`
              : ''}
          </div>
        )}
      </TableCell>
      <TableCell>
        <ScoreBar score={article.score} />
      </TableCell>
      <TableCell>
        {article.predictedOR != null ? (
          <span
            style={{
              fontVariantNumeric: 'tabular-nums',
              fontSize: '13px',
              fontWeight: 600,
            }}
          >
            {fmtOR(article.predictedOR)}
          </span>
        ) : (
          <span style={{ color: 'var(--text-tertiary)' }}>—</span>
        )}
      </TableCell>
      <TableCell>
        <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
          {article.mixPriority && (
            <Badge variant={priorityVariant}>Priorität {article.mixPriority}</Badge>
          )}
          {article.teamsAlert && (
            <Badge variant={teamsAlertVariant(article.teamsAlert)}>
              {teamsAlertLabel(article.teamsAlert)}
            </Badge>
          )}
          {article.isBreaking && <Badge variant="red">Breaking</Badge>}
          {article.isEilmeldung && <Badge variant="red">Eilmeldung</Badge>}
          {article.isSport && <Badge variant="blue">Sport</Badge>}
          {article.isPlusArticle && <Badge variant="amber">Plus</Badge>}
        </div>
      </TableCell>
      <TableCell>
        <button
          onClick={(e) => {
            e.stopPropagation()
            onPreview(article)
          }}
          title="Push-Vorschau & KI-Titel"
          style={{
            fontFamily: 'inherit',
            fontSize: '11px',
            padding: '4px 8px',
            borderRadius: '5px',
            border: '1px solid var(--border)',
            background: 'var(--white)',
            cursor: 'pointer',
            color: 'var(--accent)',
            whiteSpace: 'nowrap',
          }}
        >
          Vorschau
        </button>
      </TableCell>
    </TableRow>
  )
}

export function KandidatenPage() {
  const { data, isLoading, error, refetch } = useFeed()
  const { data: teamsAlertsData, isLoading: teamsAlertsLoading } =
    useTeamsAlerts()
  const {
    kandidatenSearch,
    kandidatenCategory,
    setKandidatenSearch,
    setKandidatenCategory,
  } = useKandidatenFilterStore()

  const [previewArticle, setPreviewArticle] = useState<Article | null>(null)

  const filtered = useMemo<Article[]>(() => {
    if (!data?.articles) return []
    let list = data.articles
    if (kandidatenCategory !== 'alle') {
      list = list.filter(
        (a) => a.category?.toLowerCase() === kandidatenCategory,
      )
    }
    if (kandidatenSearch.trim()) {
      const q = kandidatenSearch.toLowerCase()
      list = list.filter((a) => a.title.toLowerCase().includes(q))
    }
    return list
  }, [data, kandidatenSearch, kandidatenCategory])

  const [sortKey, setSortKey] = useState<'score' | 'predictedOR' | 'pubDate'>(
    'score',
  )

  const sorted = useMemo(() => {
    return [...filtered].sort((a, b) => {
      if (sortKey === 'score') return b.score - a.score
      if (sortKey === 'predictedOR')
        return (b.predictedOR ?? 0) - (a.predictedOR ?? 0)
      return new Date(b.pubDate).getTime() - new Date(a.pubDate).getTime()
    })
  }, [filtered, sortKey])

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
        <div>
          <h1 style={{ fontSize: '18px', fontWeight: 700, margin: 0 }}>
            Push-Kandidaten
          </h1>
          {data && (
            <p
              style={{
                fontSize: '12px',
                color: 'var(--text-secondary)',
                margin: '2px 0 0',
              }}
            >
              {data.count} Artikel · aktualisiert {fmtDateTime(data.fetchedAt)}
            </p>
          )}
        </div>
        <button
          onClick={() => refetch()}
          style={{
            fontFamily: 'inherit',
            fontSize: '13px',
            padding: '7px 14px',
            borderRadius: '6px',
            border: '1px solid var(--border)',
            background: 'var(--white)',
            cursor: 'pointer',
          }}
        >
          ↻ Aktualisieren
        </button>
      </div>

      <TeamsAlertHistory
        alerts={teamsAlertsData?.items ?? []}
        isLoading={teamsAlertsLoading}
      />

      {/* Filter Bar */}
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
          <Input
            placeholder="Artikel suchen…"
            value={kandidatenSearch}
            onChange={(e) => setKandidatenSearch(e.target.value)}
            style={{ width: '220px' }}
          />
          <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
            {CATEGORIES.map((cat) => (
              <FilterChip
                key={cat}
                active={kandidatenCategory === cat}
                onClick={() => setKandidatenCategory(cat)}
              >
                {cat.charAt(0).toUpperCase() + cat.slice(1)}
              </FilterChip>
            ))}
          </div>
          <div style={{ marginLeft: 'auto', display: 'flex', gap: '6px' }}>
            <span
              style={{
                fontSize: '12px',
                color: 'var(--text-secondary)',
                alignSelf: 'center',
              }}
            >
              Sortierung:
            </span>
            {(['score', 'predictedOR', 'pubDate'] as const).map((k) => (
              <FilterChip
                key={k}
                active={sortKey === k}
                onClick={() => setSortKey(k)}
              >
                {k === 'score'
                  ? 'Push Score'
                  : k === 'predictedOR'
                    ? 'XOR'
                    : 'Datum'}
              </FilterChip>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Table */}
      <Card>
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
                'Feed konnte nicht geladen werden. Läuft der Push-Balancer-Server?',
              )}
            </Alert>
          </CardContent>
        )}
        {!isLoading && !error && (
          <Table>
            <thead>
              <tr>
                <TableHeader>Artikel</TableHeader>
                <TableHeader>Push Score</TableHeader>
                <TableHeader>XOR (Prognose)</TableHeader>
                <TableHeader>Tags</TableHeader>
                <TableHeader></TableHeader>
              </tr>
            </thead>
            <tbody>
              {sorted.length === 0 ? (
                <TableRow>
                  <TableCell
                    style={{
                      textAlign: 'center',
                      color: 'var(--text-tertiary)',
                      padding: '32px',
                    }}
                    colSpan={5}
                  >
                    Keine Artikel gefunden
                  </TableCell>
                </TableRow>
              ) : (
                sorted.map((a) => (
                  <ArticleRow
                    key={a.id}
                    article={a}
                    onPreview={setPreviewArticle}
                  />
                ))
              )}
            </tbody>
          </Table>
        )}
        {!isLoading && sorted.length > 0 && (
          <div
            style={{
              padding: '10px 16px',
              borderTop: '1px solid var(--border-light)',
              fontSize: '12px',
              color: 'var(--text-tertiary)',
            }}
          >
            {sorted.length} von {data?.count ?? 0} Artikeln
          </div>
        )}
      </Card>

      {previewArticle && (
        <PushPreviewModal
          article={previewArticle}
          onClose={() => setPreviewArticle(null)}
        />
      )}
    </div>
  )
}
