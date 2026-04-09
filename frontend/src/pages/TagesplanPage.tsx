import { useState } from 'react'
import {
  useTagesplan,
  useTagesplanSuggestions,
  useTagesplanRetro,
} from '@/hooks/useApi'
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
} from '@spring-media/editorial-one-ui'
import { useAppStore } from '@/stores/app-store'
import { fmtOR, fmtNum, fmtDate, orColor } from '@/lib/format'
import type { TagesplanSlot } from '@/types/api'

function SlotCard({ slot }: { slot: TagesplanSlot }) {
  const isGolden = slot.isGoldenHour
  const hasPush = !!slot.pushed
  const now = new Date().getHours()
  const isPast = slot.hour < now

  return (
    <div
      style={{
        background: isGolden ? 'var(--amber-bg)' : 'var(--white)',
        border: `1px solid ${isGolden ? 'var(--amber-border)' : hasPush ? 'var(--green-border)' : 'var(--border)'}`,
        borderRadius: 'var(--radius)',
        padding: '14px 16px',
        opacity: isPast && !hasPush ? 0.6 : 1,
        position: 'relative',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '8px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span
            style={{
              fontSize: '15px',
              fontWeight: 700,
              color: isGolden ? 'var(--amber)' : 'var(--text)',
              fontVariantNumeric: 'tabular-nums',
            }}
          >
            {String(slot.hour).padStart(2, '0')}:00
          </span>
          {isGolden && <Badge variant="amber">Golden Hour</Badge>}
          {hasPush && <Badge variant="green">Gesendet</Badge>}
        </div>
        <span
          style={{
            fontSize: '16px',
            fontWeight: 700,
            color: orColor(slot.predictedOR),
            fontVariantNumeric: 'tabular-nums',
          }}
        >
          {fmtOR(slot.predictedOR)}
        </span>
      </div>

      {hasPush && slot.pushedTitle && (
        <div
          style={{
            fontSize: '12px',
            color: 'var(--text-secondary)',
            marginBottom: '6px',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {slot.pushedTitle}
          {slot.actualOR != null && (
            <span
              style={{
                marginLeft: '8px',
                color: 'var(--green)',
                fontWeight: 600,
              }}
            >
              Ist: {fmtOR(slot.actualOR)}
            </span>
          )}
        </div>
      )}

      <div style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>
        {slot.recommendation}
      </div>
    </div>
  )
}

function RetroSection({ mode }: { mode: 'redaktion' | 'sport' }) {
  const { data, isLoading } = useTagesplanRetro(mode)
  const [expandedDay, setExpandedDay] = useState<string | null>(null)

  if (isLoading)
    return (
      <div
        style={{ padding: '24px', display: 'flex', justifyContent: 'center' }}
      >
        <Spinner size={20} />
      </div>
    )
  if (!data?.days?.length) return null

  return (
    <Card style={{ marginTop: '24px' }}>
      <CardHeader>
        <CardTitle>Verlauf (letzte {data.days.length} Tage)</CardTitle>
        <div style={{ display: 'flex', gap: '8px' }}>
          <Badge variant="default">Ø OR {fmtOR(data.summary.avgOR)}</Badge>
          <Badge variant="default">{data.summary.totalPushes} Pushes</Badge>
          <Badge variant="default">MAE {fmtOR(data.summary.avgMAE)}</Badge>
        </div>
      </CardHeader>
      <div>
        {data.days.map((day) => {
          const isOpen = expandedDay === day.date
          return (
            <div
              key={day.date}
              style={{ borderTop: '1px solid var(--border-light)' }}
            >
              <button
                onClick={() => setExpandedDay(isOpen ? null : day.date)}
                style={{
                  width: '100%',
                  padding: '12px 20px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  fontFamily: 'inherit',
                  textAlign: 'left',
                }}
              >
                <div
                  style={{ display: 'flex', alignItems: 'center', gap: '12px' }}
                >
                  <span style={{ fontWeight: 600, fontSize: '13px' }}>
                    {fmtDate(day.date)}
                  </span>
                  <Badge
                    variant={
                      day.avgOR >= 0.06
                        ? 'green'
                        : day.avgOR >= 0.04
                          ? 'amber'
                          : 'red'
                    }
                  >
                    {fmtOR(day.avgOR)}
                  </Badge>
                  <span
                    style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}
                  >
                    {day.pushedCount} Pushes · MAE {fmtOR(day.mae)}
                  </span>
                </div>
                <span
                  style={{ color: 'var(--text-tertiary)', fontSize: '14px' }}
                >
                  {isOpen ? '▲' : '▼'}
                </span>
              </button>

              {isOpen && (
                <div
                  style={{
                    padding: '0 20px 16px',
                    display: 'grid',
                    gridTemplateColumns:
                      'repeat(auto-fill, minmax(200px, 1fr))',
                    gap: '8px',
                  }}
                >
                  {day.slots
                    .filter((s) => s.pushed || s.isGoldenHour)
                    .map((slot) => (
                      <div
                        key={slot.hour}
                        style={{
                          padding: '10px 12px',
                          borderRadius: 'var(--radius)',
                          background: slot.isGoldenHour
                            ? 'var(--amber-bg)'
                            : 'var(--bg)',
                          border: `1px solid ${slot.isGoldenHour ? 'var(--amber-border)' : 'var(--border-light)'}`,
                        }}
                      >
                        <div
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '6px',
                            marginBottom: '4px',
                          }}
                        >
                          <span
                            style={{
                              fontWeight: 700,
                              fontSize: '13px',
                              fontVariantNumeric: 'tabular-nums',
                            }}
                          >
                            {String(slot.hour).padStart(2, '0')}:00
                          </span>
                          {slot.isGoldenHour && (
                            <Badge variant="amber">Golden</Badge>
                          )}
                          {slot.pushed && (
                            <Badge variant="green">Gesendet</Badge>
                          )}
                        </div>
                        <div
                          style={{
                            display: 'flex',
                            gap: '8px',
                            fontSize: '12px',
                          }}
                        >
                          <span style={{ color: 'var(--text-secondary)' }}>
                            Prognose: <strong>{fmtOR(slot.predictedOR)}</strong>
                          </span>
                          {slot.actualOR != null && (
                            <span style={{ color: orColor(slot.actualOR) }}>
                              Ist: <strong>{fmtOR(slot.actualOR)}</strong>
                            </span>
                          )}
                        </div>
                        {slot.pushedTitle && (
                          <div
                            style={{
                              fontSize: '11px',
                              color: 'var(--text-tertiary)',
                              marginTop: '4px',
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                            }}
                          >
                            {slot.pushedTitle}
                          </div>
                        )}
                      </div>
                    ))}
                  {day.slots.filter((s) => s.pushed || s.isGoldenHour)
                    .length === 0 && (
                    <div
                      style={{
                        gridColumn: '1 / -1',
                        fontSize: '13px',
                        color: 'var(--text-tertiary)',
                        padding: '8px 0',
                      }}
                    >
                      Keine Push-Daten für diesen Tag
                    </div>
                  )}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </Card>
  )
}

export function TagesplanPage() {
  const { tagesplanMode, tagesplanDate, setTagesplanMode, setTagesplanDate } =
    useAppStore()

  const { data, isLoading, error } = useTagesplan(tagesplanDate, tagesplanMode)
  const { data: suggestions } = useTagesplanSuggestions(
    tagesplanDate,
    tagesplanMode,
  )

  return (
    <div
      style={{
        padding: '16px 24px',
        maxWidth: '1400px',
        margin: '0 auto',
        animation: 'fadeIn 0.2s ease',
      }}
    >
      {/* Header */}
      <div
        style={{
          marginBottom: '16px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '12px',
        }}
      >
        <h1 style={{ fontSize: '18px', fontWeight: 700, margin: 0 }}>
          Tagesplan
        </h1>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ display: 'flex', gap: '6px' }}>
            <FilterChip
              active={tagesplanMode === 'redaktion'}
              onClick={() => setTagesplanMode('redaktion')}
            >
              Redaktion
            </FilterChip>
            <FilterChip
              active={tagesplanMode === 'sport'}
              onClick={() => setTagesplanMode('sport')}
            >
              Sport
            </FilterChip>
          </div>
          <input
            type="date"
            value={tagesplanDate}
            onChange={(e) => setTagesplanDate(e.target.value)}
            style={{
              fontFamily: 'inherit',
              fontSize: '13px',
              padding: '6px 10px',
              borderRadius: '6px',
              border: '1px solid var(--border)',
              background: 'var(--white)',
              color: 'var(--text)',
              cursor: 'pointer',
            }}
          />
        </div>
      </div>

      {(isLoading || data?.loading) && (
        <div
          style={{ padding: '60px', display: 'flex', justifyContent: 'center' }}
        >
          <Spinner size={28} />
        </div>
      )}
      {error && (
        <Alert variant="error">Tagesplan konnte nicht geladen werden.</Alert>
      )}

      {data && (
        <>
          {/* KPI Row */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
              gap: '12px',
              marginBottom: '20px',
            }}
          >
            <StatCard
              label="Golden Hour"
              value={`${String(data.goldenHour).padStart(2, '0')}:00`}
              accent
            />
            <StatCard label="Ø XOR" value={fmtOR(data.avgOR)} />
            <StatCard label="Pushes heute" value={data.pushedCount} />
            <StatCard
              label="MAE Modell"
              value={fmtOR(data.mae)}
              sub={`${fmtNum(data.trainedOnRows)} Trainings-Rows`}
            />
          </div>

          {data.llmReview && (
            <Card style={{ marginBottom: '20px' }}>
              <CardHeader>
                <CardTitle>KI-Zusammenfassung</CardTitle>
              </CardHeader>
              <CardContent>
                <p
                  style={{
                    margin: '0 0 10px',
                    color: 'var(--text-secondary)',
                    lineHeight: 1.6,
                  }}
                >
                  {data.llmReview.summary}
                </p>
                {data.llmReview.warnings.length > 0 && (
                  <div
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '6px',
                    }}
                  >
                    {data.llmReview.warnings.map((w, i) => (
                      <Alert key={i} variant="warning">
                        {w}
                      </Alert>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Slots Grid */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
              gap: '12px',
              marginBottom: '24px',
            }}
          >
            {data.slots.map((slot) => (
              <SlotCard key={slot.hour} slot={slot} />
            ))}
          </div>

          {/* Suggestions */}
          {suggestions && suggestions.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Push-Empfehlungen</CardTitle>
                <Badge variant="purple">{suggestions.length} Vorschläge</Badge>
              </CardHeader>
              <div>
                {suggestions.map((s, i) => (
                  <div
                    key={i}
                    style={{
                      padding: '12px 20px',
                      borderBottom: '1px solid var(--border-light)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      gap: '12px',
                      cursor: 'pointer',
                    }}
                    onClick={() => window.open(s.url, '_blank')}
                  >
                    <div style={{ flex: 1, overflow: 'hidden' }}>
                      <div
                        style={{
                          fontWeight: 500,
                          fontSize: '13px',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {s.title}
                      </div>
                      <div
                        style={{
                          fontSize: '12px',
                          color: 'var(--text-tertiary)',
                          marginTop: '2px',
                        }}
                      >
                        Slot {String(s.hour).padStart(2, '0')}:00 · Score{' '}
                        {s.score.toFixed(1)}
                      </div>
                    </div>
                    <span
                      style={{
                        fontWeight: 700,
                        fontSize: '13px',
                        color: orColor(s.predictedOR),
                        fontVariantNumeric: 'tabular-nums',
                        flexShrink: 0,
                      }}
                    >
                      {fmtOR(s.predictedOR)}
                    </span>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </>
      )}

      <RetroSection mode={tagesplanMode} />
    </div>
  )
}
