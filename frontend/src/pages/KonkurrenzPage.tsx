import { useCompetitorRedaktion, useCompetitorSport } from '@/hooks/useApi'
import {
  Alert,
  Badge,
  Card,
  CardContent,
  FilterChip,
  Input,
  Spinner,
  StatCard,
} from '@spring-media/editorial-one-ui'
import { useAppStore } from '@/stores/app-store'
import { fmtDateTime } from '@/lib/format'
import type { CompetitorItem } from '@/types/api'
import { useMemo, useState } from 'react'

function ItemCard({ item }: { item: CompetitorItem }) {
  return (
    <div
      style={{
        background: 'var(--white)',
        border: `1px solid ${item.isGap ? 'var(--amber-border)' : item.isHot ? 'var(--red-border)' : 'var(--border)'}`,
        borderRadius: 'var(--radius)',
        padding: '12px 16px',
        cursor: 'pointer',
        transition: 'box-shadow 0.15s',
      }}
      onClick={() => window.open(item.url, '_blank')}
      onMouseEnter={(e) =>
        ((e.currentTarget as HTMLElement).style.boxShadow = 'var(--shadow)')
      }
      onMouseLeave={(e) =>
        ((e.currentTarget as HTMLElement).style.boxShadow = '')
      }
    >
      <div
        style={{
          marginBottom: '8px',
          display: 'flex',
          flexWrap: 'wrap',
          gap: '4px',
        }}
      >
        {item.isGap && <Badge variant="amber">Lücke</Badge>}
        {item.isExklusiv && <Badge variant="purple">Exklusiv</Badge>}
        {item.isHot && <Badge variant="red">Hot</Badge>}
        <Badge
          variant="default"
          style={{
            background: item.outletColor + '20',
            color: item.outletColor,
            borderColor: item.outletColor + '40',
          }}
        >
          {item.outlet}
        </Badge>
      </div>
      <div
        style={{
          fontSize: '13px',
          fontWeight: 500,
          marginBottom: '6px',
          lineHeight: 1.4,
        }}
      >
        {item.title}
      </div>
      <div style={{ fontSize: '11px', color: 'var(--text-tertiary)' }}>
        {fmtDateTime(item.pubDate)}
        {item.outlets.length > 1 && ` · ${item.outlets.length} Outlets`}
      </div>
    </div>
  )
}

export function KonkurrenzPage() {
  const { konkurrenzMode, setKonkurrenzMode } = useAppStore()
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState<'alle' | 'lucken' | 'exklusiv' | 'hot'>(
    'alle',
  )

  const { data: redaktion, isLoading: rLoading } = useCompetitorRedaktion()
  const { data: sport, isLoading: sLoading } = useCompetitorSport()

  const data = konkurrenzMode === 'redaktion' ? redaktion : sport
  const isLoading = konkurrenzMode === 'redaktion' ? rLoading : sLoading

  const filtered = useMemo(() => {
    if (!data?.items) return []
    let list = data.items
    if (filter === 'lucken') list = list.filter((i) => i.isGap)
    else if (filter === 'exklusiv') list = list.filter((i) => i.isExklusiv)
    else if (filter === 'hot') list = list.filter((i) => i.isHot)
    if (search.trim()) {
      const q = search.toLowerCase()
      list = list.filter((i) => i.title.toLowerCase().includes(q))
    }
    return list
  }, [data, filter, search])

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
          Konkurrenz-Monitor
        </h1>
        <div style={{ display: 'flex', gap: '6px' }}>
          <FilterChip
            active={konkurrenzMode === 'redaktion'}
            onClick={() => setKonkurrenzMode('redaktion')}
          >
            Redaktion
          </FilterChip>
          <FilterChip
            active={konkurrenzMode === 'sport'}
            onClick={() => setKonkurrenzMode('sport')}
          >
            Sport
          </FilterChip>
        </div>
      </div>

      {/* KPI Cards */}
      {data?.summary && (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
            gap: '12px',
            marginBottom: '16px',
          }}
        >
          <StatCard label="Gesamt" value={data.summary.total} />
          <StatCard label="Lücken" value={data.summary.gaps} accent />
          <StatCard label="Exklusiv" value={data.summary.exklusiv} />
          <StatCard label="Hot Topics" value={data.summary.hot} />
        </div>
      )}

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
            placeholder="Suchen…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{ width: '200px' }}
          />
          {(['alle', 'lucken', 'exklusiv', 'hot'] as const).map((f) => (
            <FilterChip
              key={f}
              active={filter === f}
              onClick={() => setFilter(f)}
            >
              {f === 'alle'
                ? 'Alle'
                : f === 'lucken'
                  ? 'Lücken'
                  : f === 'exklusiv'
                    ? 'Exklusiv'
                    : 'Hot'}
            </FilterChip>
          ))}
          {data?.fetchedAt && (
            <span
              style={{
                marginLeft: 'auto',
                fontSize: '12px',
                color: 'var(--text-tertiary)',
              }}
            >
              Abgerufen: {fmtDateTime(data.fetchedAt)}
            </span>
          )}
        </CardContent>
      </Card>

      {isLoading && (
        <div
          style={{ padding: '60px', display: 'flex', justifyContent: 'center' }}
        >
          <Spinner size={28} />
        </div>
      )}

      {!isLoading && (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
            gap: '12px',
          }}
        >
          {filtered.length === 0 ? (
            <Alert variant="info" style={{ gridColumn: '1 / -1' }}>
              Keine Einträge gefunden.
            </Alert>
          ) : (
            filtered.map((item, i) => <ItemCard key={i} item={item} />)
          )}
        </div>
      )}
    </div>
  )
}
