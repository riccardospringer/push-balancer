import { useState, useEffect, useCallback } from 'react'
import { Modal } from './Modal'
import { Button } from './Button'
import { Badge } from './Badge'
import { Spinner } from './Spinner'
import { useGenerateTitle } from '@/hooks/useApi'
import { fmtOR, fmtScore, scoreVariant } from '@/lib/format'
import type { Article, GenerateTitleResponse } from '@/types/api'

// ── Helpers ────────────────────────────────────────────────────────────────

function now(): { time: string; date: string; dateShort: string } {
  const d = new Date()
  const time = d.toLocaleTimeString('de-DE', {
    hour: '2-digit',
    minute: '2-digit',
  })
  const date = d.toLocaleDateString('de-DE', {
    weekday: 'long',
    day: 'numeric',
    month: 'long',
  })
  const dateShort = d.toLocaleDateString('de-DE', {
    weekday: 'short',
    day: 'numeric',
    month: 'long',
  })
  return { time, date, dateShort }
}

// ── iOS Phone Preview ──────────────────────────────────────────────────────

function IOSPreview({
  title,
  dachzeile,
  time,
  date,
}: {
  title: string
  dachzeile: string
  time: string
  date: string
}) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div
        style={{
          fontSize: '11px',
          color: 'var(--text-secondary)',
          marginBottom: '8px',
          fontWeight: 500,
        }}
      >
        iOS (BILD App)
      </div>
      <div
        style={{
          width: '260px',
          height: '520px',
          background: '#1a1a2e',
          borderRadius: '40px',
          border: '6px solid #2d2d3f',
          boxShadow:
            '0 20px 60px rgba(0,0,0,0.4), inset 0 0 0 1px rgba(255,255,255,0.08)',
          margin: '0 auto',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          position: 'relative',
          overflow: 'hidden',
          padding: '12px 12px 20px',
        }}
      >
        {/* Dynamic Island */}
        <div
          style={{
            width: '88px',
            height: '26px',
            background: '#000',
            borderRadius: '20px',
            marginBottom: '8px',
            flexShrink: 0,
          }}
        />

        {/* Status bar */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            width: '100%',
            padding: '0 6px',
            marginBottom: '4px',
          }}
        >
          <span
            style={{
              color: 'rgba(255,255,255,0.9)',
              fontSize: '11px',
              fontWeight: 600,
            }}
          >
            {time}
          </span>
          <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
            <svg width="14" height="10" viewBox="0 0 24 24" fill="white">
              <path d="M1 9l2 2c4.97-4.97 13.03-4.97 18 0l2-2C16.93 2.93 7.08 2.93 1 9zm8 8l3 3 3-3c-1.65-1.66-4.34-1.66-6 0zm-4-4l2 2c2.76-2.76 7.24-2.76 10 0l2-2C15.14 9.14 8.87 9.14 5 13z" />
            </svg>
            <svg width="14" height="10" viewBox="0 0 24 24" fill="white">
              <rect x="17" y="4" width="4" height="16" rx="1" />
              <rect x="11" y="8" width="4" height="12" rx="1" />
              <rect x="5" y="12" width="4" height="8" rx="1" />
            </svg>
          </div>
        </div>

        {/* Clock */}
        <div
          style={{
            color: 'white',
            fontSize: '44px',
            fontWeight: 300,
            lineHeight: 1,
            marginBottom: '2px',
            letterSpacing: '-1px',
          }}
        >
          {time}
        </div>
        <div
          style={{
            color: 'rgba(255,255,255,0.7)',
            fontSize: '13px',
            marginBottom: '16px',
          }}
        >
          {date}
        </div>

        {/* Notification */}
        <div
          style={{
            width: '100%',
            background: 'rgba(255,255,255,0.12)',
            backdropFilter: 'blur(20px)',
            borderRadius: '16px',
            padding: '10px 12px',
            border: '1px solid rgba(255,255,255,0.15)',
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              marginBottom: '6px',
            }}
          >
            <div
              style={{
                width: '24px',
                height: '24px',
                background: '#c00',
                borderRadius: '6px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '9px',
                color: 'white',
                fontWeight: 700,
                flexShrink: 0,
              }}
            >
              B
            </div>
            <span
              style={{
                color: 'rgba(255,255,255,0.8)',
                fontSize: '10px',
                fontWeight: 600,
                flex: 1,
              }}
            >
              BILD
            </span>
            <span style={{ color: 'rgba(255,255,255,0.5)', fontSize: '10px' }}>
              Jetzt
            </span>
          </div>
          {dachzeile && (
            <div
              style={{
                color: 'rgba(255,255,255,0.6)',
                fontSize: '10px',
                fontWeight: 600,
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
                marginBottom: '2px',
              }}
            >
              {dachzeile}
            </div>
          )}
          <div
            style={{
              color: 'white',
              fontSize: '12px',
              fontWeight: 500,
              lineHeight: 1.4,
              display: '-webkit-box',
              WebkitLineClamp: 3,
              WebkitBoxOrient: 'vertical',
              overflow: 'hidden',
            }}
          >
            {title || 'Push-Titel eingeben…'}
          </div>
        </div>

        {/* Home indicator */}
        <div
          style={{
            position: 'absolute',
            bottom: '8px',
            width: '100px',
            height: '4px',
            background: 'rgba(255,255,255,0.3)',
            borderRadius: '2px',
          }}
        />
      </div>
    </div>
  )
}

// ── Android Phone Preview ──────────────────────────────────────────────────

function AndroidPreview({
  title,
  dachzeile,
  time,
  dateShort,
}: {
  title: string
  dachzeile: string
  time: string
  dateShort: string
}) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div
        style={{
          fontSize: '11px',
          color: 'var(--text-secondary)',
          marginBottom: '8px',
          fontWeight: 500,
        }}
      >
        Android (BILD App)
      </div>
      <div
        style={{
          width: '260px',
          height: '520px',
          background: '#0f172a',
          borderRadius: '32px',
          border: '6px solid #1e293b',
          boxShadow:
            '0 20px 60px rgba(0,0,0,0.4), inset 0 0 0 1px rgba(255,255,255,0.06)',
          margin: '0 auto',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          padding: '10px 10px 12px',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        {/* Status bar */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            width: '100%',
            padding: '0 8px',
            marginBottom: '6px',
          }}
        >
          <span style={{ color: 'rgba(255,255,255,0.8)', fontSize: '11px' }}>
            {time}
          </span>
          <div style={{ display: 'flex', gap: '4px' }}>
            <svg
              width="12"
              height="10"
              viewBox="0 0 24 24"
              fill="rgba(255,255,255,0.7)"
            >
              <path d="M1 9l2 2c4.97-4.97 13.03-4.97 18 0l2-2C16.93 2.93 7.08 2.93 1 9zm8 8l3 3 3-3c-1.65-1.66-4.34-1.66-6 0zm-4-4l2 2c2.76-2.76 7.24-2.76 10 0l2-2C15.14 9.14 8.87 9.14 5 13z" />
            </svg>
            <svg
              width="12"
              height="10"
              viewBox="0 0 24 24"
              fill="rgba(255,255,255,0.7)"
            >
              <rect x="17" y="4" width="4" height="16" rx="1" />
              <rect x="11" y="8" width="4" height="12" rx="1" />
              <rect x="5" y="12" width="4" height="8" rx="1" />
            </svg>
          </div>
        </div>

        {/* Clock Widget */}
        <div style={{ marginBottom: '20px', textAlign: 'center' }}>
          <div
            style={{
              color: 'white',
              fontSize: '52px',
              fontWeight: 200,
              lineHeight: 1,
              letterSpacing: '-2px',
            }}
          >
            {time}
          </div>
          <div
            style={{
              color: 'rgba(255,255,255,0.6)',
              fontSize: '13px',
              marginTop: '4px',
            }}
          >
            {dateShort}
          </div>
        </div>

        {/* Notification */}
        <div
          style={{
            width: '100%',
            background: '#1e293b',
            borderRadius: '14px',
            padding: '10px 12px',
            border: '1px solid rgba(255,255,255,0.08)',
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              marginBottom: '6px',
            }}
          >
            <div
              style={{
                width: '20px',
                height: '20px',
                background: '#c00',
                borderRadius: '5px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '9px',
                color: 'white',
                fontWeight: 700,
                flexShrink: 0,
              }}
            >
              B
            </div>
            <span
              style={{
                color: 'rgba(255,255,255,0.6)',
                fontSize: '10px',
                flex: 1,
              }}
            >
              BILD
            </span>
            <div
              style={{
                width: '6px',
                height: '6px',
                borderRadius: '50%',
                background: '#3b82f6',
              }}
            />
            <span style={{ color: 'rgba(255,255,255,0.5)', fontSize: '10px' }}>
              Jetzt
            </span>
          </div>
          {dachzeile && (
            <div
              style={{
                color: 'rgba(255,255,255,0.5)',
                fontSize: '10px',
                fontWeight: 600,
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
                marginBottom: '2px',
              }}
            >
              {dachzeile}
            </div>
          )}
          <div
            style={{
              color: 'rgba(255,255,255,0.9)',
              fontSize: '12px',
              fontWeight: 400,
              lineHeight: 1.4,
              display: '-webkit-box',
              WebkitLineClamp: 3,
              WebkitBoxOrient: 'vertical',
              overflow: 'hidden',
            }}
          >
            {title || 'Push-Titel eingeben…'}
          </div>
        </div>

        {/* Nav bar */}
        <div
          style={{
            position: 'absolute',
            bottom: '8px',
            display: 'flex',
            gap: '20px',
            alignItems: 'center',
          }}
        >
          {['‹', '○', '□'].map((s, i) => (
            <span
              key={i}
              style={{
                color: 'rgba(255,255,255,0.4)',
                fontSize: '18px',
                lineHeight: 1,
              }}
            >
              {s}
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}

// ── Char counter ───────────────────────────────────────────────────────────

function CharCount({ value, max }: { value: string; max: number }) {
  const len = value.length
  const color =
    len > max
      ? 'var(--red)'
      : len > max * 0.85
        ? 'var(--amber)'
        : 'var(--text-tertiary)'
  return (
    <div
      style={{ fontSize: '11px', color, textAlign: 'right', marginTop: '3px' }}
    >
      {len} / {max} Zeichen
    </div>
  )
}

// ── Main Modal ─────────────────────────────────────────────────────────────

interface PushPreviewModalProps {
  article: Article
  onClose: () => void
}

export function PushPreviewModal({ article, onClose }: PushPreviewModalProps) {
  const [title, setTitle] = useState(article.title)
  const [dachzeile, setDachzeile] = useState('')
  const { time, date, dateShort } = now()
  const { mutate, isPending, data, error } = useGenerateTitle()
  const aiResult = data as GenerateTitleResponse | undefined

  const handleGenerateTitle = useCallback(() => {
    mutate({
      url: article.url,
      title: article.title,
      category: article.category,
    })
  }, [mutate, article])

  // Apply AI title when generated
  useEffect(() => {
    if (aiResult?.title) setTitle(aiResult.title)
  }, [aiResult])

  const scoreVar = scoreVariant(article.score)

  return (
    <Modal open onClose={onClose} title="Push-Vorschau" width={700}>
      {/* Article meta */}
      <div
        style={{
          padding: '10px 14px',
          background: 'var(--bg)',
          borderRadius: 'var(--radius)',
          marginBottom: '16px',
          display: 'flex',
          alignItems: 'flex-start',
          gap: '12px',
        }}
      >
        <div style={{ flex: 1 }}>
          <div
            style={{
              fontSize: '12px',
              fontWeight: 500,
              lineHeight: 1.4,
              marginBottom: '4px',
            }}
          >
            {article.title}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--text-tertiary)' }}>
            {article.category}
          </div>
        </div>
        <div style={{ display: 'flex', gap: '6px', flexShrink: 0 }}>
          <Badge variant={scoreVar}>Score {fmtScore(article.score)}</Badge>
          {article.predictedOR != null && (
            <Badge variant="default">XOR {fmtOR(article.predictedOR)}</Badge>
          )}
        </div>
      </div>

      {/* Dachzeile + Title inputs */}
      <div
        style={{
          marginBottom: '16px',
          display: 'flex',
          flexDirection: 'column',
          gap: '10px',
        }}
      >
        <div>
          <label
            style={{
              fontSize: '12px',
              fontWeight: 500,
              color: 'var(--text-secondary)',
              display: 'block',
              marginBottom: '4px',
            }}
          >
            Dachzeile (optional)
          </label>
          <input
            type="text"
            value={dachzeile}
            onChange={(e) => setDachzeile(e.target.value)}
            placeholder="z.B. Eilmeldung, Breaking News…"
            maxLength={80}
            style={{
              width: '100%',
              fontFamily: 'inherit',
              fontSize: '13px',
              padding: '8px 12px',
              borderRadius: '6px',
              border: '1px solid var(--border)',
              background: 'var(--white)',
              color: 'var(--text)',
              boxSizing: 'border-box',
            }}
          />
          <CharCount value={dachzeile} max={80} />
        </div>

        <div>
          <label
            style={{
              fontSize: '12px',
              fontWeight: 500,
              color: 'var(--text-secondary)',
              display: 'block',
              marginBottom: '4px',
            }}
          >
            Push-Titel
          </label>
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Push-Titel eingeben…"
            maxLength={120}
            style={{
              width: '100%',
              fontFamily: 'inherit',
              fontSize: '13px',
              padding: '8px 12px',
              borderRadius: '6px',
              border: '1px solid var(--border)',
              background: 'var(--white)',
              color: 'var(--text)',
              fontWeight: 500,
              boxSizing: 'border-box',
            }}
          />
          <CharCount value={title} max={120} />
        </div>

        {/* AI generate button */}
        <Button
          onClick={handleGenerateTitle}
          disabled={isPending}
          variant="primary"
          style={{ alignSelf: 'flex-start' }}
        >
          {isPending ? (
            <>
              <Spinner size={13} color="#fff" /> Generiere…
            </>
          ) : (
            'KI-Titel generieren'
          )}
        </Button>

        {error && (
          <div
            style={{
              fontSize: '12px',
              color: 'var(--red)',
              padding: '6px 10px',
              background: 'var(--red-bg)',
              borderRadius: '5px',
            }}
          >
            KI-Titel Generierung fehlgeschlagen
          </div>
        )}

        {/* Alternative titles */}
        {aiResult?.alternativeTitles &&
          aiResult.alternativeTitles.length > 0 && (
            <div>
              <div
                style={{
                  fontSize: '12px',
                  color: 'var(--text-secondary)',
                  marginBottom: '5px',
                  fontWeight: 500,
                }}
              >
                Alternativen:
              </div>
              <div
                style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}
              >
                {aiResult.alternativeTitles.map((t, i) => (
                  <button
                    key={i}
                    onClick={() => setTitle(t)}
                    style={{
                      textAlign: 'left',
                      fontFamily: 'inherit',
                      fontSize: '12px',
                      padding: '6px 10px',
                      borderRadius: '5px',
                      border: '1px solid var(--border)',
                      background: 'var(--white)',
                      cursor: 'pointer',
                      color: 'var(--text)',
                      transition: 'background 0.1s',
                    }}
                    onMouseEnter={(e) =>
                      ((e.currentTarget as HTMLElement).style.background =
                        'var(--accent-light)')
                    }
                    onMouseLeave={(e) =>
                      ((e.currentTarget as HTMLElement).style.background =
                        'var(--white)')
                    }
                  >
                    {t}
                  </button>
                ))}
              </div>
            </div>
          )}
      </div>

      {/* Phone Previews */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '20px',
          marginBottom: '16px',
        }}
      >
        <IOSPreview
          title={title}
          dachzeile={dachzeile}
          time={time}
          date={date}
        />
        <AndroidPreview
          title={title}
          dachzeile={dachzeile}
          time={time}
          dateShort={dateShort}
        />
      </div>

      {/* Footer actions */}
      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
        <Button
          variant="primary"
          onClick={() => window.open(article.url, '_blank')}
        >
          Artikel auf BILD.de
        </Button>
        <Button onClick={onClose}>Schließen</Button>
      </div>
    </Modal>
  )
}
