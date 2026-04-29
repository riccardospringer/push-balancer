import { useEffect, useRef } from 'react'
import { usePushAlarm, useDismissPushAlarm } from '@/hooks/use-push-alarm'
import { fmtOR } from '@/utils/format'
import type { PushAlarmArticle } from '@/types/api'

function triggerBrowserNotification(article: PushAlarmArticle) {
  if (!('Notification' in window)) return
  const fire = () => {
    new Notification('Push-Alarm — Jetzt pushen!', {
      body: `${article.title}\nScore ${article.score} · ${article.isBreaking ? 'BREAKING' : article.category}`,
      icon: '/favicon.ico',
      tag: 'push-alarm',
      requireInteraction: true,
    })
  }
  if (Notification.permission === 'granted') {
    fire()
  } else if (Notification.permission === 'default') {
    Notification.requestPermission().then((p) => { if (p === 'granted') fire() })
  }
}

export function PushAlarmBanner() {
  const { data } = usePushAlarm()
  const dismiss = useDismissPushAlarm()
  const notifiedRef = useRef<string | null>(null)

  const active = data?.active && data.article
  const article = data?.article ?? null

  useEffect(() => {
    if (active && article && notifiedRef.current !== article.title) {
      notifiedRef.current = article.title
      triggerBrowserNotification(article)
    }
    if (!active) notifiedRef.current = null
  }, [active, article])

  if (!active || !article) return null

  const isBreaking = article.isBreaking || article.isEilmeldung
  const bg = isBreaking ? 'var(--red)' : article.goldenHour ? '#d97706' : 'var(--accent)'

  return (
    <div
      style={{
        background: bg,
        color: '#fff',
        padding: '10px 20px',
        display: 'flex',
        alignItems: 'center',
        gap: '16px',
        flexWrap: 'wrap',
        position: 'sticky',
        top: 0,
        zIndex: 200,
        boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
      }}
    >
      {/* Puls-Icon */}
      <span style={{ fontSize: '18px', flexShrink: 0 }}>
        {isBreaking ? '🚨' : article.goldenHour ? '⭐' : '🔔'}
      </span>

      {/* Text */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontWeight: 700, fontSize: '14px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          Jetzt pushen: {article.title}
        </div>
        <div style={{ fontSize: '12px', opacity: 0.88, marginTop: '2px' }}>
          {article.reason}
          {article.predictedOR != null && ` · XOR ${fmtOR(article.predictedOR)}`}
          {article.pushesToday != null && ` · ${article.pushesToday} Pushes heute`}
          {article.minsSinceLastPush != null && ` · letzter Push vor ${article.minsSinceLastPush} Min.`}
        </div>
      </div>

      {/* Buttons */}
      <div style={{ display: 'flex', gap: '8px', flexShrink: 0 }}>
        <a
          href={article.url}
          target="_blank"
          rel="noreferrer"
          style={{
            background: 'rgba(255,255,255,0.25)',
            color: '#fff',
            border: '1px solid rgba(255,255,255,0.5)',
            borderRadius: '6px',
            padding: '5px 12px',
            fontSize: '12px',
            fontWeight: 600,
            cursor: 'pointer',
            textDecoration: 'none',
            fontFamily: 'inherit',
          }}
        >
          Artikel öffnen
        </a>
        <button
          onClick={() => dismiss.mutate()}
          style={{
            background: 'rgba(0,0,0,0.2)',
            color: '#fff',
            border: '1px solid rgba(255,255,255,0.3)',
            borderRadius: '6px',
            padding: '5px 12px',
            fontSize: '12px',
            cursor: 'pointer',
            fontFamily: 'inherit',
          }}
        >
          Dismiss (10 Min.)
        </button>
      </div>
    </div>
  )
}
