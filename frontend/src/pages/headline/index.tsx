import { useState } from 'react'
import type { FormEvent } from 'react'
import {
  Alert,
  Badge,
  Button,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Input,
  Spinner,
} from '@spring-media/editorial-one-ui'
import { useGenerateHeadline } from '@/hooks/use-api'
import { getApiErrorMessage } from '@/utils/api-errors'
import type { HeadlineVariant } from '@/types/api'

const CMS_ID_PATTERN = /^[0-9a-fA-F]{24}$/

function SparkleIcon() {
  return (
    <svg
      aria-hidden="true"
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
    >
      <path
        d="M12 3l1.35 4.65L18 9l-4.65 1.35L12 15l-1.35-4.65L6 9l4.65-1.35L12 3Z"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinejoin="round"
      />
      <path
        d="M19 15l.68 2.32L22 18l-2.32.68L19 21l-.68-2.32L16 18l2.32-.68L19 15Z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
    </svg>
  )
}

function VariantCard({
  variant,
  copied,
  onCopy,
}: {
  variant: HeadlineVariant
  copied: boolean
  onCopy: (variant: HeadlineVariant) => void
}) {
  return (
    <div
      style={{
        padding: '16px',
        border: `1px solid ${variant.selected ? 'var(--accent)' : 'var(--border)'}`,
        borderRadius: 'var(--radius)',
        background: variant.selected ? 'var(--accent-light)' : 'var(--white)',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '12px',
          flexWrap: 'wrap',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Badge variant={variant.selected ? 'purple' : 'default'}>
            Variante {variant.id}
          </Badge>
          <span
            style={{
              fontSize: '11px',
              fontWeight: 600,
              color: 'var(--text-secondary)',
              letterSpacing: '0.03em',
            }}
          >
            {variant.type}
          </span>
        </div>
        {variant.selected ? <Badge variant="green">Empfehlung</Badge> : null}
      </div>

      <div>
        <div
          style={{
            fontSize: '16px',
            fontWeight: 650,
            lineHeight: 1.35,
            color: 'var(--text)',
          }}
        >
          {variant.headline}
        </div>
        <div
          style={{
            marginTop: '4px',
            fontSize: '11px',
            color: 'var(--text-tertiary)',
          }}
        >
          Headline · {variant.headlineLength} Zeichen
        </div>
      </div>

      {variant.line2 ? (
        <div
          style={{
            borderTop: '1px solid var(--border-light)',
            paddingTop: '10px',
          }}
        >
          <div
            style={{
              fontSize: '14px',
              lineHeight: 1.4,
              color: 'var(--text-secondary)',
            }}
          >
            {variant.line2}
          </div>
          <div
            style={{
              marginTop: '4px',
              fontSize: '11px',
              color: 'var(--text-tertiary)',
            }}
          >
            Zeile 2 · {variant.line2Length} Zeichen
          </div>
        </div>
      ) : null}

      <div>
        <Button
          type="button"
          size="sm"
          variant={copied ? 'primary' : 'default'}
          onClick={() => onCopy(variant)}
        >
          {copied
            ? 'Kopiert'
            : variant.line2
              ? 'Beide Zeilen kopieren'
              : 'Headline kopieren'}
        </Button>
      </div>
    </div>
  )
}

export function HeadlinePage() {
  const [articleId, setArticleId] = useState('')
  const [submitted, setSubmitted] = useState(false)
  const [copiedVariant, setCopiedVariant] = useState<string | null>(null)
  const generation = useGenerateHeadline()
  const normalizedArticleId = articleId.trim()
  const articleIdValid = CMS_ID_PATTERN.test(normalizedArticleId)

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setSubmitted(true)
    setCopiedVariant(null)
    if (!articleIdValid) return
    generation.mutate({ articleId: normalizedArticleId.toLowerCase() })
  }

  const handleCopy = async (variant: HeadlineVariant) => {
    const copyText = [variant.headline, variant.line2]
      .filter(Boolean)
      .join('\n')
    try {
      await navigator.clipboard.writeText(copyText)
      setCopiedVariant(variant.id)
      window.setTimeout(() => setCopiedVariant(null), 1800)
    } catch {
      setCopiedVariant(null)
    }
  }

  const data = generation.data

  return (
    <main
      style={{
        padding: '20px 24px 40px',
        maxWidth: '980px',
        margin: '0 auto',
        animation: 'fadeIn 0.2s ease',
      }}
    >
      <header style={{ marginBottom: '20px' }}>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            flexWrap: 'wrap',
          }}
        >
          <h1 style={{ fontSize: '18px', fontWeight: 700, margin: 0 }}>
            Headline-Vorschläge
          </h1>
          <Badge variant="purple">Prompt v1.4</Badge>
        </div>
        <p
          style={{
            fontSize: '12px',
            color: 'var(--text-secondary)',
            margin: '3px 0 0',
          }}
        >
          Artikel-ID eingeben und bis zu drei beratende Push-Varianten erzeugen
        </p>
      </header>

      <Card style={{ marginBottom: '16px' }}>
        <CardHeader>
          <CardTitle>Artikel auswählen</CardTitle>
          <span style={{ fontSize: '11px', color: 'var(--text-tertiary)' }}>
            Kein Versand · keine Power-Automate-Aktion
          </span>
        </CardHeader>
        <CardContent>
          <form
            onSubmit={handleSubmit}
            style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}
          >
            <div>
              <Input
                label="Artikel-ID"
                value={articleId}
                onChange={(event) => {
                  setArticleId(event.target.value)
                  setSubmitted(false)
                  if (generation.data || generation.error) generation.reset()
                }}
                placeholder="z. B. 0123456789abcdef01234567"
                autoComplete="off"
                spellCheck={false}
                maxLength={24}
                aria-describedby="headline-article-id-hint"
                aria-invalid={submitted && !articleIdValid}
                style={{
                  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
                }}
              />
              <div
                id="headline-article-id-hint"
                style={{
                  fontSize: '11px',
                  color:
                    submitted && !articleIdValid
                      ? 'var(--red)'
                      : 'var(--text-tertiary)',
                  marginTop: '5px',
                }}
              >
                {submitted && !articleIdValid
                  ? 'Bitte eine 24-stellige CMS-ID eingeben.'
                  : 'Die CMS-ID steht in Editorial One und am Ende der BILD Artikel-URL.'}
              </div>
            </div>

            <div>
              <Button
                type="submit"
                variant="primary"
                disabled={generation.isPending || !normalizedArticleId}
              >
                {generation.isPending ? (
                  <>
                    <Spinner size={13} color="#fff" /> Vorschläge werden
                    erzeugt…
                  </>
                ) : (
                  <>
                    <SparkleIcon /> Headline-Vorschläge generieren
                  </>
                )}
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      {generation.isError ? (
        <Alert variant="error" title="Vorschläge konnten nicht erzeugt werden">
          {getApiErrorMessage(
            generation.error,
            'Bitte Artikel-ID prüfen und erneut versuchen.',
          )}
        </Alert>
      ) : null}

      {data ? (
        <div
          aria-live="polite"
          style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}
        >
          <Card>
            <CardHeader>
              <CardTitle>Gefundener Artikel</CardTitle>
              <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                <Badge variant="default">{data.article.category}</Badge>
                <Badge variant="blue">{data.article.contentType}</Badge>
              </div>
            </CardHeader>
            <CardContent>
              <a
                href={data.article.url}
                target="_blank"
                rel="noreferrer"
                style={{
                  color: 'var(--text)',
                  fontSize: '15px',
                  fontWeight: 600,
                  lineHeight: 1.4,
                }}
              >
                {data.article.title}
              </a>
              <div
                style={{
                  marginTop: '6px',
                  fontSize: '11px',
                  color: 'var(--text-tertiary)',
                  overflowWrap: 'anywhere',
                }}
              >
                Artikel-ID: {data.article.articleId}
              </div>
            </CardContent>
          </Card>

          {!data.promptActive ? (
            <Alert variant="warning" title="Lokaler Fallback aktiv">
              {data.reviewPoint}
            </Alert>
          ) : null}

          {data.escalation ? (
            <Alert variant="warning" title="CvD-Prüfung erforderlich">
              {data.reviewPoint ?? data.reasoning}
            </Alert>
          ) : null}

          {data.variants.length > 0 ? (
            <Card>
              <CardHeader>
                <CardTitle>Vorschläge</CardTitle>
                <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                  {data.stage ? (
                    <Badge variant="blue">Stufe {data.stage}</Badge>
                  ) : null}
                  <Badge variant="purple">Prompt {data.promptVersion}</Badge>
                </div>
              </CardHeader>
              <CardContent>
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
                    gap: '12px',
                  }}
                >
                  {data.variants.map((variant) => (
                    <VariantCard
                      key={variant.id}
                      variant={variant}
                      copied={copiedVariant === variant.id}
                      onCopy={handleCopy}
                    />
                  ))}
                </div>
              </CardContent>
            </Card>
          ) : null}

          {data.reasoning || data.stageReason || data.reviewPoint ? (
            <Card>
              <CardHeader>
                <CardTitle>Redaktionelle Einordnung</CardTitle>
              </CardHeader>
              <CardContent>
                <div
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '8px',
                    fontSize: '12px',
                    color: 'var(--text-secondary)',
                  }}
                >
                  {data.stageReason ? (
                    <div>
                      <strong style={{ color: 'var(--text)' }}>Stufe:</strong>{' '}
                      {data.stageReason}
                    </div>
                  ) : null}
                  {data.reasoning ? (
                    <div>
                      <strong style={{ color: 'var(--text)' }}>
                        Empfehlung:
                      </strong>{' '}
                      {data.reasoning}
                    </div>
                  ) : null}
                  {data.reviewPoint ? (
                    <div>
                      <strong style={{ color: 'var(--text)' }}>
                        Prüfpunkt:
                      </strong>{' '}
                      {data.reviewPoint}
                    </div>
                  ) : null}
                </div>
              </CardContent>
            </Card>
          ) : null}
        </div>
      ) : null}
    </main>
  )
}
