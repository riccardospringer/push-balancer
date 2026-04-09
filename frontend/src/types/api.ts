// ── System ────────────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy'
  uptime: number
  checks: Record<string, { ok: boolean; latencyMs?: number; error?: string }>
  research?: {
    version: number
    lastUpdate: string
    rollingAccuracy?: number
  }
}

// ── Articles / Candidates ─────────────────────────────────────────────────────

export interface Article {
  id: string
  url: string
  title: string
  category: string
  pubDate: string
  modDate?: string
  score: number
  predictedOR?: number
  isBreaking?: boolean
  isEilmeldung?: boolean
  isSport?: boolean
  isPlusArticle?: boolean
}

export interface FeedResponse {
  articles: Article[]
  fetchedAt: string
  count: number
}

// ── Push Statistics ───────────────────────────────────────────────────────────

export interface Push {
  id: string
  title: string
  channel: string
  sentAt: string
  recipients: number
  opened: number
  openRate: number
  predictedOR?: number
  url?: string
}

export interface PushStatsResponse {
  pushes: Push[]
  today: {
    count: number
    avgOR: number
    topOR: number
    recipients: number
  }
  channels: string[]
}

// ── ML ────────────────────────────────────────────────────────────────────────

export interface MlStatusResponse {
  modelVersion: string
  trainedAt: string
  mae: number
  rmse: number
  r2: number
  trainingRows: number
  features: string[]
  isEnsemble: boolean
  advisoryOnly: true
  actionAllowed: false
}

export interface MlPredictRequest {
  title: string
  category: string
  hour?: number
  weekday?: number
  channel?: string
}

export interface MlPredictResponse {
  predictedOR: number
  confidence: number
  advisoryOnly: true
  actionAllowed: false
}

export interface MlMonitoringResponse {
  recentPredictions: Array<{
    id: string
    predictedOR: number
    actualOR?: number
    error?: number
    timestamp: string
  }>
  rollingMAE: number
  drift: number
}

export interface MlExperiment {
  id: string
  name: string
  mae: number
  rmse: number
  r2: number
  trainedAt: string
  notes?: string
}

// ── GBRT ──────────────────────────────────────────────────────────────────────

export interface GbrtStatusResponse {
  active: boolean
  modelVersion: string
  mae: number
  trainingRows: number
  features: string[]
  lastRetrain: string
}

// ── Tagesplan ─────────────────────────────────────────────────────────────────

export type TagesplanMode = 'redaktion' | 'sport'

export interface TagesplanSlot {
  hour: number
  label: string
  predictedOR: number
  actualOR?: number
  pushed?: boolean
  pushedTitle?: string
  pushedAt?: string
  isGoldenHour: boolean
  recommendation: string
}

export interface TagesplanResponse {
  date: string
  mode: TagesplanMode
  slots: TagesplanSlot[]
  goldenHour: number
  avgOR: number
  pushedCount: number
  mae: number
  trainedOnRows: number
  pushedToday: Array<{ title: string; hour: number; or: number }>
  loading?: boolean
  llmReview?: {
    summary: string
    warnings: string[]
  }
}

export interface TagesplanRetroResponse {
  days: Array<{
    date: string
    avgOR: number
    pushedCount: number
    mae: number
    slots: TagesplanSlot[]
  }>
  summary: {
    avgOR: number
    totalPushes: number
    avgMAE: number
  }
}

export interface TagesplanSuggestion {
  hour: number
  title: string
  url: string
  score: number
  predictedOR: number
}

// ── Competitor ────────────────────────────────────────────────────────────────

export interface CompetitorItem {
  title: string
  url: string
  pubDate: string
  outlet: string
  outletColor: string
  isGap: boolean
  isExklusiv: boolean
  isHot: boolean
  outlets: string[]
}

export interface CompetitorResponse {
  items: CompetitorItem[]
  summary: {
    total: number
    gaps: number
    exklusiv: number
    hot: number
  }
  fetchedAt: string
}

// ── Forschung ─────────────────────────────────────────────────────────────────

export interface ResearchRule {
  id: string
  category: string
  rule: string
  confidence: number
  supportCount: number
  createdAt: string
}

export interface ResearchRulesResponse {
  version: number
  rules: ResearchRule[]
  rollingAccuracy: number
  generatedAt: string
}

export interface ForschungResponse {
  learnings: Array<{
    id: string
    text: string
    impact: 'high' | 'medium' | 'low'
    createdAt: string
  }>
  experiments: MlExperiment[]
  abTest?: {
    control: { name: string; mae: number }
    treatment: { name: string; mae: number }
    pValue: number
    winner?: 'control' | 'treatment'
  }
}

// ── Analyse / Adobe ───────────────────────────────────────────────────────────

export interface AdobeTrafficResponse {
  hourly: Array<{ hour: number; pageviews: number; visitors: number }>
  topArticles: Array<{ title: string; url: string; pageviews: number }>
  fetchedAt: string
}

// ── Push Title / Schwab ────────────────────────────────────────────────────────

export interface GenerateTitleRequest {
  url: string
  title?: string
  category?: string
}

export interface GenerateTitleResponse {
  title: string
  alternativeTitles: string[]
  reasoning: string
  advisoryOnly: true
}
