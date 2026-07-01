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
  scoreReason?: string
  performanceDrivers?: string[]
  risks?: string[]
  recommendedText?: string
  mixPriority?: 'hoch' | 'mittel' | 'niedrig' | string
  scoreBreakdown?: {
    bildFit?: number
    historicalTiming?: number
    mixBalance?: number
    openingRatePotential?: number
    riskAndFatigue?: number
    freshness?: number
    bildReiz?: number
    headlineStrength?: number
    politicsContext?: number
    videoFit?: number
    editorialFeedback?: number
  }
  teamsAlert?: {
    candidateId: string
    articleId: string
    articleUrl: string
    shouldNotify: boolean
    status: 'notify' | 'skip' | 'observe' | 'sent' | 'failed' | string
    summary?: string
    reasons?: string[]
    blockingReasons?: string[]
    scoreReason?: string
    performanceDrivers?: string[]
    risks?: string[]
    scoreBreakdown?: Article['scoreBreakdown']
    minutesSinceLastPush?: number | null
    lastPushAt?: string | null
    lastTeamsAlertAt?: string | null
    alertCount?: number
    evaluatedAt?: string
  }
  isBreaking?: boolean
  isEilmeldung?: boolean
  isSport?: boolean
  isPlusArticle?: boolean
}

export interface FeedResponse {
  articles: Article[]
  fetchedAt: string
  total: number
  count: number
  offset: number
  limit: number
}

export interface TeamsAlertRecord {
  articleKey: string
  articleId: string
  articleUrl: string
  articleTitle: string
  status: 'sent' | 'failed' | 'notify' | 'skip' | 'observe' | string
  score: number
  predictedOR: number
  reason: string
  lastError?: string
  alertCount: number
  lastAlertAt?: string | null
  lastDecisionAt?: string | null
  isBreaking?: boolean
}

export interface TeamsAlertsResponse {
  items: TeamsAlertRecord[]
  total: number
  fetchedAt: string
}

export interface TeamsRecommendationRecord {
  id: string
  articleKey: string
  articleId: string
  articleUrl: string
  articleTitle: string
  section: string
  type: 'teams_alert' | 'daily_plan' | string
  status: string
  shouldNotify: boolean
  score: number
  teamsAlertScore: number
  teamsAlertThreshold: number
  editorialScore: number
  predictedOR: number
  predictedORLabel: string
  expectedVisits: number
  dashboardRank: number
  summary: string
  reasons: string[]
  blockingReasons: string[]
  decision: Record<string, unknown>
  sendStatus: string
  sendError: string
  decidedAt?: string | null
  scheduledFor?: string | null
  sentAt?: string | null
}

export interface TeamsRecommendationsResponse {
  items: TeamsRecommendationRecord[]
  total: number
  type?: 'all' | 'teams_alert' | 'daily_plan' | string
  fetchedAt: string
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
  total: number
  offset: number
  limit: number
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

export interface TagesplanSuggestionsResponse {
  items: TagesplanSuggestion[]
  total: number
  offset: number
  limit: number
  grouped: Record<string, TagesplanSuggestion[]>
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


export interface PushAlarmArticle {
  title: string
  url: string
  score: number
  category: string
  predictedOR?: number
  reason?: string
  isBreaking?: boolean
  isEilmeldung?: boolean
  goldenHour?: boolean
  pushesToday?: number
  minsSinceLastPush?: number
}

export interface PushAlarmResponse {
  active: boolean
  article?: PushAlarmArticle
}
