import type {
  HealthResponse,
  FeedResponse,
  PushStatsResponse,
  MlStatusResponse,
  MlPredictRequest,
  MlPredictResponse,
  MlMonitoringResponse,
  MlExperiment,
  GbrtStatusResponse,
  TagesplanResponse,
  TagesplanRetroResponse,
  TagesplanSuggestion,
  TagesplanMode,
  CompetitorResponse,
  ResearchRulesResponse,
  ForschungResponse,
  AdobeTrafficResponse,
  GenerateTitleRequest,
  GenerateTitleResponse,
} from '@/types/api'

const BASE = import.meta.env.VITE_API_BASE ?? ''

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    signal,
    headers: { Accept: 'application/json' },
  })
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText} — ${path}`)
  }
  return res.json() as Promise<T>
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText} — ${path}`)
  }
  return res.json() as Promise<T>
}

// ── System ────────────────────────────────────────────────────────────────────

export const api = {
  health: (signal?: AbortSignal) =>
    get<HealthResponse>('/api/health', signal),

  // ── Feed ───────────────────────────────────────────────────────────────────

  feed: (signal?: AbortSignal) =>
    get<FeedResponse>('/api/feed', signal),

  checkPlus: (urls: string[]) =>
    post<Record<string, boolean>>('/api/check-plus', { urls }),

  // ── Push Stats ─────────────────────────────────────────────────────────────

  pushStats: (signal?: AbortSignal) =>
    get<PushStatsResponse>('/api/push', signal),

  syncPush: () => post<{ ok: boolean; synced: number }>('/api/pushes/sync', {}),

  // ── ML ─────────────────────────────────────────────────────────────────────

  mlStatus: (signal?: AbortSignal) =>
    get<MlStatusResponse>('/api/ml/status', signal),

  mlSafety: (signal?: AbortSignal) =>
    get<{ safe: boolean; reason?: string }>('/api/ml/safety', signal),

  mlPredict: (body: MlPredictRequest) =>
    post<MlPredictResponse>('/api/ml/predict', body),

  mlPredictBatch: (articles: MlPredictRequest[]) =>
    post<MlPredictResponse[]>('/api/ml/predict/batch', { articles }),

  mlMonitoring: (signal?: AbortSignal) =>
    get<MlMonitoringResponse>('/api/ml/monitoring', signal),

  mlExperiments: (signal?: AbortSignal) =>
    get<MlExperiment[]>('/api/ml/experiments', signal),

  mlCompareExperiments: (ids: string[]) =>
    post<{ comparison: MlExperiment[] }>('/api/ml/experiments/compare', { ids }),

  mlAbStatus: (signal?: AbortSignal) =>
    get<{ active: boolean; variant?: string }>('/api/ml/ab-status', signal),

  mlRetrain: () =>
    post<{ ok: boolean; jobId: string }>('/api/ml/retrain', {}),

  mlFeedback: (predictionId: string, actualOR: number) =>
    post<{ ok: boolean }>('/api/ml/feedback', { predictionId, actualOR }),

  // ── GBRT ───────────────────────────────────────────────────────────────────

  gbrtStatus: (signal?: AbortSignal) =>
    get<GbrtStatusResponse>('/api/gbrt/status', signal),

  gbrtModelJson: (signal?: AbortSignal) =>
    get<unknown>('/api/gbrt/model.json', signal),

  gbrtPredict: (body: MlPredictRequest) =>
    post<MlPredictResponse>('/api/gbrt/predict', body),

  gbrtRetrain: () =>
    post<{ ok: boolean }>('/api/gbrt/retrain', {}),

  gbrtPromote: () =>
    post<{ ok: boolean }>('/api/gbrt/promote', {}),

  // ── Tagesplan ──────────────────────────────────────────────────────────────

  tagesplan: (date?: string, mode?: TagesplanMode, signal?: AbortSignal) => {
    const params = new URLSearchParams()
    if (date) params.set('date', date)
    if (mode) params.set('mode', mode)
    const qs = params.toString()
    return get<TagesplanResponse>(`/api/tagesplan${qs ? `?${qs}` : ''}`, signal)
  },

  tagesplanRetro: (mode?: TagesplanMode, signal?: AbortSignal) => {
    const params = new URLSearchParams()
    if (mode) params.set('mode', mode)
    const qs = params.toString()
    return get<TagesplanRetroResponse>(`/api/tagesplan/retro${qs ? `?${qs}` : ''}`, signal)
  },

  tagesplanHistory: (days?: number, signal?: AbortSignal) => {
    const params = new URLSearchParams()
    if (days) params.set('days', String(days))
    const qs = params.toString()
    return get<TagesplanRetroResponse>(`/api/tagesplan/history${qs ? `?${qs}` : ''}`, signal)
  },

  tagesplanSuggestions: (date?: string, mode?: TagesplanMode, signal?: AbortSignal) => {
    const params = new URLSearchParams()
    if (date) params.set('date', date)
    if (mode) params.set('mode', mode)
    const qs = params.toString()
    return get<TagesplanSuggestion[]>(`/api/tagesplan/suggestions${qs ? `?${qs}` : ''}`, signal)
  },

  tagesplanLogSuggestions: (suggestions: TagesplanSuggestion[]) =>
    post<{ ok: boolean }>('/api/tagesplan/suggestions/log', { suggestions }),

  // ── Competitor ─────────────────────────────────────────────────────────────

  competitorRedaktion: (signal?: AbortSignal) =>
    get<CompetitorResponse>('/api/feeds/competitor', signal),

  competitorSport: (signal?: AbortSignal) =>
    get<CompetitorResponse>('/api/feeds/competitor/sport', signal),

  competitorXor: (channel: string, signal?: AbortSignal) =>
    get<{ xor: number; channel: string }>(`/api/competitors/xor?channel=${channel}`, signal),

  // ── Forschung / Analytics ──────────────────────────────────────────────────

  forschung: (signal?: AbortSignal) =>
    get<ForschungResponse>('/api/forschung', signal),

  researchRules: (signal?: AbortSignal) =>
    get<ResearchRulesResponse>('/api/research-rules', signal),

  adobeTraffic: (signal?: AbortSignal) =>
    get<AdobeTrafficResponse>('/api/adobe/traffic', signal),

  learnings: (signal?: AbortSignal) =>
    get<{ learnings: string[] }>('/api/learnings', signal),

  // ── AI / Push-Title ────────────────────────────────────────────────────────

  generateTitle: (body: GenerateTitleRequest) =>
    post<GenerateTitleResponse>('/api/push-title/generate', body),

  monitoringTick: () =>
    post<{ ok: boolean }>('/api/monitoring/tick', {}),
}
