import type {
  AdobeTrafficResponse,
  CompetitorResponse,
  FeedResponse,
  ForschungResponse,
  GbrtStatusResponse,
  GenerateTitleRequest,
  GenerateTitleResponse,
  HealthResponse,
  MlExperiment,
  MlMonitoringResponse,
  MlPredictRequest,
  MlPredictResponse,
  MlStatusResponse,
  PushStatsResponse,
  ResearchRulesResponse,
  TagesplanMode,
  TagesplanResponse,
  TagesplanRetroResponse,
  TagesplanSuggestion,
} from '@/types/api'

const BASE = import.meta.env.VITE_API_BASE ?? ''

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    signal,
    headers: { Accept: 'application/json' },
  })

  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText} - ${path}`)
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
    throw new Error(`${res.status} ${res.statusText} - ${path}`)
  }

  return res.json() as Promise<T>
}

function buildQuery(params: Record<string, string | number | undefined>) {
  const search = new URLSearchParams()
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== '') {
      search.set(key, String(value))
    }
  })
  const query = search.toString()
  return query ? `?${query}` : ''
}

export const api = {
  health: async (signal?: AbortSignal): Promise<HealthResponse> => {
    const payload = await get<{
      status: string
      uptimeSeconds?: number
      endpoints?: Record<
        string,
        { ok: boolean; status?: number; error?: string }
      >
      researchLastAnalysis?: number
      researchDataPoints?: number
    }>('/api/health', signal)

    const status =
      payload.status === 'ok'
        ? 'healthy'
        : payload.status === 'degraded'
          ? 'degraded'
          : 'unhealthy'

    return {
      status,
      uptime: payload.uptimeSeconds ?? 0,
      checks: Object.fromEntries(
        Object.entries(payload.endpoints ?? {}).map(([key, value]) => [
          key,
          {
            ok: value.ok,
            error: value.error,
          },
        ]),
      ),
      research: {
        version: payload.researchDataPoints ?? 0,
        lastUpdate: payload.researchLastAnalysis
          ? new Date(payload.researchLastAnalysis * 1000).toISOString()
          : '',
      },
    }
  },

  feed: (signal?: AbortSignal) => get<FeedResponse>('/api/articles', signal),

  checkPlus: (urls: string[]) =>
    post<Record<string, boolean>>('/api/check-plus', { urls }),

  pushStats: (signal?: AbortSignal) =>
    get<PushStatsResponse>('/api/pushes', signal),

  syncPush: () =>
    post<{ ok: boolean; synced: number }>('/api/pushes/refresh', {}),

  mlStatus: (signal?: AbortSignal) =>
    get<MlStatusResponse>('/api/ml-model', signal),

  mlSafety: (signal?: AbortSignal) =>
    get<{ safetyMode: string; advisoryOnly: boolean; actionAllowed: boolean }>(
      '/api/ml/safety-status',
      signal,
    ),

  mlPredict: (body: MlPredictRequest) =>
    post<MlPredictResponse>('/api/ml/predict', {
      title: body.title,
      cat: body.category,
      hour: body.hour,
    }),

  mlPredictBatch: (articles: MlPredictRequest[]) =>
    post<MlPredictResponse[]>('/api/ml/predict-batch', { articles }),

  mlMonitoring: (signal?: AbortSignal) =>
    get<MlMonitoringResponse>('/api/ml-model/monitoring', signal),

  mlExperiments: async (signal?: AbortSignal): Promise<MlExperiment[]> => {
    const payload = await get<{ items?: MlExperiment[] }>(
      '/api/ml/experiments',
      signal,
    )
    return payload.items ?? []
  },

  mlCompareExperiments: (ids: string[]) =>
    post<{ comparison: MlExperiment[] }>('/api/ml/experiments/compare', {
      ids,
    }),

  mlAbStatus: (signal?: AbortSignal) =>
    get<{ active: boolean; variant?: string }>('/api/ml/ab-status', signal),

  mlRetrain: () =>
    post<{ ok: boolean; message?: string; jobId?: string }>(
      '/api/ml-model/retraining-jobs',
      {},
    ),

  mlFeedback: (predictionId: string, actualOR: number) =>
    post<{ ok: boolean }>('/api/predictions/feedback', {
      pushId: predictionId,
      actualOr: actualOR,
    }),

  gbrtStatus: (signal?: AbortSignal) =>
    get<GbrtStatusResponse>('/api/gbrt-model', signal),

  gbrtModelJson: (signal?: AbortSignal) =>
    get<unknown>('/api/gbrt/model.json', signal),

  gbrtPredict: (body: MlPredictRequest) =>
    post<MlPredictResponse>('/api/gbrt/predict', {
      title: body.title,
      cat: body.category,
      hour: body.hour,
    }),

  gbrtRetrain: () =>
    post<{ ok: boolean }>('/api/gbrt-model/retraining-jobs', {}),

  gbrtPromote: () => post<{ ok: boolean }>('/api/gbrt-model/promotions', {}),

  tagesplan: (date?: string, mode?: TagesplanMode, signal?: AbortSignal) =>
    get<TagesplanResponse>(
      `/api/tagesplan${buildQuery({ date, mode })}`,
      signal,
    ),

  tagesplanRetro: (mode?: TagesplanMode, signal?: AbortSignal) =>
    get<TagesplanRetroResponse>(
      `/api/tagesplan/retro${buildQuery({ mode })}`,
      signal,
    ),

  tagesplanHistory: (days?: number, signal?: AbortSignal) =>
    get<TagesplanRetroResponse>(
      `/api/tagesplan/history${buildQuery({ days })}`,
      signal,
    ),

  tagesplanSuggestions: (
    date?: string,
    mode?: TagesplanMode,
    signal?: AbortSignal,
  ) =>
    get<TagesplanSuggestion[]>(
      `/api/tagesplan/suggestions${buildQuery({ date, mode })}`,
      signal,
    ),

  tagesplanLogSuggestions: (suggestions: TagesplanSuggestion[]) =>
    post<{ ok: boolean }>('/api/tagesplan/log-suggestions', { suggestions }),

  competitorRedaktion: (signal?: AbortSignal) =>
    get<CompetitorResponse>('/api/feeds/competitor', signal),

  competitorSport: (signal?: AbortSignal) =>
    get<CompetitorResponse>('/api/feeds/competitor/sport', signal),

  competitorXor: (titles: string[]) =>
    post<Record<string, { xor: number }>>('/api/competitors/xor', { titles }),

  forschung: (signal?: AbortSignal) =>
    get<ForschungResponse>('/api/research-insights', signal),

  researchRules: async (
    signal?: AbortSignal,
  ): Promise<ResearchRulesResponse> => {
    const payload = await get<{
      version: number
      items?: Array<Record<string, unknown>>
      accuracy?: number
      lastUpdate?: number
    }>('/api/research-rules', signal)

    return {
      version: payload.version ?? 0,
      rules: (payload.items ?? []).map((item, index) => ({
        id: String(item.id ?? index),
        category: String(item.category ?? item.cat ?? 'news'),
        rule: String(
          item.title ?? item.rule ?? item.message ?? 'Research rule',
        ),
        confidence: Number(item.confidence ?? 0),
        supportCount: Number(item.supportCount ?? item.n ?? 0),
        createdAt: payload.lastUpdate
          ? new Date(payload.lastUpdate * 1000).toISOString()
          : new Date(0).toISOString(),
      })),
      rollingAccuracy: Number(payload.accuracy ?? 0),
      generatedAt: payload.lastUpdate
        ? new Date(payload.lastUpdate * 1000).toISOString()
        : new Date(0).toISOString(),
    }
  },

  adobeTraffic: async (signal?: AbortSignal): Promise<AdobeTrafficResponse> => {
    const payload = await get<AdobeTrafficResponse>(
      '/api/analytics/adobe-traffic',
      signal,
    )
    return {
      hourly: payload.hourly ?? [],
      topArticles: payload.topArticles ?? [],
      fetchedAt: payload.fetchedAt ?? '',
    }
  },

  learnings: (signal?: AbortSignal) =>
    get<{ learnings: string[] }>('/api/learnings', signal),

  generateTitle: (body: GenerateTitleRequest) =>
    post<GenerateTitleResponse>('/api/push-title/generate', body),

  monitoringTick: () => post<{ ok: boolean }>('/api/ml/monitoring/tick', {}),
}
