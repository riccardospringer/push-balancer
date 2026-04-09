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
import { rawClient } from './client-base'

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
    const payload = (await rawClient.getHealth(signal)) as {
      status: string
      uptimeSeconds?: number
      endpoints?: Record<
        string,
        { ok: boolean; status?: number; error?: string }
      >
      researchLastAnalysis?: number
      researchDataPoints?: number
    }

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

  feed: (signal?: AbortSignal) =>
    rawClient.listArticles({}, signal) as Promise<FeedResponse>,

  checkPlus: (urls: string[]) =>
    post<Record<string, boolean>>('/api/check-plus', { urls }),

  pushStats: (signal?: AbortSignal) =>
    rawClient.listPushes({}, signal) as Promise<PushStatsResponse>,

  syncPush: () =>
    rawClient.refreshPushes({}) as Promise<{ ok: boolean; synced: number }>,

  mlStatus: (signal?: AbortSignal) =>
    rawClient.getMlModelStatus(signal) as Promise<MlStatusResponse>,

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
    rawClient.getMlModelMonitoring(signal) as Promise<MlMonitoringResponse>,

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
    rawClient.createMlRetrainingJob({}) as Promise<{
      ok: boolean
      message?: string
      jobId?: string
    }>,

  mlFeedback: (predictionId: string, actualOR: number) =>
    post<{ ok: boolean }>('/api/predictions/feedback', {
      pushId: predictionId,
      actualOr: actualOR,
    }),

  gbrtStatus: (signal?: AbortSignal) =>
    rawClient.getGbrtModelStatus(signal) as Promise<GbrtStatusResponse>,

  gbrtModelJson: (signal?: AbortSignal) =>
    get<unknown>('/api/gbrt/model.json', signal),

  gbrtPredict: (body: MlPredictRequest) =>
    post<MlPredictResponse>('/api/gbrt/predict', {
      title: body.title,
      cat: body.category,
      hour: body.hour,
    }),

  gbrtRetrain: () =>
    rawClient.createGbrtRetrainingJob({}) as Promise<{ ok: boolean }>,

  gbrtPromote: () =>
    rawClient.createGbrtPromotion({}) as Promise<{ ok: boolean }>,

  tagesplan: (date?: string, mode?: TagesplanMode, signal?: AbortSignal) =>
    rawClient.getDailyPlan(
      { date, mode },
      signal,
    ) as Promise<TagesplanResponse>,

  tagesplanRetro: (mode?: TagesplanMode, signal?: AbortSignal) =>
    rawClient.getDailyPlanRetro(
      { mode },
      signal,
    ) as Promise<TagesplanRetroResponse>,

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
    rawClient.listDailyPlanSuggestions({ date, mode }, signal) as Promise<
      TagesplanSuggestion[]
    >,

  tagesplanLogSuggestions: (suggestions: TagesplanSuggestion[]) =>
    post<{ ok: boolean }>('/api/tagesplan/log-suggestions', { suggestions }),

  competitorRedaktion: (signal?: AbortSignal) =>
    rawClient.listEditorialCompetitorItems(
      signal,
    ) as Promise<CompetitorResponse>,

  competitorSport: (signal?: AbortSignal) =>
    rawClient.listSportCompetitorItems(signal) as Promise<CompetitorResponse>,

  competitorXor: (titles: string[]) =>
    post<Record<string, { xor: number }>>('/api/competitors/xor', { titles }),

  forschung: (signal?: AbortSignal) =>
    rawClient.getResearchInsights(signal) as Promise<ForschungResponse>,

  researchRules: async (
    signal?: AbortSignal,
  ): Promise<ResearchRulesResponse> => {
    const payload = (await rawClient.listResearchRules({}, signal)) as {
      version: number
      items?: Array<Record<string, unknown>>
      accuracy?: number
      lastUpdate?: number
    }

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
    const payload = (await rawClient.getAdobeTraffic(
      signal,
    )) as AdobeTrafficResponse
    return {
      hourly: payload.hourly ?? [],
      topArticles: payload.topArticles ?? [],
      fetchedAt: payload.fetchedAt ?? '',
    }
  },

  learnings: (signal?: AbortSignal) =>
    get<{ learnings: string[] }>('/api/learnings', signal),

  generateTitle: (body: GenerateTitleRequest) =>
    rawClient.generatePushTitle(body) as Promise<GenerateTitleResponse>,

  monitoringTick: () => post<{ ok: boolean }>('/api/ml/monitoring/tick', {}),
}
