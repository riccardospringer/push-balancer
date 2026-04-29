import type {
  AdobeTrafficResponse,
  PushAlarmResponse,
  CompetitorResponse,
  FeedResponse,
  ForschungResponse,
  GbrtStatusResponse,
  GenerateTitleRequest,
  GenerateTitleResponse,
  HealthResponse,
  MlMonitoringResponse,
  MlStatusResponse,
  PushStatsResponse,
  ResearchRulesResponse,
  TagesplanMode,
  TagesplanResponse,
  TagesplanRetroResponse,
  TagesplanSuggestionsResponse,
} from '@/types/api'
import { rawClient, ApiError } from './api-client-base'

async function fetchJson<T>(path: string, method: 'GET' | 'POST' = 'GET', signal?: AbortSignal): Promise<T> {
  const res = await fetch(path, {
    method,
    signal,
    headers: { Accept: 'application/json' },
  })
  if (!res.ok) throw new ApiError(res.status, `${res.status} ${path}`)
  return res.json() as Promise<T>
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

  pushAlarm: (signal?: AbortSignal) =>
    fetchJson<PushAlarmResponse>('/api/push-alarm', 'GET', signal),

  dismissPushAlarm: () =>
    fetchJson<{ ok: boolean }>('/api/push-alarm/dismiss', 'POST'),

  pushStats: (signal?: AbortSignal) =>
    rawClient.listPushes({}, signal) as Promise<PushStatsResponse>,

  syncPush: () =>
    rawClient.createPushRefreshJob({}) as Promise<{
      ok: boolean
      synced: number
    }>,

  mlStatus: (signal?: AbortSignal) =>
    rawClient.getMlModelStatus(signal) as Promise<MlStatusResponse>,

  mlMonitoring: (signal?: AbortSignal) =>
    rawClient.getMlModelMonitoring(signal) as Promise<MlMonitoringResponse>,

  mlRetrain: () =>
    rawClient.createMlRetrainingJob({}) as Promise<{
      ok: boolean
      message?: string
      jobId?: string
    }>,

  gbrtStatus: (signal?: AbortSignal) =>
    rawClient.getGbrtModelStatus(signal) as Promise<GbrtStatusResponse>,

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

  tagesplanSuggestions: (
    date?: string,
    mode?: TagesplanMode,
    signal?: AbortSignal,
  ) =>
    rawClient.listDailyPlanSuggestions(
      { date, mode },
      signal,
    ) as Promise<TagesplanSuggestionsResponse>,

  competitorRedaktion: (signal?: AbortSignal) =>
    rawClient.listEditorialCompetitorItems(
      signal,
    ) as Promise<CompetitorResponse>,

  competitorSport: (signal?: AbortSignal) =>
    rawClient.listSportCompetitorItems(signal) as Promise<CompetitorResponse>,

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

  generateTitle: (body: GenerateTitleRequest) =>
    rawClient.createPushTitleGeneration(body) as Promise<GenerateTitleResponse>,
}
