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

function normalizeTagesplan(payload: unknown, mode: TagesplanMode): TagesplanResponse {
  const data = (payload ?? {}) as Record<string, unknown>
  const slotsRaw = Array.isArray(data.slots) ? data.slots : []
  const goldenHour = Number(data.golden_hour ?? 0)
  const pushedRaw = Array.isArray(data.already_pushed_today)
    ? (data.already_pushed_today as Array<Record<string, unknown>>)
    : []

  const slots = slotsRaw.map((slotRaw) => {
    const slot = slotRaw as Record<string, unknown>
    const pushedThisHour = Array.isArray(slot.pushed_this_hour)
      ? (slot.pushed_this_hour as Array<Record<string, unknown>>)
      : []
    const firstPushed = pushedThisHour[0]
    return {
      hour: Number(slot.hour ?? 0),
      label: `${String(slot.hour ?? 0).padStart(2, '0')}:00`,
      predictedOR: Number(slot.expected_or ?? 0) / 100,
      actualOR: firstPushed?.or != null ? Number(firstPushed.or) / 100 : undefined,
      pushed: pushedThisHour.length > 0,
      pushedTitle: firstPushed?.title ? String(firstPushed.title) : undefined,
      isGoldenHour: Number(slot.hour ?? -1) === goldenHour,
      recommendation: String(slot.mood ?? ''),
    }
  })

  const metrics = (data.ml_metrics as Record<string, unknown> | undefined) ?? {}

  return {
    date: String(data.date ?? ''),
    mode,
    slots,
    goldenHour,
    avgOR: Number(data.avg_or_today ?? 0) / 100,
    pushedCount: Number(data.n_pushed_today ?? 0),
    mae: Number(metrics.mae ?? 0) / 100,
    trainedOnRows: Number(data.total_pushes_db ?? 0),
    pushedToday: pushedRaw.map((row) => ({
      title: String(row.title ?? ''),
      hour: Number(row.hour ?? 0),
      or: Number(row.or ?? row.actual_or ?? 0) / 100,
    })),
    loading: Boolean(data.loading),
    llmReview: data.llm_review
      ? (data.llm_review as TagesplanResponse['llmReview'])
      : undefined,
  }
}

function normalizeTagesplanRetro(payload: unknown): TagesplanRetroResponse {
  const data = (payload ?? {}) as Record<string, unknown>
  const daysRaw = Array.isArray(data.days) ? data.days : []
  const summary = (data.summary ?? {}) as Record<string, unknown>

  return {
    days: daysRaw.map((raw) => {
      const day = raw as Record<string, unknown>
      const hours = (day.hours ?? {}) as Record<string, { pushes?: Array<Record<string, unknown>> }>
      const slots = Object.entries(hours).map(([hour, hourData]) => {
        const pushes = Array.isArray(hourData?.pushes) ? hourData.pushes : []
        const first = pushes[0]
        return {
          hour: Number(hour),
          label: `${String(hour).padStart(2, '0')}:00`,
          predictedOR: Number(first?.predicted_or ?? 0) / 100,
          actualOR: first?.actual_or != null ? Number(first.actual_or) / 100 : undefined,
          pushed: pushes.length > 0,
          pushedTitle: first?.title ? String(first.title) : undefined,
          isGoldenHour: false,
          recommendation: '',
        }
      })
      return {
        date: String(day.date_iso ?? day.date ?? ''),
        avgOR: Number(day.avg_or ?? 0) / 100,
        pushedCount: Number(day.n_pushed ?? 0),
        mae: Number(day.prediction_mae ?? 0) / 100,
        slots,
      }
    }),
    summary: {
      avgOR: Number(summary.avg_or_7d ?? 0) / 100,
      totalPushes: Number(summary.total_pushes ?? 0),
      avgMAE: Number(summary.prediction_mae_7d ?? 0) / 100,
    },
  }
}

function normalizeTagesplanSuggestions(payload: unknown): TagesplanSuggestionsResponse {
  const data = (payload ?? {}) as Record<string, unknown>
  const itemsRaw = Array.isArray(data.items) ? data.items : []

  const items = itemsRaw.map((raw) => {
    const row = raw as Record<string, unknown>
    return {
      hour: Number(row.hour ?? row.slot_hour ?? 0),
      title: String(row.title ?? ''),
      url: String(row.url ?? row.link ?? ''),
      score: Number(row.score ?? 0),
      predictedOR: Number(row.predictedOR ?? row.expected_or ?? 0),
    }
  })

  const grouped: Record<string, typeof items> = {}
  for (const item of items) {
    const key = String(item.hour)
    if (!grouped[key]) grouped[key] = []
    grouped[key].push(item)
  }

  return {
    items,
    total: Number(data.total ?? items.length),
    offset: Number(data.offset ?? 0),
    limit: Number(data.limit ?? items.length),
    grouped,
  }
}

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

  tagesplan: async (date?: string, mode: TagesplanMode = 'redaktion', signal?: AbortSignal) => {
    const payload = await rawClient.getDailyPlan({ date, mode }, signal)
    return normalizeTagesplan(payload, mode)
  },

  tagesplanRetro: async (mode: TagesplanMode = 'redaktion', signal?: AbortSignal) => {
    const payload = await rawClient.getDailyPlanRetro({ mode }, signal)
    return normalizeTagesplanRetro(payload)
  },

  tagesplanSuggestions: async (
    date?: string,
    mode?: TagesplanMode,
    signal?: AbortSignal,
  ) => {
    const payload = await rawClient.listDailyPlanSuggestions({ date, mode }, signal)
    return normalizeTagesplanSuggestions(payload)
  },

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
