import { api as generatedApi } from './generated/api-client'
import { rawClient } from './generated/api-client-base'
import type {
  PushStatsResponse,
  TagesplanMode,
  TagesplanResponse,
  TagesplanRetroResponse,
  TagesplanSuggestionsResponse,
} from '@/types/api'

function normalizeTagesplan(payload: unknown, mode: TagesplanMode): TagesplanResponse {
  const data = (payload ?? {}) as Record<string, unknown>
  const slotsRaw = Array.isArray(data.slots) ? data.slots : []
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
      predictedOR: Number(slot.expected_or ?? 0),
      actualOR:
        firstPushed && firstPushed.or != null
          ? Number(firstPushed.or)
          : undefined,
      pushed: pushedThisHour.length > 0,
      pushedTitle: firstPushed?.title ? String(firstPushed.title) : undefined,
      isGoldenHour: Number(slot.hour ?? -1) === Number(data.golden_hour ?? -2),
      recommendation: String(slot.mood ?? ''),
    }
  })

  return {
    date: String(data.date ?? ''),
    mode,
    slots,
    goldenHour: Number(data.golden_hour ?? data.goldenHour ?? 0),
    avgOR: Number(data.avg_or_today ?? data.avgOR ?? 0),
    pushedCount: Number(data.n_pushed_today ?? data.pushedCount ?? 0),
    mae: Number(
      ((data.ml_metrics as Record<string, unknown> | undefined)?.mae as number) ??
        data.mae ??
        0,
    ),
    trainedOnRows: Number(data.total_pushes_db ?? data.trainedOnRows ?? 0),
    pushedToday: pushedRaw.map((row) => ({
      title: String(row.title ?? ''),
      hour: Number(row.hour ?? 0),
      or: Number(row.or ?? row.actual_or ?? 0),
    })),
    loading: Boolean(data.loading),
    llmReview: (data.llm_review as TagesplanResponse['llmReview']) ?? undefined,
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
          predictedOR: Number(first?.predicted_or ?? 0),
          actualOR: first?.actual_or != null ? Number(first.actual_or) : undefined,
          pushed: pushes.length > 0,
          pushedTitle: first?.title ? String(first.title) : undefined,
          isGoldenHour: false,
          recommendation: '',
        }
      })
      return {
        date: String(day.date_iso ?? day.date ?? ''),
        avgOR: Number(day.avg_or ?? 0),
        pushedCount: Number(day.n_pushed ?? 0),
        mae: Number(day.prediction_mae ?? 0),
        slots,
      }
    }),
    summary: {
      avgOR: Number(summary.avg_or_7d ?? 0),
      totalPushes: Number(summary.total_pushes ?? 0),
      avgMAE: Number(summary.prediction_mae_7d ?? 0),
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

export const api = {
  ...generatedApi,
  pushStats: (
    params?: {
      limit?: number
      days?: number
      sort?: 'sentAt' | 'openRate' | 'performanceDelta' | 'recipients'
      category?: string
    },
    signal?: AbortSignal,
  ) => rawClient.listPushes(params ?? {}, signal) as Promise<PushStatsResponse>,
  tagesplan: async (date?: string, mode: TagesplanMode = 'redaktion', signal?: AbortSignal) => {
    const payload = await generatedApi.tagesplan(date, mode, signal)
    return normalizeTagesplan(payload, mode)
  },
  tagesplanRetro: async (mode: TagesplanMode = 'redaktion', signal?: AbortSignal) => {
    const payload = await generatedApi.tagesplanRetro(mode, signal)
    return normalizeTagesplanRetro(payload)
  },
  tagesplanSuggestions: async (
    date?: string,
    mode?: TagesplanMode,
    signal?: AbortSignal,
  ) => {
    const payload = await generatedApi.tagesplanSuggestions(date, mode, signal)
    return normalizeTagesplanSuggestions(payload)
  },
}
