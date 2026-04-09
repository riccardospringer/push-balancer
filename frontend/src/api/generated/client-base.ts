/*
 * Auto-generated from ../openapi.yaml (OpenAPI 3.0.3, API 3.0.1).
 * Run `pnpm generate:api-client` after changing the API spec.
 */

const BASE = import.meta.env.VITE_API_BASE ?? ''

type QueryParams = Record<string, string | number | undefined>

function withQuery(path: string, params: QueryParams = {}) {
  const search = new URLSearchParams()
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== '') {
      search.set(key, String(value))
    }
  })
  const query = search.toString()
  return query ? `${path}?${query}` : path
}

async function request<T>(
  path: string,
  method: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE',
  options: { body?: unknown; signal?: AbortSignal } = {},
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method,
    signal: options.signal,
    headers: {
      Accept: 'application/json',
      ...(options.body !== undefined
        ? { 'Content-Type': 'application/json' }
        : {}),
    },
    ...(options.body !== undefined
      ? { body: JSON.stringify(options.body) }
      : {}),
  })

  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText} - ${path}`)
  }

  return res.json() as Promise<T>
}

export const rawClient = {
  getHealth: (signal?: AbortSignal) =>
    request<unknown>('/api/health', 'GET', { signal }),

  listArticles: (params: ListArticlesParams = {}, signal?: AbortSignal) =>
    request<unknown>(withQuery('/api/articles', params), 'GET', { signal }),

  listPushes: (params: ListPushesParams = {}, signal?: AbortSignal) =>
    request<unknown>(withQuery('/api/pushes', params), 'GET', { signal }),

  refreshPushes: (body: unknown = {}, signal?: AbortSignal) =>
    request<unknown>('/api/pushes/refresh', 'POST', { body, signal }),

  listEditorialCompetitorItems: (signal?: AbortSignal) =>
    request<unknown>('/api/feeds/competitor', 'GET', { signal }),

  listSportCompetitorItems: (signal?: AbortSignal) =>
    request<unknown>('/api/feeds/competitor/sport', 'GET', { signal }),

  getResearchInsights: (signal?: AbortSignal) =>
    request<unknown>('/api/research-insights', 'GET', { signal }),

  listResearchRules: (
    params: ListResearchRulesParams = {},
    signal?: AbortSignal,
  ) =>
    request<unknown>(withQuery('/api/research-rules', params), 'GET', {
      signal,
    }),

  getAdobeTraffic: (signal?: AbortSignal) =>
    request<unknown>('/api/analytics/adobe-traffic', 'GET', { signal }),

  getMlModelStatus: (signal?: AbortSignal) =>
    request<unknown>('/api/ml-model', 'GET', { signal }),

  getMlModelMonitoring: (signal?: AbortSignal) =>
    request<unknown>('/api/ml-model/monitoring', 'GET', { signal }),

  createMlRetrainingJob: (body: unknown = {}, signal?: AbortSignal) =>
    request<unknown>('/api/ml-model/retraining-jobs', 'POST', { body, signal }),

  getGbrtModelStatus: (signal?: AbortSignal) =>
    request<unknown>('/api/gbrt-model', 'GET', { signal }),

  createGbrtRetrainingJob: (body: unknown = {}, signal?: AbortSignal) =>
    request<unknown>('/api/gbrt-model/retraining-jobs', 'POST', {
      body,
      signal,
    }),

  createGbrtPromotion: (body: unknown = {}, signal?: AbortSignal) =>
    request<unknown>('/api/gbrt-model/promotions', 'POST', { body, signal }),

  getDailyPlan: (params: GetDailyPlanParams = {}, signal?: AbortSignal) =>
    request<unknown>(withQuery('/api/tagesplan', params), 'GET', { signal }),

  getDailyPlanRetro: (
    params: GetDailyPlanRetroParams = {},
    signal?: AbortSignal,
  ) =>
    request<unknown>(withQuery('/api/tagesplan/retro', params), 'GET', {
      signal,
    }),

  listDailyPlanSuggestions: (
    params: ListDailyPlanSuggestionsParams = {},
    signal?: AbortSignal,
  ) =>
    request<unknown>(withQuery('/api/tagesplan/suggestions', params), 'GET', {
      signal,
    }),

  generatePushTitle: (body: unknown = {}, signal?: AbortSignal) =>
    request<unknown>('/api/push-title/generate', 'POST', { body, signal }),
} as const

export interface ListArticlesParams extends QueryParams {
  offset?: string | number
  limit?: string | number
}

export interface ListPushesParams extends QueryParams {
  limit?: string | number
}

export interface ListResearchRulesParams extends QueryParams {
  offset?: string | number
  limit?: string | number
}

export interface GetDailyPlanParams extends QueryParams {
  date?: string | number
  mode?: string | number
}

export interface GetDailyPlanRetroParams extends QueryParams {
  mode?: string | number
}

export interface ListDailyPlanSuggestionsParams extends QueryParams {
  date?: string | number
  mode?: string | number
}
