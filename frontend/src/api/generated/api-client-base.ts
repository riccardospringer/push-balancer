/*
 * Auto-generated from ../push-balancer-api-v3.1.0.yaml (OpenAPI 3.1.0, API 3.1.0).
 * Run `pnpm generate:api-client` after changing the API spec.
 */

const BASE = import.meta.env.VITE_API_BASE ?? ''

type QueryParams = Record<string, string | number | undefined>

export interface ProblemDetails {
  type?: string
  title?: string
  status?: number
  detail?: string
  instance?: string
}

export class ApiError extends Error {
  status: number
  problem?: ProblemDetails

  constructor(status: number, message: string, problem?: ProblemDetails) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.problem = problem
  }
}

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
      ...(options.body !== undefined ? { 'Content-Type': 'application/json' } : {}),
    },
    ...(options.body !== undefined ? { body: JSON.stringify(options.body) } : {}),
  })

  if (!res.ok) {
    let problem: ProblemDetails | undefined
    try {
      problem = (await res.json()) as ProblemDetails
    } catch {
      problem = undefined
    }
    const message =
      problem?.detail ??
      problem?.title ??
      `${res.status} ${res.statusText} - ${path}`
    throw new ApiError(res.status, message, problem)
  }

  return res.json() as Promise<T>
}

export const rawClient = {
  getHealth: (signal?: AbortSignal) =>
    request<unknown>('/api/health', 'GET', { signal }),

  getMemoryStats: (signal?: AbortSignal) =>
    request<unknown>('/api/memory-stats', 'GET', { signal }),

  listArticles: (
    params: ListArticlesParams = {},
    signal?: AbortSignal,
  ) =>
    request<unknown>(withQuery('/api/articles', params), 'GET', { signal }),

  listPushes: (
    params: ListPushesParams = {},
    signal?: AbortSignal,
  ) =>
    request<unknown>(withQuery('/api/pushes', params), 'GET', { signal }),

  refreshPushes: (body: unknown = {}, signal?: AbortSignal) =>
    request<unknown>('/api/pushes/refresh', 'POST', { body, signal }),

  syncPushRelay: (body: unknown = {}, signal?: AbortSignal) =>
    request<unknown>('/api/pushes/sync', 'POST', { body, signal }),

  createPushRefreshJob: (body: unknown = {}, signal?: AbortSignal) =>
    request<unknown>('/api/push-refresh-jobs', 'POST', { body, signal }),

  createPredictionFeedback: (body: unknown = {}, signal?: AbortSignal) =>
    request<unknown>('/api/predictions/feedback', 'POST', { body, signal }),

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
    request<unknown>(withQuery('/api/research-rules', params), 'GET', { signal }),

  checkPlusByUrl: (
    params: CheckPlusByUrlParams = {},
    signal?: AbortSignal,
  ) =>
    request<unknown>(withQuery('/api/check-plus', params), 'GET', { signal }),

  checkPlusBatch: (body: unknown = {}, signal?: AbortSignal) =>
    request<unknown>('/api/check-plus', 'POST', { body, signal }),

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
    request<unknown>('/api/gbrt-model/retraining-jobs', 'POST', { body, signal }),

  createGbrtPromotion: (body: unknown = {}, signal?: AbortSignal) =>
    request<unknown>('/api/gbrt-model/promotions', 'POST', { body, signal }),

  getDailyPlan: (
    params: GetDailyPlanParams = {},
    signal?: AbortSignal,
  ) =>
    request<unknown>(withQuery('/api/tagesplan', params), 'GET', { signal }),

  getDailyPlanRetro: (
    params: GetDailyPlanRetroParams = {},
    signal?: AbortSignal,
  ) =>
    request<unknown>(withQuery('/api/tagesplan/retro', params), 'GET', { signal }),

  listDailyPlanHistory: (
    params: ListDailyPlanHistoryParams = {},
    signal?: AbortSignal,
  ) =>
    request<unknown>(withQuery('/api/tagesplan/history', params), 'GET', { signal }),

  listDailyPlanSuggestions: (
    params: ListDailyPlanSuggestionsParams = {},
    signal?: AbortSignal,
  ) =>
    request<unknown>(withQuery('/api/tagesplan/suggestions', params), 'GET', { signal }),

  createDailyPlanSuggestionsLog: (body: unknown = {}, signal?: AbortSignal) =>
    request<unknown>('/api/tagesplan/log-suggestions', 'POST', { body, signal }),

  generatePushTitle: (body: unknown = {}, signal?: AbortSignal) =>
    request<unknown>('/api/push-title/generate', 'POST', { body, signal }),

  createPushTitleGeneration: (body: unknown = {}, signal?: AbortSignal) =>
    request<unknown>('/api/push-title-generations', 'POST', { body, signal }),

  getBildSitemap: (signal?: AbortSignal) =>
    request<unknown>('/api/feed', 'GET', { signal }),

  proxyPushApi: (signal?: AbortSignal) =>
    request<unknown>('/api/push/{path}', 'GET', { signal }),

  listCompetitorFeeds: (signal?: AbortSignal) =>
    request<unknown>('/api/competitors', 'GET', { signal }),

  getCompetitorFeed: (signal?: AbortSignal) =>
    request<unknown>('/api/competitor/{name}', 'GET', { signal }),

  listSportCompetitorFeeds: (signal?: AbortSignal) =>
    request<unknown>('/api/sport-competitors', 'GET', { signal }),

  listSportEuropaFeeds: (
    params: ListSportEuropaFeedsParams = {},
    signal?: AbortSignal,
  ) =>
    request<unknown>(withQuery('/api/sport-europa', params), 'GET', { signal }),

  listSportGlobalFeeds: (
    params: ListSportGlobalFeedsParams = {},
    signal?: AbortSignal,
  ) =>
    request<unknown>(withQuery('/api/sport-global', params), 'GET', { signal }),

  listInternationalFeeds: (
    params: ListInternationalFeedsParams = {},
    signal?: AbortSignal,
  ) =>
    request<unknown>(withQuery('/api/international', params), 'GET', { signal }),

  getInternationalFeed: (signal?: AbortSignal) =>
    request<unknown>('/api/international/{name}', 'GET', { signal }),

  createCompetitorXorAnalysis: (body: unknown = {}, signal?: AbortSignal) =>
    request<unknown>('/api/competitors/xor', 'POST', { body, signal }),

  getResearchState: (signal?: AbortSignal) =>
    request<unknown>('/api/forschung', 'GET', { signal }),

  getResearchLearnings: (signal?: AbortSignal) =>
    request<unknown>('/api/learnings', 'GET', { signal }),

  getAdobeTrafficLegacy: (signal?: AbortSignal) =>
    request<unknown>('/api/adobe/traffic', 'GET', { signal }),

  getMlSafetyStatus: (signal?: AbortSignal) =>
    request<unknown>('/api/ml/safety-status', 'GET', { signal }),

  getMlLegacyStatus: (signal?: AbortSignal) =>
    request<unknown>('/api/ml/status', 'GET', { signal }),

  getMlPrediction: (
    params: GetMlPredictionParams = {},
    signal?: AbortSignal,
  ) =>
    request<unknown>(withQuery('/api/ml/predict', params), 'GET', { signal }),

  listMlExperiments: (
    params: ListMlExperimentsParams = {},
    signal?: AbortSignal,
  ) =>
    request<unknown>(withQuery('/api/ml/experiments', params), 'GET', { signal }),

  compareMlExperiments: (
    params: CompareMlExperimentsParams = {},
    signal?: AbortSignal,
  ) =>
    request<unknown>(withQuery('/api/ml/experiments/compare', params), 'GET', { signal }),

  getMlAbStatus: (signal?: AbortSignal) =>
    request<unknown>('/api/ml/ab-status', 'GET', { signal }),

  listMlMonitoringEvents: (
    params: ListMlMonitoringEventsParams = {},
    signal?: AbortSignal,
  ) =>
    request<unknown>(withQuery('/api/ml/monitoring', params), 'GET', { signal }),

  createMlLegacyRetrainingJob: (body: unknown = {}, signal?: AbortSignal) =>
    request<unknown>('/api/ml/retrain', 'POST', { body, signal }),

  createMlMonitoringTick: (body: unknown = {}, signal?: AbortSignal) =>
    request<unknown>('/api/ml/monitoring/tick', 'POST', { body, signal }),

  createMlBatchPrediction: (body: unknown = {}, signal?: AbortSignal) =>
    request<unknown>('/api/ml/predict-batch', 'POST', { body, signal }),

  createMlBatchPredictionAlias: (body: unknown = {}, signal?: AbortSignal) =>
    request<unknown>('/api/predict-batch', 'POST', { body, signal }),

  getGbrtLegacyStatus: (signal?: AbortSignal) =>
    request<unknown>('/api/gbrt/status', 'GET', { signal }),

  getGbrtModelJson: (signal?: AbortSignal) =>
    request<unknown>('/api/gbrt/model.json', 'GET', { signal }),

  getGbrtPrediction: (
    params: GetGbrtPredictionParams = {},
    signal?: AbortSignal,
  ) =>
    request<unknown>(withQuery('/api/gbrt/predict', params), 'GET', { signal }),

  createGbrtLegacyRetrainingJob: (body: unknown = {}, signal?: AbortSignal) =>
    request<unknown>('/api/gbrt/retrain', 'POST', { body, signal }),

  createGbrtLegacyPromotion: (body: unknown = {}, signal?: AbortSignal) =>
    request<unknown>('/api/gbrt/force-promote', 'POST', { body, signal }),

  createSchwabChatMessage: (body: unknown = {}, signal?: AbortSignal) =>
    request<unknown>('/api/schwab-chat', 'POST', { body, signal }),

  createSchwabApproval: (body: unknown = {}, signal?: AbortSignal) =>
    request<unknown>('/api/schwab-approval', 'POST', { body, signal }),

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

export interface CheckPlusByUrlParams extends QueryParams {
  url?: string | number
}

export interface GetDailyPlanParams extends QueryParams {
  date?: string | number
  mode?: string | number
}

export interface GetDailyPlanRetroParams extends QueryParams {
  mode?: string | number
}

export interface ListDailyPlanHistoryParams extends QueryParams {
  days?: string | number
  offset?: string | number
  limit?: string | number
}

export interface ListDailyPlanSuggestionsParams extends QueryParams {
  date?: string | number
  mode?: string | number
}

export interface ListSportEuropaFeedsParams extends QueryParams {
  offset?: string | number
  limit?: string | number
}

export interface ListSportGlobalFeedsParams extends QueryParams {
  offset?: string | number
  limit?: string | number
}

export interface ListInternationalFeedsParams extends QueryParams {
  offset?: string | number
  limit?: string | number
}

export interface GetMlPredictionParams extends QueryParams {
  title?: string | number
  cat?: string | number
  hour?: string | number
  is_eilmeldung?: string | number
  link?: string | number
  push_id?: string | number
}

export interface ListMlExperimentsParams extends QueryParams {
  offset?: string | number
  limit?: string | number
}

export interface CompareMlExperimentsParams extends QueryParams {
  a?: string | number
  b?: string | number
}

export interface ListMlMonitoringEventsParams extends QueryParams {
  offset?: string | number
  limit?: string | number
}

export interface GetGbrtPredictionParams extends QueryParams {
  title?: string | number
  cat?: string | number
  hour?: string | number
  is_eilmeldung?: string | number
  plus?: string | number
}

