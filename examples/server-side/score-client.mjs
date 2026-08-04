/**
 * Minimal Node.js 20+ backend client for the Push Balancer score API.
 * Keep apiKey server-side and never log the request URL or response body.
 */

const CMS_ID_PATTERN = /^[A-Za-z0-9_-]{1,128}$/;
const BATCH_CMS_ID_PATTERN = /^[0-9a-fA-F]{24}$/;
const RETRYABLE_STATUS = new Set([500, 502, 503]);
const RETRYABLE_BATCH_STATUS = new Set([...RETRYABLE_STATUS, 429]);
const MAX_RESPONSE_BYTES = 64 * 1024;
const MAX_BATCH_RESPONSE_BYTES = 1024 * 1024;
const MAX_BATCH_SIZE = 500;

export class ScoreApiError extends Error {
  constructor(message, { status = undefined, retryable = false } = {}) {
    super(message);
    this.name = "ScoreApiError";
    this.status = status;
    this.retryable = retryable;
  }
}

function scoreUrl(baseUrl, cmsId) {
  if (!CMS_ID_PATTERN.test(cmsId)) {
    throw new ScoreApiError("CMS ID has an invalid format");
  }
  let base;
  try {
    base = new URL(baseUrl);
  } catch {
    throw new ScoreApiError("Score API base URL is invalid");
  }
  const localHttp =
    base.protocol === "http:" && ["127.0.0.1", "[::1]", "localhost"].includes(base.hostname);
  if (base.protocol !== "https:" && !localHttp) {
    throw new ScoreApiError("Score API base URL must use HTTPS");
  }
  if (base.username || base.password || base.search || base.hash) {
    throw new ScoreApiError("Score API base URL must not contain credentials, query, or fragment");
  }
  return new URL(`api/v1/scores/${encodeURIComponent(cmsId)}`, `${base.href.replace(/\/$/, "")}/`);
}

function batchScoreUrl(baseUrl) {
  let base;
  try {
    base = new URL(baseUrl);
  } catch {
    throw new ScoreApiError("Score API base URL is invalid");
  }
  const localHttp =
    base.protocol === "http:" && ["127.0.0.1", "[::1]", "localhost"].includes(base.hostname);
  if (base.protocol !== "https:" && !localHttp) {
    throw new ScoreApiError("Score API base URL must use HTTPS");
  }
  if (base.username || base.password || base.search || base.hash) {
    throw new ScoreApiError("Score API base URL must not contain credentials, query, or fragment");
  }
  return new URL("api/v1/scores/batch", `${base.href.replace(/\/$/, "")}/`);
}

function isBoundedNumber(value, minimum, maximum) {
  return (
    typeof value === "number" &&
    Number.isFinite(value) &&
    value >= minimum &&
    value <= maximum
  );
}

function hasExactKeys(value, expectedKeys) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  const actualKeys = Object.keys(value).sort();
  return (
    actualKeys.length === expectedKeys.length &&
    actualKeys.every((key, index) => key === [...expectedKeys].sort()[index])
  );
}

function validateScoreBreakdown(value) {
  if (value?.kind === "engagement") {
    const keys = [
      "kind",
      "relevance",
      "urgency",
      "curiosity",
      "freshness",
      "timing",
      "titleBoost",
      "breaking",
      "research",
      "pushHistory",
      "topicSaturation",
    ];
    if (
      !hasExactKeys(value, keys) ||
      !isBoundedNumber(value.relevance, 0, 30) ||
      !isBoundedNumber(value.urgency, 0, 25) ||
      !isBoundedNumber(value.curiosity, 0, 25) ||
      !isBoundedNumber(value.freshness, 0, 20) ||
      !isBoundedNumber(value.timing, 0, 15) ||
      !isBoundedNumber(value.titleBoost, 0, 15) ||
      !isBoundedNumber(value.breaking, 0, 15) ||
      !isBoundedNumber(value.research, 0, 12) ||
      !isBoundedNumber(value.pushHistory, -4, 8) ||
      !isBoundedNumber(value.topicSaturation, -30, 0)
    ) {
      throw new ScoreApiError("Score API response violates the v1 contract");
    }
    return Object.fromEntries(keys.map((key) => [key, value[key]]));
  }
  if (value?.kind === "sport") {
    const keys = ["kind", "sportRelevance", "timing", "drama", "freshness"];
    if (
      !hasExactKeys(value, keys) ||
      !isBoundedNumber(value.sportRelevance, 0, 35) ||
      !isBoundedNumber(value.timing, 0, 30) ||
      !isBoundedNumber(value.drama, 0, 25) ||
      !isBoundedNumber(value.freshness, 0, 10)
    ) {
      throw new ScoreApiError("Score API response violates the v1 contract");
    }
    return Object.fromEntries(keys.map((key) => [key, value[key]]));
  }
  throw new ScoreApiError("Score API response violates the v1 contract");
}

function validatePayload(payload, expectedCmsId) {
  if (
    payload === null ||
    typeof payload !== "object" ||
    payload.cmsId !== expectedCmsId ||
    typeof payload.score !== "number" ||
    !Number.isFinite(payload.score) ||
    payload.score < 0 ||
    payload.score > 100 ||
    typeof payload.scoredAt !== "string" ||
    payload.scoredAt.length === 0
  ) {
    throw new ScoreApiError("Score API response violates the v1 contract");
  }

  const hasBreakdown = Object.hasOwn(payload, "scoreBreakdown");
  const hasOrFactor = Object.hasOwn(payload, "orFactor");
  if (!hasBreakdown && !hasOrFactor) {
    return {
      cmsId: payload.cmsId,
      score: payload.score,
      scoredAt: payload.scoredAt,
      scoreBreakdown: null,
      orFactor: null,
    };
  }
  if (!hasBreakdown || !hasOrFactor) {
    throw new ScoreApiError("Score API response violates the v1 contract");
  }
  if (payload.scoreBreakdown === null && payload.orFactor === null) {
    return {
      cmsId: payload.cmsId,
      score: payload.score,
      scoredAt: payload.scoredAt,
      scoreBreakdown: null,
      orFactor: null,
    };
  }
  if (!isBoundedNumber(payload.orFactor, 0.6, 1.5)) {
    throw new ScoreApiError("Score API response violates the v1 contract");
  }
  const scoreBreakdown = validateScoreBreakdown(payload.scoreBreakdown);
  return {
    cmsId: payload.cmsId,
    score: payload.score,
    scoredAt: payload.scoredAt,
    scoreBreakdown,
    orFactor: payload.orFactor,
  };
}

async function readJsonPayload(response, maxResponseBytes = MAX_RESPONSE_BYTES) {
  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.toLowerCase().includes("application/json")) {
    throw new ScoreApiError("Score API response has an invalid content type");
  }
  const declaredLength = Number(response.headers.get("content-length"));
  if (Number.isFinite(declaredLength) && declaredLength > maxResponseBytes) {
    throw new ScoreApiError("Score API response exceeds the size limit");
  }
  if (!response.body || typeof response.body.getReader !== "function") {
    throw new ScoreApiError("Score API response body is unavailable");
  }

  const reader = response.body.getReader();
  const chunks = [];
  let size = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.byteLength;
    if (size > maxResponseBytes) {
      await reader.cancel().catch(() => {});
      throw new ScoreApiError("Score API response exceeds the size limit");
    }
    chunks.push(value);
  }

  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  try {
    return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch {
    throw new ScoreApiError("Score API response is not valid JSON");
  }
}

function validateBatchPayload(payload, cmsIds) {
  const topLevelKeys = [
    "requestedCount",
    "uniqueCount",
    "foundCount",
    "notFoundCount",
    "results",
  ];
  if (!hasExactKeys(payload, topLevelKeys) || !Array.isArray(payload.results)) {
    throw new ScoreApiError("Score API response violates the v1 batch contract");
  }
  const uniqueCount = new Set(cmsIds.map((cmsId) => cmsId.toLowerCase())).size;
  if (
    !Number.isInteger(payload.requestedCount) ||
    payload.requestedCount !== cmsIds.length ||
    !Number.isInteger(payload.uniqueCount) ||
    payload.uniqueCount !== uniqueCount ||
    !Number.isInteger(payload.foundCount) ||
    !Number.isInteger(payload.notFoundCount) ||
    payload.results.length !== cmsIds.length
  ) {
    throw new ScoreApiError("Score API response violates the v1 batch contract");
  }

  let foundCount = 0;
  const results = payload.results.map((item, index) => {
    const expectedCmsId = cmsIds[index];
    if (item?.status === "notFound") {
      if (!hasExactKeys(item, ["status", "cmsId"]) || item.cmsId !== expectedCmsId) {
        throw new ScoreApiError("Score API response violates the v1 batch contract");
      }
      return { status: "notFound", cmsId: item.cmsId };
    }
    if (
      item?.status !== "found" ||
      !hasExactKeys(item, [
        "status",
        "cmsId",
        "score",
        "scoredAt",
        "scoreBreakdown",
        "orFactor",
      ])
    ) {
      throw new ScoreApiError("Score API response violates the v1 batch contract");
    }
    let score;
    try {
      score = validatePayload(item, expectedCmsId);
    } catch (error) {
      if (error instanceof ScoreApiError) {
        throw new ScoreApiError("Score API response violates the v1 batch contract");
      }
      throw error;
    }
    foundCount += 1;
    return { status: "found", ...score };
  });

  if (
    payload.foundCount !== foundCount ||
    payload.notFoundCount !== cmsIds.length - foundCount
  ) {
    throw new ScoreApiError("Score API response violates the v1 batch contract");
  }
  return {
    requestedCount: payload.requestedCount,
    uniqueCount: payload.uniqueCount,
    foundCount: payload.foundCount,
    notFoundCount: payload.notFoundCount,
    results,
  };
}

const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

export async function getScoreByCmsId({
  baseUrl,
  cmsId,
  apiKey,
  timeoutMs = 35_000,
  maxAttempts = 3,
  fetchImpl = fetch,
}) {
  if (!apiKey) {
    throw new ScoreApiError("Score API key is required");
  }
  if (!Number.isInteger(maxAttempts) || maxAttempts < 1 || maxAttempts > 3) {
    throw new ScoreApiError("maxAttempts must be between 1 and 3");
  }
  if (!Number.isInteger(timeoutMs) || timeoutMs < 1_000 || timeoutMs > 120_000) {
    throw new ScoreApiError("timeoutMs must be between 1000 and 120000");
  }
  const url = scoreUrl(baseUrl, cmsId);

  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      const response = await fetchImpl(url, {
        method: "GET",
        headers: { Accept: "application/json", "X-Score-Key": apiKey },
        cache: "no-store",
        redirect: "error",
        signal: AbortSignal.timeout(timeoutMs),
      });
      if (response.status === 404) return null;
      if (!response.ok) {
        const retryable = RETRYABLE_STATUS.has(response.status);
        if (!retryable || attempt === maxAttempts) {
          throw new ScoreApiError("Score API request failed", {
            status: response.status,
            retryable,
          });
        }
      } else {
        const payload = await readJsonPayload(response);
        return validatePayload(payload, cmsId);
      }
    } catch (error) {
      if (error instanceof ScoreApiError && !error.retryable) throw error;
      if (attempt === maxAttempts) {
        throw error instanceof ScoreApiError
          ? error
          : new ScoreApiError("Score API network request failed", { retryable: true });
      }
    }
    const backoffMs = 250 * 2 ** (attempt - 1) + Math.floor(Math.random() * 100);
    await delay(backoffMs);
  }
  throw new ScoreApiError("Score API request failed");
}

export async function getScoresByCmsIds({
  baseUrl,
  cmsIds,
  apiKey,
  timeoutMs = 35_000,
  maxAttempts = 3,
  fetchImpl = fetch,
  sleepImpl = delay,
}) {
  if (!apiKey) {
    throw new ScoreApiError("Score API key is required");
  }
  if (
    !Array.isArray(cmsIds) ||
    cmsIds.length < 1 ||
    cmsIds.length > MAX_BATCH_SIZE ||
    cmsIds.some((cmsId) => typeof cmsId !== "string" || !BATCH_CMS_ID_PATTERN.test(cmsId))
  ) {
    throw new ScoreApiError("cmsIds must contain between 1 and 500 24-hex CMS IDs");
  }
  if (!Number.isInteger(maxAttempts) || maxAttempts < 1 || maxAttempts > 3) {
    throw new ScoreApiError("maxAttempts must be between 1 and 3");
  }
  if (!Number.isInteger(timeoutMs) || timeoutMs < 1_000 || timeoutMs > 120_000) {
    throw new ScoreApiError("timeoutMs must be between 1000 and 120000");
  }
  if (typeof sleepImpl !== "function") {
    throw new ScoreApiError("sleepImpl must be a function");
  }
  const url = batchScoreUrl(baseUrl);

  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    let retryDelayMs;
    try {
      const response = await fetchImpl(url, {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
          "X-Score-Key": apiKey,
        },
        body: JSON.stringify({ cmsIds }),
        cache: "no-store",
        redirect: "error",
        signal: AbortSignal.timeout(timeoutMs),
      });
      if (!response.ok) {
        const retryable = RETRYABLE_BATCH_STATUS.has(response.status);
        if (response.status === 429) {
          if (response.headers?.get("retry-after") !== "1") {
            throw new ScoreApiError("Score API batch response violates retry contract", {
              status: 429,
              retryable: false,
            });
          }
          retryDelayMs = 1_000;
        }
        if (!retryable || attempt === maxAttempts) {
          throw new ScoreApiError("Score API batch request failed", {
            status: response.status,
            retryable,
          });
        }
      } else {
        const payload = await readJsonPayload(response, MAX_BATCH_RESPONSE_BYTES);
        return validateBatchPayload(payload, cmsIds);
      }
    } catch (error) {
      if (error instanceof ScoreApiError && !error.retryable) throw error;
      if (attempt === maxAttempts) {
        throw error instanceof ScoreApiError
          ? error
          : new ScoreApiError("Score API batch network request failed", { retryable: true });
      }
    }
    const backoffMs =
      retryDelayMs ?? 250 * 2 ** (attempt - 1) + Math.floor(Math.random() * 100);
    await sleepImpl(backoffMs);
  }
  throw new ScoreApiError("Score API batch request failed");
}
