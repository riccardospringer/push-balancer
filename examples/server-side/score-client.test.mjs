import assert from "node:assert/strict";
import test from "node:test";

import {
  ScoreApiError,
  getScoreByCmsId,
  getScoresByCmsIds,
} from "./score-client.mjs";

const CMS_ID = "0123456789abcdef01234567";
const OTHER_CMS_ID = "fedcba987654321001234567";
const API_KEY = "synthetic-score-key";
const BASE_URL = "https://score.example.invalid";

test("returns a validated score without putting the key in the URL", async () => {
  let capturedUrl;
  let capturedOptions;
  const result = await getScoreByCmsId({
    baseUrl: BASE_URL,
    cmsId: CMS_ID,
    apiKey: API_KEY,
    maxAttempts: 1,
    fetchImpl: async (url, options) => {
      capturedUrl = url;
      capturedOptions = options;
      return new Response(
        JSON.stringify({ cmsId: CMS_ID, score: 87.4, scoredAt: "2026-01-01T12:00:00" }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    },
  });

  assert.deepEqual(result, {
    cmsId: CMS_ID,
    score: 87.4,
    scoredAt: "2026-01-01T12:00:00",
    scoreBreakdown: null,
    orFactor: null,
  });
  assert.equal(capturedUrl.href, `${BASE_URL}/api/v1/scores/${CMS_ID}`);
  assert.equal(capturedUrl.href.includes(API_KEY), false);
  assert.equal(capturedOptions.headers["X-Score-Key"], API_KEY);
  assert.equal(capturedOptions.redirect, "error");
  assert.equal(capturedOptions.cache, "no-store");
});

test("returns the captured engagement breakdown without recomputing the score", async () => {
  const scoreBreakdown = {
    kind: "engagement",
    relevance: 30,
    urgency: 0,
    curiosity: 7.6,
    freshness: 11.7,
    timing: 6,
    titleBoost: 3,
    breaking: 0,
    research: 0,
    pushHistory: 0,
    topicSaturation: 0,
  };
  const result = await getScoreByCmsId({
    baseUrl: BASE_URL,
    cmsId: CMS_ID,
    apiKey: API_KEY,
    maxAttempts: 1,
    fetchImpl: async () =>
      new Response(
        JSON.stringify({
          cmsId: CMS_ID,
          score: 55,
          scoredAt: "2026-01-01T12:00:00Z",
          scoreBreakdown,
          orFactor: 1.06,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
  });

  assert.deepEqual(result, {
    cmsId: CMS_ID,
    score: 55,
    scoredAt: "2026-01-01T12:00:00Z",
    scoreBreakdown,
    orFactor: 1.06,
  });
});

test("accepts explicit null explanation fields from a legacy snapshot", async () => {
  const result = await getScoreByCmsId({
    baseUrl: BASE_URL,
    cmsId: CMS_ID,
    apiKey: API_KEY,
    maxAttempts: 1,
    fetchImpl: async () =>
      new Response(
        JSON.stringify({
          cmsId: CMS_ID,
          score: 55,
          scoredAt: "2026-01-01T12:00:00Z",
          scoreBreakdown: null,
          orFactor: null,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
  });

  assert.deepEqual(result, {
    cmsId: CMS_ID,
    score: 55,
    scoredAt: "2026-01-01T12:00:00Z",
    scoreBreakdown: null,
    orFactor: null,
  });
});

test("ignores unknown future top-level fields", async () => {
  const result = await getScoreByCmsId({
    baseUrl: BASE_URL,
    cmsId: CMS_ID,
    apiKey: API_KEY,
    maxAttempts: 1,
    fetchImpl: async () =>
      new Response(
        JSON.stringify({
          cmsId: CMS_ID,
          score: 55,
          scoredAt: "2026-01-01T12:00:00Z",
          scoreBreakdown: null,
          orFactor: null,
          futureAdditiveField: "ignored",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
  });

  assert.deepEqual(result, {
    cmsId: CMS_ID,
    score: 55,
    scoredAt: "2026-01-01T12:00:00Z",
    scoreBreakdown: null,
    orFactor: null,
  });
});

test("rejects a partial score explanation", async () => {
  await assert.rejects(
    getScoreByCmsId({
      baseUrl: BASE_URL,
      cmsId: CMS_ID,
      apiKey: API_KEY,
      maxAttempts: 1,
      fetchImpl: async () =>
        new Response(
          JSON.stringify({
            cmsId: CMS_ID,
            score: 55,
            scoredAt: "2026-01-01T12:00:00Z",
            orFactor: 1.06,
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
    }),
    (error) =>
      error instanceof ScoreApiError &&
      error.message === "Score API response violates the v1 contract",
  );
});

test("rejects an out-of-range breakdown component", async () => {
  await assert.rejects(
    getScoreByCmsId({
      baseUrl: BASE_URL,
      cmsId: CMS_ID,
      apiKey: API_KEY,
      maxAttempts: 1,
      fetchImpl: async () =>
        new Response(
          JSON.stringify({
            cmsId: CMS_ID,
            score: 55,
            scoredAt: "2026-01-01T12:00:00Z",
            scoreBreakdown: {
              kind: "sport",
              sportRelevance: 36,
              timing: 18,
              drama: 12,
              freshness: 8,
            },
            orFactor: 1.06,
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
    }),
    (error) =>
      error instanceof ScoreApiError &&
      error.message === "Score API response violates the v1 contract",
  );
});

test("maps 404 to no current score without retrying", async () => {
  let calls = 0;
  const result = await getScoreByCmsId({
    baseUrl: BASE_URL,
    cmsId: CMS_ID,
    apiKey: API_KEY,
    fetchImpl: async () => {
      calls += 1;
      return { ok: false, status: 404 };
    },
  });

  assert.equal(result, null);
  assert.equal(calls, 1);
});

test("does not retry permanent authentication errors", async () => {
  let calls = 0;
  await assert.rejects(
    getScoreByCmsId({
      baseUrl: BASE_URL,
      cmsId: CMS_ID,
      apiKey: API_KEY,
      fetchImpl: async () => {
        calls += 1;
        return { ok: false, status: 401 };
      },
    }),
    (error) =>
      error instanceof ScoreApiError && error.status === 401 && error.retryable === false,
  );
  assert.equal(calls, 1);
});

test("rejects a non-TLS remote URL before sending the secret", async () => {
  let calls = 0;
  await assert.rejects(
    getScoreByCmsId({
      baseUrl: "http://score.example.invalid",
      cmsId: CMS_ID,
      apiKey: API_KEY,
      fetchImpl: async () => {
        calls += 1;
      },
    }),
    ScoreApiError,
  );
  assert.equal(calls, 0);
});

test("rejects an oversized success response", async () => {
  await assert.rejects(
    getScoreByCmsId({
      baseUrl: BASE_URL,
      cmsId: CMS_ID,
      apiKey: API_KEY,
      maxAttempts: 1,
      fetchImpl: async () =>
        new Response("x".repeat(64 * 1024 + 1), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
    }),
    (error) =>
      error instanceof ScoreApiError &&
      error.message === "Score API response exceeds the size limit" &&
      error.retryable === false,
  );
});

test("batch sends one body request and preserves duplicate positions", async () => {
  let capturedUrl;
  let capturedOptions;
  const responsePayload = {
    requestedCount: 3,
    uniqueCount: 2,
    foundCount: 2,
    notFoundCount: 1,
    results: [
      {
        status: "found",
        cmsId: CMS_ID.toUpperCase(),
        score: 55,
        scoredAt: "2026-01-01T12:00:00Z",
        scoreBreakdown: null,
        orFactor: null,
      },
      { status: "notFound", cmsId: OTHER_CMS_ID },
      {
        status: "found",
        cmsId: CMS_ID,
        score: 55,
        scoredAt: "2026-01-01T12:00:00Z",
        scoreBreakdown: null,
        orFactor: null,
      },
    ],
  };

  const result = await getScoresByCmsIds({
    baseUrl: BASE_URL,
    cmsIds: [CMS_ID.toUpperCase(), OTHER_CMS_ID, CMS_ID],
    apiKey: API_KEY,
    maxAttempts: 1,
    fetchImpl: async (url, options) => {
      capturedUrl = url;
      capturedOptions = options;
      return new Response(JSON.stringify(responsePayload), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    },
  });

  assert.deepEqual(result, responsePayload);
  assert.equal(capturedUrl.href, `${BASE_URL}/api/v1/scores/batch`);
  assert.equal(capturedOptions.method, "POST");
  assert.equal(capturedOptions.headers["X-Score-Key"], API_KEY);
  assert.equal(capturedOptions.redirect, "error");
  assert.equal(capturedOptions.cache, "no-store");
  assert.deepEqual(JSON.parse(capturedOptions.body), {
    cmsIds: [CMS_ID.toUpperCase(), OTHER_CMS_ID, CMS_ID],
  });
});

test("batch retries 429 once using the exact one-second Retry-After", async () => {
  let calls = 0;
  const delays = [];
  const result = await getScoresByCmsIds({
    baseUrl: BASE_URL,
    cmsIds: [CMS_ID],
    apiKey: API_KEY,
    maxAttempts: 2,
    sleepImpl: async (milliseconds) => {
      delays.push(milliseconds);
    },
    fetchImpl: async () => {
      calls += 1;
      if (calls === 1) {
        return new Response("", {
          status: 429,
          headers: { "Retry-After": "1" },
        });
      }
      return new Response(
        JSON.stringify({
          requestedCount: 1,
          uniqueCount: 1,
          foundCount: 0,
          notFoundCount: 1,
          results: [{ status: "notFound", cmsId: CMS_ID }],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    },
  });

  assert.equal(calls, 2);
  assert.deepEqual(delays, [1_000]);
  assert.deepEqual(result.results, [{ status: "notFound", cmsId: CMS_ID }]);
});

test("batch rejects malformed counts, order, and extra per-item fields", async () => {
  for (const payload of [
    {
      requestedCount: 1,
      uniqueCount: 1,
      foundCount: 0,
      notFoundCount: 1,
      results: [{ status: "notFound", cmsId: OTHER_CMS_ID }],
    },
    {
      requestedCount: 1,
      uniqueCount: 1,
      foundCount: 1,
      notFoundCount: 0,
      results: [{ status: "notFound", cmsId: CMS_ID }],
    },
    {
      requestedCount: 1,
      uniqueCount: 1,
      foundCount: 0,
      notFoundCount: 1,
      results: [{ status: "notFound", cmsId: CMS_ID, extra: true }],
    },
  ]) {
    await assert.rejects(
      getScoresByCmsIds({
        baseUrl: BASE_URL,
        cmsIds: [CMS_ID],
        apiKey: API_KEY,
        maxAttempts: 1,
        fetchImpl: async () =>
          new Response(JSON.stringify(payload), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }),
      }),
      (error) =>
        error instanceof ScoreApiError &&
        error.message === "Score API response violates the v1 batch contract",
    );
  }
});

test("batch rejects invalid input without a request", async () => {
  let calls = 0;
  for (const cmsIds of [[], ["not-a-cms-id"], Array(501).fill(CMS_ID)]) {
    await assert.rejects(
      getScoresByCmsIds({
        baseUrl: BASE_URL,
        cmsIds,
        apiKey: API_KEY,
        fetchImpl: async () => {
          calls += 1;
        },
      }),
      ScoreApiError,
    );
  }
  assert.equal(calls, 0);
});

test("batch rejects a response larger than one MiB", async () => {
  await assert.rejects(
    getScoresByCmsIds({
      baseUrl: BASE_URL,
      cmsIds: [CMS_ID],
      apiKey: API_KEY,
      maxAttempts: 1,
      fetchImpl: async () =>
        new Response("x".repeat(1024 * 1024 + 1), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
    }),
    (error) =>
      error instanceof ScoreApiError &&
      error.message === "Score API response exceeds the size limit",
  );
});
