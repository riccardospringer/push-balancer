# Consume the CMS-ID score API

This is the handoff contract for an approved internal backend. The production
base URL and credential are supplied by the platform and secret-management
owners after the activation gates in `SCORE_API_ACTIVATION.md` are approved.
Do not use the Render URL for this interface.

## Request and response

Send one CMS ID as a URL-encoded path segment and the dedicated credential as a
header:

```http
GET /api/v1/scores/0123456789abcdef01234567 HTTP/1.1
Host: <approved-private-host>
Accept: application/json
X-Score-Key: <injected-secret>
```

```json
{
  "cmsId": "0123456789abcdef01234567",
  "score": 58.3,
  "scoredAt": "2026-07-15T12:00:00Z",
  "scoreBreakdown": {
    "kind": "engagement",
    "relevance": 30,
    "urgency": 0,
    "curiosity": 7.6,
    "freshness": 11.7,
    "timing": 6,
    "titleBoost": 3,
    "breaking": 0,
    "research": 0,
    "pushHistory": 0,
    "topicSaturation": 0
  },
  "orFactor": 1.06
}
```

- `cms_id` accepts `A-Z`, `a-z`, `0-9`, `_`, and `-`, with a maximum length of
  128 characters.
- `score` is the score already calculated and displayed by the legacy Render
  candidate UI as **Gesamt**, on a scale from 0 to 100.
- `scoredAt` is the UTC timestamp of that browser-generated UI snapshot.
- `scoreBreakdown` contains the allowlisted captured numeric explanation. For engagement
  candidates, `relevance`, `urgency`, `curiosity`, `freshness`, `timing`, and
  `titleBoost` map to Relevanz, Dringlichkeit, Neugier, Aktualitaet, Timing, and
  Titel-Boost. `breaking`, `research`, `pushHistory`, and `topicSaturation` are
  included when those existing adjustments apply. Sport candidates instead use
  `kind: "sport"` with `sportRelevance`, `timing`, `drama`, and `freshness`.
- `orFactor` is the captured OR-Faktor used for candidate sorting. It is not
  added to `score`.
- `score` is authoritative. Existing score caps, age multipliers, and TV
  adjustments can prevent a naive sum of the explanation values from matching
  the total. The API forwards values exactly and never sums or recalculates
  them. Legacy snapshots return `scoreBreakdown: null` and `orFactor: null`.
- The latest captured value remains available for up to eight hours; use
  `scoredAt` when the consumer needs a stricter business freshness rule.
- The response intentionally excludes article title, URL, prose explanations,
  model metadata, and predicted opening-rate data.
- The service returns no zero or alternate fallback score.

The stable machine-readable contract is
[`../openapi/score-api-v1.yaml`](../openapi/score-api-v1.yaml). The endpoint is
versioned as `/api/v1`; additive fields are possible, so consumers should read
the documented fields and ignore unknown future top-level fields.

## Batch use

For article lists, send one request containing 1–500 CMS IDs:

```http
POST /api/v1/scores/batch HTTP/1.1
Host: <approved-private-host>
Accept: application/json
Content-Type: application/json
X-Score-Key: <injected-secret>

{"cmsIds":["0123456789abcdef01234567","fedcba987654321001234567"]}
```

```json
{
  "requestedCount": 2,
  "uniqueCount": 2,
  "foundCount": 1,
  "notFoundCount": 1,
  "results": [
    {
      "status": "found",
      "cmsId": "0123456789abcdef01234567",
      "score": 58.3,
      "scoredAt": "2026-07-15T12:00:00Z",
      "scoreBreakdown": null,
      "orFactor": null
    },
    {"status": "notFound", "cmsId": "fedcba987654321001234567"}
  ]
}
```

The body and result variants reject extra fields. IDs must be 24 hexadecimal
characters. Duplicates remain in their original positions; found/not-found
counts are position-based, while `uniqueCount` is case-insensitive. Any source,
network, ordering, size, or contract failure fails the whole call with `502`.
The service makes exactly one deduplicated Render batch request and never fans
out to single source requests. `notFound` means that no fresh candidate-UI
snapshot exists in Render's eight-hour window; it does not mean that the article
itself does not exist. A valid batch in which every item is `notFound` still
returns HTTP `200`.

## Server-side use only

Read `NEXT_PUSH_BALANCER_URL` and `SCORE_API_KEY` from the approved workload
configuration and secret manager. Never put the key in browser code, source
control, a query parameter, command-line argument, screenshot, ticket, or log.
Do not log the request URL because the CMS ID is part of its path.

The repository contains two references:

- [`../scripts/smoke_score_api.py`](../scripts/smoke_score_api.py) is an
  operational smoke check that reads the URL, key, and approved CMS ID only
  from environment variables and prints only the numeric score.
- [`../examples/server-side/score-client.mjs`](../examples/server-side/score-client.mjs)
  is a Node.js 20+ backend client with schema validation, redirect rejection,
  timeout, and bounded retry behavior.

Example smoke check:

```bash
export NEXT_PUSH_BALANCER_URL="https://<approved-private-host>"
read -r -s SCORE_API_KEY
export SCORE_API_KEY
read -r CMS_ID
export CMS_ID
python3 scripts/smoke_score_api.py
unset SCORE_API_KEY CMS_ID
```

Prefer workload secret injection over an interactive shell in normal use. The
example deliberately avoids placing either value in shell history.

## Error handling

| Status | Meaning | Consumer action |
|---|---|---|
| `200` | Current score returned | Validate the v1 shape and use `score` |
| `401` | Key missing or invalid | Do not retry; check secret injection/rotation |
| `404` | No Render UI score captured within the workday window exists for this ID, or hidden network denial | Treat as no score; do not invent a fallback |
| `422` | Invalid CMS-ID format | Do not retry; fix the caller |
| `429` | Two batch calls are already running on this worker | Honor `Retry-After`; retry with jitter |
| `500` | Unexpected runtime failure | Retry at most twice with backoff and jitter |
| `502` | Render UI score source unavailable | Retry at most twice with backoff and jitter |
| `503` | Score lookup unavailable | Retry at most twice with backoff and jitter |

Do not include the response body in logs. Error bodies use
`application/problem+json` and redact the CMS ID from `instance`, but callers
should still log only a status, internal correlation identifier, and coarse
outcome.

The Render score-source request times out after 25 seconds. Use a per-attempt
timeout of 35 seconds. For lists, call
`POST /api/v1/scores/batch` with the exact `{ "cmsIds": [...] }` body documented
above instead of issuing parallel single requests. At most two batches run per
worker; a third is rejected immediately instead of entering an unbounded queue.

## Handoff checklist

The consuming team must receive, through approved channels:

- the environment-specific private base URL
- workload access or approved source ranges
- mTLS/workload-identity configuration when required
- a separate staging and production `SCORE_API_KEY` secret reference
- the OpenAPI file and error-handling policy above
- an approved non-personal test CMS ID for the controlled live smoke test
- the key-rotation and incident contact

Production access is not complete until a positive request from the real
consumer network and a negative request from an unapproved network have both
been verified without CMS IDs appearing in logs.
