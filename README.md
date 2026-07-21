# Push Balancer

An ML-powered advisory system for optimizing BILD push notification scheduling. It predicts the expected Opening Rate (OR) for news articles using a multi-method ensemble and provides an 18-slot daily plan with article recommendations.

> **Safety mode: ADVISORY ONLY.** The system never sends push notifications autonomously. All predictions are recommendations for editorial staff.

---

## Contents

1. [Prerequisites](#prerequisites)
2. [Local Setup](#local-setup)
3. [Docker Setup](#docker-setup)
4. [Architecture Overview](#architecture-overview)
5. [API Endpoints](#api-endpoints)
6. [Database](#database)
7. [Deployment (Render)](#deployment-render)
8. [Environment Variables](#environment-variables)
9. [Development](#development)

---

## Prerequisites

- Python 3.11+ (3.13 recommended; matches the Docker image)
- pip
- Node.js 20+ and `pnpm` 10.x for the React frontend
- Access to the BILD Push Statistics API (`push-frontend.bildcms.de`) — internal network only
- Optional: OpenAI API key (manual title generation only when explicitly enabled; prediction-time LLM scoring stays off by default), Adobe Analytics credentials, Football-Data.org key, The Odds API key

---

## Local Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd push-balancer

# 2. Install dependencies
pip install -r requirements.txt
pnpm --dir frontend install --frozen-lockfile

# 3. Configure environment variables
cp .env.example .env
# Edit .env and fill in the required values
# Minimal setup: PUSH_API_BASE
# Optional features: OPENAI_API_KEY, Adobe credentials, admin keys, sports APIs
# Cost guard: all paid external APIs are globally disabled unless explicitly enabled, and each feature has its own additional opt-in

# 4. Start backend and frontend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8050
pnpm --dir frontend dev
```

The API starts on `http://localhost:8050` by default, and the Vite frontend on `http://localhost:5173`. For production-like local checks, run `pnpm --dir frontend build`; the generated assets are written to `dist-frontend/` and served by FastAPI.

### Editorial One UI Registry Setup

The frontend is prepared for the private `@spring-media/editorial-one-ui` package via [frontend/.npmrc](/Users/riccardo.longo/push-balancer/frontend/.npmrc). The committed `.npmrc` only declares the `spring-media` registry, so local builds stay quiet even when no token is present. As soon as the package is installed locally, Vite resolves the real package automatically; otherwise it transparently falls back to the local shim. Render is unaffected because the deployment only serves the prebuilt [dist-frontend/](/Users/riccardo.longo/push-balancer/dist-frontend) assets. To install the real private package when access is available:

```bash
export NPM_TOKEN=ghp_your_token_here
pnpm config set //npm.pkg.github.com/:_authToken "$NPM_TOKEN"
pnpm --dir frontend info @spring-media/editorial-one-ui
```

If the package is not yet available in your environment, the app uses the local shim in [frontend/src/editorial-one-ui-shim/index.tsx](/Users/riccardo.longo/push-balancer/frontend/src/editorial-one-ui-shim/index.tsx) while app code already imports `@spring-media/editorial-one-ui`. This is a temporary fallback and not a full replacement for validating against the real private package.

Application code should never import the shim directly. Use `@spring-media/editorial-one-ui` and `@spring-media/editorial-one-ui/fonts.css`; the Vite and TypeScript aliases keep the local fallback transparent until the private package can be installed.

### GitHub Ownership

The intended long-term home for this project is the `spring-media` GitHub organization rather than personal namespaces. If you create or migrate the repository there, prefer a canonical org URL such as `https://github.com/spring-media/push-balancer` and update the local `spring-media` remote to match.

When `push-balancer-api-v3.1.0.yaml` changes, regenerate the frontend base client with:

```bash
pnpm --dir frontend generate:api-client
```

### Supported Runtime Path

The supported application path for handover and production work is:

- FastAPI backend in [app/](/Users/riccardo.longo/push-balancer/app)
- React frontend in [frontend/](/Users/riccardo.longo/push-balancer/frontend)
- production assets in [dist-frontend/](/Users/riccardo.longo/push-balancer/dist-frontend)

### macOS: libomp for LightGBM

On macOS, SIP blocks `DYLD_LIBRARY_PATH`. The server auto-loads `~/.local/lib/libomp.dylib` at startup if present. To install:

```bash
brew install libomp
cp $(brew --prefix libomp)/lib/libomp.dylib ~/.local/lib/libomp.dylib
```

---

## Docker Setup

```bash
# Build the image
docker build -t push-balancer .

# Run with environment variables
docker run -p 8050:8050 \
  -e OPENAI_API_KEY=sk-... \
  -e PUSH_API_BASE=http://push-frontend.bildcms.de \
  -e FOOTBALL_DATA_KEY=... \
  -e ODDS_API_KEY=... \
  -e ADOBE_CLIENT_ID=... \
  -e ADOBE_CLIENT_SECRET=... \
  push-balancer
```

The Dockerfile is based on `python:3.13-slim` and exposes port `8050`. If you need a startup seed, provide a sanitized file via `PUSH_SNAPSHOT_PATH` at runtime instead of committing production data into the repository or image.

---

## Architecture Overview

### ML Pipeline

The OR prediction uses a 9-method ensemble resolved in priority order:

| Priority | Method | Description |
|---|---|---|
| 1 | **Unified Stacking Ensemble** | LightGBM + XGBoost + CatBoost → Ridge Meta-Learner (OOF, 5-fold TimeSeriesSplit) |
| 2 | **LightGBM** | Gradient boosted trees, ~60+ features, SHAP explanations |
| 3 | **GBRT** | Pure-Python Gradient Boosted Regression Trees (no numpy dependency) |
| 4 | **Stacking Heuristic** | Bayesian cat×hour baseline + similarity weighting |
| 5 | **TF-IDF Similarity** | Cosine similarity against historical pushes |
| 6 | **Sentence Embeddings** | Semantic similarity via transformer model |
| 7 | **Category×Hour Baseline** | Historical average OR per category and hour |
| 8 | **Optional LLM Scoring** | Opt-in quality check for prediction heuristics; disabled by default and cached when enabled |
| 9 | **Keyword Heuristic** | Rule-based fallback when no model is available |

The Stacking Ensemble is only activated when its MAE is within 2% of the single LightGBM baseline (safety gate). An Online Residual Corrector applies real-time bias correction per category and hour group.

**Features include:** title length, emotional word counts, BILD topic clusters (crime, royals, costs, health, auto, relationships, extreme weather), temporal features (hour sin/cos, weekday, prime time, Bundesliga windows), historical OR baselines, TF-IDF and embedding similarities, optional cached LLM scores, and sport-specific magnitude signals.

### Research Worker

An autonomous background thread runs every **20 seconds**:

- Fetches new push data from the BILD Push Statistics API (or Render sync endpoint)
- Updates push history in SQLite
- Runs OR prediction and logs to `prediction_log`
- Applies online residual correction

Periodic tasks triggered by cycle counter:

| Interval | Task |
|---|---|
| Every 30 cycles (~10 min) | Stacking meta-model retrain |
| Every 60 cycles (~20 min) | GBRT drift detection, monitoring tick |
| Every 90 cycles (~30 min) | GBRT online learning update |
| Every 360 cycles (~2h) | GBRT full retrain |
| Every 1080 cycles (~6h) | LightGBM full retrain |
| Every 1440 cycles (~8h) | Unified Stacking full retrain |
| Cycle 1 | GBRT + LightGBM first train |
| Cycle 5 | Unified Stacking first train |

### Tagesplan (Daily Schedule)

The Tagesplan covers **18 hourly slots (06:00–23:00)**. For each slot it computes:

- Historical OR baseline for category × hour
- Top article recommendations from the current BILD sitemap
- Expected OR forecast per article (using the active ML model)
- Slot-level confidence and category diversity score

Results are cached and refreshed every 5 minutes in the background. Suggestion snapshots are persisted to the `tagesplan_suggestions` table for retrospective analysis.

### Competitor & Context Intelligence

- **German competitors:** Welt, Spiegel, Focus, n-tv, Tagesschau, FAZ, SZ, Stern, T-Online, Zeit (+ sport-specific feeds)
- **International:** 24 outlets across Europe, US, Middle East, Asia, South America, Australia
- **Sports APIs:** Football-Data.org (Bundesliga, Champions League), The Odds API (betting context)
- **Adobe Analytics:** Traffic source breakdown (push / home / social / search / direct) matched to push headlines via fuzzy string matching, refreshed every 30 minutes

---

## API Endpoints

A full OpenAPI specification is maintained in [`push-balancer-api-v3.1.0.yaml`](push-balancer-api-v3.1.0.yaml). The documented, frontend-stable contract currently includes:

### GET Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/health` | Service health, endpoint checks, and research metadata |
| `GET /api/articles` | Article candidates from the BILD sitemap |
| `GET /api/v1/status` | Smoke test for the read-only consumer API |
| `GET /api/v1/recommendations` | Drop-in read-only consumer API for ranked recommendations |
| `GET /api/v1/articles` | Read-only consumer API for ranked articles and advisory scores |
| `GET /api/v1/scores` | Compact read-only consumer API for article score projections |
| `GET /api/pushes` | Recent push history with same-day aggregates |
| `GET /api/feeds/competitor` | Editorial competitor monitoring feed |
| `GET /api/feeds/competitor/sport` | Sports competitor monitoring feed |
| `GET /api/research-insights` | Current research learnings and experiment summary |
| `GET /api/research-rules` | Active research rules with pagination metadata |
| `GET /api/check-plus` | Check a single BILD URL for a paywall indication |
| `GET /api/analytics/adobe-traffic` | Adobe traffic analytics payload |
| `GET /api/ml-model` | Stable ML model status contract |
| `GET /api/ml-model/monitoring` | ML monitoring and recent prediction comparisons |
| `GET /api/gbrt-model` | Stable GBRT model status contract |
| `GET /api/tagesplan` | Daily planning slots with recommendations |
| `GET /api/tagesplan/retro` | Retrospective planning summary |
| `GET /api/tagesplan/history` | Historical slot-level planning aggregates |
| `GET /api/tagesplan/suggestions` | Suggested articles for the current plan |

### POST Endpoints

| Endpoint | Description |
|---|---|
| `POST /api/pushes/refresh` | Refresh the live push view |
| `POST /api/predictions/feedback` | Store observed OR feedback for a predicted push |
| `POST /api/check-plus` | Check multiple BILD URLs for a paywall indication |
| `POST /api/ml-model/retraining-jobs` | Trigger an ML retraining job |
| `POST /api/gbrt-model/retraining-jobs` | Trigger a GBRT retraining job |
| `POST /api/gbrt-model/promotions` | Promote the current GBRT candidate |
| `POST /api/tagesplan/log-suggestions` | Persist daily-plan suggestion snapshots |
| `POST /api/push-title-generations` | Generate advisory push headline variants |

Additional compatibility and operational helper endpoints still exist, but the frontend contract should prefer the documented endpoints above.

Compatibility endpoints are also marked at runtime with `Deprecation: true` and a `Sunset` header so downstream clients can detect that they should migrate to the stable contract.

Protected mutation endpoints require the `X-Admin-Key` header and remain unavailable when `ADMIN_API_KEY` is not configured. Downstream consumer endpoints require `Authorization: Bearer <CONSUMER_API_KEY>` or `X-Consumer-Key` and remain unavailable when `CONSUMER_API_KEY` is not configured.

### Consumer API

Use the versioned consumer API for backend-to-backend integrations that need current article candidates and scores without depending on the React frontend contract:

```bash
curl -H "Authorization: Bearer $CONSUMER_API_KEY" \
  "https://push-balancer.onrender.com/api/v1/recommendations?limit=20&minScore=70"

curl -H "Authorization: Bearer $CONSUMER_API_KEY" \
  "https://push-balancer.onrender.com/api/v1/scores?category=sport&limit=20"
```

The responses are read-only and advisory-only (`actionAllowed=false`). Production deployments should expose `/api/health` for platform checks and `/api/v1/*` for authenticated consumers only. Keep `/api/docs`, `/api/openapi.json`, and legacy `/api/*` routes behind the internal CIDR allowlist.

### Editorial Push Scoring

Article candidates are ranked by an editorial push score in [`app/scoring/editorial.py`](app/scoring/editorial.py). The score combines predicted opening-rate potential with BILD-specific signals: freshness, real news development, headline clarity, outrage/curiosity/emotion, broad audience relevance, video fit, and section mix. Politics remains eligible for top ranks when there is a concrete current development, but stale, abstract, complex, or debate-only politics receives explicit penalties.

The ranking is rebalanced after scoring so strong non-politics candidates from news, sport, entertainment, crime, consumer, service, and curiosity have a realistic chance when the top field is otherwise dominated by politics. Each article returns `scoreReason`, `performanceDrivers`, `risks`, `mixPriority`, `recommendedText`, and a structured `scoreBreakdown` so editors can see why the candidate is high or low.

---

## Database

SQLite database at `.push_history.db` in the repository root by default (override via `DB_PATH`). WAL mode is enabled for concurrent reads.

| Table | Description |
|---|---|
| `pushes` | Push notification history (OR, title, category, hour, channel stats, LLM scores) |
| `prediction_log` | Per-push predictions with actual OR feedback for ML training |
| `experiments` | Hyperparameter and metric log for each training run |
| `promotion_log` | Challenger vs. champion gate results |
| `embedding_cache` | Title embedding cache (hash → vector) |
| `monitoring_events` | Drift, calibration shift, MAE spike, and A/B events |
| `tagesplan_suggestions` | Saved article recommendations per date and slot hour |

Indexes cover `ts_num`, `cat`, `or_val × ts_num`, `hour × or_val`, and `date_iso`.

---

## Deployment (Render)

The service is defined in [`render.yaml`](render.yaml) as a Docker web service.

---

## Privacy & Governance

This repository includes privacy guardrails in [AGENTS.md](/Users/riccardo.longo/push-balancer/AGENTS.md) and a project-specific overview in [PRIVACY.md](/Users/riccardo.longo/push-balancer/PRIVACY.md).

Privacy-relevant implementation work should document:

- purpose
- data categories and data subjects
- external recipients or transfers
- retention and deletion approach
- safeguards and required approvals

Operational privacy rules in this repository:

- Do not commit production snapshots, raw push exports, or analytics dumps.
- Use `PUSH_SNAPSHOT_PATH` only for sanitized startup seeds mounted outside the repository.
- Keep `OPENAI_API_KEY`, `ADMIN_API_KEY`, `CONSUMER_API_KEY`, `PUSH_SYNC_SECRET`, Adobe credentials, and `NPM_TOKEN` out of source control.
- Admin mutation endpoints stay disabled unless `ADMIN_API_KEY` is explicitly configured.
- Relay sync stays disabled unless `PUSH_SYNC_SECRET` is configured on both sides.

```yaml
# render.yaml (excerpt)
services:
  - type: web
    name: push-balancer
    runtime: docker
    dockerfilePath: ./Dockerfile
```

### Push Data Sync

Because Render instances cannot reach the internal BILD Push Statistics API directly, a two-path strategy is used:

1. **Direct fetch** (`_push_auto_fetch_worker`): The Render instance tries to fetch `PUSH_API_BASE` directly every 120 seconds.
2. **Relay sync** (`POST /api/pushes/sync`): The local Mac server posts fresh push data to the Render instance every cycle, authenticated via `PUSH_SYNC_SECRET`. Set `RENDER_SYNC_URL` on the local server to enable this.
3. **Optional startup seed**: if you mount a sanitized snapshot file and point `PUSH_SNAPSHOT_PATH` at it, the service seeds SQLite at startup before any live fetch succeeds.

### Microsoft Teams Push Recommendations

`PUSH_TEAMS_ALERTS_ENABLED=1` starts a background worker that evaluates an expanded article field from the Push Balancer and sends Power Automate / Teams recommendations. The webhook secret belongs in `PUSH_TEAMS_WEBHOOK_URL` and must stay in Render secrets or `.env`, never in Git.

The live cadence is based on the full weekday/hour OR matrix and the Teams channel's own sent-message counter. Every day starts with binding decisions at `06:15` and `06:45`. Every red/yellow weekday cell also gets a `:15` Top-1 decision and a fresh `:45` re-ranking. Strong non-peak `:45` cells fill the plan to at least 15; days with many golden hours contain up to 18 binding decisions. The 10/11 o'clock dead zone is avoided unless recovery is mathematically necessary. Every decision keeps the raw Push Score 75 floor and all hard quality gates. A 30-minute independent Teams cooldown makes consecutive `:15`/`:45` decisions executable. If missed slots mean that the remaining binding plan can no longer reach 15, projected-shortfall recovery evaluates the strongest candidate at the next free cooldown edge instead of allowing another multi-hour gap.

After a normal Teams recommendation is sent, the next raw-score threshold rises without becoming impossible: at the first eligible moment after the 30-minute cooldown it is 80.0, then it falls linearly to the 75.0 baseline by minute 90. The curve uses only the last successfully sent Teams recommendation, never a live-push timestamp. A canonical Push Balancer score strictly above 80 always clears soft Alert Score, CvD-total, forecast, dashboard-rank, and daily-fatigue gates. It never clears quiet hours, the slot plan, exact live-push or Teams article/topic duplicates, publication/factual integrity, section rules, the daily cap, the global Teams cooldown, missing news-event protection, grounded-title approval, or transport checks. The decision and Teams payload state the current adaptive threshold and how many soft gates were waived.

When `PUSH_BALANCER_SCORE_API_ENABLED=1`, every current article is resolved to a CMS ID and queried through the internal score-only API. After the hard gates, the highest fresh API score wins strictly; Alert Score, CvD score, forecast, response estimate, and section fit may break only an exact API-score tie. HTTP 404, timeout/5xx, stale or malformed data, missing credentials, and 401/403 never fall back to a local score. Factual integrity, publication time, destination, section allowlists, morning fit, confirmed sport-event state, grounded title approval, hard quiet hours, Teams cooldown, and Teams article/topic deduplication remain mandatory. A missing score-qualified candidate is reported through aggregate title-free diagnostics instead of being hidden as a silent empty cycle. Verified breaking still bypasses slot waiting and the global Teams cooldown after all of its hard gates pass.

Between 05:00 and 09:59 Berlin time, an additional reader-value gate rejects isolated death or accident stories without an acute warning, concrete action, major public event, nationally prominent subject, or broader victim scope. This prevents a high model OR from turning a distressing but narrow local tragedy into a morning recommendation. The gate is deterministic and local; it does not send article data to an external model.

When explicitly approved and enabled, a versioned local review network fans the same ephemeral, minimized signal snapshot out to 17 deterministic specialists immediately before a real recommendation becomes eligible: context integrity, destination, live-push deduplication, Teams story/re-alert deduplication, freshness, factual risk, Teams cooldown, recommendation load, news value, forecast, response potential, slot timing, section fit, confirmed sport event, headline clarity, daily balance, and an adversarial skeptic. Safety and policy checks never inflate support. Consensus is calculated only from five evidence families: article forecast, expected push openings, slot timing, section fit, and headline clarity. At least three must explicitly approve and the evidence score must reach 60/100; cautions and abstentions contribute zero. One hard veto always wins, and `:45` lowers neither the evidence requirement nor any hard gate. Missing Teams deduplication state or unavailable/stale live-push history fails closed. The review performs no network or model call and is measured against a 50 ms local budget. Exact live-push deduplication and Teams duplicate approval are checked again immediately before the webhook POST. It is disabled by default and in `render.yaml` until the privacy incident described in `PRIVACY.md` is resolved and Privacy Manager, Product/System Owner, DPO, and Legal approve the new article-scoring purpose and payload wording.

Push titles use a second local jury. It compares every available source together: explicit feed suggestions, optional model output, locally generated variants, and the article headline. No source wins merely because it arrived first or came from a model. Each title is scored for clarity, specificity, reader relevance, grounded curiosity, and honesty. A strong title keeps the topic or actor visible, creates one specific answer the article can deliver, and avoids revealing every useful detail. Generic teasers, vague pronouns, `im Fokus` prefixes, manipulative bait, and unsupported numbers are rejected. This jury is a mandatory local core gate for every live recommendation, independent of the optional 17-reviewer network. A title below 68/100 or without a concrete click reason blocks delivery before the webhook; urgent facts still remain explicit instead of being hidden for curiosity.

After title selection, a final local CvD jury uses raw Push Score as its strongest direct dimension, followed by article strength, OR-forecast quality, exact weekday/slot fit, title quality, and the winner's margin over the next eligible candidate. Approved reviewer consensus becomes an additional dimension only when the optional network is enabled; otherwise that weight is removed and the remaining dimensions are reweighted, so a missing review is never represented as synthetic positive evidence. The weighted recommendation must reach `PUSH_TEAMS_MIN_RECOMMENDATION_QUALITY` (72/100 by default) and retain every hard approval. This final gate is mandatory for every live Teams recommendation. It runs before the in-memory reservation and database send claim, so a rejected recommendation cannot consume the cooldown for the next strong article. Breaking and fresh material sport states receive a three-minute send window; normal approved recommendations receive a five-minute window. The Teams message leads with that exact deadline, the raw Push Score and one overall recommendation-strength verdict, then separates `Warum dieser Push?`, `Warum jetzt?`, and a single `Gegencheck` instead of exposing the full internal scorecard.

Once per day, the worker sends a compact schedule containing all 15-18 binding `:15`/`:45` decisions, clearly marked golden-hour pairs and recovery cells, the strongest section for each matrix hour, sport context, and intentionally deprioritized hours. This planning message does not consume the recommendation cooldown. Sport is classified into separate states: prematch/starting-time notices wait, material live events must be at most 10 minutes old, finals at most 60 minutes old, and confirmed transfers/personnel decisions at most 180 minutes old. Only fresh material states can bypass slot waiting; they never bypass factual, duplicate, publication-time, or cooldown gates. This sport check is local and does not introduce another external API.

Before every worker cycle, the service refreshes actual live-push history for exact-article duplicate protection, retrospective story comparison, and aggregate reach baselines. A successful direct refresh or a relay snapshot no older than five minutes makes the duplicate check authoritative; stale, empty, or unavailable history blocks delivery fail-closed. Canonical article URLs and CMS document IDs are compared across the retained history, including the `urlId`-only representation returned by the live API. The same exact check runs again immediately before delivery, preventing a race when an article is live-pushed after selection. Similar titles or story slugs under a different URL and CMS ID remain comparison signals only. Actual push count, density, and timing never affect Teams pacing, ranking, slot deficit, or cooldown.

Duplicate protection combines actual live-push history with Teams send history. An article whose canonical URL or CMS document ID already appeared in a real push is never eligible for a Teams recommendation, including at `:45`, with a score above 80, or as breaking. A successfully recommended Teams article also cannot be recommended again while its 45-day Teams state is retained, even after a headline, score, forecast, or breaking flag changes. Different articles about the same topic are suppressed inside the configured Teams topic window. The database claim enforces exact Teams-article uniqueness across workers, while the in-memory guard covers rapid concurrent cycles. Candidates already blocked by that guard are removed before ranking so they cannot repeatedly occupy Top 1 and starve the next eligible article; rejected database claims and failed webhook attempts release their process-local reservation. Normal recommendations use the Teams channel's own sent-message counter, 30-minute global cooldown, and maximum of 18 daily messages. Verified breaking requires an explicit breaking flag plus an EIL/BREAKING title marker or trusted editorial provenance; once all factual, publication-time, section, quality, raw-score, live-push and Teams-dedup gates pass, it bypasses slot waiting, the global Teams cooldown, and the daily cap for immediate recommendation. Breaking never bypasses the hard 00:00-05:29 Berlin quiet window. In internal-score mode the final transport gate additionally requires `pushScoreSource=internal_score_api`; a local fallback cannot be sent.

Germany relevance is evaluated locally from the existing public article headline, URL path, section, and verified breaking state. Broadly relevant domestic policy, consumer, infrastructure, warning, and German-sport topics receive a transparent editorial adjustment, but in internal-score mode that adjustment cannot reorder unequal API scores. A separate bounded `germany_people` class recognizes only a named holder of a German public role plus a confirmed parenthood event in the People section. It treats that shape as concrete positive People news instead of abstract politics and still requires the API Push Score floor, Teams duplicate checks, and a suitable slot. Partner sex, marital status, sexual orientation, and other identity traits are neither matched nor inferred. Pure US domestic crime/people stories without a direct Germany angle are blocked. Other non-breaking international stories need an exceptional Push Score (85 for major geopolitical events, otherwise 90); verified worldwide breaking remains eligible for immediate review.

For offline editorial stress testing, `app.research.synthetic_reader_panel` provides a shadow-only matrix of 144 non-personal reader situations: twelve editorial interests, four attention states, and three usage motives. It accepts explicit article dictionaries and performs no network, database, model, or file access. Its `wouldOpenCells`, `wouldConsiderCells`, and `syntheticInterestIndex` fields describe deterministic test coverage only; every result is permanently marked `representsObservedUsers=false`, `canEstimateOpeningRate=false`, and `productionUseAllowed=false`. The bundled runner uses only dummy headlines and `example.invalid` URLs, including a synthetic German public-figure parenthood case and an anonymous counterexample. Its findings can inform a human editorial review, but they are not a Teams gate, an OR forecast, audience research, or permission to learn from real users.

Sent/attempted live recommendations and generated daily-plan entries are persisted for 45 days in `teams_recommendations`. If the review network is approved and enabled, the internal `GET /api/teams-recommendations` endpoint also returns compact local reviewer verdicts, consensus score, latency measurement, and strongest counterargument. It contains article and decision metadata, not the ephemeral reviewer snapshot, raw push history, or recipient- or employee-level activity. The individual verdicts are not placed in the Power Automate payload.

For Power Automate, use the trigger body field `messageHtml` as the Teams message content:

```text
@{triggerBody()?['messageHtml']}
```

The payload also includes structured fields such as `articleTitle`, `articleUrl`, `pushScore`, `pushScoreSource`, `pushScoreScoredAt`, `predictedORLabel`, `expectedOpens`, `responseMetric=expected_opens`, `livePushComparison`, `whyNow`, `whyPushworthy`, `recommendedPushText`, `recommendedSendWindow`, `recommendedSendBy`, `editorialReview`, `selectionScore`, `scoreReason`, `performanceDrivers`, `risks`, and `scoreBreakdown`. `livePushComparison` contains only `available`, `matched`, and `matchType`; raw live history is never sent to Teams. Reach multiplied by open rate is presented as expected push openings, never as visits; the legacy `expectedVisits` field remains only as a compatibility alias during consumer migration. When the local review network is enabled, only its concise consensus line and strongest counterargument are rendered in the existing message text/HTML; all individual verdicts remain local. The title jury always contributes only the compact `pushTitleReview` fields `approved`, `score`, and `clickReason`; its dimensions, grounding analysis, and risks remain local. The final jury always contributes only `recommendationQuality.approved`, `score`, `confidence`, and the minimized dispatch window; its full scorecard and blockers remain local. An internally built but unapproved object is marked `push_recommendation_preview`, has no recommended action, and is rejected by the sender before any webhook request. Low-confidence global-average prediction fallbacks are not shown as article-specific OR forecasts; they are rendered as "keine belastbare Prognose". `PUSH_TEAMS_DASHBOARD_TOP_LIMIT` remains the normal top-field guardrail, while `PUSH_TEAMS_CANDIDATE_LIMIT` controls how many candidates the automatic Teams worker inspects. In internal-score mode, the highest fresh API score among candidates that pass the hard gates is the final recommendation; response potential and local models can only break an exact API-score tie.

### CORS

Allowed origins are computed automatically from `PORT`, `RAILWAY_PUBLIC_DOMAIN`, `RENDER_EXTERNAL_HOSTNAME`, and the local network IP. The Render hostname `push-balancer.onrender.com` is always included.

### Internal Network Access

Use `INTERNAL_ACCESS_ENABLED=1` together with `INTERNAL_ACCESS_ALLOWED_CIDRS` to restrict the app to AS/VPN egress IPs. On Render this protection is enabled by default, so non-exempt routes stay closed until the AS network CIDRs are configured. Keep `/api/health,/api/v1` in `INTERNAL_ACCESS_EXEMPT_PATHS` so platform health checks and authenticated consumer API calls can work while docs and legacy routes remain internal. `SCORE_CAPTURE_CONSUMER_ALLOWED_CIDRS` is a separate least-privilege allowlist: it grants only `GET /api/score-capture/health` and `GET /api/score-capture/by-cms-id/{cms_id}` to the approved BILD Next consumer and does not grant UI, debug, or write access.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `PAID_EXTERNAL_APIS_ENABLED` | No | `false` | Master kill switch for all paid external API usage in the active runtime; local fallbacks stay available |
| `BACKGROUND_AUTOMATIONS_ENABLED` | No | `false` | Disables autonomous background polling, research loops, training ticks, push auto-fetch, and cache warmups; endpoints fall back to on-demand work where available |
| `HEALTH_ACTIVE_CHECKS_ENABLED` | No | `false` | Disables continuous outbound health probes; `/api/health` still reports passive runtime status |
| `ECONOMY_MODE` | No | `true` on Render, otherwise `false` | Render-first low-cost profile that keeps the service usable while avoiding expensive live fetches and external context lookups by default |
| `PUSH_LIVE_FETCH_ENABLED` | No | `false` in economy mode | Allows direct live polling of the internal Push Statistics API; when disabled the service uses cache/DB fallbacks only |
| `LIVE_FEED_FALLBACK_ENABLED` | No | `false` in economy mode | Allows live competitor/international feed fetches on cache miss; when disabled those endpoints return cached data or empty results |
| `RESEARCH_EXTERNAL_CONTEXT_ENABLED` | No | `false` in economy mode | Allows live weather and trend fetches for research analysis; when disabled research uses local defaults |
| `ARTICLE_PREDICTION_ENRICHMENT_ENABLED` | No | `false` in economy mode | Controls whether `/api/articles` enriches each item with on-the-fly OR predictions |
| `PUSH_BALANCER_CAPTURED_SCORE_MAX_AGE_SECONDS` | No | `180` | Maximum age of a candidate-view Push Balancer rating before Teams falls back to the local editorial score |
| `PUSH_BALANCER_SCORE_API_ENABLED` | Approval required | `false` | Makes the internal score API canonical for Teams and fails closed without a fresh valid response |
| `PUSH_BALANCER_SCORE_API_BASE_URL` | Yes, when enabled | — | HTTPS origin of the internal score-only API; never receives article text or reader data |
| `PUSH_BALANCER_SCORE_API_KEY` | Yes, when enabled | — | Runtime-only `X-Score-Key` secret; never put it in Git, URLs, logs, or Teams payloads |
| `PUSH_BALANCER_SCORE_API_MAX_AGE_SECONDS` | No | `900` | Maximum accepted age of `scoredAt`; stale scores are excluded without fallback |
| `PUSH_BALANCER_SCORE_API_TIMEOUT_SECONDS` | No | `2.5` | Per-request timeout before the one bounded retry |
| `PUSH_BALANCER_SCORE_API_CACHE_TTL_SECONDS` | No | `45` | Process-memory cache; short enough to force a fresh `:45` re-ranking after `:15` |
| `PUSH_BALANCER_SCORE_API_MAX_CONCURRENCY` | No | `16` | Hard-bounded parallel score lookups for the current candidate field |
| `TAGESPLAN_ON_DEMAND_BUILD_ENABLED` | No | `false` in economy mode | Controls whether `/api/tagesplan` builds a fresh plan on request; when disabled it returns a lightweight loading payload |
| `PUSH_TEAMS_ALERTS_ENABLED` | No | `false` | Enables editorial Teams recommendation alerts for only the strongest eligible push candidate |
| `PUSH_TEAMS_WEBHOOK_URL` | Yes, when alerts enabled | — | Power Automate or Teams webhook URL; configure as a secret |
| `PUSH_TEAMS_MIN_SCORE` | No | `75` | Raw push score floor before the weighted Teams Alert Score is evaluated |
| `PUSH_TEAMS_MIN_ALERT_SCORE` | No | `78` | Minimum weighted Teams Alert Score for a normal recommendation before deadline fallback |
| `PUSH_TEAMS_TARGET_PUSHES_PER_DAY` | No | `15` | Daily floor; `06:15/06:45` and every golden-hour `:15/:45` pair are binding, so strong weekdays may schedule up to 18 |
| `PUSH_TEAMS_MIN_ALERTS_PER_DAY` | No | `15` | Independent Teams-message minimum used for deficit and catch-up logic; actual live-push count never replaces it |
| `PUSH_TEAMS_MAX_ALERTS_PER_DAY` | No | `18` | Daily cap including optional double opportunities; breaking can still use its configured override |
| `PUSH_TEAMS_QUIET_HOURS_START` | No | `00:00` | Berlin-local start of the hard no-send window for every Teams payload type |
| `PUSH_TEAMS_QUIET_HOURS_END` | No | `05:30` | Berlin-local end of the hard no-send window; sending is allowed again at exactly 05:30 |
| `PUSH_TEAMS_SLOT_GATE_ENABLED` | No | `true` | Enables weekday-specific `:15/:45` golden-hour pairs, daily morning base slots, and cooldown-edge recovery when the remaining plan cannot reach 15 |
| `PUSH_TEAMS_SLOT_DEADLINE_MINUTE` | No | `45` | Minute at which the worker stops collecting and selects the best eligible candidate when behind |
| `PUSH_TEAMS_PEAK_SLOT_MIN_OR` | No | `6.0` | Historical OR threshold for mandatory peak cells and first-priority double opportunities; reserves still require at least 5.0% |
| `PUSH_TEAMS_DEADLINE_FALLBACK_MIN_SCORE` | No | `75` | Hard raw-score floor for normal `:45` recommendations; pacing and timing can never lower it. Verified breaking keeps its separate floor |
| `PUSH_TEAMS_DEADLINE_FALLBACK_MIN_ALERT_SCORE` | No | `73` | Reference value for the deadline countercheck and ranking; raw Push Score 75 is the binding numeric floor |
| `PUSH_TEAMS_DEADLINE_FALLBACK_MIN_EDITORIAL_SCORE` | No | `69` | Reference value for the deadline countercheck and ranking; all factual, event, timing, title, and duplicate hard gates remain active |
| `PUSH_TEAMS_DAILY_SCHEDULE_SEND_ENABLED` | No | `false` | Sends one restart-safe daily Teams timing plan when enabled; production Render config enables it |
| `PUSH_TEAMS_DAILY_SCHEDULE_SEND_TIME` | No | `05:45` | Berlin-local earliest send time for the daily timing plan |
| `PUSH_TEAMS_AGENT_REVIEW_ENABLED` | Approval required | `false` | Adds the versioned 17-specialist local consensus as an extra veto/evidence layer; mandatory title, final-quality, and Teams-dedup gates remain active without it; enable only after privacy, product, DPO, and legal approval |
| `PUSH_TEAMS_AGENT_REVIEW_MIN_EVIDENCE_APPROVALS` | No | `3` | Required approvals among the five independent evidence families; passed safety checks and policy pressure do not count as positive evidence |
| `PUSH_TEAMS_AGENT_REVIEW_MIN_CONSENSUS_SCORE` | No | `60` | Minimum share of approving evidence families; cautions and abstentions add no support and every hard veto blocks |
| `PUSH_TEAMS_MIN_RECOMMENDATION_QUALITY` | No | `72` | Mandatory final CvD quality with raw Push Score as the strongest direct dimension; reviewer consensus is included only when the optional network is enabled |
| `PUSH_TEAMS_VISIT_SELECTION_WEIGHT` | No | `0.10` | Share of response potential inside the three-point raw-score winner band; the runtime share is capped at 15% |
| `PUSH_TEAMS_AGENT_REVIEW_MAX_LATENCY_MS` | No | `50` | Per-candidate local review budget; an overrun is logged and fails closed |
| `PUSH_TEAMS_SCORE_ONLY_MODE` | No | `false` | When enabled, forecast is treated as a context signal; the weighted Teams Alert Score and independent Teams cooldown still decide eligibility |
| `PUSH_TEAMS_DASHBOARD_TOP_LIMIT` | No | `20` | Normal top-field guardrail for Teams decisions and dashboard transparency |
| `PUSH_TEAMS_CANDIDATE_LIMIT` | No | `80` | Maximum number of article candidates inspected by the automatic Teams worker; candidates beyond the dashboard top field need the stricter Expanded Field gate |
| `PUSH_TEAMS_NO_FORECAST_MIN_ALERT_SCORE` | No | `76` | Higher Teams Alert Score required when no reliable article-specific OR forecast is available |
| `PUSH_TEAMS_EDITORIAL_GATE_ENABLED` | No | `true` | Enables the hard CvD review layer before any Teams recommendation can be sent |
| `PUSH_TEAMS_EDITORIAL_TOP_LIMIT` | No | `10` | Normal non-breaking recommendations must be in the top N dashboard candidates |
| `PUSH_TEAMS_MIN_EDITORIAL_SCORE` | No | `74` | Minimum CvD score based on news value, urgency, public need, timing, clarity, and user load |
| `PUSH_TEAMS_MIN_EDITORIAL_NEWS_VALUE` | No | `24` | Minimum hard-news value required before Teams can recommend a push |
| `PUSH_TEAMS_MIN_TIME_FIT_SCORE` | No | `4` | Minimum CvD time-fit score; blocks normal pushes in weak daypart/weekday windows while still allowing breaking-news overrides |
| `PUSH_TEAMS_MIN_OR` | No | `5.0` | Minimum predicted OR percentage for a standard Teams recommendation |
| `PUSH_TEAMS_MIN_MINUTES_SINCE_LAST_PUSH` | No | `30` | Legacy compatibility floor for non-independent operation; live-push timing is not checked in the fixed independent Teams policy |
| `PUSH_TEAMS_ALERT_COOLDOWN_MINUTES` | No | `90` | Retry/memory safety interval; a successfully sent article remains non-repeatable for the full retained Teams state |
| `PUSH_TEAMS_GLOBAL_COOLDOWN_MINUTES` | No | `30` | Minimum pause between normal Teams recommendations, allowing binding `:15` and `:45` decisions |
| `PUSH_TEAMS_POST_SEND_THRESHOLD_ENABLED` | No | `true` | Raises the raw-score floor after a sent Teams recommendation; the live-push history never drives this curve |
| `PUSH_TEAMS_POST_SEND_PEAK_SCORE` | No | `80` | Raw-score threshold at the first eligible moment after the Teams cooldown; capped by the high-score always threshold |
| `PUSH_TEAMS_POST_SEND_DECAY_MINUTES` | No | `90` | Minutes after the last Teams send by which the elevated threshold has decayed linearly back to `PUSH_TEAMS_MIN_SCORE` |
| `PUSH_TEAMS_HIGH_SCORE_ALWAYS_THRESHOLD` | No | `80` | A canonical score strictly above this value waives soft quality/fatigue gates only; all hard safety, timing, duplicate, title, and transport gates remain mandatory |
| `PUSH_TEAMS_REQUIRE_ARTICLE_FORECAST` | No | `true` | Requires article-model OR forecasts for normal non-breaking Teams recommendations; breaking and clear public warning/usefulness cases can still pass |
| `PUSH_TEAMS_REALERT_SCORE_DELTA` | No | `8` | Required score improvement for a re-alert |
| `PUSH_TEAMS_REALERT_OR_DELTA` | No | `0.75` | Required OR percentage-point improvement for a re-alert |
| `PUSH_TEAMS_ALLOWED_SECTIONS` | No | `News,Politik,Wirtschaft,Geld,Regional,Digital,Unterhaltung,Sport` | Comma-separated section allowlist; Sport still requires a confirmed event |
| `PUSH_TEAMS_EXCLUDED_SECTIONS` | No | empty | Hard section exclusions applied even during deadline fallback |
| `PUSH_TEAMS_BREAKING_OVERRIDE` | No | `true` | Lets verified breaking bypass slot wait, global Teams cooldown, and daily cap after all quality, factual, quiet-hour, and Teams-dedup gates pass |
| `PUSH_TEAMS_BREAKING_MIN_SCORE` | No | `72` | Breaking-news raw score floor outside score-only mode; weighted Teams Alert Score still decides final eligibility |
| `PUSH_TEAMS_BREAKING_MIN_MINUTES_SINCE_LAST_PUSH` | No | `45` | Legacy compatibility value for non-independent pacing; live-push timing is ignored in the fixed independent Teams policy |
| `OPENAI_API_KEY` | No | — | OpenAI API key for optional editorial assistant features |
| `OPENAI_TITLE_GENERATION_ENABLED` | No | `false` | Enables the higher-quality LLM path for manual push-title generation; without it the endpoint uses a local fallback |
| `OPENAI_TITLE_GENERATION_MODEL` | No | `gpt-5.6-luna` | Efficient GPT-5.6 model used for interactive manual title generation |
| `OPENAI_TITLE_GENERATION_TIMEOUT_S` | No | `8.0` | Hard timeout for the interactive title generation request |
| `OPENAI_TITLE_GENERATION_MAX_TOKENS` | No | `600` | Max completion tokens; each of the four titles appears only once in the compact model response |
| `OPENAI_TITLE_GENERATION_REASONING_EFFORT` | No | `none` | Low-latency GPT-5.6 mode; the editorial prompt still generates and ranks all four variants |
| `OPENAI_TITLE_GENERATION_MAX_CALLS_PER_HOUR` | No | `0` | Hard hourly budget for paid title generation; `0` keeps the local fallback active |
| `OPENAI_TITLE_GENERATION_MAX_CALLS_PER_DAY` | No | `0` | Hard daily budget for paid title generation; `0` keeps the local fallback active |
| `OPENAI_BACKFILL_ENABLED` | No | `false` | Keeps the dormant LLM backfill worker disabled unless it is explicitly needed |
| `OPENAI_PREDICTION_SCORING_ENABLED` | No | `false` | Hard cost guard for OR prediction: only when set to `1`/`true` may the runtime use OpenAI during prediction |
| `OPENAI_PREDICTION_SCORING_MODEL` | No | `gpt-4o-mini` | Model used for opt-in prediction scoring |
| `OPENAI_PREDICTION_SCORING_TIMEOUT_S` | No | `4.0` | Timeout for opt-in prediction scoring requests |
| `OPENAI_PREDICTION_SCORING_MAX_TOKENS` | No | `60` | Max completion tokens for opt-in prediction scoring |
| `OPENAI_PREDICTION_SCORING_CACHE_TTL_S` | No | `3600` | Cache lifetime in seconds for identical opt-in prediction scoring prompts |
| `OPENAI_PREDICTION_SCORING_MAX_CALLS_PER_HOUR` | No | `0` | Hard hourly budget for paid prediction scoring; `0` disables OpenAI scoring entirely |
| `OPENAI_PREDICTION_SCORING_MAX_CALLS_PER_DAY` | No | `0` | Hard daily budget for paid prediction scoring; `0` disables OpenAI scoring entirely |
| `PUSH_API_BASE` | Yes | `http://push-frontend.bildcms.de` | Base URL of the BILD Push Statistics API (internal network) |
| `FOOTBALL_DATA_KEY` | No | — | Reserved for future sport integrations; currently not used by the active FastAPI runtime |
| `ODDS_API_KEY` | No | — | Reserved for future betting-context integrations; currently not used by the active FastAPI runtime |
| `ADOBE_CLIENT_ID` | No | — | Adobe Analytics OAuth2 client ID |
| `ADOBE_CLIENT_SECRET` | No | — | Adobe Analytics OAuth2 client secret |
| `ADOBE_TRAFFIC_ENABLED` | No | `false` | Hard cost guard for Adobe traffic fetching and related background work |
| `ADOBE_GLOBAL_COMPANY_ID` | No | `axelsp2` | Adobe Analytics company ID |
| `BILD_SITEMAP_URL` | No | `https://www.bild.de/sitemap-news.xml` | BILD news sitemap URL |
| `PUSH_SYNC_SECRET` | No | — | Strong random shared secret for the push data relay between local server and Render |
| `RENDER_SYNC_URL` | No | — | Render deployment URL; if set, the local server relays push data to it (e.g. `https://push-balancer.onrender.com`) |
| `PORT` | No | `8050` | Server listen port |
| `BIND_HOST` | No | `0.0.0.0` | Server bind host |
| `ALLOW_INSECURE_SSL` | No | `0` | Set to `1` to disable SSL certificate verification (development only) |
| `ADMIN_API_KEY` | No | — | Strong random admin key for protected retraining and promotion endpoints; required to enable admin mutations |
| `CONSUMER_API_KEY` | No | — | Strong random read-only key for downstream consumer endpoints (`/api/v1/recommendations`, `/api/v1/articles`, `/api/v1/scores`); required to enable consumer API access |
| `INTERNAL_ACCESS_ENABLED` | No | `true` on Render, `false` locally | Restrict non-exempt routes to the CIDRs listed in `INTERNAL_ACCESS_ALLOWED_CIDRS` |
| `INTERNAL_ACCESS_ALLOWED_CIDRS` | No | `127.0.0.1/32,::1/128,145.243.0.0/16,91.220.134.0/24` | Comma-separated AS/VPN egress CIDRs or individual IPs in `/32` or `/128` notation |
| `SCORE_CAPTURE_CONSUMER_ALLOWED_CIDRS` | No | BILD Next staging NAT `/32` address | Dedicated egress allowlist for the two read-only score-capture source routes; does not grant UI, debug, or POST access |
| `INTERNAL_ACCESS_EXEMPT_PATHS` | No | `/api/health` | Comma-separated route list that remains reachable without the internal allowlist; production should use `/api/health,/api/v1` so only health checks and authenticated consumer routes are externally reachable |
| `DB_PATH` | No | `.push_history.db` | Override SQLite location, e.g. on a persistent disk |
| `PUSH_DB_MAX_DAYS` | No | `90` | Maximum age of push rows loaded from SQLite into memory for analysis/runtime paths |
| `PUSH_DB_MAX_ROWS` | No | `15000` locally, lower on Render | Maximum number of push rows loaded from SQLite into memory |
| `PUSH_SNAPSHOT_PATH` | No | — | Optional path to a sanitized startup seed file mounted outside the repository |
| `NPM_TOKEN` | No | — | GitHub Packages token for installing `@spring-media/editorial-one-ui` locally |

Variables are loaded from a `.env` file in the project directory at startup (via a lightweight built-in parser — no `python-dotenv` required).

---

## Development

### Project Structure

```
push-balancer/
├── app/                      # FastAPI application, routers, ML and research modules
├── frontend/                 # React/Vite client
├── tests/                    # Pytest suite
├── requirements.txt
├── pyproject.toml            # Python test/lint configuration
├── Dockerfile
├── .editorconfig
└── .push_history.db          # SQLite database (created at runtime, git-ignored)
```

Frontend source layout:

```
frontend/src/
├── app.tsx
├── main.tsx
├── api/
├── components/
│   ├── main-layout/
│   ├── top-nav/
│   └── ui/
├── editorial-one-ui-shim/
├── hooks/
├── pages/
│   ├── analyse/
│   ├── forschung/
│   ├── kandidaten/
│   ├── konkurrenz/
│   ├── live-pushes/
│   └── tagesplan/
├── router/
├── stores/
├── types/
└── utils/
```

### Adding a Feature

1. Implement backend logic in `app/` and prefer a dedicated router/module over extending legacy files.
2. Add or update the corresponding frontend page/component in `frontend/src/` when the feature is user-facing.
3. Document API changes in `push-balancer-api-v3.1.0.yaml`.
4. If the feature introduces a new environment variable, add it to `.env.example` and the table in this README.
5. Run the relevant checks before pushing (`pytest`, frontend lint, frontend typecheck/build).

### Runtime guardrails

- The active runtime uses bounded SQLite loads via `PUSH_DB_MAX_DAYS` and `PUSH_DB_MAX_ROWS` to avoid loading the full history into memory on smaller instances.
- The Tagesplan prediction path includes a guard against saturated OR forecasts when a model output looks incorrectly back-transformed.

### Running Tests

```bash
# Backend tests
python -m pytest

# Frontend quality gates
pnpm --dir frontend lint
pnpm --dir frontend typecheck
pnpm --dir frontend build

# Manual smoke test
curl http://localhost:8050/api/health
curl "http://localhost:8050/api/ml/predict?title=Scholz%20tritt%20zurück&cat=politik&hour=18"
```

### Retrain a Model Manually

```bash
# LightGBM
curl -X POST \
  -H "X-Admin-Key: $ADMIN_API_KEY" \
  http://localhost:8050/api/ml-model/retraining-jobs

# GBRT
curl -X POST \
  -H "X-Admin-Key: $ADMIN_API_KEY" \
  http://localhost:8050/api/gbrt-model/retraining-jobs

# Promote GBRT challenger
curl -X POST \
  -H "X-Admin-Key: $ADMIN_API_KEY" \
  http://localhost:8050/api/gbrt-model/promotions
```

### Linting and Formatting

```bash
ruff check app tests
ruff format app tests
pnpm --dir frontend lint
```
