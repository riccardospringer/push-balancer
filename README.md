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

`PUSH_TEAMS_ALERTS_ENABLED=1` starts a background worker that evaluates the same top article field shown in the Push Balancer dashboard and sends a Power Automate / Teams recommendation only when the central decision model says a redakteur should act now. The webhook secret belongs in `PUSH_TEAMS_WEBHOOK_URL` and must stay in Render secrets or `.env`, never in Git.

For Power Automate, use the trigger body field `messageHtml` as the Teams message content:

```text
@{triggerBody()?['messageHtml']}
```

The payload also includes structured fields such as `articleTitle`, `articleUrl`, `pushScore`, `predictedORLabel`, `whyNow`, `whyPushworthy`, `recommendedPushText`, `editorialReview`, and `selectionScore`. Low-confidence global-average prediction fallbacks are not shown as article-specific OR forecasts; they are rendered as "keine belastbare Prognose". Candidates outside `PUSH_TEAMS_DASHBOARD_TOP_LIMIT` are ignored for Teams. The CvD gate additionally limits normal recommendations to the editorial top field, checks hard news value, and blocks soft topics unless they have a clear current public-interest angle. The final recommendation is not the dashboard rank 1 by default; it is the eligible candidate with the best CvD-led selection score across the top field. If no reliable OR forecast is available, the article needs the stricter `PUSH_TEAMS_NO_FORECAST_MIN_ALERT_SCORE`.

### CORS

Allowed origins are computed automatically from `PORT`, `RAILWAY_PUBLIC_DOMAIN`, `RENDER_EXTERNAL_HOSTNAME`, and the local network IP. The Render hostname `push-balancer.onrender.com` is always included.

### Internal Network Access

Use `INTERNAL_ACCESS_ENABLED=1` together with `INTERNAL_ACCESS_ALLOWED_CIDRS` to restrict the app to AS/VPN egress IPs. On Render this protection is enabled by default, so non-exempt routes stay closed until the AS network CIDRs are configured. Keep `/api/health,/api/v1` in `INTERNAL_ACCESS_EXEMPT_PATHS` so platform health checks and authenticated consumer API calls can work while docs and legacy routes remain internal.

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
| `TAGESPLAN_ON_DEMAND_BUILD_ENABLED` | No | `false` in economy mode | Controls whether `/api/tagesplan` builds a fresh plan on request; when disabled it returns a lightweight loading payload |
| `PUSH_TEAMS_ALERTS_ENABLED` | No | `false` | Enables editorial Teams recommendation alerts for only the strongest eligible push candidate |
| `PUSH_TEAMS_WEBHOOK_URL` | Yes, when alerts enabled | — | Power Automate or Teams webhook URL; configure as a secret |
| `PUSH_TEAMS_MIN_SCORE` | No | `75` | Raw push score floor before the weighted Teams Alert Score is evaluated |
| `PUSH_TEAMS_MIN_ALERT_SCORE` | No | `78` | Minimum weighted Teams Alert Score for a Teams recommendation; combines raw score, news value, freshness, timing, competition, and user-load penalty |
| `PUSH_TEAMS_SCORE_ONLY_MODE` | No | `false` | When enabled, forecast is treated as a context signal; the weighted Teams Alert Score, known last-push timing, and pause rules still decide final notification eligibility |
| `PUSH_TEAMS_DASHBOARD_TOP_LIMIT` | No | `20` | Limits Teams evaluation to the top N article candidates from the same payload used by the dashboard |
| `PUSH_TEAMS_NO_FORECAST_MIN_ALERT_SCORE` | No | `88` | Higher Teams Alert Score required when no reliable article-specific OR forecast is available |
| `PUSH_TEAMS_EDITORIAL_GATE_ENABLED` | No | `true` | Enables the hard CvD review layer before any Teams recommendation can be sent |
| `PUSH_TEAMS_EDITORIAL_TOP_LIMIT` | No | `10` | Normal non-breaking recommendations must be in the top N dashboard candidates |
| `PUSH_TEAMS_MIN_EDITORIAL_SCORE` | No | `82` | Minimum CvD score based on news value, urgency, public need, timing, clarity, and user load |
| `PUSH_TEAMS_MIN_EDITORIAL_NEWS_VALUE` | No | `24` | Minimum hard-news value required before Teams can recommend a push |
| `PUSH_TEAMS_MIN_OR` | No | `5.0` | Minimum predicted OR percentage for a standard Teams recommendation |
| `PUSH_TEAMS_MIN_MINUTES_SINCE_LAST_PUSH` | No | `30` | Minimum pause after the previous push |
| `PUSH_TEAMS_ALERT_COOLDOWN_MINUTES` | No | `90` | Cooldown before the same article can be re-alerted |
| `PUSH_TEAMS_GLOBAL_COOLDOWN_MINUTES` | No | `30` | Minimum pause between any two Teams recommendations, even for different articles |
| `PUSH_TEAMS_REALERT_SCORE_DELTA` | No | `8` | Required score improvement for a re-alert |
| `PUSH_TEAMS_REALERT_OR_DELTA` | No | `0.75` | Required OR percentage-point improvement for a re-alert |
| `PUSH_TEAMS_ALLOWED_SECTIONS` | No | empty | Comma-separated section allowlist, e.g. `News,Politik,Sport,Regional` |
| `PUSH_TEAMS_BREAKING_OVERRIDE` | No | `true` | Allows lower configured breaking-news thresholds |
| `PUSH_TEAMS_BREAKING_MIN_SCORE` | No | `72` | Breaking-news raw score floor outside score-only mode; weighted Teams Alert Score still decides final eligibility |
| `OPENAI_API_KEY` | No | — | OpenAI API key for optional editorial assistant features |
| `OPENAI_TITLE_GENERATION_ENABLED` | No | `false` | Enables the higher-quality LLM path for manual push-title generation; without it the endpoint uses a local fallback |
| `OPENAI_TITLE_GENERATION_MODEL` | No | `gpt-4o-mini` | Model used for manual title generation when enabled |
| `OPENAI_TITLE_GENERATION_TIMEOUT_S` | No | `8.0` | Timeout for manual title generation requests |
| `OPENAI_TITLE_GENERATION_MAX_TOKENS` | No | `320` | Max completion tokens for manual title generation |
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
