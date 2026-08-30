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
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8050 --no-proxy-headers --no-access-log
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
| `POST /api/headline-generations` | Resolve one current CMS document ID and generate advisory Prompt v1.4 headline variants for the standalone `/dist-frontend/headline` tab |

The standalone Headline tab keeps the complete supplied Prompt v1.4 in
`app/prompts/push_headline_v1_4.md` with a checked SHA-256 digest. The request
contains only a 24-character CMS document ID. The backend resolves the existing
public title, category, URL and content type from local recommendation metadata
or the BILD news sitemap; only the minimized title and allowlisted category can
enter the optional Prompt v1.4 provider call. The CMS ID, URL, article body,
Power Automate flow and Teams transport are not part of that call. When the
approved external path is off, the tab stays usable with an explicitly marked
local fallback.

Additional compatibility and operational helper endpoints still exist, but the frontend contract should prefer the documented endpoints above.

The scheduled Teams integration additionally uses these operational endpoints, which are intentionally excluded from the public OpenAPI document:

| Endpoint | Description |
|---|---|
| `GET /api/teams-readiness` | Internal readiness proof for transport, score source, authoritative history, and the fixed slot plan |
| `GET /api/v1/power-automate/teams/readiness` | Authenticated, data-minimized readiness and latest-slot delivery proof |
| `POST /api/v1/power-automate/teams/claim` | Idempotently claim the current scheduled Teams recommendation |
| `POST /api/v1/power-automate/teams/receipt` | Record the final Teams delivery result for a claimed slot |

Compatibility endpoints are also marked at runtime with `Deprecation: true` and a `Sunset` header so downstream clients can detect that they should migrate to the stable contract.

Protected mutation endpoints require the `X-Admin-Key` header and remain unavailable when `ADMIN_API_KEY` is not configured. Downstream consumer endpoints require `Authorization: Bearer <CONSUMER_API_KEY>` or `X-Consumer-Key` and remain unavailable when `CONSUMER_API_KEY` is not configured. The scheduled Teams readiness, claim, and receipt routes use only their dedicated `X-Power-Automate-Key` and remain unavailable when `POWER_AUTOMATE_API_KEY` is not configured. The authenticated readiness response is an explicit data-minimized allowlist; `/api/teams-readiness` remains the complete diagnostic behind internal CIDR/VPN access control.

### Consumer API

Use the versioned consumer API for backend-to-backend integrations that need current article candidates and scores without depending on the React frontend contract:

```bash
curl -H "Authorization: Bearer $CONSUMER_API_KEY" \
  "https://push-balancer.onrender.com/api/v1/recommendations?limit=20&minScore=70"

curl -H "Authorization: Bearer $CONSUMER_API_KEY" \
  "https://push-balancer.onrender.com/api/v1/scores?category=sport&limit=20"
```

Recommendation, article, and score responses include the last 24 hours of
already-sent pushes in a separate `livePushes` array. Each entry is marked with
`isLivePush=true`, `alreadySent=true`, and `flags.livePush=true`; unsent article
candidates remain in `articles`/`scores` and are explicitly marked as not live.
`livePushStatus.authoritative=true` confirms that the relay snapshot is trusted,
persisted, and no more than five minutes old.

The responses are read-only and advisory-only (`actionAllowed=false`). Production deployments should expose `/api/health` for platform checks and `/api/v1/*` for authenticated consumers only. Keep `/api/docs`, `/api/openapi.json`, and legacy `/api/*` routes behind the internal CIDR allowlist.

### Editorial Push Scoring

Article candidates are ranked by an editorial push score in [`app/scoring/editorial.py`](app/scoring/editorial.py). Since the score rebuild of 2026-08-30 the composition is:

| Component | Weight | Source |
|---|---|---|
| BILD-Reiz (LLM reader score) | 40 % | One LLM call per article from a BILD reader's perspective ([`app/scoring/reader_score.py`](app/scoring/reader_score.py)), persistently cached; bounded heuristic fallback without an API key |
| Öffnungs-Potenzial | 20 % | ML-predicted opening rate (LightGBM ensemble) plus content signals |
| Aktualität | 15 % | Publication freshness (first publication counts, not re-publish) |
| Mix-Balance | 10 % | Section/topic/tone fatigue over the last 6 hours |
| Historie | 10 % | Historical opening behaviour (category and hour as separate signals; the category×hour interaction was removed) |
| Headline-Stärke | 3 % | Clarity and concreteness |
| Risiko | 2 % | Clickbait/fatigue guard |

All candidates pass the same hard freshness gate: older than 12 h scores 0; before that an age multiplier applies (≤90 min ×1.0 · 3 h ×0.95 · 6 h ×0.6 · 9 h ×0.4 · 12 h ×0.3). Breaking news is detected via the "Breaking News" taxonomy node and enters immediately; all other articles enter 3 minutes after publication so the LLM check can complete first. BILD-Fit, the Germany-relevance adjustment, video special-casing, the corporate-PR malus, and the Reuters overload malus were removed — the LLM reader score covers those signals itself. Politics remains eligible for top ranks when there is a concrete current development, but stale, abstract, complex, or debate-only politics receives explicit penalties.

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
- Keep `OPENAI_API_KEY`, `ADMIN_API_KEY`, `CONSUMER_API_KEY`, `POWER_AUTOMATE_API_KEY`, `PUSH_SYNC_SECRET`, Adobe credentials, and `NPM_TOKEN` out of source control.
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
2. **Relay sync** (`POST /api/pushes/sync`): The local Mac server posts fresh push data to the Render instance every 30 seconds, authenticated via `PUSH_SYNC_SECRET`. Render parses and persists the complete snapshot before acknowledging it; parse or database failures return a non-2xx response so the relay retries. An authoritative relay payload must carry `source=live` or `source=relay` together with the original `snapshotTs`; receipt time on Render never renews an old snapshot. Set `RENDER_SYNC_URL` on the local server to enable this.
3. **Optional startup seed**: if you mount a sanitized snapshot file and point `PUSH_SNAPSHOT_PATH` at it, the service seeds SQLite at startup before any live fetch succeeds.

### Microsoft Teams Push Recommendations

Power Automate owns the production schedule and Teams transport. A one-minute
Recurrence trigger polls every minute from `+0` through `+14` for each fixed
Berlin-local slot. The first five minutes are the primary interval; the next
ten minutes are a bounded automatic recovery extension. Monday through Friday
use:

The Product/System Owner, Privacy Manager, DPO, and Legal/Group Legal approval for the exactly-five contract and `POWER_AUTOMATE_REQUIRE_LIVE_PUSH_HISTORY=false` cloud-only mode was recorded on 2026-08-09 in `PRIVACY.md`. This authorizes the scoped backend rollout. Production must edit only the existing canonical Exact-5 flow; do not create another scheduled flow. Legacy transports stay off, and the canonical flow may stay active only while the production readiness proof is green.

`06:00`, `06:36`, `07:12`, `07:47`, `08:23`, `08:59`, `12:30`, `17:30`, `18:49`, `20:08`, `21:26`, `22:45`.

Saturday and Sunday move only the six morning slots two hours later:

`08:00`, `08:36`, `09:12`, `09:47`, `10:23`, `10:59`, `12:30`, `17:30`, `18:49`, `20:08`, `21:26`, `22:45`.

These weekday/weekend times are the complete scheduled-flow plan. The Power Automate path ignores `PUSH_TEAMS_SLOT_DELAY_DATE`, `PUSH_TEAMS_SLOT_DELAY_FROM`, `PUSH_TEAMS_SLOT_DELAY_MINUTES`, the legacy golden-hour plan, catch-up logic, and daily Sport quotas. `POWER_AUTOMATE_RECOVERY_GRACE_SECONDS=600` extends the original five-minute primary interval to a hard maximum total of 15 minutes. Invalid, negative, or greater-than-600 values disable only recovery and cannot widen that bound. An initial claim is issued only while at least 30 seconds remain before the total window expires; a later request returns HTTP 200 with `ready=false` and `reason=slot_closed`. The Power Automate trigger condition must include minutes `+0` through `+14`; retaining the old five-minute condition prevents the backend recovery from being called.

During a window, the flow calls `POST /api/v1/power-automate/teams/claim` with its dedicated `X-Power-Automate-Key` and a unique `requestId`. `top` is the candidate with the absolute highest fresh, technically valid `internal_score_api` score after Teams-article duplicate removal. Exact live-push duplicate removal is additionally mandatory only when `POWER_AUTOMATE_REQUIRE_LIVE_PUSH_HISTORY=true`; the approved cloud-only mode does not claim that comparison. Lower-scored Sport, Breaking, section-mix, OR, local quality, pacing, or quota signals can never replace it; secondary signals only break an exact API-score tie. The separate nullable `alternative` is the highest valid candidate from the opposite Sport/non-Sport class for display and is not necessarily overall rank 2.

The scheduled contract is all-or-nothing: one ready response contains exactly five URL- and CMS-deduplicated recommendations. Top 1 always has the fresh canonical internal score. Places 2–5 first use further technically valid canonical candidates; when the only technical blocker is a missing fresh canonical score, they may be filled from the current, publication-age-weighted article field for display only. That local fallback value never authorizes Top 1 and never crosses into Teams as a numeric Push Score. Any other blocker still excludes the article. If fewer than five safe candidates exist, the API returns HTTP 200 with `ready=false` and `reason=insufficient_recommendations`, creates no slot or article-group claim, and the next minute run may try again while the window remains open. A legacy short replay is released fail-closed with `reason=claim_contract_stale`. The flow must not post to Teams for either no-send outcome.

The Push Balancer remains responsible for the narrow technical gates and a single atomic claim covering the slot plus all five displayed article identities. A conflict on any one identity rolls back the whole group. In the configured cloud-only mode, the scheduled claim deliberately does not call or promise an authoritative AS-network live-push history; the five durable article claims, slot, request, and receipt prevent repeated Teams recommendations. When `POWER_AUTOMATE_REQUIRE_LIVE_PUSH_HISTORY=true`, the claim performs one approved refresh and fails closed unless the direct/relay snapshot is authoritative and fresh; it does not renew that snapshot a second time during the same claim.

Only when the claim returns `ready="yes"`, `contractVersion=2`, and `recommendationCount=5` does the flow post `messageHtml` to Teams exactly once and then call `POST /api/v1/power-automate/teams/receipt` with the original `requestId`. The receipt atomically finalizes all five article identities with the slot; repeated receipts remain idempotent, and the five-item group counts as one Teams message. Claim no-ops continue to use the JSON boolean `ready=false`; the separate `/api/teams-readiness` endpoint also continues to use a JSON boolean. A successful Teams action uses `status=sent`; every failed, timed-out, or skipped Teams action uses the terminal `status=delivery_uncertain`. The Teams connector retry policy must be `None`, because Teams may already have accepted a message even when the connector does not report success.

Keep `PUSH_TEAMS_ALERTS_ENABLED=true` so recommendation claims remain enabled, and set `PUSH_TEAMS_BACKGROUND_SENDER_ENABLED=false` at cutover so the legacy webhook worker cannot race the scheduled flow. `PUSH_TEAMS_WEBHOOK_URL` is needed only for the legacy rollback path. Never run both transport owners at the same time. A process with Teams alerts enabled now refuses to start when both the legacy background sender and `POWER_AUTOMATE_API_KEY` are configured; the claim endpoint independently returns HTTP 503 for the same conflict.

At every transport-owner cutover, audit both **Meine Flows → Cloud-Flows** and
**Für mich freigegeben** in Power Automate; the owned-flow list alone does not
show shared schedulers that can claim a slot first. The legacy instant-webhook
flow and every legacy/shared scheduled flow must be **Off**, leaving exactly one
active scheduler: the canonical Exact-5 flow. Rotate `POWER_AUTOMATE_API_KEY`
for the owner change and distribute the new value only to the deployment secret
store and that canonical flow's protected Claim/Receipt configuration. Legacy
and shared flows must retain stale credentials so re-enabling one cannot make it
an authorized claim owner.

The adaptive schedule, cooldown, deadline, delay, Sport-quota, and daily-plan environment variables below remain available to the legacy worker and internal diagnostics. They do not affect or replace the fixed 12-slot Power Automate schedule.

Before cutover, `/api/v1/power-automate/teams/readiness` (or the complete internal `/api/teams-readiness` diagnostic) must report top-level `ready=true`, `teamsAlertsEnabled=true`, `transportMode=power_automate_scheduled`, `backgroundSenderEnabled=false`, `powerAutomateConfigured=true`, `durableStorage.required=true`, `durableStorage.durable=true`, `durableStorage.mode=persistent_disk`, `scoreApi.ok=true`, `exactFive.contractOk=true`, `exactFive.recommendationCount=5`, `exactFive.top1Canonical=true`, `recovery.enabled=true`, `recovery.configurationValid=true`, `recovery.graceSeconds=600`, `deliveryHealth.ok=true`, `deliveryHealth.attentionRequired=false`, `pushHistory.ok=true`, `slots.ok=true`, `slots.plannedToday=12`, and the exact weekday or weekend labels above. The Exact-5 probe runs the same read-only filter/evaluation path as the claim and creates no slot/article claim. The claim returns 503, and Render startup fails, instead of falling back to ephemeral storage when the `/data` disk is missing or unwritable. `pushHistory.historyAuthoritative=false` together with `fallbackMode=durable_slot_and_receipt_dedup` is the expected cloud-only state when no AS-network relay is available. The protected route additionally exposes only the latest due slot's safe state: `sent` plus `receiptRecorded=true` and `timingState=terminal` is the successful delivery proof; `recovery_open` is still automatically eligible, `missed` is an expired unclaimed slot, and `delivery_uncertain`, `awaiting_receipt`, or `overdue_unresolved` must never trigger another Teams post. `awaiting_receipt` remains healthy only while the claim window is open; `missed`, `delivery_uncertain`, `overdue_unresolved`, and blocked or unknown states force top-level `ready=false`, so a monitor cannot silently report a failed slot as healthy.

The dedicated `POWER_AUTOMATE_API_KEY` is a strong random opaque shared secret, not a derived key or KDF output. It must exist only in the deployment secret store and the protected Power Automate secret/configuration. `render.yaml` declares it with `sync: false`, so configure it manually in the Render dashboard. Enable Secure Inputs and Secure Outputs on every HTTP and Teams action that handles the key or recommendation payload. Neither the key nor a signed webhook URL may appear in Git, flow names, URLs, Teams messages, logs, screenshots, or run-history output.

Only the minimized hand-off crosses into Microsoft 365: non-personal contract/count metadata, slot timestamps, and exactly five public article titles, URLs, and latest publication times in the rendered recommendation HTML. A numeric Push Score is shown only where a fresh canonical score exists; a display-only filler says that its canonical score is pending, while its local ordering value remains backend-only. Raw push history, candidates outside those five recommendations, reviewer details, secrets, and recipient- or employee-level activity are excluded. Backend recommendation state is retained for 45 days; Power Automate run-history retention must follow the approved tenant policy and be kept as short as operationally necessary.

The complete action-by-action setup, trigger expression, API examples, cutover checklist, and rollback procedure are documented in [`integrations/power-automate/README.md`](integrations/power-automate/README.md).

### CORS

Allowed origins are computed automatically from `PORT`, `RAILWAY_PUBLIC_DOMAIN`, `RENDER_EXTERNAL_HOSTNAME`, and the local network IP. The Render hostname `push-balancer.onrender.com` is always included.

### Internal Network Access

Use `INTERNAL_ACCESS_ENABLED=1` together with `INTERNAL_ACCESS_ALLOWED_CIDRS` to restrict the app to AS/VPN egress IPs. On Render this protection is enabled by default, so non-exempt routes stay closed until the AS network CIDRs are configured. Keep `/api/health,/api/v1` in `INTERNAL_ACCESS_EXEMPT_PATHS` so platform health checks and authenticated consumer API calls can work while docs and legacy routes remain internal. `SCORE_CAPTURE_CONSUMER_ALLOWED_CIDRS` is a separate least-privilege allowlist: it grants only `GET /api/score-capture/health`, `GET /api/score-capture/by-cms-id/{cms_id}`, and the exact read-only `POST /api/score-capture/by-cms-id/batch?includeBreakdown=1` to the approved BILD Next consumer. It does not grant UI, debug, or browser-capture write access.

On Render, CIDR authorization trusts only one syntactically valid `CF-Connecting-IP` value supplied by the guaranteed Cloudflare public ingress. A missing, malformed, duplicate, or comma-separated value fails closed. `True-Client-IP`, `X-Real-IP`, and `X-Forwarded-For` are never authorization sources. Outside Render, forwarded headers are ignored and only the validated socket peer (`request.client.host`) is used. The container starts Uvicorn with proxy-header rewriting disabled, so spoofed forwarding headers cannot replace that socket peer before application authorization. Uvicorn access logging is disabled because request paths can contain CMS IDs; application logs use a redacted score-source path instead.

The CMS-ID source route remains backward compatible: without a query parameter it returns exactly `score` and `capturedAt`. The fixed opt-in query `?includeBreakdown=1` additionally returns `scoreBreakdown` and `orFactor` when the capture contains that complete pair; legacy captures keep the exact two-field response. New captures contain only numeric fields already displayed or applied by the candidate view. Engagement fields are `relevance`, `urgency`, `curiosity`, `freshness`, `timing`, `titleBoost`, `breaking`, `research`, `pushHistory`, and `topicSaturation`. Sport captures use `sportRelevance`, `timing`, `drama`, and `freshness`. `score` remains the displayed total, while `orFactor` is the separate sorting factor. The source neither recalculates nor changes the existing score.

For lists, send 1 to 500 unique lowercase 24-character hexadecimal IDs to `POST /api/score-capture/by-cms-id/batch?includeBreakdown=1` as the exact JSON body `{"cmsIds":["…"]}`. The exact response is `{"results":[…]}` in request order. A found item contains `cmsId`, `status:"found"`, `score`, and `capturedAt`, plus the complete `scoreBreakdown`/`orFactor` pair when it was captured. A missing item contains only `cmsId` and `status:"notFound"`. `notFound` means that no fresh captured candidate-view score exists in the current eight-hour workday window; it does not mean that the CMS article itself is missing. The source performs one memory scan and one database scan for the complete batch. A database/read failure returns `503` for the whole request and is never converted into individual `notFound` results.

The browser capture sends candidates in bounded chunks of 100. Its 30-second throttle and fingerprint advance only after every chunk receives an HTTP success response, so a failed capture remains retryable. If optional explanation values are outside their approved numeric bounds, the browser omits that enrichment pair and still sends the valid legacy total score.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `PAID_EXTERNAL_APIS_ENABLED` | No | `false` | Master kill switch for all paid external API usage in the active runtime; local fallbacks stay available |
| `BACKGROUND_AUTOMATIONS_ENABLED` | No | `false` | Disables autonomous background polling, research loops, training ticks, push auto-fetch, and cache warmups; endpoints fall back to on-demand work where available |
| `PUSH_AUTO_FETCH_ENABLED` | No | follows `BACKGROUND_AUTOMATIONS_ENABLED` | On the approved relay host, refreshes the live Push Statistics cache; keep disabled on Render |
| `PUSH_RENDER_SYNC_ENABLED` | No | follows `BACKGROUND_AUTOMATIONS_ENABLED` | On the approved relay host, forwards the cache with its original snapshot timestamp; requires `RENDER_SYNC_URL` and `PUSH_SYNC_SECRET` |
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
| `PUSH_BALANCER_SCORE_API_MAX_AGE_SECONDS` | No | `28800` | Maximum source-valid workday age of `scoredAt`; scores at or beyond eight hours are excluded without fallback |
| `PUSH_BALANCER_SCORE_API_TIMEOUT_SECONDS` | No | `2.5` | Legacy single-score request timeout; productive article fields use the batch timeout |
| `PUSH_BALANCER_SCORE_API_BATCH_TIMEOUT_SECONDS` | No | `35` | Timeout for the single score batch POST; at most one retry is allowed |
| `PUSH_BALANCER_SCORE_API_CACHE_TTL_SECONDS` | No | `45` | Legacy single-score process cache; productive article-field batches are not cached |
| `PUSH_BALANCER_SCORE_API_MAX_CONCURRENCY` | No | `16` | Compatibility setting retained for older callers; productive fields use one bounded batch instead of parallel single lookups |
| `TAGESPLAN_ON_DEMAND_BUILD_ENABLED` | No | `false` in economy mode | Controls whether `/api/tagesplan` builds a fresh plan on request; when disabled it returns a lightweight loading payload |
| `PUSH_TEAMS_ALERTS_ENABLED` | No | `false` | Enables the Teams recommendation policy and scheduled claim API; keep enabled when Power Automate owns transport |
| `PUSH_TEAMS_BACKGROUND_SENDER_ENABLED` | No | `false` | Enables the legacy in-process webhook sender; keep `false` while the scheduled Power Automate flow owns transport |
| `POWER_AUTOMATE_API_KEY` | Yes, for scheduled Teams flow | — | Dedicated random shared secret for `/api/v1/power-automate/teams/{claim,receipt}`; set manually in Render (`sync: false`) and send only as `X-Power-Automate-Key` |
| `POWER_AUTOMATE_RECOVERY_GRACE_SECONDS` | No | `600` | Adds at most ten recovery minutes after the five-minute primary claim interval; `0` disables recovery, invalid or greater-than-600 values fail closed, and the total window never exceeds 15 minutes |
| `PUSH_TEAMS_WEBHOOK_URL` | Yes, for legacy sender only | — | Legacy Power Automate or Teams webhook URL; keep only as a protected rollback secret |
| `PUSH_TEAMS_MIN_SCORE` | No | `75` | Legacy/non-mandatory diagnostic threshold; no floor applies in a binding Top-1 slot |
| `PUSH_TEAMS_MIN_ALERT_SCORE` | No | `78` | Minimum weighted Teams Alert Score for a normal recommendation before deadline fallback |
| `PUSH_TEAMS_TARGET_PUSHES_PER_DAY` | No | `15` | Daily floor; `06:15/06:45` and every golden-hour `:15/:45` pair are binding, so strong weekdays may schedule up to 18 |
| `PUSH_TEAMS_MIN_ALERTS_PER_DAY` | No | `15` | Independent Teams-message minimum used for deficit and catch-up logic; actual live-push count never replaces it |
| `PUSH_TEAMS_MAX_ALERTS_PER_DAY` | No | `18` | Daily cap including optional double opportunities; breaking can still use its configured override |
| `PUSH_TEAMS_QUIET_HOURS_START` | No | `00:00` | Berlin-local start of the hard no-send window for every Teams payload type |
| `PUSH_TEAMS_QUIET_HOURS_END` | No | `05:30` | Berlin-local end of the hard no-send window; sending is allowed again at exactly 05:30 |
| `PUSH_TEAMS_SLOT_GATE_ENABLED` | No | `true` | Enables weekday-specific `:15/:45` golden-hour pairs, daily morning base slots, and cooldown-edge recovery when the remaining plan cannot reach 15 |
| `PUSH_TEAMS_SLOT_DEADLINE_MINUTE` | No | `45` | Minute at which the worker stops collecting and selects the best eligible candidate when behind |
| `PUSH_TEAMS_SLOT_DELAY_DATE` | No | — | Berlin date (`YYYY-MM-DD`) on which the remaining recovery delay applies |
| `PUSH_TEAMS_SLOT_DELAY_FROM` | No | — | First original binding time (`HH:MM`) to move on the configured date |
| `PUSH_TEAMS_SLOT_DELAY_MINUTES` | No | `0` | Minutes added to each remaining slot; unsafe or duplicate results fail back to the original plan |
| `PUSH_TEAMS_EXTERNALLY_RECOMMENDED_URLS` | No | — | Public article URLs already recommended by an approved external recovery; used only for exact deduplication |
| `PUSH_TEAMS_PEAK_SLOT_MIN_OR` | No | `6.0` | Historical OR threshold for mandatory peak cells and first-priority double opportunities; reserves still require at least 5.0% |
| `PUSH_TEAMS_DEADLINE_FALLBACK_MIN_SCORE` | No | `75` | Hard raw-score floor for normal `:45` recommendations; pacing and timing can never lower it. Verified breaking keeps its separate floor |
| `PUSH_TEAMS_DEADLINE_FALLBACK_MIN_ALERT_SCORE` | No | `73` | Reference value for the deadline countercheck and ranking; raw Push Score 75 is the binding numeric floor |
| `PUSH_TEAMS_DEADLINE_FALLBACK_MIN_EDITORIAL_SCORE` | No | `69` | Reference value for the deadline countercheck and ranking; all factual, event, timing, title, and duplicate hard gates remain active |
| `PUSH_TEAMS_DAILY_SCHEDULE_SEND_ENABLED` | No | `false` | Sends one restart-safe daily Teams timing plan when enabled; production Render config enables it |
| `PUSH_TEAMS_DAILY_SCHEDULE_SEND_TIME` | No | `05:45` | Berlin-local earliest send time for the daily timing plan |
| `PUSH_TEAMS_AGENT_REVIEW_ENABLED` | Approval required | `false` | Adds the versioned 17-specialist local consensus as advisory evidence in binding Top-1 slots; enable only after privacy, product, DPO, and legal approval |
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
| `PUSH_TEAMS_BREAKING_OVERRIDE` | No | `true` | Prioritizes verified breaking inside an open binding slot; it cannot bypass the global Teams cooldown, raster, or transport-time slot check |
| `PUSH_TEAMS_BREAKING_MIN_SCORE` | No | `72` | Breaking-news raw score floor outside score-only mode; weighted Teams Alert Score still decides final eligibility |
| `PUSH_TEAMS_BREAKING_MIN_MINUTES_SINCE_LAST_PUSH` | No | `45` | Legacy compatibility value for non-independent pacing; live-push timing is ignored in the fixed independent Teams policy |
| `OPENAI_API_KEY` | No | — | OpenAI API key for optional editorial assistant features |
| `OPENAI_TITLE_GENERATION_ENABLED` | No | `false` | Enables the higher-quality LLM path for manual push-title generation; without it the endpoint uses a local fallback |
| `OPENAI_TITLE_GENERATION_MODEL` | No | `gpt-5.6-luna` | Efficient GPT-5.6 model used for interactive manual title generation |
| `OPENAI_TITLE_GENERATION_TIMEOUT_S` | No | `45.0` | Total deadline for the interactive title-generation request, including its single bounded correction attempt |
| `OPENAI_TITLE_GENERATION_MAX_TOKENS` | No | `2000` | Completion budget for three complete structured v1.4 headline/line-2 pairs, including bounded reasoning |
| `OPENAI_TITLE_GENERATION_REASONING_EFFORT` | No | `low` | Small GPT-5.6 reasoning budget for reliable v1.4 contract and length checks |
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
| `SCORE_CAPTURE_CONSUMER_ALLOWED_CIDRS` | No | BILD Next staging NAT `/32` address | Dedicated egress allowlist for the two minimal GET routes and exact read-only batch POST; does not grant UI, debug, or browser-capture POST access |
| `INTERNAL_ACCESS_EXEMPT_PATHS` | No | `/api/health` | Comma-separated route list that remains reachable without the internal allowlist; production should use `/api/health,/api/v1` so only health checks and authenticated consumer routes are externally reachable |
| `DB_PATH` | No | `.push_history.db` | SQLite location; the scheduled Render flow requires `/data/.push_history.db` on the persistent disk |
| `PUSH_DB_DURABILITY_REQUIRED` | No | `true` on Render | Fail startup instead of using `/tmp` when durable storage is missing; must remain `true` for scheduled Teams delivery |
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

---

## Deployment (Flux/CD)

To match the deployment path expected by `spring-media/bildnext-flux-cd`
(application repository: https://github.com/spring-media/next-push.balancer),
this repository includes the application-side artifacts a consuming Flux repo needs:

- [`VERSION`](VERSION) for semver image and chart publishing
- [`.github/workflows/docker-build.yaml`](.github/workflows/docker-build.yaml)
  for atomic ARM64 image and matching Helm-chart releases with semver tags
- [`helm/`](helm/) for the application chart
- [`deploy/flux-examples/`](deploy/flux-examples/) for sanitized `HelmRelease`,
  `HelmRepository`, and `ImageUpdateAutomation` examples

Use [`deploy/README.md`](deploy/README.md) as the repo-local checklist. The
Flux/Next runtime starts `app.score_main:app` via the contract-bound
[`Dockerfile`](Dockerfile); the Render deployment of the full Push Balancer
uses [`Dockerfile.render`](Dockerfile.render) with `app.main:app`.
