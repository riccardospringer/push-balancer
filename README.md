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
- Optional: OpenAI API key (GPT-4o title scoring), Adobe Analytics credentials, Football-Data.org key, The Odds API key

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
# Edit .env and fill in the required values (at minimum OPENAI_API_KEY and PUSH_API_BASE)

# 4. Start backend and frontend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8050
pnpm --dir frontend dev
```

The API starts on `http://localhost:8050` by default, and the Vite frontend on `http://localhost:5173`. For production-like local checks, run `pnpm --dir frontend build`; the generated assets are written to `dist-frontend/` and served by FastAPI.

### Editorial One UI Registry Setup

The frontend is prepared for the private `@spring-media/editorial-one-ui` package via [frontend/.npmrc](/Users/riccardo.longo/push-balancer/frontend/.npmrc). To install the package when access is available:

```bash
export NPM_TOKEN=ghp_your_token_here
pnpm --dir frontend info @spring-media/editorial-one-ui
```

If the package is not yet available in your environment, the app uses the local shim in [frontend/src/editorial-one-ui-shim/index.tsx](/Users/riccardo.longo/push-balancer/frontend/src/editorial-one-ui-shim/index.tsx) while app code already imports `@spring-media/editorial-one-ui`.

When `openapi.yaml` changes, regenerate the frontend base client with:

```bash
pnpm --dir frontend generate:api-client
```

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

The Dockerfile is based on `python:3.13-slim` and exposes port `8050`. It bundles the pre-computed `push-snapshot.json` as a startup seed so Render instances have data before the first API poll completes.

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
| 8 | **GPT-4o LLM Scoring** | 5-dimension title quality scores (magnitude, clickability, relevance, urgency, emotionality) |
| 9 | **Keyword Heuristic** | Rule-based fallback when no model is available |

The Stacking Ensemble is only activated when its MAE is within 2% of the single LightGBM baseline (safety gate). An Online Residual Corrector applies real-time bias correction per category and hour group.

**Features include:** title length, emotional word counts, BILD topic clusters (crime, royals, costs, health, auto, relationships, extreme weather), temporal features (hour sin/cos, weekday, prime time, Bundesliga windows), historical OR baselines, TF-IDF and embedding similarities, GPT-4o LLM scores, and sport-specific magnitude signals.

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

A full OpenAPI specification is maintained in [`openapi.yaml`](openapi.yaml). The documented, frontend-stable contract currently includes:

### GET Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/health` | Service health, endpoint checks, and research metadata |
| `GET /api/articles` | Article candidates from the BILD sitemap |
| `GET /api/pushes` | Recent push history with same-day aggregates |
| `GET /api/feeds/competitor` | Editorial competitor monitoring feed |
| `GET /api/feeds/competitor/sport` | Sports competitor monitoring feed |
| `GET /api/research-insights` | Current research learnings and experiment summary |
| `GET /api/research-rules` | Active research rules with pagination metadata |
| `GET /api/analytics/adobe-traffic` | Adobe traffic analytics payload |
| `GET /api/ml-model` | Stable ML model status contract |
| `GET /api/ml-model/monitoring` | ML monitoring and recent prediction comparisons |
| `GET /api/gbrt-model` | Stable GBRT model status contract |
| `GET /api/tagesplan` | Daily planning slots with recommendations |
| `GET /api/tagesplan/retro` | Retrospective planning summary |
| `GET /api/tagesplan/suggestions` | Suggested articles for the current plan |

### POST Endpoints

| Endpoint | Description |
|---|---|
| `POST /api/pushes/refresh` | Refresh the live push view |
| `POST /api/ml-model/retraining-jobs` | Trigger an ML retraining job |
| `POST /api/gbrt-model/retraining-jobs` | Trigger a GBRT retraining job |
| `POST /api/gbrt-model/promotions` | Promote the current GBRT candidate |
| `POST /api/push-title/generate` | Generate optimized push headline variants |

Legacy or internal helper endpoints still exist for operational compatibility, but the frontend contract should prefer the documented endpoints above.

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
3. **Startup seed**: `push-snapshot.json` is bundled into the Docker image and seeded into SQLite at startup so the instance has baseline data before any live fetch succeeds.

### CORS

Allowed origins are computed automatically from `PORT`, `RAILWAY_PUBLIC_DOMAIN`, `RENDER_EXTERNAL_HOSTNAME`, and the local network IP. The Render hostname `push-balancer.onrender.com` is always included.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key for GPT-4o title scoring and editorial assistant |
| `PUSH_API_BASE` | Yes | `http://push-frontend.bildcms.de` | Base URL of the BILD Push Statistics API (internal network) |
| `FOOTBALL_DATA_KEY` | No | — | API key for Football-Data.org (Bundesliga, Champions League fixtures) |
| `ODDS_API_KEY` | No | — | API key for The Odds API (betting odds as sport context signal) |
| `ADOBE_CLIENT_ID` | No | — | Adobe Analytics OAuth2 client ID |
| `ADOBE_CLIENT_SECRET` | No | — | Adobe Analytics OAuth2 client secret |
| `ADOBE_GLOBAL_COMPANY_ID` | No | `axelsp2` | Adobe Analytics company ID |
| `BILD_SITEMAP_URL` | No | `https://www.bild.de/sitemap-news.xml` | BILD news sitemap URL |
| `PUSH_SYNC_SECRET` | No | `bild-push-sync-2026` | Shared secret for the push data relay between local server and Render |
| `RENDER_SYNC_URL` | No | — | Render deployment URL; if set, the local server relays push data to it (e.g. `https://push-balancer.onrender.com`) |
| `PORT` | No | `8050` | Server listen port |
| `BIND_HOST` | No | `0.0.0.0` | Server bind host |
| `ALLOW_INSECURE_SSL` | No | `0` | Set to `1` to disable SSL certificate verification (development only) |

Variables are loaded from a `.env` file in the project directory at startup (via a lightweight built-in parser — no `python-dotenv` required).

---

## Development

### Project Structure

```
push-balancer/
├── app/                      # FastAPI application, routers, ML and research modules
├── frontend/                 # React/Vite client
├── tests/                    # Pytest suite
├── push-balancer-server.py   # Legacy monolith kept for compatibility/migration
├── push-snapshot.json        # Startup data seed for Render
├── requirements.txt
├── pyproject.toml            # Python test/lint configuration
├── Dockerfile
├── .editorconfig
└── .push_history.db          # SQLite database (created at runtime, git-ignored)
```

### Adding a Feature

1. Implement backend logic in `app/` and prefer a dedicated router/module over extending legacy files.
2. Add or update the corresponding frontend page/component in `frontend/src/` when the feature is user-facing.
3. Document API changes in `openapi.yaml`.
4. If the feature introduces a new environment variable, add it to `.env.example` and the table in this README.
5. Run the relevant checks before pushing (`pytest`, frontend lint, frontend typecheck/build).

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
curl -X POST http://localhost:8050/api/ml/retrain

# GBRT
curl -X POST http://localhost:8050/api/gbrt/retrain
```

### Linting and Formatting

```bash
ruff check app tests
ruff format app tests
pnpm --dir frontend lint
```
