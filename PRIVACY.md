# Privacy Overview

This project processes editorial and analytics-related data in an Axel Springer context and therefore follows privacy-by-design and privacy-by-default principles.

## Purpose

The service supports editorial planning for push notifications. It predicts opening-rate trends, analyses historical push performance, and provides decision support for editors.

## Data categories

- Push metadata such as title, category, send time, channel, opened count, and recipient count
- Derived analytics such as predictions, confidence intervals, trend metrics, and performance groupings
- Optional Adobe Analytics traffic aggregates
- Optional OpenAI input for title optimisation when explicitly configured

## Data subjects

- Editorial users working with push campaigns
- End users receiving push notifications, only insofar as aggregated performance data is present

## External systems

- BILD Push Statistics API
- BILD sitemap feed
- Adobe Analytics API when configured
- OpenAI API when configured for title generation

## Runtime safeguards in this repository

- Production snapshots and raw analytics dumps must not be committed to git or baked into the Docker image.
- `PUSH_SNAPSHOT_PATH` is only for sanitized startup seed files mounted at runtime.
- `ADMIN_API_KEY` protects admin mutation endpoints and they should remain disabled when the key is unset.
- `PUSH_SYNC_SECRET` protects relay sync and must be set on both sides before `POST /api/pushes/sync` is exposed.
- Secrets such as `OPENAI_API_KEY`, Adobe credentials, `ADMIN_API_KEY`, `PUSH_SYNC_SECRET`, and `NPM_TOKEN` are runtime-only values.

## External transfer notes

- OpenAI is only contacted when title generation is explicitly configured and invoked.
- Adobe Analytics is only contacted when the Adobe credentials are configured.
- Both integrations should be treated as external recipients and reviewed when payload scope changes.

## Retention and deletion

- SQLite data lives in the local database configured via `DB_PATH`.
- Daily plan suggestion snapshots are persisted for retrospective analysis and should be reviewed when retention needs change.
- Any new persisted data category should document retention and deletion behavior in the corresponding handover or PR note.

## Engineering rules

- Do not use production data in prompts, tests, screenshots, or examples
- Do not log secrets, tokens, or raw payloads unnecessarily
- Keep retention, deletion paths, and role-based access in mind when changing persistence or observability
- Escalate any new external integration, profiling logic, or privacy-relevant processing before rollout

## PR / handover template

Use this block for privacy-relevant changes:

```text
PRIVACY NOTE
- Purpose:
- Data categories:
- Data subjects:
- Legal basis: [or: TBD – Legal Review required]
- Roles:
- External recipients / international transfer:
- Retention / deletion:
- Safeguards:
- Required documentation / approvals:
```
