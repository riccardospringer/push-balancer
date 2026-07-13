# Privacy Overview

This project processes editorial and analytics-related data in an Axel Springer context and therefore follows privacy-by-design and privacy-by-default principles.

## Purpose

The service supports editorial planning for push notifications. It predicts opening-rate trends, analyses historical push performance, and provides decision support for editors.

## Data categories

- Push metadata such as title, category, send time, channel, opened count, and recipient count
- Derived analytics such as predictions, confidence intervals, trend metrics, and performance groupings
- Teams recommendation metadata such as article URL/title, decision scores, and technical delivery status
- Daily Teams schedule delivery metadata: calendar date, claim/send timestamp, item count, status, and truncated error text
- Optional Adobe Analytics traffic aggregates
- Optional OpenAI input for title optimisation when explicitly configured

## Data subjects

- Editorial users working with push campaigns
- End users receiving push notifications, only insofar as aggregated performance data is present
- Persons named in published editorial headlines, incidentally and only as part of the existing article metadata

## External systems

- BILD Push Statistics API
- BILD sitemap feed
- Adobe Analytics API when configured
- Microsoft Power Automate and Microsoft Teams for the configured editorial group chat
- OpenAI API when configured for title generation

## Runtime safeguards in this repository

- Production snapshots and raw analytics dumps must not be committed to git or baked into the Docker image.
- `PUSH_SNAPSHOT_PATH` is only for sanitized startup seed files mounted at runtime.
- `ADMIN_API_KEY` protects admin mutation endpoints and they should remain disabled when the key is unset.
- `PUSH_SYNC_SECRET` protects relay sync and must be set on both sides before `POST /api/pushes/sync` is exposed.
- `PUSH_TEAMS_WEBHOOK_URL` is a runtime-only secret. Teams payloads contain the minimum editorial recommendation fields and never include the webhook URL.
- Daily schedule sends are atomically claimed by date, preventing duplicate cloud delivery after restarts or concurrent workers.
- `GET /api/teams-recommendations` returns article/decision snapshots only and must remain behind the service's existing internal-access controls.
- Secrets such as `OPENAI_API_KEY`, Adobe credentials, `ADMIN_API_KEY`, `PUSH_SYNC_SECRET`, `PUSH_TEAMS_WEBHOOK_URL`, and `NPM_TOKEN` are runtime-only values.

## External transfer notes

- OpenAI is only contacted when title generation is explicitly configured and invoked.
- Adobe Analytics is only contacted when the Adobe credentials are configured.
- Microsoft Power Automate / Teams receives the existing editorial article recommendation and schedule fields when Teams alerts are enabled.
- These integrations should be treated as external recipients and reviewed when payload scope, tenant routing, or transfer conditions change.

## Retention and deletion

- SQLite data lives in the local database configured via `DB_PATH`.
- Actual push history is retained for 90 days; exact article URLs are deduplicated within that retained operating history.
- Teams alert metadata, recommendation snapshots, and daily schedule delivery metadata are deleted after 45 days by the database cleanup job.
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
