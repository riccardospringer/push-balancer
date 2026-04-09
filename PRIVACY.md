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
