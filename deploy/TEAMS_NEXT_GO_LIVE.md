# Legacy Teams/Next go-live runbook — obsolete for Exact-5

> **Do not use this file to deploy, configure, enable, test, or roll back the
> current Exact-5 Power Automate channel.** Its former K8s/Next background-worker
> and incoming-webhook instructions describe a retired transport architecture
> and conflict with the single-owner scheduled flow.

The authoritative production contract and operating procedure is
[`integrations/power-automate/README.md`](../integrations/power-automate/README.md).
It defines the fixed Europe/Berlin schedule, exactly-five condition, one Teams
action, idempotent claim/receipt flow, persistent storage gate, monitoring,
cutover, and rollback.

For the current channel, these invariants apply:

- Power Automate is the only scheduler and Teams transport.
- `PUSH_TEAMS_BACKGROUND_SENDER_ENABLED=false` while that flow is enabled.
- The legacy incoming-webhook/Next sender must remain off and must never be
  enabled concurrently as a fallback or test.
- Operators must audit both **Meine Flows → Cloud-Flows** and **Für mich
  freigegeben**: the legacy instant-webhook flow and every legacy/shared
  scheduled flow remain **Off**, and exactly one canonical Exact-5 scheduler is
  active.
- A transport-owner cutover rotates `POWER_AUTOMATE_API_KEY`; only the
  canonical Exact-5 flow receives the new value. Retired, copied, and shared
  flows retain stale credentials and cannot be used as a live fallback.
- Production activation requires the complete readiness proof in the
  authoritative runbook; old image tags, digests, Helm values, and smoke-check
  results from this file are not valid evidence.

Historical K8s/Next details are intentionally removed from this active branch
so operators cannot accidentally follow contradictory go-live instructions.
