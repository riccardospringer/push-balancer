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
- Production activation requires the complete readiness proof in the
  authoritative runbook; old image tags, digests, Helm values, and smoke-check
  results from this file are not valid evidence.

Historical K8s/Next details are intentionally removed from this active branch
so operators cannot accidentally follow contradictory go-live instructions.
