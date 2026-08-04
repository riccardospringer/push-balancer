# CMS score API activation and rollback

This runbook keeps the existing Render scoring calculation unchanged while the
read-only CMS-ID adapter is introduced on Next. The Render capture contract is
extended only with an allowlisted numeric explanation of already calculated
values. Production
activation is a separate, explicitly approved change.

The authoritative approval checklist is
[`spring-media/next-push.balancer#2`](https://github.com/spring-media/next-push.balancer/issues/2).

## Required gates

Before any live request, record all of the following in the approved systems:

- Product/System Owner and Privacy Manager approval, including purpose, legal
  basis, roles, recipient, rights handling, and log retention
- confirmation that the approval covers the additive numeric breakdown and OR
  factor, plus the named consumer repository and namespace/service account or
  approved egress IP/CIDR
- an authorised Render metadata inventory and an approved minimal live score
  comparison; never copy Render secrets, logs, databases, or trained models
- a named owner and documented physical deletion/expiry process for Render's
  `article_score_log`
- incident-owner closure or an explicitly documented, scoped release decision
- the immutable application image digest and Helm chart version produced after
  merge, entered as Helm value `digest`, plus confirmation that the target has
  ARM64 capacity
- separate staging and production secret-manager references for
  `SCORE_API_KEY` and the dedicated service `URL_API_KEY`
- the approved internal `URL_API_BASE`
- private DNS, TLS/mTLS or equivalent workload identity, ingress class,
  consumer ranges, and trusted proxy ranges
- access-log redaction or disabling at every proxy; the CMS ID is in the path
- approved ingress and egress policy, including cluster DNS, UrlServer, and the
  BILD sitemap source

Standard Kubernetes NetworkPolicy cannot restrict HTTPS by FQDN. Use internal
service selectors, an approved egress proxy with stable ranges, or a supported
FQDN policy such as Cilium. Do not open unrestricted TCP/443 egress.

## Additive rollout order

After approval, deletion and incident gates are documented, use this order:

1. Deploy the Render source extension first. Its existing default single-score
   response remains legacy-minimal, while the explanation and true batch
   contracts are available only through their fixed opt-in routes.
2. Before changing Next, verify from the approved Next egress path that Render's
   default single lookup is unchanged and that the new Render batch route
   returns one strict ordered result per requested unique CMS ID.
   `/api/ready` alone does not prove batch capability. The currently deployed
   Next version does not call the new route.
3. Deploy this Next adapter only after the Render batch source is ready. Verify
   the public single response and `POST /api/v1/scores/batch`, including exact
   total, component, factor, timestamp, order, duplicate, and not-found parity.
4. Rolling Next back first is safe because the previous Next version ignores
   Render's opt-in batch route and Render's default response remains unchanged.
5. For rollback, disable or revert Next's public batch use before, or in the
   same coordinated change as, reverting Render. Never revert Render alone
   while this Next batch endpoint is serving traffic: that would break public
   batch calls. The public single lookup remains compatible with Render's
   legacy `{score,capturedAt}` shape.

## Verified BILD Next platform conventions

The documented private hostnames are:

- `push-balancer.internal.stg.bildnext.as-infra.de`
- `push-balancer.internal.prd.bildnext.as-infra.de`

MaPS uses the `skipper` ingress class with
`zalando.org/aws-load-balancer-scheme: internal` and a `SourceFromLast`
predicate for approved internal source ranges. The platform documents ACM
wildcard termination for team ingress rather than a per-application Kubernetes
TLS Secret. Do not invent a TLS Secret name: the chart's TLS rollout gate must
first be aligned with this platform-managed certificate model, or a real Secret
must be confirmed by the platform owner.

The MaPS VPC-CNI component configures NetworkPolicy support, but enforcement,
the actual Skipper peer address, trusted-proxy ranges, and consumer source path
must still be verified in the live cluster with synthetic traffic. Platform
CIDRs are not automatically approved consumer or trusted-proxy ranges.

## Staging

1. Merge the application PR and verify the published image and chart, including
   the ARM64 manifest and immutable digest. Do not activate a version that has
   not been published.
2. Merge the suspended Flux scaffold. Keep `ImageUpdateAutomation` suspended.
3. Record Product/System Owner and Privacy approval before the first workload
   start, even when it cannot receive traffic or process live data.
4. Render the exact dark-start values, lint the chart, validate schemas, and
   perform a server-side dry run. Keep runtime `ExternalSecret` disabled,
   `URL_API_BASE` empty, `SCORE_API_KEY` absent, ingress disabled, application
   allowlists loopback-only, and NetworkPolicy ingress and egress empty.
5. Dark-deploy staging by setting only `HelmRelease.spec.suspend: false`.
   Verify Deployment, Pod, frozen seed load, image digest, architecture, and
   `/api/health`. Do not read application logs unless an approved, redacted
   diagnostic path is required.
6. If this dark start fails, reconcile `replicaCount: 0`, verify that no Pod is
   running, and only then suspend the HelmRelease. Suspending alone does not
   stop a Pod that already exists.
7. In a separate approved change, provision secret values outside Git, add only
   their `ExternalSecret` references, and configure UrlServer, allowlists,
   trusted proxies, and least-privilege egress. Never put secret values in Git.
8. Use an authorised internal path for one functional score check. Health alone
   does not verify UrlServer, sitemap, or scoring.
9. In a separate approved PR, activate private reachability with final
   NetworkPolicy, allowlists, TLS/mTLS, log redaction,
   `ingress.rolloutGateApproved: true`, and `ingress.enabled: true`. If the
   consumer is in-cluster, prefer Service DNS plus mesh identity and keep
   ingress disabled.
10. Run all positive and negative contract checks below from the real network
   locations, then complete the approved minimal Render comparison.

## Production

1. Repeat the dark deployment with separate production secrets and production
   network values. Never reuse staging keys.
2. Verify the exact image digest before enabling reachability.
3. Activate production reachability in a separate, explicitly approved PR.
4. Switch the consumer URL or feature flag only after production verification.
5. Keep Render and the old Push Balancer untouched as the score reference; a
   rollback of this adapter must not mutate their state.
6. Keep image automation suspended until compatibility and rollback behavior
   for independent chart/image updates have been approved.

## Contract verification

Use only synthetic identifiers for negative tests and an explicitly approved
minimal CMS-ID sample for the positive live check.

- after deploying Render first, its default single response remains exactly
  legacy-minimal and its opt-in batch source returns strict ordered results;
  the old Next deployment remains unaffected
- after deploying Next, an existing legacy snapshot returns `200` with
  `scoreBreakdown: null` and `orFactor: null`, while a fresh engagement or sport
  snapshot returns `200` with `cmsId`, `score`, `scoredAt`, `scoreBreakdown`,
  and `orFactor`; compare every numeric value with the same Render UI candidate
- `score` remains the captured Gesamt value; do not sum the breakdown or apply
  `orFactor` to it in the adapter or consumer; existing caps, age multipliers,
  and TV adjustments can make a naive component sum differ from Gesamt
- a legacy two-field Render snapshot still returns `200` with both explanation
  fields set to `null`
- response headers: `Cache-Control: no-store` and `Vary: X-Score-Key`
- missing or invalid key from an allowed network: `401`
- source outside the allowlist: denied by policy; application fallback is `404`
- unknown valid identifier: `404`
- invalid identifier: `422`
- upstream unavailable: `502`; disabled/unavailable calculation: `503`
- batch with 1–500 24-hex IDs: one result per original position, including
  duplicates, plus exact requested/unique/found/not-found counts
- batch source: exactly one deduplicated Render POST; malformed or partial
  source output makes the whole call `502`
- third simultaneous batch per worker: immediate `429` with `Retry-After: 1`
- `/docs`, `/openapi.json`, admin routes, and mutations: `404`
- TLS chain and mTLS/workload identity: positive and negative checks
- synthetic marker ID: absent from ingress, application, platform, and
  UrlServer logs under the approved retention configuration

Run `scripts/smoke_score_api.py` from the approved consumer network. It prints
only the score and returns `2` for no current score, `3` for a permanent
configuration/contract error, and `4` for a transient service/network error.

## Fail-closed rollback

`HelmRelease.spec.suspend: true` does not stop running traffic. It only stops
reconciliation. Never combine fail-closed values and `suspend: true` in one
rollback and assume the values were applied.

1. Stop the consumer feature flag or calls.
2. For an incident, immediately block private ingress or NetworkPolicy ingress
   and revoke/rotate `SCORE_API_KEY`.
3. While `HelmRelease.spec.suspend` is still `false`, apply:
   `ingress.enabled: false`, an empty NetworkPolicy ingress list, and loopback-
   only application allowlists.
4. Wait for reconciliation and verify that access is denied.
5. Remove runtime `ExternalSecret` references while reconciliation is active.
6. If the workload must stop, reconcile `replicaCount: 0` and verify that no
   Pod remains.
7. Only then set `HelmRelease.spec.suspend: true` in a separate change.
8. Delete secret-manager values according to the approved retention procedure.
9. Keep image automation suspended throughout rollback.

The legacy Render score path is not changed by this rollback.
