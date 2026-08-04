# Push Balancer Flux Alignment

This repository contains the application-side artifacts that match the deployment
path described in `spring-media/bildnext-flux-cd`.

## Canonical Sources

- Application repository: `https://github.com/spring-media/next-push.balancer`
- Flux repository: `https://github.com/spring-media/bildnext-flux-cd`
- Published Docker image: `ghcr.io/spring-media/next-push.balancer`
- Published Helm chart repository: `oci://ghcr.io/spring-media`

## Simple Deployment Path

Use this as the minimum path to get Push Balancer deployed through a Flux repo.

1. Create a `HelmRelease` in the Flux repo.
   Use the manifests under `deploy/flux-examples/` as the reference.
   Reuse the existing `next-helm-charts` source when it is already present;
   do not create a duplicate HelmRepository.

2. Make sure this application repo builds and pushes an ARM64 Docker image with
   semver tags.
   The workflow is implemented in `.github/workflows/docker-build.yaml`.

3. Make sure this application repo contains a Helm chart that follows the
   example chart setup and publishes that chart.
   The chart lives in `helm/`; `.github/workflows/docker-build.yaml` publishes
   it from the exact same immutable release tree as the Docker image.

4. Add image update automation in the Flux repo.
   Use `deploy/flux-examples/stg/push-balancer-automation.yaml` and
   `deploy/flux-examples/prd/push-balancer-automation.yaml` as the reference.

At the time of the score migration audit, the target Flux repository did not
yet contain a Push Balancer path. Create the staging and production
`push-balancer/` directories only after the rollout gates below are approved.

## CMS score rollout gates

Before copying these examples into the target Flux repository:

- confirm Product/System Owner and Privacy Manager approval for the consuming team
- confirm legal basis, roles, access/export/correction handling, and log retention
- provision separate `SCORE_API_KEY` and `URL_API_KEY` secret references
- replace `URL_API_BASE` with the approved internal UrlServer endpoint
- keep `ASGI_APP=app.score_main:app`; the full application ASGI entrypoint is
  outside the score-consumer purpose
- keep `BACKGROUND_AUTOMATIONS_ENABLED=false`, `PUSH_LIVE_FETCH_ENABLED=false`,
  `LIVE_FEED_FALLBACK_ENABLED=false`, and paid external APIs disabled
- keep production at one replica until stateless numeric parity is approved
- keep only `/api/health` exempt from internal network access
- replace both loopback-only CIDR placeholders with the approved consumer and
  trusted-ingress ranges; the ingress must overwrite forwarding headers
- configure private ingress for only `/api/v1/scores`, with TLS and either mTLS
  or an equivalently approved workload-authentication control
- keep `ingress.enabled=false` and `ingress.rolloutGateApproved=false` until the
  ingress host, TLS secret, authentication, and access-log redaction/retention
  have been approved
- keep the NetworkPolicy enabled; replace its empty default-deny lists with
  target-cluster rules for the approved ingress, cluster DNS, the BILD sitemap,
  and UrlServer only
- when the isolated egress proxy is enabled, keep its hostname allowlist exact;
  it additionally accepts only globally routable public IPv4 unicast and
  rejects IPv6, private, CGNAT, metadata and other special-use ranges
- if an additive cluster policy still permits direct egress on AWS, enable the
  optional ClusterNetworkPolicy layer only after verifying the target CRD, VPC
  CNI enforcement, helm-controller cluster RBAC and reserved Admin priorities
  900/901; it is currently approved only for the validated IPv4 cluster mode
- account for AWS VPC CNI standard mode's initial default-allow interval: the
  ClusterNetworkPolicy option must retain its mandatory secretless startup gate,
  immutable image digest and fixed allowlisted `www.bild.de` CONNECT target; do
  not bypass the gate if the same Service's sentinel port is not dropped by
  three consecutive policy timeouts within 120 seconds
- verify a compatible Reloader controller is operating. ConfigMap template
  changes trigger a checksum rollout; ExternalSecret value rotation needs the
  Reloader annotation (or a documented equivalent rollout mechanism)
- provision only `SCORE_API_KEY` and `URL_API_KEY`; do not enable admin or
  push-sync credentials in the score-only workload
- do not copy the Render database, snapshots, logs, trained runtime models, or secrets
- in an authorised private Render shell, record only software/model metadata
  (commit/image identifier, Python and package versions, architecture,
  `libgomp1` version, and active-model SHA-256); never capture `env`, secrets,
  raw logs, CMS IDs, or database contents
- derive immutable base-image and hashed dependency locks from that inventory;
  do not treat the open `>=` ranges in `requirements.txt` as a parity lock
- run the synthetic golden fixture in both the Render-compatible amd64 image and
  the ARM64 Next image, then compare an approved minimal live sample before cutover

The `Score Runtime Parity` workflow executes
`scripts/score-parity-fixture.py` inside native amd64 and ARM64 release
containers. Both jobs must report the same dependency versions, seed digest,
prediction and final score before the application PR can leave draft state.
This validates the committed cold-start runtime; the separately approved live
Render model/state comparison remains mandatory for production activation.

The score route fails closed while `SCORE_API_KEY` is absent. `URL_API_BASE`
and `URL_API_KEY` are optional legacy-URL fallback settings.
The chart also refuses to render an enabled ingress without explicit rollout
approval and TLS configuration. The Flux examples deliberately combine a
disabled ingress, loopback-only application allowlists, and a default-deny
NetworkPolicy. Do not bypass those controls merely to make the endpoint
reachable, expose it through a public ingress, or embed its key in frontend code.

Use [`SCORE_API_CONSUMER.md`](SCORE_API_CONSUMER.md) for the consuming team's
contract and [`SCORE_API_ACTIVATION.md`](SCORE_API_ACTIVATION.md) for the exact
dark-deploy, verification, production activation, and fail-closed rollback
sequence. In particular, `HelmRelease.spec.suspend: true` pauses reconciliation
but does not stop already-running traffic.

## Deployment Checklist

- Do I have a `HelmRelease` in the Flux repo?
- Does that `HelmRelease` include the image tag setter comment,
  for example `# {"$imagepolicy": "namespace:app:tag"}`?
- Do I have a Docker build process in this application repo that builds
  `linux/arm64` images?
- Are the published Docker image tags semver, for example `1.2.3`?
- Do I have a Helm chart action in this application repo?
- Does the application Helm chart follow the example chart setup?
- Do I have image update automation in the Flux repo?
- Do the workflows follow the example structure?

## What "Follows The Examples" Means

For this application repo:

- A Docker workflow builds and pushes an ARM64 image with semver tags.
- A Helm chart exists in `helm/` and follows the same setup expected by the
  `bildnext-flux-cd` examples.
- A Helm workflow publishes the chart after the image workflow or when
  `helm/**` changes.

For the Flux repo that consumes this application:

- A `HelmRelease` points to the published chart version.
- An `ImageUpdateAutomation` updates the deployment path in that repo.
- The `HelmRelease` image tag is managed by Flux image automation.
- The Flux image policy follows the published semver image tags.

## Fail-closed example files

- `deploy/flux-examples/stg/ghcr.yaml`
- `deploy/flux-examples/stg/push-balancer.yaml`
- `deploy/flux-examples/stg/push-balancer-automation.yaml`
- `deploy/flux-examples/prd/ghcr.yaml`
- `deploy/flux-examples/prd/push-balancer.yaml`
- `deploy/flux-examples/prd/push-balancer-automation.yaml`

All secret-store names, remote secret keys, ingress hosts, URLs, and CIDRs in
those examples are placeholders so the repository can stay public. They are
scaffolds, not deployable manifests: ingress is disabled, TLS is unset, and the
NetworkPolicy denies all ingress and egress until private target-cluster values
are supplied and approved.

The chart version and image tag track the current `VERSION`, and CI updates the
Flux example versions together with the chart. That mechanical alignment does
not satisfy the rollout gates above.

## Teams Channel Runtime (push-balancer-teams)

Der Teams-Kanal (voller Push Balancer, `ASGI_APP=app.main:app`) laeuft als
zweites HelmRelease neben der Score-API auf Next - 24/7 im AS-Netz mit
direkter Erreichbarkeit von BILD-Push-API und interner Score-API. Auf Render
ist `PUSH_TEAMS_ALERTS_ENABLED=false`, damit es keine Doppelposts gibt.

Beispiele: `deploy/flux-examples/stg/push-balancer-teams.yaml` und
`deploy/flux-examples/prd/push-balancer-teams.yaml`.

Checkliste vor Aktivierung:

1. Secrets im Secret-Store anlegen: Teams-Webhook-URL und Score-API-Consumer-Key
   (Pfade in den Beispielen ersetzen).
2. `PUSH_API_BASE` und `PUSH_BALANCER_SCORE_API_BASE_URL` im Cluster-Repo
   setzen (interne Hostnamen bleiben aus dem oeffentlichen Repo draussen).
3. NetworkPolicy-Egress erlauben: DNS, BILD-Push-API, Score-API, www.bild.de
   (Sitemap), Teams/Power-Automate-Webhook (HTTPS). Ingress bleibt leer.
4. PVC beibehalten: SQLite-State verhindert doppelte Live-Push-Posts nach
   Neustarts.
5. Nachweis nach dem Rollout (Port-Forward):

   ```bash
   kubectl -n bildnext port-forward svc/push-balancer-teams 8050:8050 &
   python3 scripts/teams_smoke_check.py --base-url http://localhost:8050
   ```

   Der Smoke-Check prueft end-to-end: kanonische Score-API-Kette, frische
   Push-Historie, korrekten Tagesplan (Morgen-Doppel, Mittagsslot,
   Abend-Hot-Hours) und Webhook-Konfiguration. Exit-Code 0 = einsatzbereit.
