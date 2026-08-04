# Push Balancer Helm Chart

This Helm chart deploys Push Balancer to a Kubernetes cluster. Its privacy-safe
default runs the isolated CMS score ASGI application; an explicitly approved
deployment can override `ASGI_APP` for another runtime.

## Prerequisites

- Kubernetes 1.27+
- Helm 3.0+
- kubectl configured to communicate with your cluster
- External Secrets Operator for runtime secret references
- a compatible Reloader controller, or an approved equivalent that restarts
  pods after ExternalSecret value rotation

## Fail-closed score defaults

The chart defaults to `ASGI_APP=app.score_main:app`, one replica, no persistence,
no live push fetch, and no legacy background automation. It creates a
default-deny NetworkPolicy with empty ingress and egress rules. Ingress is
disabled, and Helm refuses to render it when enabled unless
`ingress.rolloutGateApproved=true`, `ingress.tls` is configured, and `digest`
contains the approved immutable `sha256:` image digest.

These defaults are intentional rollout gates. Supply private target-cluster
rules for the approved ingress, DNS, BILD sitemap and UrlServer; never disable
the policy merely to make the service reachable. Configure mTLS or an
equivalently approved workload-authentication control at the ingress and verify
that request-path logs are redacted with an approved retention period.

For clusters without an FQDN-aware NetworkPolicy implementation, enable
`egressProxy`. It runs a separate, non-root CONNECT-only workload that accepts
only the exact names in `allowedConnectHosts`, never terminates TLS, and emits
no request logs. Resolved proxy targets must also be globally routable public
IPv4 unicast addresses; IPv6, private, VPC, cluster, CGNAT, metadata,
documentation, multicast and other special-use results are rejected. This
fail-closed IPv4 scope matches the approved upstreams and avoids ambiguous IPv6
transition-address handling. The application receives `HTTPS_PROXY`
automatically and may egress only to cluster DNS and the proxy. Configure
`networkPolicy.dns` for the actual cluster DNS workload and
`networkPolicy.upstreamCidrs` only on the proxy; never add direct TCP/443 egress
to the application policy.

On an IPv4 AWS VPC CNI cluster where a broader additive policy would otherwise
permit direct application egress, `egressProxy.clusterNetworkPolicy.enabled`
can add a second, cluster-scoped enforcement layer. It is disabled by default
and may be enabled only after the `networking.k8s.aws/v1alpha1`
`ClusterNetworkPolicy` CRD, VPC CNI enforcement, reserved priorities 900/901,
and helm-controller create/delete RBAC have been verified in that cluster. The
chart then renders disjoint namespace-and-release selectors: Admin priority 900
accepts app DNS plus proxy traffic and proxy DNS plus TCP/443, while priority
901 denies every other IPv4 destination. Cluster-scoped names contain the
namespace, Helm release and a collision-resistant identity hash. Do not enable
this mode on an IPv6 or dual-stack cluster until equivalent IPv6 enforcement
has been separately validated. The proxy's exact-host and globally-routable-IP
checks remain mandatory because the proxy's TCP/443 Admin rule covers IPv4.

AWS VPC CNI standard enforcement mode can briefly leave a newly created pod in
default-allow state before its policies are attached. Enabling the cluster
policy therefore also adds a mandatory, secretless init container to the
application pod. The same proxy Service exposes a fixed sentinel TCP port only
in this mode. The gate resolves that Service once per round, requires a
credential-free CONNECT to the already-approved `www.bild.de:443` target on the
real proxy port, and requires a TCP timeout on the sentinel port at that same
Service address in three consecutive rounds. Only a timeout counts as a policy
drop; refused, unreachable, or other errors are indeterminate and fail closed.
The main container cannot start before this same-path evidence exists. The gate
sends no HTTP path, TLS content or credentials and emits no request or address
logs. Keep `www.bild.de` in `allowedConnectHosts` whenever this mode is enabled.

## Installation

### Quick Start

```bash
kubectl create namespace bildnext

helm install push-balancer ./helm \
  --namespace bildnext
```

This installs a network-isolated workload with the score route disabled until
its runtime key, UrlServer endpoint, allowlists and NetworkPolicy rules exist.

### Installation with Ingress

```bash
helm upgrade --install push-balancer ./helm \
  --namespace bildnext \
  --values approved-score-values.yaml
```

The approved values file must keep the path limited to `/api/v1/scores`, set
the private host and ingress class, configure TLS (and mTLS or the approved
equivalent), set `ingress.rolloutGateApproved=true`, and provide non-empty
NetworkPolicy ingress/egress rules. Do not commit private hosts, CIDRs or secret
references to this public example repository.

On MaPS, `platformManagedTls=true` may be used instead of a Kubernetes TLS
Secret only with `className=skipper` and the `internal` load-balancer scheme.
The chart rejects every other platform-managed TLS combination.

### Using an Existing GHCR Pull Secret

```bash
helm install push-balancer ./helm \
  --namespace bildnext \
  --set imagePullSecrets[0].name=ghcr-imagepull-secret \
  --set imagePullSecret.enabled=false
```

## Validation

```bash
helm lint ./helm

helm template push-balancer ./helm \
  --namespace bildnext
```

## Flux Alignment

This chart follows the same structure expected by the `bildnext-flux-cd`
examples:

- app chart is versioned and published separately from the Docker image
- image tag is exposed as a top-level `tag` value
- Flux `ImageRepository` and `ImagePolicy` can be rendered by the chart
- image pull secret and application secrets can be sourced via `ExternalSecret`
- ConfigMap/ExternalSecret template changes alter pod checksums, while the
  Reloader annotation covers ExternalSecret value rotations
- Flux examples remain fail closed until private TLS, authentication and
  NetworkPolicy values have passed the rollout gates in `deploy/README.md`
