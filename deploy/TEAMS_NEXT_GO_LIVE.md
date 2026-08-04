# Teams-Kanal auf Next/K8s scharfschalten — Go-Live-Runbook

Dieser Runbook bringt den **vollen Teams-Kanal** (deterministisches Raster,
Hot-Fresh, Robustheit, **kanonische Push-Balancer-Top-1**) 24/7 im AS-Netz live.
Es ist der einzige Pfad, auf dem die echte Score-API-Top-1 möglich ist — Render
kann die interne Score-API nicht erreichen.

**Die Applikationsseite ist vollständig fertig und publiziert.** Alles unten
braucht Zugriff auf `spring-media/bildnext-flux-cd` und den Cluster-Secret-Store.

---

## 0. Was bereits fertig ist (keine Aktion nötig)

| Baustein | Status |
|---|---|
| Voll-System-Code (Raster 06:00–23:00, Hot-Fresh, Retry, Watchdog, Wirkungsmessung) | ✅ `spring-media/next-push.balancer@main`, v0.1.34 |
| Docker-Image ARM64 | ✅ `ghcr.io/spring-media/next-push.balancer:0.1.34` |
| **Immutable Digest** | ✅ `sha256:6c834f5275d50d57974afdedc184fb2027e9dff2651629c30f09ae7f0985fb99` |
| Helm-Chart | ✅ `next-push-balancer-chart` **0.1.34** (OCI, `oci://ghcr.io/spring-media`) |
| HelmRelease-Vorlagen (stg + prd) | ✅ `deploy/flux-examples/{stg,prd}/push-balancer-teams.yaml` |
| Smoke-Check | ✅ `scripts/teams_smoke_check.py` |

CI-Beweis: „Docker Build and Push" für den v0.1.34-Merge auf `main` war grün
(2026-07-31 14:04 UTC).

---

## 1. Secrets im Cluster-Secret-Store anlegen

Zwei Werte, **niemals in Git**, an den in eurem Store üblichen Pfaden:

| secretKey (im Pod) | Inhalt |
|---|---|
| `PUSH_TEAMS_WEBHOOK_URL` | Power-Automate/Teams Incoming-Webhook-URL des Ziel-Kanals |
| `PUSH_BALANCER_SCORE_API_KEY` | Consumer-Key für die interne Score-API |

Dann in der HelmRelease unter `externalSecret`:
- `secretStoreRef.name` → euren realen `ClusterSecretStore` (ersetzt
  `example-secret-manager`)
- die zwei `remoteRef.key`-Pfade → eure realen Store-Pfade
  (ersetzt `/example/project/push-balancer/...`)

## 2. Interne Basis-URLs setzen (bleiben aus dem öffentlichen Repo draußen)

Im Cluster-Repo unter `values.config`:

| Key | Wert |
|---|---|
| `PUSH_API_BASE` | interne BILD-Push-API-Basis-URL |
| `PUSH_BALANCER_SCORE_API_BASE_URL` | interner Score-API-Service (z. B. Cluster-DNS `http://push-balancer.bildnext.svc.cluster.local:8050` oder der bestätigte interne Host) |

`PUSH_BALANCER_SCORE_API_ENABLED: "true"` ist bereits gesetzt → damit zieht der
Kanal die **kanonische Top-1** statt Fallback.

## 3. Digest pinnen

In `values.digest` den approved Digest eintragen:
```
digest: "sha256:6c834f5275d50d57974afdedc184fb2027e9dff2651629c30f09ae7f0985fb99"
```
(Chart erzwingt `sha256:<64 hex>`; leerer Wert = Tag-basiert, für Produktion
Digest pinnen.)

## 4. NetworkPolicy-Egress freigeben

Default-deny bleibt. In `values.networkPolicy.egress` genau diese Ziele erlauben
(Standard-K8s-NetworkPolicy kann HTTPS nicht per FQDN filtern → interne
Service-Selektoren bzw. approved Egress-Ranges nutzen, wie im Score-Release):

1. **Cluster-DNS** (kube-dns, UDP/TCP 53)
2. **BILD-Push-API** (`PUSH_API_BASE`)
3. **Interne Score-API** (`PUSH_BALANCER_SCORE_API_BASE_URL`)
4. **www.bild.de** (Sitemap, HTTPS 443)
5. **Teams/Power-Automate-Webhook** (HTTPS 443)

Ingress bleibt leer — der Kanal ist rein outbound. `ingress.enabled: false`
bleibt so.

## 5. HelmRelease in den Flux-Pfad kopieren

- `deploy/flux-examples/stg/push-balancer-teams.yaml` → Staging-Pfad im
  `bildnext-flux-cd`-Repo
- `deploy/flux-examples/prd/push-balancer-teams.yaml` → Produktions-Pfad
- Beide referenzieren Chart **0.1.34** und die `next-helm-charts`-HelmRepository
  (vorhandene wiederverwenden, keine Dublette anlegen).
- Der PVC (`persistence.enabled: true`, 2Gi) ist Pflicht: SQLite-State verhindert
  doppelte Live-Push-Posts und verlorene Tageszähler nach Neustarts.

## 6. Rollout & Verifikation

Nach dem Reconcile:
```bash
kubectl -n bildnext rollout status deploy/push-balancer-teams
kubectl -n bildnext port-forward svc/push-balancer-teams 8050:8050 &
python3 scripts/teams_smoke_check.py --base-url http://localhost:8050
```
Exit-Code **0** = einsatzbereit. Der Smoke-Check prüft end-to-end: kanonische
Score-API-Kette, frische Push-Historie, korrekten Tagesplan (Morgen-Doppel
06/07/08, Mittagsslot 12:30, Abend-Hot-Hours) und Webhook-Konfiguration.

Zusätzlich:
```bash
curl -s localhost:8050/api/health          # teamsChannel: healthy
curl -s localhost:8050/api/teams-readiness  # 200, keine configurationProblems, scoreApi ok
```

## 7. Render abschalten (Doppelposts vermeiden)

Sobald der Next-Kanal grün postet: im Render-Service
`PUSH_TEAMS_ALERTS_ENABLED=false` setzen. Dann läuft der Teams-Kanal
ausschließlich auf Next (kanonische Top-1), Render bleibt reiner Score-Rechner.

## 8. Rollback

- Kanal stumm schalten: `PUSH_TEAMS_ALERTS_ENABLED=false` im Teams-Release
  (Reloader triggert Rollout) **oder** `replicaCount: 0` reconcilen.
- `HelmRelease.spec.suspend: true` stoppt nur Reconciliation, **nicht** laufenden
  Traffic — erst Replicas/Flag ziehen, dann suspend.
- Render + Score-API bleiben davon unberührt.

---

### Der eine offene menschliche Schritt
Schritte 1–7 brauchen Cluster- und Secret-Store-Zugriff. Sobald die zwei Secrets
und die zwei internen Basis-URLs gesetzt sind und die HelmRelease im Flux-Repo
liegt, postet der Kanal Montag früh ab 06:00 die **kanonische Push-Balancer-Top-1**.
