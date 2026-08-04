# Render-Deploy des Teams-Kanals (v0.1.32)

## Problem, das dieser Stand löst

Render deployte bisher aus `riccardospringer/push-balancer` — einem **separaten
Repo mit altem Code** (Ruhezeit bis 05:30 → der 5:31-Post; kein Hot-Fresh,
keine Robustheit). Alle Fixes liegen hier in `riccardospringer/next-push.balancer`.

## Sofort-Fix (heute, ohne Deploy)

Stoppt den 5:31-Post auf dem AKTUELLEN alten Render sofort. Im Render-Dashboard
→ Environment zwei Variablen setzen und speichern:

```
PUSH_TEAMS_QUIET_HOURS_START = 22:00
PUSH_TEAMS_QUIET_HOURS_END   = 06:00
```

Verifiziert gegen den exakten alten Code: blockt 22:00–06:00, erster Push
frühestens 06:00.

## Kompletter Fix (bringt Hot-Fresh, Raster, Robustheit live)

Ein Deploy-Schritt im Render-Dashboard:

1. **Settings → Repository**: von `riccardospringer/push-balancer` auf
   **`riccardospringer/next-push.balancer`**, Branch **`main`** umstellen.
   (Der Persistent-Disk-State unter `/data` bleibt am Service erhalten —
   Push-Historie und Dubletten-Claims überleben.)
2. **Settings → Docker**: der Blueprint nutzt bereits `./Dockerfile.render`
   (voller Kanal, `app.main:app`). Nichts zu tun.
3. **Environment**: sicherstellen, dass `PUSH_TEAMS_WEBHOOK_URL` (Secret)
   gesetzt ist. Die Timing-/Score-Werte kommen aus `render.yaml`:
   - `PUSH_TEAMS_ALERTS_ENABLED=true`
   - `PUSH_TEAMS_QUIET_HOURS_START=22:00`, `_END=06:00`
   - `PUSH_BALANCER_SCORE_API_ENABLED=false` (bewährte Capture-/Editorial-
     Score-Quelle; die interne Score-API ist aus Renders Netz nicht garantiert
     erreichbar und würde den Kanal sonst fail-closed verstummen lassen. Erst
     einschalten, wenn die interne URL aus Render nachweislich erreichbar ist.)
4. **Manual Deploy → Deploy latest commit.**

## Verifikation nach dem Deploy

```bash
curl -s https://<render-url>/api/teams-readiness | jq
python3 scripts/teams_smoke_check.py --base-url https://<render-url>
```

`/api/teams-readiness` muss `200` liefern (altes Code = 404); `runtime.status`
sollte `ok`/`starting` sein, `configurationProblems` leer.

## Danach

Der Kanal läuft dann mit deterministischem Raster (06/07/08-Doppel, 12:30,
Abend-Hot-Hours), Ruhezeit 22–06, Hot-Fresh-Sofort-Eskalation für brandaktuelle
Top-Stories, Zustellungs-Retry und Watchdog. Wirkung: `/api/teams-effectiveness`.
