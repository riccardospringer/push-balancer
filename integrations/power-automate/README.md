# Scheduled Power Automate Teams delivery

This runbook configures Power Automate as the only production scheduler and Microsoft Teams transport for Push Balancer recommendations. Push Balancer keeps ownership of the absolute internal-API ranking, the five displayed recommendations, exact duplicate protection, and the durable per-slot claim.

## Production contract

The flow prepares each recommendation two minutes before these fixed
Europe/Berlin delivery slots and waits until `scheduledAtUtc` before posting it to
Teams. Monday through Friday:

| Slot | Recurrence/claim-attempt window |
|---|---|
| `06:00` | `06:00`–`06:04` |
| `06:36` | `06:36`–`06:40` |
| `07:12` | `07:12`–`07:16` |
| `07:47` | `07:47`–`07:51` |
| `08:23` | `08:23`–`08:27` |
| `08:59` | `08:59`–`09:03` |
| `12:30` | `12:30`–`12:34` |
| `17:30` | `17:30`–`17:34` |
| `18:49` | `18:49`–`18:53` |
| `20:08` | `20:08`–`20:12` |
| `21:26` | `21:26`–`21:30` |
| `22:45` | `22:45`–`22:49` |

On Saturday and Sunday, only the six morning slots move two hours later:

| Slot | Recurrence/claim-attempt window |
|---|---|
| `08:00` | `08:00`–`08:04` |
| `08:36` | `08:36`–`08:40` |
| `09:12` | `09:12`–`09:16` |
| `09:47` | `09:47`–`09:51` |
| `10:23` | `10:23`–`10:27` |
| `10:59` | `10:59`–`11:03` |

The common `12:30`, `17:30`, `18:49`, `20:08`, `21:26`, and `22:45` slots remain unchanged on weekends.

All claim windows are half-open (`slot - 2 minutes <= now < slot + 5 minutes`). The explicit Berlin conversion below handles both CET and CEST. The flow runs once per minute inside each window, not just at the first minute: this provides bounded recovery opportunities when a Microsoft 365 trigger is delayed or the claim API is temporarily unavailable before a slot is reserved. A successful early claim is held by a Power Automate **Delay until** action using the UTC `scheduledAtUtc` value; the Teams action therefore still starts at the official slot. The backend slot claim makes repeated runs idempotent.

The weekday/weekend labels above are the entire Power Automate schedule. `PUSH_TEAMS_SLOT_DELAY_DATE`, `PUSH_TEAMS_SLOT_DELAY_FROM`, `PUSH_TEAMS_SLOT_DELAY_MINUTES`, legacy golden-hour/catch-up rules, and daily Sport quotas are ignored by the claim path. An initial claim additionally requires at least 30 seconds of delivery budget before the five-minute window expires. If fewer than 30 seconds remain, the API returns HTTP 200 with `{"ready":false,"reason":"slot_closed"}` even though the Recurrence trigger is still inside its listed minute.

The operational endpoints are deliberately excluded from the public OpenAPI document:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/teams-readiness` | Internal pre-cutover proof for transport, score source, authoritative history, and the fixed 12-slot plan |
| `POST` | `/api/v1/power-automate/teams/claim` | Claim the one recommendation for the current slot |
| `POST` | `/api/v1/power-automate/teams/receipt` | Finalize a claimed slot as `sent`, `failed`, or `delivery_uncertain` |

Every request must send:

```http
Content-Type: application/json
X-Power-Automate-Key: <protected secret reference>
```

Claim and receipt require the dedicated header above. Do not use `CONSUMER_API_KEY`, `X-Consumer-Key`, a signed webhook URL, or a Teams connection token for them. `/api/teams-readiness` is not a public health route; access it only from the approved VPN/CIDR network.

## Prerequisites and secrets

- The deployment has `PUSH_TEAMS_ALERTS_ENABLED=true`.
- The deployment has a strong random `POWER_AUTOMATE_API_KEY` in its secret store. It is an opaque shared secret, not a derived key or KDF output.
- The flow can use the HTTP connector and the Microsoft Teams action **Post message in a chat or channel**.
- The flow has a tenant-approved protected secret/configuration for the same API key. A solution secret backed by Azure Key Vault is preferred.
- Only the minimum necessary owners can edit the flow or its connections.

To generate one new key on macOS and copy it directly to the clipboard without printing it, run:

```bash
python3 -c 'import secrets; print(secrets.token_urlsafe(48), end="")' | pbcopy
```

`render.yaml` declares `POWER_AUTOMATE_API_KEY` with `sync: false`; create or update it manually in the Render dashboard. Paste the same value into that deployment secret and the protected Power Automate secret/configuration. Then clear the clipboard:

```bash
pbcopy </dev/null
```

Never commit the value, paste it into a URL or Teams message, or place it in a normal Compose action, plain-text variable, flow name, screenshot, or support log. Secure Inputs/Outputs hide run-history values; they do not replace least-privilege flow ownership or an approved secret store.

## 1. Configure the Recurrence trigger

Create a **Scheduled cloud flow** and configure its **Recurrence** trigger:

| Setting | Value |
|---|---|
| Frequency | `Minute` |
| Interval | `1` |
| Time zone | `(UTC+01:00) Amsterdam, Berlin, Bern, Rome, Stockholm, Vienna` / `W. Europe Standard Time` |
| Concurrency control | `On` |
| Degree of parallelism | `1` |

Under **Settings → Trigger conditions**, add this single condition:

```text
@contains(if(or(equals(dayOfWeek(convertTimeZone(utcNow(),'UTC','W. Europe Standard Time')),0),equals(dayOfWeek(convertTimeZone(utcNow(),'UTC','W. Europe Standard Time')),6)),'|08:00|08:01|08:02|08:03|08:04|08:36|08:37|08:38|08:39|08:40|09:12|09:13|09:14|09:15|09:16|09:47|09:48|09:49|09:50|09:51|10:23|10:24|10:25|10:26|10:27|10:59|11:00|11:01|11:02|11:03|12:30|12:31|12:32|12:33|12:34|17:30|17:31|17:32|17:33|17:34|18:49|18:50|18:51|18:52|18:53|20:08|20:09|20:10|20:11|20:12|21:26|21:27|21:28|21:29|21:30|22:45|22:46|22:47|22:48|22:49|','|06:00|06:01|06:02|06:03|06:04|06:36|06:37|06:38|06:39|06:40|07:12|07:13|07:14|07:15|07:16|07:47|07:48|07:49|07:50|07:51|08:23|08:24|08:25|08:26|08:27|08:59|09:00|09:01|09:02|09:03|12:30|12:31|12:32|12:33|12:34|17:30|17:31|17:32|17:33|17:34|18:49|18:50|18:51|18:52|18:53|20:08|20:09|20:10|20:11|20:12|21:26|21:27|21:28|21:29|21:30|22:45|22:46|22:47|22:48|22:49|'),concat('|',formatDateTime(convertTimeZone(utcNow(),'UTC','W. Europe Standard Time'),'HH:mm'),'|'))
```

The delimiters make the string lookup exact. Do not replace the explicit time-zone conversion with the flow owner's local time zone.

## 2. Claim the recommendation

Add an **HTTP** action and rename it `Claim_recommendation`:

| Setting | Value |
|---|---|
| Method | `POST` |
| URI | `https://push-balancer.onrender.com/api/v1/power-automate/teams/claim` |
| `Content-Type` header | `application/json` |
| `X-Power-Automate-Key` header | Select the protected secret/configuration; never type a committed value |
| Body | See below |

```json
{
  "requestId": "@{workflow()?['run']?['name']}"
}
```

The Power Automate run name is a non-personal, per-run idempotency value. Do not use an editor name, email address, Teams user ID, or message recipient as `requestId`.

Enable **Secure Inputs** and **Secure Outputs** in this HTTP action's settings. The claim response has `Cache-Control: no-store` and one of two normal shapes.

For the scheduled path, `top` always means the candidate with the absolute highest fresh, technically valid `internal_score_api` score after exact live-push and Teams-article duplicate removal. Sport quota, Sport corridor, Breaking, section mix, OR, quality models, pacing, and legacy daily targets cannot promote a lower API score; secondary signals may only break an exact API-score tie. The separate `alternative` field never changes `top`.

Ready response (synthetic example):

```json
{
  "ready": true,
  "slotId": "teams-recommendation-1785753000",
  "scheduledAt": "2026-08-03T12:30:00+02:00",
  "scheduledAtUtc": "2026-08-03T10:30:00Z",
  "expiresAt": "2026-08-03T12:35:00+02:00",
  "top": {
    "title": "Synthetische Top-Meldung",
    "url": "https://example.invalid/news/top",
    "category": "news",
    "pushScore": 91.4,
    "isSport": false
  },
  "alternative": {
    "title": "Synthetische Sport-Alternative",
    "url": "https://example.invalid/sport/alternative",
    "category": "sport",
    "pushScore": 88.2,
    "isSport": true
  },
  "messageHtml": "<h2>🔵 JETZT MÜSSEN (!) WIR PUSHEN</h2><p>Das sind meine 5 Empfehlungen.</p>"
}
```

`alternative` is the highest valid Sport candidate when `top.isSport=false`, or the highest valid non-Sport candidate when `top.isSport=true`. It is `null` when no safe opposite alternative is available. It is a display alternative only, does not implement a Sport quota, and does not claim to be the overall second-ranked article.

Expected no-op response:

```json
{
  "ready": false,
  "reason": "slot_already_claimed"
}
```

`outside_window`, `slot_closed`, `no_candidate`, `candidate_not_approved`, `already_live_pushed`, `slot_already_claimed`, `article_already_claimed`, and `article_claim_unavailable` are operational reasons, not Teams content. `slot_closed` also covers an initial request with less than the required 30-second delivery budget. Expected no-send outcomes use HTTP 200 with `ready=false`. Branch only on `ready`; do not build flow behavior around a particular reason string.

Each claim attempt loads authoritative live-push history once, then reuses that same immutable decision snapshot for selection and its final exact-duplicate comparison. It deliberately performs no second network refresh during the claim. Direct live data is authoritative; relayed data is authoritative only when the complete snapshot parses and persists, its transmitted `source` is `live` or `relay`, and its age is below 300 seconds using the original `snapshotTs`. Render receipt time must never reset that age.

A retry of the claim action within the same Power Automate run uses the same `requestId` and can replay the original response. A different minute run cannot take over a live or completed claim and normally receives `ready=false`. Non-2xx responses are fail-closed: `401` means missing/wrong auth, `422` means an invalid request, and `503` means the integration or authoritative input is unavailable. None of those responses may post to Teams.

## 3. Branch on `ready`

Add a **Condition** named `Recommendation_ready` with this expression:

```text
@equals(body('Claim_recommendation')?['ready'], true)
```

- **If no:** end successfully without a Teams action. A Terminate action with status `Succeeded` is optional.
- **If yes:** continue with the Teams action below.

Enable Secure Inputs/Outputs on the condition and on downstream actions that expose claim fields in their inputs or outputs.

## 4. Post to Teams exactly once

In the **If yes** branch, add **Microsoft Teams → Post message in a chat or channel**, rename it `Post_to_Teams`, and select the approved bot identity and target chat/channel. Use only this value for the message body:

```text
@{body('Claim_recommendation')?['messageHtml']}
```

In the Teams action settings:

- set **Retry policy** to `None`;
- enable **Secure Inputs** and **Secure Outputs**;
- do not add a second Teams action, connector fallback, or webhook retry;
- do not append `requestId`, secrets, raw candidate data, or run diagnostics to the message.

A failed, timed-out, or skipped Teams action is acknowledgement-ambiguous: Microsoft may have accepted the message even if the flow did not receive confirmation. Do not retry the Teams action. The error branch must record `delivery_uncertain`, making the slot terminal without claiming that delivery definitely failed.

## 5. Record the delivery receipt

Add two parallel HTTP actions after `Post_to_Teams`. Both use the same protected `X-Power-Automate-Key`, `Content-Type: application/json`, Secure Inputs/Outputs, the original claim `requestId`, and this URI:

```text
https://push-balancer.onrender.com/api/v1/power-automate/teams/receipt
```

### Successful receipt

Name the first action `Receipt_sent`, configure **Run after** only for `Post_to_Teams` **is successful**, and use:

```json
{
  "slotId": "@{body('Claim_recommendation')?['slotId']}",
  "requestId": "@{workflow()?['run']?['name']}",
  "status": "sent"
}
```

### Uncertain-delivery receipt

Name the second action `Receipt_delivery_uncertain`, configure **Run after** for `Post_to_Teams` **has failed**, **has timed out**, and **is skipped**, and use:

```json
{
  "slotId": "@{body('Claim_recommendation')?['slotId']}",
  "requestId": "@{workflow()?['run']?['name']}",
  "status": "delivery_uncertain"
}
```

All three non-success states are terminal because the connector outcome cannot prove that Teams did not accept the message. Every receipt is bound to the run that acquired the claim: its `requestId` must exactly match the claim request or the API returns HTTP 409 without changing delivery state. The API also supports `status=failed` for a separately verified, definite pre-delivery failure, but the standard Teams action error branch must not infer that state.

A successful receipt returns a minimized acknowledgement:

```json
{
  "slotId": "teams-recommendation-1785753000",
  "status": "sent",
  "recordedAt": "2026-08-03T12:30:30+02:00"
}
```

Repeated identical receipts are safe. A success can never be downgraded to failure.

## Secure configuration checklist

Before saving or enabling the flow, verify all of the following:

- The API key comes from the approved protected secret/configuration and is sent only in `X-Power-Automate-Key`.
- Secure Inputs and Secure Outputs are enabled on claim, condition, Teams, and both receipt actions.
- The Teams retry policy is `None`.
- Recurrence concurrency is `1`.
- An initial claim is accepted only with at least 30 seconds remaining in its slot window.
- The false branch contains no Teams action.
- `Receipt_sent` and `Receipt_delivery_uncertain` both carry the original `requestId`.
- `Receipt_delivery_uncertain` runs after failed, timed-out, and skipped Teams outcomes.
- The flow sends `messageHtml` directly and does not reconstruct a message from the full response.
- The target chat/channel and bot identity are the approved production connection.
- Run-history retention follows the approved tenant policy and is no longer than operationally necessary.
- Test data, screenshots, and support cases use synthetic articles and `example.invalid` URLs.

## Cutover

Never activate the scheduled flow while the legacy background sender is active for the same slot.

1. Deploy the claim/receipt endpoints with `POWER_AUTOMATE_API_KEY` configured, while the new flow remains off.
2. Keep `PUSH_TEAMS_ALERTS_ENABLED=true`; this is the overall recommendation and claim-API gate.
3. Update and restart the approved Mac/Next relay before enabling the flow. The outgoing relay body must send `source=live` or `source=relay` plus the original `snapshotTs`; receipt time must never renew stale data. For the existing macOS LaunchAgents:

   ```bash
   launchctl kickstart -k "gui/$(id -u)/com.bild.push-balancer"
   launchctl kickstart -k "gui/$(id -u)/com.bild.push-sync"
   ```

4. Verify locally, without printing messages or secrets, that the refresh is authoritative and the age derived from the original snapshot is below 300 seconds. A relay-backed response is reported as `source=cache->db`; it can be authoritative only after the receiver validated the transmitted `live`/`relay` lineage, original timestamp, complete parse, and persistence:

   ```bash
   python3 - <<'PY'
   import json, urllib.request
   request = urllib.request.Request(
       "http://127.0.0.1:8050/api/pushes/refresh", method="POST"
   )
   data = json.load(urllib.request.urlopen(request, timeout=60))
   print({
       "source": data.get("source"),
       "ageSeconds": data.get("snapshot_age_seconds"),
       "authoritative": data.get("history_authoritative"),
   })
   PY
   ```

5. Choose a cutover before the next scheduled window and set `PUSH_TEAMS_BACKGROUND_SENDER_ENABLED=false` in the deployment. From the approved VPN/CIDR network, inspect only the minimized readiness fields:

   ```bash
   curl -fsS "https://push-balancer.onrender.com/api/teams-readiness" | \
     python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps({"ready":d.get("ready"),"teamsAlertsEnabled":d.get("teamsAlertsEnabled"),"transportMode":d.get("transportMode"),"backgroundSenderEnabled":d.get("backgroundSenderEnabled"),"powerAutomateConfigured":d.get("powerAutomateConfigured"),"scoreApiOk":(d.get("scoreApi") or {}).get("ok"),"historyAuthoritative":(d.get("pushHistory") or {}).get("historyAuthoritative"),"slotsOk":(d.get("slots") or {}).get("ok"),"plannedToday":(d.get("slots") or {}).get("plannedToday"),"labels":(d.get("slots") or {}).get("labels")},indent=2))'
   ```

   Continue only when `ready=true`, `teamsAlertsEnabled=true`, `transportMode=power_automate_scheduled`, `backgroundSenderEnabled=false`, `powerAutomateConfigured=true`, `scoreApiOk=true`, `historyAuthoritative=true`, `slotsOk=true`, `plannedToday=12`, and `labels` exactly matches the 12 slots in this document. A partial green check is not sufficient.
6. Confirm the old incoming-webhook Power Automate flow is off, then turn on the new scheduled flow.
7. At the next slot, verify exactly one `ready=true` run, one successful Teams action, and one `Receipt_sent`. The other minute runs should be normal `ready=false` no-ops.
8. Observe at least the agreed validation period before retiring the protected `PUSH_TEAMS_WEBHOOK_URL` rollback secret. Rotate/remove it afterward according to the approved secret process.

Production state after cutover:

```env
PUSH_TEAMS_ALERTS_ENABLED=true
PUSH_TEAMS_BACKGROUND_SENDER_ENABLED=false
POWER_AUTOMATE_API_KEY=<deployment secret; never commit>
```

## Rollback

Rollback changes the transport owner; it must not bypass ranking, duplicate protection, or privacy controls.

1. Turn off the scheduled Power Automate flow.
2. Confirm no scheduled run or Teams action is still executing.
3. Restore the protected legacy `PUSH_TEAMS_WEBHOOK_URL` if it was removed.
4. Set `PUSH_TEAMS_BACKGROUND_SENDER_ENABLED=true` and deploy.
5. Confirm the service is healthy before the next slot and that only the legacy sender is active.
6. Record the rollback reason and inspect the failed flow using only secured/minimized run history.

Do not enable both senders as a temporary test. The durable slot claim reduces races, but a single explicit transport owner is the operational invariant.

## Privacy and retention

The integration's purpose is to deliver an editorial recommendation, not to send a push notification or monitor employees. Its Microsoft 365 payload is limited to:

- non-personal slot and delivery timestamps;
- public article title and URL;
- section, Sport/non-Sport marker, latest publication time, and advisory Push Score;
- rendered `messageHtml`;
- non-personal `slotId` and Power Automate `requestId` for idempotency.

It excludes raw push history, candidates outside the five displayed recommendations, audience or recipient data, employee identities or activity, model prompts, reviewer scorecards, connection tokens, and API secrets. Backend Teams recommendation and slot state is retained for 45 days. Microsoft 365 run-history and message retention remain subject to the approved tenant policy, controller/processor roles, transfer path, and deletion process documented in the project privacy record.
