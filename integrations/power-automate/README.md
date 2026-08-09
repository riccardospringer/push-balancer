# Scheduled Power Automate Teams delivery

This runbook configures Power Automate as the only production scheduler and Microsoft Teams transport for Push Balancer recommendations. Push Balancer keeps ownership of the absolute internal-API Top-1, the exactly five displayed recommendations, exact duplicate protection, and the durable per-slot claim.

> **Approval record:** On 2026-08-09, the Product/System Owner, Privacy Manager, DPO, and Legal/Group Legal approval for this exactly-five contract and `POWER_AUTOMATE_REQUIRE_LIVE_PUSH_HISTORY=false` cloud-only mode was recorded in `PRIVACY.md`. The scoped backend rollout is approved. Keep the new flow off until the legacy transport is proven unable to race and production readiness is green.

## Production contract

The flow claims and posts each recommendation as soon as its fixed
Europe/Berlin slot opens. Monday through Friday:

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

All claim windows are half-open (`slot <= now < slot + 5 minutes`). The explicit Berlin conversion below handles both CET and CEST. The flow runs once per minute inside each window, not just at the first minute: this provides bounded recovery opportunities when a Microsoft 365 trigger is delayed, the claim API is temporarily unavailable, or fewer than five safe recommendations exist before a slot is reserved. The first successful exactly-five claim posts immediately; the backend slot claim makes repeated runs idempotent. Do not use a **Delay until** action.

The weekday/weekend labels above are the entire Power Automate schedule. `PUSH_TEAMS_SLOT_DELAY_DATE`, `PUSH_TEAMS_SLOT_DELAY_FROM`, `PUSH_TEAMS_SLOT_DELAY_MINUTES`, legacy golden-hour/catch-up rules, and daily Sport quotas are ignored by the claim path. An initial claim additionally requires at least 30 seconds of delivery budget before the five-minute window expires. If fewer than 30 seconds remain, the API returns HTTP 200 with `{"ready":false,"reason":"slot_closed"}` even though the Recurrence trigger is still inside its listed minute.

The operational endpoints are deliberately excluded from the public OpenAPI document:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/teams-readiness` | Internal pre-cutover proof for transport, score source, configured history mode, and the fixed 12-slot plan |
| `GET` | `/api/v1/power-automate/teams/readiness` | Authenticated, data-minimized proof with the same readiness calculation plus the latest due slot's delivery state |
| `POST` | `/api/v1/power-automate/teams/claim` | Claim one Teams message containing exactly five recommendations for the current slot |
| `POST` | `/api/v1/power-automate/teams/receipt` | Finalize a claimed slot as `sent`, `failed`, or `delivery_uncertain` |

Every authenticated request must send:

```http
X-Power-Automate-Key: <protected secret reference>
```

The two `POST` requests additionally send `Content-Type: application/json`.

The authenticated readiness, claim, and receipt endpoints require the dedicated header above. Do not use `CONSUMER_API_KEY`, `X-Consumer-Key`, a signed webhook URL, or a Teams connection token for them. `/api/teams-readiness` remains a full internal diagnostic and is available only from the approved VPN/CIDR network. The authenticated readiness endpoint exposes only a fixed allowlist of booleans, modes, the 12 public slot labels, configuration enums, and `latestSlot`; it never returns candidates, article metadata, request IDs, account data, or secrets.

## Prerequisites and secrets

- The deployment has `PUSH_TEAMS_ALERTS_ENABLED=true`.
- The deployment uses the mounted persistent disk with `DB_PATH=/data/.push_history.db` and `PUSH_DB_DURABILITY_REQUIRED=true`; an ephemeral `/tmp` database is forbidden for this flow.
- The deployment has a strong random `POWER_AUTOMATE_API_KEY` in its secret store. It is an opaque shared secret, not a derived key or KDF output.
- The flow can use the HTTP connector and the Microsoft Teams action **Post message in a chat or channel**.
- The flow has a tenant-approved protected secret/configuration for the same API key. A solution secret backed by Azure Key Vault is preferred.
- The dedicated key is authorized only for the canonical Exact-5 flow. Legacy,
  copied, and shared flows retain stale credentials after a transport-owner
  cutover and therefore cannot claim a slot even if someone re-enables them.
- Only the minimum necessary owners can edit the flow or its connections.

To generate one new key on macOS and copy it directly to the clipboard without printing it, run:

```bash
python3 -c 'import secrets; print(secrets.token_urlsafe(48), end="")' | pbcopy
```

`render.yaml` declares `POWER_AUTOMATE_API_KEY` with `sync: false`; create or update it manually in the Render dashboard. After every rotation, update the `X-Power-Automate-Key` header in both `Claim_recommendation` and `Receipt_delivery_result` with the **classic Power Automate designer**. The new designer can display the edited HTTP header and appear to save it but reload the value as empty. Save in the classic designer, fully reload/reopen the flow and both HTTP actions, and verify that both protected header values remain populated and match the deployment secret without displaying or logging the value. Only then clear the clipboard:

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
| Timeout | `PT45S` |
| Retry policy | `Fixed interval` |
| Retry count | `2` |
| Retry interval | `PT5S` |

```json
{
  "requestId": "@{workflow()?['run']?['name']}"
}
```

The Power Automate run name is a non-personal, per-run idempotency value. Do not use an editor name, email address, Teams user ID, or message recipient as `requestId`.

Enable **Secure Inputs** and **Secure Outputs** in this HTTP action's settings. The claim response has `Cache-Control: no-store` and one of two normal shapes.

For the scheduled path, `top` always means the candidate with the absolute highest fresh, technically valid `internal_score_api` score after Teams-article duplicate removal. Exact live-push duplicate removal is additionally mandatory only when `POWER_AUTOMATE_REQUIRE_LIVE_PUSH_HISTORY=true`; the approved cloud-only mode does not claim that comparison. Sport quota, Sport corridor, Breaking, section mix, OR, quality models, pacing, and legacy daily targets cannot promote a lower API score; secondary signals may only break an exact API-score tie. The separate `alternative` field never changes `top`.

Ready response (synthetic example):

```json
{
  "ready": "yes",
  "contractVersion": 2,
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
  "recommendationCount": 5,
  "messageHtml": "<h2>🔵 JETZT MÜSSEN (!) WIR PUSHEN</h2><p>Das sind meine 5 Empfehlungen.</p><p><strong>Top 1:</strong> Synthetisch</p><p><strong>Top 2:</strong> Synthetisch</p><p><strong>Top 3:</strong> Synthetisch</p><p><strong>Top 4:</strong> Synthetisch</p><p><strong>Top 5:</strong> Synthetisch</p>"
}
```

`alternative` is the highest valid Sport candidate when `top.isSport=false`, or the highest valid non-Sport candidate when `top.isSport=true`. It is `null` when no safe opposite alternative is available. It is a display alternative only, does not implement a Sport quota, and does not claim to be the overall second-ranked article.

The ready contract is all-or-nothing. Top 1 always has a fresh canonical `internal_score_api` score. Places 2–5 first use further technically valid canonical candidates. A place may be filled from the current, publication-age-weighted article field only when its sole technical blocker is the missing fresh canonical score; promotional/fiction, publication, URL, exact Teams-duplicate, and every other hard blocker still exclude it. URL and CMS identity are both deduplicated. The backend-only fallback value orders display places but is never returned or rendered as a numeric Push Score; the message instead says `Kanonischer Push Score steht noch aus.` If fewer than five safe unique articles exist, no slot/article-group claim is created and the next minute run may retry inside the same window. A ready response atomically reserves the slot and all five visible identities; one conflicting identity rolls back the complete group.

Expected no-op response:

```json
{
  "ready": false,
  "reason": "slot_already_claimed"
}
```

`outside_window`, `slot_closed`, `no_candidate`, `candidate_not_approved`, `already_live_pushed`, `live_history_unavailable`, `insufficient_recommendations`, `claim_contract_stale`, `slot_already_claimed`, `article_already_claimed`, and `article_claim_unavailable` are operational reasons, not Teams content. `slot_closed` also covers an initial request with less than the required 30-second delivery budget. Expected no-send outcomes use HTTP 200 with the JSON boolean `ready=false`. Do not build flow behavior around a particular reason string; verify the complete ready contract below.

In the configured cloud-only mode, the scheduled claim does not call the AS-network live-push feed. Durable exact Teams/article and per-slot claims remain available in that mode. When an approved fresh history relay is available elsewhere in the application it remains best-effort context, but it is not renewed or treated as an authoritative prerequisite by the cloud-only claim. With `POWER_AUTOMATE_REQUIRE_LIVE_PUSH_HISTORY=true`, the scheduled claim instead performs one approved refresh and fails closed unless the result is authoritative and fresh. Enable that mode only after its separate review; do not claim that it is active without runtime evidence.

A retry of the claim action within the same Power Automate run uses the same `requestId` and can replay the original response only when contract version, slot ID, recommendation count, five rendered Top blocks, five pseudonymous group rows, and all five owned article claims still match. The fixed two-retry policy above is therefore bounded and idempotent; do not increase it or use an unbounded/custom retry loop. A legacy, short, incomplete, or orphaned replay is released fail-closed and cannot produce a one-item message. A different minute run cannot take over a live or completed claim and normally receives `ready=false`. Non-2xx responses are fail-closed: `401` means missing/wrong auth, `422` means an invalid request, and `503` means the integration or required input is unavailable. None of those responses may post to Teams.

## 3. Branch on the complete ready contract

Add a **Condition** named `Exact_five_ready` with this complete expression:

```text
@and(equals(body('Claim_recommendation')?['ready'],'yes'),equals(body('Claim_recommendation')?['contractVersion'],2),equals(body('Claim_recommendation')?['recommendationCount'],5),not(empty(body('Claim_recommendation')?['slotId'])),not(empty(body('Claim_recommendation')?['messageHtml'])),contains(body('Claim_recommendation')?['messageHtml'],'<strong>Top 1:</strong>'),contains(body('Claim_recommendation')?['messageHtml'],'<strong>Top 2:</strong>'),contains(body('Claim_recommendation')?['messageHtml'],'<strong>Top 3:</strong>'),contains(body('Claim_recommendation')?['messageHtml'],'<strong>Top 4:</strong>'),contains(body('Claim_recommendation')?['messageHtml'],'<strong>Top 5:</strong>'))
```

- **If no:** leave the branch completely empty. It must contain no Teams action, receipt, loop, fallback, or Terminate action.
- **If yes:** continue with the Teams action below.

Keep Secure Outputs enabled on `Claim_recommendation`; do not copy its response into a Compose action, variable, log, or diagnostic branch. If the tenant UI offers Secure Inputs/Outputs for the condition, enable them there as well.

## 4. Post to Teams exactly once

In the **If yes** branch, add **Microsoft Teams → Post message in a chat or channel**, rename it `Post_to_Teams`, and select the approved bot identity and target chat/channel. Use only this value for the message body:

```text
@{body('Claim_recommendation')?['messageHtml']}
```

In the Teams action settings:

- set **Timeout** to `PT30S`;
- set **Retry policy** to `None`;
- enable **Secure Inputs** and **Secure Outputs**;
- do not add a second Teams action, connector fallback, or webhook retry;
- do not append `requestId`, secrets, raw candidate data, or run diagnostics to the message.

A failed, timed-out, or skipped Teams action is acknowledgement-ambiguous: Microsoft may have accepted the message even if the flow did not receive confirmation. Do not retry the Teams action. The error branch must record `delivery_uncertain`, making the slot terminal without claiming that delivery definitely failed.

## 5. Record the delivery receipt

Add exactly one HTTP action after `Post_to_Teams` and name it `Receipt_delivery_result`. It uses the same protected `X-Power-Automate-Key`, `Content-Type: application/json`, Secure Inputs/Outputs, and this URI:

```text
https://push-balancer.onrender.com/api/v1/power-automate/teams/receipt
```

Configure its settings exactly as follows:

| Setting | Value |
|---|---|
| Timeout | `PT30S` |
| Retry policy | `Fixed interval` |
| Retry count | `3` |
| Retry interval | `PT5S` |
| Run after `Post_to_Teams` | `is successful`, `has failed`, `has timed out`, and `is skipped` |

Use one dynamic status in the body so those four mutually exclusive outcomes cannot race two parallel receipt branches:

```json
{
  "slotId": "@{body('Claim_recommendation')?['slotId']}",
  "requestId": "@{workflow()?['run']?['name']}",
  "status": "@{if(equals(actions('Post_to_Teams').status,'Succeeded'),'sent','delivery_uncertain')}"
}
```

Only the literal Teams action status `Succeeded` maps to `sent`. Failed, timed-out, and skipped outcomes map to `delivery_uncertain` because the connector outcome cannot prove that Teams did not accept the message. Every receipt is bound to the run that acquired the claim: its `requestId` must exactly match the claim request or the API returns HTTP 409 without changing delivery state. One transaction finalizes the parent slot, all five pseudonymous group rows, and all five article claims; a partial finalization is rolled back, repeated receipts are idempotent, and the group counts as one Teams message. The fixed receipt retries are safe because the endpoint is idempotent. The API also supports `status=failed` for a separately verified, definite pre-delivery failure, but the standard Teams action must not infer that state.

A successful receipt returns a minimized acknowledgement:

```json
{
  "slotId": "teams-recommendation-1785753000",
  "status": "sent",
  "recordedAt": "2026-08-03T12:30:30+02:00"
}
```

Repeated identical receipts are safe. A success can never be downgraded to failure, and a terminal `delivery_uncertain` receipt must not be rewritten as `sent` later.

If an entire flow run terminates after the claim but before the receipt action can execute, the backend cannot know whether Teams accepted the post. The unresolved five-item group therefore remains fail-closed and none of its article identities may be recycled into a later slot merely because the five-minute lease elapsed. Reconcile that run in the protected Power Automate history and record `sent` only when `Post_to_Teams` is proven `Succeeded`; otherwise record `delivery_uncertain`, always with the original `requestId`. Never rerun the Teams action or release the group as definitely failed when delivery is ambiguous. This trades availability for duplicate prevention.

## Monitoring and reconciliation

Monitoring must be external to the delivery branch so an alerting failure cannot create another Teams post. Configure the Power Platform operations monitor to alert immediately when the flow is turned off or suspended, either connector loses authorization, a trigger/action is throttled, or any enabled run fails. After every five-minute slot window, reconcile the protected run history and minimized backend state against exactly one of these outcomes: one `sent` receipt for one five-item message, one terminal `delivery_uncertain` receipt, or no claim because every attempt returned `ready=false`.

Use this checklist without exposing the API key, message body, article data, connection token, or raw run inputs/outputs:

1. **Missing `sent` receipt:** if `Exact_five_ready` entered its true branch and `Post_to_Teams` is proven `Succeeded`, but `Receipt_delivery_result` has no successful acknowledgement by the end of the slot window, alert and replay only the receipt request with the original `slotId` and `requestId` obtained through the approved minimized reconciliation view. Never resubmit the Teams action, expose secured action payloads, or rerun the complete flow. If the Teams status is unavailable or ambiguous, record `delivery_uncertain` instead of `sent`.
2. **`delivery_uncertain`:** alert immediately and treat all five identities as terminally delivered for duplicate prevention. Do not repost, release, or upgrade the receipt. An authorized operator may verify whether the message is visible in the target channel for the incident record, but absence from the current view is not proof that Microsoft never accepted it.
3. **No exact five:** normal minute attempts may return `ready=false` and retry automatically until the window closes. If no run reaches the full Exact-5 condition during the whole window, record the missed slot and inspect only minimized readiness fields for candidate count, canonical Top-1, score API, durable storage, and configured history mode. Do not weaken the condition, fill fewer than five places, or manually create a message. Escalate repeated missed windows to the System Owner.
4. **Connector authorization or suspension:** keep both the scheduled flow and legacy sender from posting while the approved flow owner restores the least-privilege HTTP/Teams connections and clears the suspension. Re-run the flow checker, confirm all secure settings and readiness gates, and resume only for a future slot; do not replay a past scheduled run.
5. **Unexpected claim owner:** audit both **Meine Flows → Cloud-Flows** and
   **Für mich freigegeben** (tenant labels may appear in English). A shared
   scheduled flow can claim the slot before the visible owned flow and make the
   canonical run look like a harmless `ready=false` no-op. Keep the legacy
   instant-webhook flow and every shared or copied scheduled flow **Off**. If an
   unexpected flow has ever held the current key, rotate the key as described
   under Cutover; disabling that flow alone is not a durable authorization
   boundary.

A `401` from `Claim_recommendation` is a failed slot and must alert immediately.
Correct and persist both HTTP headers for a future slot; never replay or
manually resend the past slot.

A run that claimed five items but remains in backend `sending` without a terminal receipt after its Power Automate run has ended is an incident even if a later slot succeeds. Preserve only the non-personal `slotId`, `requestId`, action statuses, and timestamps needed for reconciliation, subject to the approved retention period.

## Secure configuration checklist

Before saving or enabling the flow, verify all of the following:

- The API key comes from the approved protected secret/configuration and is sent only in `X-Power-Automate-Key`.
- Secure Inputs and Secure Outputs are enabled on claim, Teams, and the single receipt action (and on the condition when the tenant UI offers those settings).
- Claim uses timeout `PT45S` and fixed retry count `2` at interval `PT5S`.
- Teams uses timeout `PT30S` and retry policy `None`.
- `Receipt_delivery_result` uses timeout `PT30S` and fixed retry count `3` at interval `PT5S`.
- Recurrence concurrency is `1`.
- An initial claim is accepted only with at least 30 seconds remaining in its slot window.
- `Exact_five_ready` checks `ready="yes"`, `contractVersion=2`, `recommendationCount=5`, nonempty `slotId` and `messageHtml`, and all five `Top 1` through `Top 5` markers together.
- The false branch is empty.
- `Receipt_delivery_result` carries the original `requestId` and runs after successful, failed, timed-out, and skipped Teams outcomes.
- The flow sends `messageHtml` directly and does not reconstruct a message from the full response.
- The target chat/channel and bot identity are the approved production connection.
- Both **Meine Flows → Cloud-Flows** and **Für mich freigegeben** have been
  audited: the legacy instant-webhook flow and every legacy/shared scheduled
  flow are **Off**, and the canonical Exact-5 flow is the only active scheduler.
- The current `POWER_AUTOMATE_API_KEY` is present only in the deployment secret
  store and the canonical Exact-5 flow; legacy/shared flows retain stale keys.
- After a key rotation, both HTTP headers were saved with the classic designer,
  then the flow and both actions were fully reloaded and reopened to prove the
  protected values remained populated and matched; the clipboard was cleared
  only after that proof.
- Run-history retention follows the approved tenant policy and is no longer than operationally necessary.
- Test data, screenshots, and support cases use synthetic articles and `example.invalid` URLs.

## Cutover

Never activate the scheduled flow while any other transport owner can claim or
post for the same slot. Power Automate's owned Cloud-Flows view is not a complete
inventory: shared scheduled flows appear under **Für mich freigegeben** and can
race the canonical flow.

1. Audit both **Meine Flows → Cloud-Flows** and **Für mich freigegeben**. Keep
   the canonical Exact-5 flow off during preparation; turn off the legacy
   instant-webhook flow and every other owned, copied, or shared scheduled flow,
   and confirm that none has a run or Teams action still executing. The required
   steady state is exactly one enabled scheduler after cutover.
2. Treat the transport-owner change as a credential rotation. Generate a new
   random `POWER_AUTOMATE_API_KEY`, place it in the deployment secret store and
   only the canonical Exact-5 flow's protected Claim and Receipt configuration,
   and leave every legacy/shared flow with its old stale value. Never distribute
   the rotated key to a fallback flow. Complete the update outside a scheduled
   window while the canonical flow remains off, clear the clipboard, and verify
   that the secret was not exposed in run history or screenshots.
3. Deploy the claim/receipt endpoints with the rotated key configured, while the canonical flow remains off.
4. Keep `PUSH_TEAMS_ALERTS_ENABLED=true`; this is the overall recommendation and claim-API gate.
5. Only when the separately reviewed `POWER_AUTOMATE_REQUIRE_LIVE_PUSH_HISTORY=true` mode is intentionally enabled, update and restart the approved Mac/Next relay before enabling the flow. The outgoing relay body must send `source=live` or `source=relay` plus the original `snapshotTs`; receipt time must never renew stale data. For the existing macOS LaunchAgents:

   ```bash
   launchctl kickstart -k "gui/$(id -u)/com.bild.push-balancer"
   launchctl kickstart -k "gui/$(id -u)/com.bild.push-sync"
   ```

6. In that fail-closed live-history mode only, verify locally, without printing messages or secrets, that the refresh is authoritative and the age derived from the original snapshot is below 300 seconds. A relay-backed response is reported as `source=cache->db`; it can be authoritative only after the receiver validated the transmitted `live`/`relay` lineage, original timestamp, complete parse, and persistence. Skip steps 5–6 when readiness deliberately reports `required=false` with the durable fallback mode:

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

7. Choose a cutover before the next scheduled window and set `PUSH_TEAMS_BACKGROUND_SENDER_ENABLED=false` in the deployment. Prefer the authenticated, already minimized readiness route. Inject the rotated key from the approved secret store; never type it into the command line, shell history, or URL:

   ```bash
   curl -fsS \
     -H "X-Power-Automate-Key: ${POWER_AUTOMATE_API_KEY:?load from approved secret store}" \
     "https://push-balancer.onrender.com/api/v1/power-automate/teams/readiness" | jq
   ```

   From an approved VPN/CIDR network, the complete internal diagnostic can still be reduced locally before display:

   ```bash
   curl -fsS "https://push-balancer.onrender.com/api/teams-readiness" | \
     python3 -c 'import json,sys; d=json.load(sys.stdin); h=d.get("pushHistory") or {}; x=d.get("exactFive") or {}; s=d.get("durableStorage") or {}; print(json.dumps({"ready":d.get("ready"),"teamsAlertsEnabled":d.get("teamsAlertsEnabled"),"transportMode":d.get("transportMode"),"backgroundSenderEnabled":d.get("backgroundSenderEnabled"),"powerAutomateConfigured":d.get("powerAutomateConfigured"),"durableStorageRequired":s.get("required"),"durableStorageOk":s.get("durable"),"durableStorageMode":s.get("mode"),"scoreApiOk":(d.get("scoreApi") or {}).get("ok"),"exactFiveContractOk":x.get("contractOk"),"exactFiveCount":x.get("recommendationCount"),"top1Canonical":x.get("top1Canonical"),"historyOk":h.get("ok"),"historyRequired":h.get("required"),"historyAuthoritative":h.get("historyAuthoritative"),"fallbackMode":h.get("fallbackMode"),"slotsOk":(d.get("slots") or {}).get("ok"),"plannedToday":(d.get("slots") or {}).get("plannedToday"),"labels":(d.get("slots") or {}).get("labels")},indent=2))'
   ```

   Continue only when `ready=true`, `teamsAlertsEnabled=true`, `transportMode=power_automate_scheduled`, `backgroundSenderEnabled=false`, `powerAutomateConfigured=true`, `durableStorage.required=true`, `durableStorage.durable=true`, `durableStorage.mode=persistent_disk`, `scoreApi.ok=true`, `exactFive.contractOk=true`, `exactFive.recommendationCount=5`, `exactFive.top1Canonical=true`, `slots.ok=true`, `slots.plannedToday=12`, and `slots.labels` exactly matches the 12 slots in this document. The Exact-5 readiness probe uses the same read-only candidate preparation as the claim and creates no slot or article claim. A missing/unwritable persistent disk must stop startup or make the claim return 503; never accept a `/tmp` fallback. `pushHistory.historyAuthoritative=true` is acceptable; when live history is deliberately not required, `historyAuthoritative=false` is acceptable only together with `pushHistory.ok=true`, `pushHistory.required=false`, and `fallbackMode=durable_slot_and_receipt_dedup`. A partial green check is not sufficient.

   `latestSlot.state=sent` with `receiptRecorded=true` proves the latest due slot reached a successful receipt. `delivery_uncertain` requires human reconciliation and must never be retried automatically through Teams. `sending` after the five-minute lease or `unclaimed` after the window indicates a missing receipt or no successful Exact-5 claim and must alert the operator.
8. Recheck both Power Automate inventories immediately before activation. The
   legacy instant-webhook flow and every shared/legacy scheduled flow must show
   **Off**. Turn on only the canonical Exact-5 flow and confirm that it is the
   sole active scheduled transport owner.
9. At the next slot, verify exactly one claim with `ready="yes"`, `contractVersion=2`, and `recommendationCount=5`; one Teams message containing Top 1 through Top 5; one successful Teams action; and one `Receipt_delivery_result` acknowledgement with `status=sent`. The other minute runs should be normal `ready=false` no-ops. Confirm that no disabled shared/legacy flow created a run in that window. A contract mismatch must take the empty no branch and send nothing.
10. Observe at least the agreed validation period before retiring the protected `PUSH_TEAMS_WEBHOOK_URL` rollback secret. Rotate/remove it afterward according to the approved secret process.

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
4. Remove `POWER_AUTOMATE_API_KEY` from the deployment secret store so no
   scheduled flow remains an authorized claim owner. Do not expose its former
   value while removing it.
5. Set `PUSH_TEAMS_BACKGROUND_SENDER_ENABLED=true` and deploy. Startup fails
   closed if the Power Automate key is still configured, and the claim endpoint
   independently returns HTTP 503 while the legacy sender is enabled.
6. Confirm the service is healthy before the next slot and that only the legacy sender is active.
7. Record the rollback reason and inspect the failed flow using only secured/minimized run history.

Do not enable both senders as a temporary test. The durable slot claim reduces races, but a single explicit transport owner is the operational invariant.

## Privacy and retention

The integration's purpose is to deliver an editorial recommendation, not to send a push notification or monitor employees. Its Microsoft 365 payload is limited to:

- non-personal slot and delivery timestamps;
- exactly five public article titles, URLs, and latest publication times;
- section and Sport/non-Sport marker, plus an advisory Push Score only when it is canonical and fresh;
- non-personal `contractVersion` and `recommendationCount` fields;
- rendered `messageHtml`;
- non-personal `slotId` and Power Automate `requestId` for idempotency.

For display-only fillers, the local pre-API ordering value remains in the backend and Teams receives only the pending-score text. The payload excludes raw push history, candidates outside the five displayed recommendations, audience or recipient data, employee identities or activity, model prompts, reviewer scorecards, connection tokens, and API secrets. Backend Teams recommendation and slot state is retained for 45 days. Microsoft 365 run-history and message retention remain subject to the approved tenant policy, controller/processor roles, transfer path, and deletion process documented in the project privacy record.
