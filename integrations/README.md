# Push Balancer Consumer API

Server-side apps can use the small ESM client in
`integrations/push-balancer-client.js`.

```js
import { PushBalancerClient } from "./push-balancer-client.js";

const pushBalancer = new PushBalancerClient({
  baseUrl: process.env.PUSH_BALANCER_URL,
  apiKey: process.env.PUSH_BALANCER_API_KEY,
});

const recommendations = await pushBalancer.recommendations({
  limit: 10,
  minScore: 70,
});

console.log(recommendations.articles);
console.log(recommendations.livePushes);
```

Every article response also contains the already-sent pushes from the last 24
hours in `livePushes`. These entries are separate from recommendations and are
marked with `isLivePush: true`, `alreadySent: true`, and
`flags.livePush: true` so consumers cannot mistake them for unsent candidates.

The API also works with plain `fetch`:

```js
const response = await fetch(`${process.env.PUSH_BALANCER_URL}/api/v1/recommendations?limit=10`, {
  headers: {
    Authorization: `Bearer ${process.env.PUSH_BALANCER_API_KEY}`,
  },
});

if (!response.ok) {
  throw new Error(`Push Balancer failed with HTTP ${response.status}`);
}

const data = await response.json();
```

Recommended environment variables for consuming apps:

```env
PUSH_BALANCER_URL=https://push-balancer.onrender.com
PUSH_BALANCER_API_KEY=...
```

Available methods:

| Method | Endpoint | Purpose |
|---|---|---|
| `status()` | `GET /api/v1/status` | Smoke test auth and API readiness |
| `recommendations()` | `GET /api/v1/recommendations` | Drop-in ranked article recommendations |
| `articles()` | `GET /api/v1/articles` | Full article candidate payload |
| `scores()` | `GET /api/v1/scores` | Compact score projection |

Keep the API key on the server side. Browser-side usage exposes the key to users.

## Scheduled Teams delivery

The production Microsoft Teams hand-off uses a scheduled Power Automate flow and a separate least-privilege API key. It does not use the consumer client above. See [`power-automate/README.md`](power-automate/README.md) for the exact Recurrence trigger, claim/receipt contract, secure action settings, cutover, and rollback.
