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
```

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
