export class PushBalancerError extends Error {
  constructor(message, { status, payload } = {}) {
    super(message);
    this.name = "PushBalancerError";
    this.status = status || 0;
    this.payload = payload || null;
  }
}

export class PushBalancerClient {
  constructor({ baseUrl, apiKey, fetchImpl = globalThis.fetch } = {}) {
    if (!baseUrl) {
      throw new PushBalancerError("Push Balancer baseUrl is required.");
    }
    if (!apiKey) {
      throw new PushBalancerError("Push Balancer apiKey is required.");
    }
    if (typeof fetchImpl !== "function") {
      throw new PushBalancerError("A fetch implementation is required.");
    }
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.apiKey = apiKey;
    this.fetchImpl = fetchImpl;
  }

  async status() {
    return this.#get("/api/v1/status");
  }

  async recommendations(params = {}) {
    return this.#get("/api/v1/recommendations", {
      limit: params.limit ?? 20,
      offset: params.offset ?? 0,
      category: params.category,
      minScore: params.minScore ?? 60,
      includeExplanations: params.includeExplanations ?? false,
    });
  }

  async articles(params = {}) {
    return this.#get("/api/v1/articles", {
      limit: params.limit ?? 50,
      offset: params.offset ?? 0,
      category: params.category,
      minScore: params.minScore,
      includeExplanations: params.includeExplanations ?? true,
    });
  }

  async scores(params = {}) {
    return this.#get("/api/v1/scores", {
      limit: params.limit ?? 100,
      offset: params.offset ?? 0,
      category: params.category,
      minScore: params.minScore,
    });
  }

  async #get(path, params = {}) {
    const url = new URL(`${this.baseUrl}${path}`);
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== null && value !== "") {
        url.searchParams.set(key, String(value));
      }
    }

    const response = await this.fetchImpl(url, {
      method: "GET",
      headers: {
        Accept: "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
    });

    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json")
      ? await response.json()
      : await response.text();

    if (!response.ok) {
      const detail = typeof payload === "object" && payload !== null
        ? payload.detail || payload.title
        : payload;
      throw new PushBalancerError(
        detail || `Push Balancer request failed with HTTP ${response.status}.`,
        { status: response.status, payload },
      );
    }

    return payload;
  }
}
