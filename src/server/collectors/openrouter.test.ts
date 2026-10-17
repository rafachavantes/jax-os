import { createHash } from "node:crypto";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { compactRouting, percentileMatches } from "../../lib/agent-settings";
import { evaluateEndpoint } from "../../lib/openrouter-comparison";
import {
  compareEndpoints,
  composeCompare,
  invalidateOpenRouterCache,
  productionCompareComposition,
  type CompareComposition,
  type CompareEndpointsInput,
  type CompareSnapshotConnection,
} from "./openrouter";

const TOKEN = "sk-or-v1-SECRET_TOKEN_SENTINEL";
const MODEL = "deepseek/deepseek-v4-flash-0731";
const URL = "https://openrouter.ai/api/v1/models/deepseek/deepseek-v4-flash-0731/endpoints";
const NOW = Date.parse("2026-09-13T12:00:00.000Z");
const CAP = 2 * 1024 * 1024;

function leak(value: unknown) {
  expect(JSON.stringify(value)).not.toContain(TOKEN);
}

function input(over: Partial<CompareEndpointsInput> = {}): CompareEndpointsInput {
  return {
    connection: "openrouter",
    model: MODEL,
    models: [MODEL],
    credential: { token: TOKEN, identity: "OPENROUTER_API_KEY", revision: "r1" },
    ...over,
  };
}

function endpoint(over: Record<string, unknown> = {}) {
  return {
    name: "Together: DeepSeek",
    tag: "together",
    provider_name: "Together",
    pricing: { prompt: "0.000001", completion: "0.000002" },
    latency_last_30m: { p50: 250, p75: 350, p90: 480, p99: 850 },
    throughput_last_30m: { p50: 45.2, p75: 38.5, p90: 28.3, p99: 15.1 },
    supported_parameters: ["temperature", "top_p"],
    ...over,
  };
}

function payload(endpoints: unknown) {
  return { data: { id: MODEL, name: "DeepSeek", created: 1, description: "x", architecture: {}, endpoints } };
}

function jsonRes(body: unknown, status = 200, headers?: HeadersInit) {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...headers } });
}

function fetchOf(res: Response | (() => Response | Promise<Response>)) {
  return vi.fn(async () => (typeof res === "function" ? res() : res));
}

async function compare(res: Response | (() => Response | Promise<Response>), over?: Partial<CompareEndpointsInput>, now = NOW) {
  return compareEndpoints(input(over), { fetch: fetchOf(res) as unknown as typeof fetch, now: () => now });
}

beforeEach(() => invalidateOpenRouterCache());
afterEach(() => vi.restoreAllMocks());

describe("compareEndpoints projection", () => {
  it("converts 0.000001 USD/token to 1 USD/million, latency ms to seconds, and labels the 30m window", async () => {
    const row = (await compare(jsonRes(payload([endpoint()])))).endpoints[0];
    expect(row).toMatchObject({
      tag: "together",
      provider: "Together",
      prices: { prompt: 1, completion: 2 },
      latency: { p50: 0.25, p75: 0.35, p90: 0.48, p99: 0.85, window: "30m" },
      throughput: { p50: 45.2, p75: 38.5, p90: 28.3, p99: 15.1, window: "30m" },
      supported_parameters: ["temperature", "top_p"],
      partial: false,
      eligible: true,
      quantization: null,
    });
  });

  it("trims a reported quantization and treats a blank or missing one as null", async () => {
    const withValue = (await compare(jsonRes(payload([endpoint({ quantization: " fp8 " })])))).endpoints[0];
    expect(withValue.quantization).toBe("fp8");
    invalidateOpenRouterCache();
    const blank = (await compare(jsonRes(payload([endpoint({ quantization: "  " })])))).endpoints[0];
    expect(blank.quantization).toBeNull();
    invalidateOpenRouterCache();
    const missing = (await compare(jsonRes(payload([endpoint()])))).endpoints[0];
    expect(missing.quantization).toBeNull();
  });

  it("keeps a real zero price as zero and leaves absent metrics null", async () => {
    const row = (await compare(jsonRes(payload([endpoint({
      pricing: { prompt: "0", completion: "0" },
      latency_last_30m: null,
      throughput_last_30m: undefined,
    })])))).endpoints[0];
    expect(row.prices).toEqual({ prompt: 0, completion: 0 });
    expect(row.latency).toBeNull();
    expect(row.throughput).toBeNull();
    expect(compactRouting({ sort: "price", allow_fallbacks: true, max_price: { prompt: 0 } })).toEqual({
      sort: "price",
      allow_fallbacks: true,
      max_price: { prompt: 0 },
    });
  });

  it("a p50-only observation cannot pass a p90 goal", async () => {
    const row = (await compare(jsonRes(payload([endpoint({
      latency_last_30m: { p50: 80 },
      throughput_last_30m: { p50: 10, p75: 9, p90: 8, p99: 7 },
    })])))).endpoints[0];
    expect(percentileMatches({ p90: 40 }, row.latency)).toBe(false);
    expect(percentileMatches({ p90: 8 }, row.throughput)).toBe(true);
  });

  it("compares a seconds latency goal against the millisecond feed", async () => {
    const row = (await compare(jsonRes(payload([endpoint({ latency_last_30m: { p90: 2731.8 } })])))).endpoints[0];
    const goal = (p90: number) => evaluateEndpoint(row, { sort: "price", allow_fallbacks: true, preferred_max_latency: { p90 } }).goals;
    expect(goal(3)).toBe("yes");
    expect(goal(2)).toBe("no");
  });

  it("marks invalid prices as named partial data", async () => {
    const row = (await compare(jsonRes(payload([endpoint({ pricing: { prompt: "nope", completion: "0.000001" } })])))).endpoints[0];
    expect(row.prices).toEqual({ prompt: null, completion: 1 });
    expect(row.partial).toBe(true);
  });

  it("never constructs a routing slug from a display label and treats missing tags as read-only", async () => {
    const row = (await compare(jsonRes(payload([endpoint({ tag: "", name: "OpenAI: GPT-4", provider_name: "OpenAI" })])))).endpoints[0];
    expect(row.tag).toBeNull();
    expect(row.provider).toBe("OpenAI");
    expect(row.eligible).toBe(false);
  });

  it("deduplicates by endpoint tag and keeps untagged rows", async () => {
    const endpoints = (await compare(jsonRes(payload([
      endpoint({ tag: "openai", provider_name: "OpenAI" }),
      endpoint({ tag: "openai", provider_name: "Duplicate" }),
      endpoint({ tag: "", provider_name: "Ghost" }),
      endpoint({ tag: "", provider_name: "Other" }),
    ])))).endpoints;
    expect(endpoints.map((e) => e.provider)).toEqual(["OpenAI", "Ghost", "Other"]);
  });

  it("empty endpoints is quiet empty, not unavailable", async () => {
    const r = await compare(jsonRes(payload([])));
    expect(r).toMatchObject({ endpoints: [], error: null, stale: false, fetched_at: "2026-09-13T12:00:00.000Z" });
  });
});

describe("compareEndpoints transport", () => {
  it("fetches only the fixed official URL with encoded segments, bearer credential, timeout and redirect rejection", async () => {
    const timeout = vi.spyOn(AbortSignal, "timeout");
    const fetchFn = fetchOf(jsonRes(payload([])));
    await compareEndpoints(input({ model: "acme/foo/bar", models: ["acme/foo/bar"] }), {
      fetch: fetchFn as unknown as typeof fetch,
      now: () => NOW,
    });
    const [url, init] = fetchFn.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("https://openrouter.ai/api/v1/models/acme/foo%2Fbar/endpoints");
    expect((init.headers as Record<string, string>).Authorization).toBe(`Bearer ${TOKEN}`);
    expect(init.redirect).toBe("error");
    expect(timeout).toHaveBeenCalledWith(10_000);
    expect(init.signal).toBeInstanceOf(AbortSignal);
  });

  it("does not fetch when the model is not configured on that connection", async () => {
    const fetchFn = fetchOf(jsonRes(payload([])));
    await expect(compareEndpoints(input({ models: ["other/model"] }), {
      fetch: fetchFn as unknown as typeof fetch,
      now: () => NOW,
    })).rejects.toThrow("unavailable");
    expect(fetchFn).not.toHaveBeenCalled();
  });

  it("forbidden/error/malformed responses are unavailable, not empty", async () => {
    await expect(compare(jsonRes({ error: { code: 403, message: "Only management keys can perform this operation" } }, 403)))
      .rejects.toThrow("unavailable");
    await expect(compare(jsonRes({ data: { endpoints: "nope" } }))).rejects.toThrow("unavailable");
    await expect(compare(new Response("not-json", { status: 200 }))).rejects.toThrow("unavailable");
    await expect(compare(jsonRes({ data: {} }))).rejects.toThrow("unavailable");
  });

  it("rejects overflow, timeout and redirect without leaking the credential", async () => {
    const oversize = new Response("{}", { status: 200, headers: { "content-length": String(CAP + 1) } });
    await expect(compare(oversize)).rejects.toThrow("unavailable");

    let remain = CAP + 1;
    const stream = new ReadableStream<Uint8Array>({
      pull(controller) {
        const n = Math.min(65_536, remain);
        remain -= n;
        controller.enqueue(new Uint8Array(n).fill(0x78));
        if (remain <= 0) controller.close();
      },
    });
    await expect(compare(new Response(stream, { status: 200 }))).rejects.toThrow("unavailable");

    const timeoutFetch = vi.fn(async () => {
      throw Object.assign(new Error("aborted"), { name: "AbortError" });
    });
    await expect(compareEndpoints(input(), { fetch: timeoutFetch as unknown as typeof fetch, now: () => NOW }))
      .rejects.toThrow("unavailable");

    const redirectFetch = vi.fn(async () => {
      throw Object.assign(new TypeError("redirect"), { name: "TypeError" });
    });
    await expect(compareEndpoints(input(), { fetch: redirectFetch as unknown as typeof fetch, now: () => NOW }))
      .rejects.toThrow("unavailable");

    for (const err of [oversize, "stream", timeoutFetch.mock.results, redirectFetch.mock.results]) leak(err);
  });

  it("never puts Authorization into the DTO or thrown errors", async () => {
    leak(await compare(jsonRes(payload([endpoint()]))));
    invalidateOpenRouterCache();
    await expect(compare(jsonRes({ error: { message: "nope" } }, 500))).rejects.toThrow("unavailable");
    try {
      await compare(jsonRes({ error: { message: "nope" } }, 500));
    } catch (e) {
      leak(e instanceof Error ? e.message : e);
    }
  });
});

describe("compareEndpoints cache", () => {
  it("reuses a 60s cache keyed by connection/model and non-secret identity, not the token", async () => {
    const fetchFn = fetchOf(() => jsonRes(payload([endpoint()])));
    const deps = { fetch: fetchFn as unknown as typeof fetch, now: () => NOW };
    await compareEndpoints(input(), deps);
    await compareEndpoints(input({ credential: { token: "other-secret", identity: "OPENROUTER_API_KEY", revision: "r1" } }), deps);
    expect(fetchFn).toHaveBeenCalledTimes(1);
    await compareEndpoints(input({ credential: { token: TOKEN, identity: "OPENROUTER_API_KEY", revision: "r2" } }), deps);
    expect(fetchFn).toHaveBeenCalledTimes(2);
    await compareEndpoints(input(), { fetch: fetchFn as unknown as typeof fetch, now: () => NOW + 60_000 });
    expect(fetchFn).toHaveBeenCalledTimes(3);
  });

  it("deduplicates in-flight fetches", async () => {
    let release!: (value: Response) => void;
    const gate = new Promise<Response>((resolve) => { release = resolve; });
    const fetchFn = vi.fn(() => gate);
    const deps = { fetch: fetchFn as unknown as typeof fetch, now: () => NOW };
    const a = compareEndpoints(input(), deps);
    const b = compareEndpoints(input(), deps);
    expect(fetchFn).toHaveBeenCalledTimes(1);
    release(jsonRes(payload([endpoint()])));
    expect((await a).endpoints).toHaveLength(1);
    expect((await b).endpoints).toHaveLength(1);
  });

  it("returns last-good as stale on failure and invalidates per connection", async () => {
    const fetchFn = vi.fn()
      .mockResolvedValueOnce(jsonRes(payload([endpoint()])))
      .mockResolvedValueOnce(jsonRes({ error: { code: 403 } }, 403))
      .mockResolvedValueOnce(jsonRes(payload([endpoint({ tag: "fresh" })])));
    const deps = { fetch: fetchFn as unknown as typeof fetch };
    const first = await compareEndpoints(input(), { ...deps, now: () => NOW });
    const stale = await compareEndpoints(input(), { ...deps, now: () => NOW + 60_000 });
    expect(stale).toMatchObject({
      stale: true,
      error: "unavailable",
      fetched_at: first.fetched_at,
      endpoints: [{ tag: "together" }],
    });
    leak(stale);
    invalidateOpenRouterCache("openrouter");
    const next = await compareEndpoints(input(), { ...deps, now: () => NOW + 120_000 });
    expect(next).toMatchObject({ stale: false, error: null, endpoints: [{ tag: "fresh" }] });
  });

  it("manual refresh uses the same bounded fetch and bypasses freshness", async () => {
    const fetchFn = fetchOf(() => jsonRes(payload([endpoint()])));
    const deps = { fetch: fetchFn as unknown as typeof fetch, now: () => NOW };
    await compareEndpoints(input(), deps);
    await compareEndpoints(input({ refresh: true }), deps);
    expect(fetchFn).toHaveBeenCalledTimes(2);
    const [, init] = fetchFn.mock.calls[1] as unknown as [string, RequestInit];
    expect(init.redirect).toBe("error");
    expect((init.headers as Record<string, string>).Authorization).toBe(`Bearer ${TOKEN}`);
  });
});

describe("composeCompare production composition", () => {
  const models = [{ id: "m/1" }, { id: "m/2" }];
  const REV = {
    settings: "absent",
    sources: { "opencode.json": "ab".repeat(32).slice(0, 64), "opencode.jsonc": "absent" },
  };

  function conn(over: Partial<CompareSnapshotConnection> = {}): CompareSnapshotConnection {
    return {
      id: "conn-a",
      adapter: "openrouter",
      auth: "api-key",
      credential: { kind: "native" },
      models,
      ...over,
    };
  }

  function compOf(rows: Array<ReturnType<typeof conn>>, editor_revision: unknown = REV) {
    const read = vi.fn(async (b: { kind: string; connection?: string }): Promise<string> =>
      b.kind === "native" ? "sk-native" : "sk-bws");
    const compare = vi.fn(async (i: unknown) => ({ credential: (i as { credential: unknown }).credential }));
    const composition = {
      snapshot: async () => ({ connections: rows, editor_revision }),
      read,
      compare,
    } as CompareComposition;
    return { composition, read, compare };
  }

  it("compares only an openrouter connection by its own native credential with a real source revision", async () => {
    const { composition, compare } = compOf([conn()]);
    const out = await composeCompare(composition, "conn-a", "m/1", false);
    expect(out).toEqual({
      credential: { token: "sk-native", identity: "native", revision: REV.sources["opencode.json"] },
    });
    expect(compare).toHaveBeenCalledWith(expect.objectContaining({ connection: "conn-a", model: "m/1", models: ["m/1", "m/2"] }));
    const next: CompareComposition = {
      snapshot: async () => ({ connections: [conn()] }),
      read: vi.fn(async () => "sk-native"),
      compare: vi.fn(async (i) => ({ credential: (i as { credential: unknown }).credential })),
    };
    const fallback = await composeCompare(next, "conn-a", "m/1", false);
    expect(fallback).toEqual({ credential: { token: "sk-native", identity: "native", revision: "absent" } });
  });

  it("keys an env-kind credential by its env name and a SHA-256 of the read value (MOA-498 D4)", async () => {
    const { composition, read, compare } = compOf([conn({ credential: { kind: "env", env: "JAX_PROVIDER_CONN_A_API_KEY" } })]);
    read.mockResolvedValue("secret-value-1");
    await composeCompare(composition, "conn-a", "m/1", false);
    expect(read).toHaveBeenCalledWith({ kind: "env", env: "JAX_PROVIDER_CONN_A_API_KEY" });
    const expectedRevision = createHash("sha256").update("secret-value-1").digest("hex");
    expect(compare).toHaveBeenCalledWith(expect.objectContaining({
      credential: expect.objectContaining({ identity: "JAX_PROVIDER_CONN_A_API_KEY", revision: expectedRevision }),
    }));
  });

  it("rejects oauth-only, non-openrouter, missing credentials, and unknown models without reading or fetching", async () => {
    const rejected: Array<ReturnType<typeof conn>> = [
      conn({ auth: "oauth" }),
      conn({ adapter: "xai" }),
      conn({ credential: null }),
    ];
    for (const row of rejected) {
      const { composition, read, compare } = compOf([row]);
      await expect(composeCompare(composition, row.id, "m/1", false)).rejects.toThrow("unavailable");
      expect(read).not.toHaveBeenCalled();
      expect(compare).not.toHaveBeenCalled();
    }
    const { composition, read, compare } = compOf([conn()]);
    await expect(composeCompare(composition, "conn-a", "m/unknown", false)).rejects.toThrow("unavailable");
    expect(read).not.toHaveBeenCalled();
    expect(compare).not.toHaveBeenCalled();
  });

  it("production composition wires snapshot, private read and the bounded collector", async () => {
    const composition = productionCompareComposition();
    expect(typeof composition.snapshot).toBe("function");
    expect(typeof composition.read).toBe("function");
    expect(typeof composition.compare).toBe("function");
  });
});
