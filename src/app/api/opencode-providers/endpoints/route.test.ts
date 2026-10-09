import { describe, expect, it, vi } from "vitest";
import { GET, dynamic, type EndpointsRouteDeps } from "./route";

function handleGet(req: Request, deps: EndpointsRouteDeps) {
  Object.defineProperty(req, "jaxDeps", { value: deps });
  return GET(req);
}

const DATA = {
  connection: "openrouter",
  model: "deepseek/deepseek-v4-flash-0731",
  fetched_at: "2026-09-13T00:00:00.000Z",
  stale: false,
  error: null,
  endpoints: [],
};

function get(url: string) {
  return new Request(url);
}

function deps(over: Partial<EndpointsRouteDeps> = {}): EndpointsRouteDeps {
  return { compare: vi.fn(async () => DATA), ...over };
}

describe("opencode-providers/endpoints route", () => {
  it("is force-dynamic", () => {
    expect(dynamic).toBe("force-dynamic");
  });

  it("GET compares a selected connection/model and sets no-store", async () => {
    const d = deps();
    const res = await handleGet(get(
      "http://127.0.0.1/api/opencode-providers/endpoints?connection=openrouter&model=deepseek/deepseek-v4-flash-0731",
    ), d);
    expect(res.status).toBe(200);
    expect(res.headers.get("cache-control")).toBe("no-store");
    expect(await res.json()).toEqual({ ok: true, data: DATA });
    expect(d.compare).toHaveBeenCalledWith("openrouter", "deepseek/deepseek-v4-flash-0731", false);
  });

  it("passes refresh=1", async () => {
    const d = deps();
    await handleGet(get(
      "http://127.0.0.1/api/opencode-providers/endpoints?connection=openrouter&model=acme/model&refresh=1",
    ), d);
    expect(d.compare).toHaveBeenCalledWith("openrouter", "acme/model", true);
  });

  it("rejects missing/invalid connection or model without calling compare", async () => {
    const d = deps();
    const urls = [
      "http://127.0.0.1/api/opencode-providers/endpoints",
      "http://127.0.0.1/api/opencode-providers/endpoints?connection=openrouter",
      "http://127.0.0.1/api/opencode-providers/endpoints?connection=x%0ay&model=m",
      "http://127.0.0.1/api/opencode-providers/endpoints?connection=&model=m",
    ];
    for (const url of urls) {
      const res = await handleGet(get(url), d);
      expect(res.status).toBe(200);
      expect(await res.json()).toEqual({ ok: false, error: "invalid payload" });
    }
    expect(d.compare).not.toHaveBeenCalled();
  });

  it("collector failure is unavailable, never a 500", async () => {
    const d = deps({ compare: vi.fn(async () => { throw new Error("token leaked?"); }) });
    const res = await handleGet(get(
      "http://127.0.0.1/api/opencode-providers/endpoints?connection=openrouter&model=acme/model",
    ), d);
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: false, error: "unavailable" });
  });
});
