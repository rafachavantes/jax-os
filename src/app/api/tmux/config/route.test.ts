import { afterEach, describe, expect, it, vi } from "vitest";

const settingsState = vi.hoisted(() => ({ ttyd: false }));
vi.mock("../../../../server/settings", () => ({
  readGeneralSettings: () => ({ ok: true, data: { integrations: { ttyd: settingsState.ttyd } } }),
}));

import { GET } from "./route";

describe("GET /api/tmux/config", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    delete process.env.TTYD_RO_URL;
    delete process.env.TTYD_RW_URL;
    settingsState.ttyd = false;
  });

  it("returns configured:false immediately when ttyd is disabled, never reading TTYD_RO_URL/reaching the network", async () => {
    process.env.TTYD_RO_URL = "http://127.0.0.1:9";
    process.env.TTYD_RW_URL = "http://127.0.0.1:9";
    const fetchSpy = vi.spyOn(global, "fetch");
    const json = await (await GET()).json();
    // `reachable:false` stays alongside `configured:false`: TtydConfig's existing shape requires
    // both (src/lib/api.ts, outside this plan's whitelist) — the panel's first branch keys off
    // `!configured` and never reaches `reachable` for this response.
    expect(json).toEqual({ ok: true, data: { configured: false, reachable: false } });
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("reachable is false on an HTTP error response, not just a network error (decision 10)", async () => {
    settingsState.ttyd = true;
    process.env.TTYD_RO_URL = "http://127.0.0.1:9999";
    process.env.TTYD_RW_URL = "http://127.0.0.1:9999";
    vi.spyOn(global, "fetch").mockResolvedValue(new Response("", { status: 500 }));
    const json = await (await GET()).json();
    expect(json.data.reachable).toBe(false);
  });

  it("reachable is true on a 2xx response (regression)", async () => {
    settingsState.ttyd = true;
    process.env.TTYD_RO_URL = "http://127.0.0.1:9999";
    process.env.TTYD_RW_URL = "http://127.0.0.1:9999";
    vi.spyOn(global, "fetch").mockResolvedValue(new Response("", { status: 200 }));
    const json = await (await GET()).json();
    expect(json.data.reachable).toBe(true);
  });
});
