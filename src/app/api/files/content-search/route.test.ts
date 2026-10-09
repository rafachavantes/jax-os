import { beforeEach, describe, expect, it, vi } from "vitest";

const settingsState = vi.hoisted(() => ({ vault: true, vaultPath: "/tmp/vault" as string | null }));
vi.mock("../../../../server/settings", () => ({
  readGeneralSettings: () => ({
    ok: true,
    data: { vaultPath: settingsState.vaultPath, integrations: { vault: settingsState.vault } },
  }),
}));

vi.mock("../../../../server/collectors/fileContentSearch", () => ({ searchContent: vi.fn() }));

import { searchContent } from "../../../../server/collectors/fileContentSearch";
import { GET } from "./route";

const hit = { root: "vault" as const, rel: "daily/2026-09-26.md", line: 3, snippet: "hit" };

describe("GET /api/files/content-search", () => {
  beforeEach(() => {
    vi.mocked(searchContent).mockReset();
    settingsState.vault = true;
    settingsState.vaultPath = "/tmp/vault";
  });

  it.each([
    { label: "scope=vault refuses when vault is off, never calling searchContent", vault: false, vaultPath: "/tmp/vault", scope: "vault", refused: true },
    { label: "scope=vault refuses when vault is enabled but unconfigured (null vaultPath)", vault: true, vaultPath: null, scope: "vault", refused: true },
    { label: "scope=all narrows to reposOnly when vault is off, searchContent gets the narrowed scope", vault: false, vaultPath: "/tmp/vault", scope: "all", refused: false },
  ])("$label (spec per-integration table, decision 16)", async ({ vault, vaultPath, scope, refused }) => {
    settingsState.vault = vault;
    settingsState.vaultPath = vaultPath;
    if (!refused) vi.mocked(searchContent).mockResolvedValueOnce({ hits: [hit], truncated: false });
    const req = new Request(`http://127.0.0.1/api/files/content-search?q=hit&scope=${scope}`);
    const res = await GET(req);
    expect(res.status).toBe(200);
    if (refused) {
      expect(await res.json()).toEqual({ ok: false, error: "disabled" });
      expect(vi.mocked(searchContent)).not.toHaveBeenCalled();
    } else {
      await res.json();
      expect(vi.mocked(searchContent)).toHaveBeenCalledWith({ kind: "reposOnly" }, "hit", req.signal);
    }
  });

  it("forwards q/scope and an incomplete envelope", async () => {
    vi.mocked(searchContent).mockResolvedValueOnce({ hits: [hit], truncated: true });
    const req = new Request("http://127.0.0.1/api/files/content-search?q=hit&scope=vault");
    expect(await (await GET(req)).json()).toEqual({ ok: true, data: [hit], truncated: true });
    expect(searchContent).toHaveBeenCalledWith({ kind: "vault" }, "hit", req.signal);
  });
  it("refuses a 257-byte query or an invalid scope without calling the collector", async () => {
    expect(await (await GET(new Request(`http://127.0.0.1/api/files/content-search?q=${"a".repeat(257)}`))).json())
      .toEqual({ ok: false, error: "invalid payload" });
    expect(await (await GET(new Request("http://127.0.0.1/api/files/content-search?q=hit&scope=repo:.."))).json())
      .toEqual({ ok: false, error: "invalid payload" });
    expect(searchContent).not.toHaveBeenCalled();
  });
  it("degrades to 'content search unavailable' on collector failure; a cancelled signal reports 'search cancelled'", async () => {
    vi.mocked(searchContent).mockRejectedValueOnce(new Error("boom"));
    const res = await GET(new Request("http://127.0.0.1/api/files/content-search?q=hit"));
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: false, error: "content search unavailable" });
    const controller = new AbortController();
    const req = new Request("http://127.0.0.1/api/files/content-search?q=hit", { signal: controller.signal });
    controller.abort(new Error("gone"));
    vi.mocked(searchContent).mockRejectedValueOnce(controller.signal.reason);
    expect(await (await GET(req)).json()).toEqual({ ok: false, error: "search cancelled" });
  });
});
