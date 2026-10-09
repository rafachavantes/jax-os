import { beforeEach, describe, expect, it, vi } from "vitest";

const settingsState = vi.hoisted(() => ({ vault: true, vaultPath: "/tmp/vault" as string | null }));
vi.mock("../../../../server/settings", () => ({
  readGeneralSettings: () => ({
    ok: true,
    data: { vaultPath: settingsState.vaultPath, integrations: { vault: settingsState.vault } },
  }),
}));

vi.mock("../../../../server/collectors/files", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../../../server/collectors/files")>()),
  buildIndex: vi.fn(),
}));

import { buildIndex } from "../../../../server/collectors/files";
import { GET } from "./route";

describe("GET /api/files/index", () => {
  beforeEach(() => {
    vi.mocked(buildIndex).mockReset();
    settingsState.vault = true;
    settingsState.vaultPath = "/tmp/vault";
  });

  it.each([
    { label: "scope=vault refuses when vault is off, never calling buildIndex", vault: false, vaultPath: "/tmp/vault", scope: "vault", refused: true },
    { label: "scope=vault refuses when vault is enabled but unconfigured (null vaultPath)", vault: true, vaultPath: null, scope: "vault", refused: true },
    { label: "scope=all narrows to reposOnly when vault is off, buildIndex gets the narrowed scope", vault: false, vaultPath: "/tmp/vault", scope: "all", refused: false },
  ])("$label (spec per-integration table, decision 16)", async ({ vault, vaultPath, scope, refused }) => {
    settingsState.vault = vault;
    settingsState.vaultPath = vaultPath;
    if (!refused) {
      vi.mocked(buildIndex).mockResolvedValueOnce({ data: [], truncated: false, truncatedRoots: [], builtAt: "x" });
    }
    const res = await GET(new Request(`http://127.0.0.1/api/files/index?scope=${scope}`));
    expect(res.status).toBe(200);
    if (refused) {
      expect(await res.json()).toEqual({ ok: false, error: "disabled" });
      expect(vi.mocked(buildIndex)).not.toHaveBeenCalled();
    } else {
      await res.json();
      expect(vi.mocked(buildIndex)).toHaveBeenCalledWith({ kind: "reposOnly" });
    }
  });

  it("returns the index envelope with truncated/truncatedRoots/builtAt passed through", async () => {
    vi.mocked(buildIndex).mockResolvedValueOnce({
      data: [{ root: "repos", rel: "a.ts", name: "a.ts" }],
      truncated: true,
      truncatedRoots: ["repos"],
      builtAt: "2026-09-18T18:30:00.000Z",
    });
    const res = await GET(new Request("http://127.0.0.1/api/files/index"));
    expect(await res.json()).toEqual({
      ok: true,
      data: [{ root: "repos", rel: "a.ts", name: "a.ts" }],
      truncated: true,
      truncatedRoots: ["repos"],
      builtAt: "2026-09-18T18:30:00.000Z",
    });
    expect(buildIndex).toHaveBeenCalledWith({ kind: "all" });
  });

  it("degrades to {ok:false} on an unexpected collector failure, never a 5xx", async () => {
    vi.mocked(buildIndex).mockRejectedValueOnce(new Error("boom"));
    const res = await GET(new Request("http://127.0.0.1/api/files/index"));
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: false, error: "boom" });
  });

  it("refuses an invalid scope without calling the collector", async () => {
    const req = new Request("http://127.0.0.1/api/files/index?scope=repo:..");
    expect(await (await GET(req)).json()).toEqual({ ok: false, error: "invalid payload" });
    expect(buildIndex).not.toHaveBeenCalled();
  });
});
