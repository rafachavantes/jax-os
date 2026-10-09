import { beforeEach, describe, expect, it, vi } from "vitest";

// A real temp repos root with a "jax-os" dir: parseScope("repo:jax-os") runs for real in this
// suite (importOriginal below), and it checks listRepoNames() on disk — never Rafa's machine.
const settingsState = vi.hoisted(() => {
  const { mkdirSync, mkdtempSync } = require("node:fs");
  const { tmpdir } = require("node:os");
  const { join } = require("node:path");
  const reposRoot = mkdtempSync(join(tmpdir(), "jaxos-search-repos-"));
  mkdirSync(join(reposRoot, "jax-os"));
  return { vault: true, vaultPath: "/tmp/vault" as string | null, reposRoot };
});
vi.mock("../../../../server/settings", () => ({
  readGeneralSettings: () => ({
    ok: true,
    data: { reposRoot: settingsState.reposRoot, vaultPath: settingsState.vaultPath, integrations: { vault: settingsState.vault } },
  }),
}));

vi.mock("../../../../server/collectors/files", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../../../server/collectors/files")>()),
  searchByName: vi.fn(),
}));

import { searchByName } from "../../../../server/collectors/files";
import { GET } from "./route";

const hit = { root: "repos" as const, rel: "a.ts", name: "a.ts" };

describe("GET /api/files/search", () => {
  beforeEach(() => {
    vi.mocked(searchByName).mockReset();
    settingsState.vault = true;
    settingsState.vaultPath = "/tmp/vault";
  });

  it.each([
    { label: "scope=vault refuses when vault is off, never calling searchByName", vault: false, vaultPath: "/tmp/vault", scope: "vault", refused: true },
    { label: "scope=vault refuses when vault is enabled but unconfigured (null vaultPath)", vault: true, vaultPath: null, scope: "vault", refused: true },
    { label: "scope=all narrows to reposOnly when vault is off, searchByName gets the narrowed scope", vault: false, vaultPath: "/tmp/vault", scope: "all", refused: false },
  ])("$label (spec per-integration table, decision 16)", async ({ vault, vaultPath, scope, refused }) => {
    settingsState.vault = vault;
    settingsState.vaultPath = vaultPath;
    if (!refused) vi.mocked(searchByName).mockResolvedValueOnce({ hits: [], truncated: false });
    const req = new Request(`http://127.0.0.1/api/files/search?q=foo&scope=${scope}`);
    const res = await GET(req);
    expect(res.status).toBe(200);
    if (refused) {
      expect(await res.json()).toEqual({ ok: false, error: "disabled" });
      expect(vi.mocked(searchByName)).not.toHaveBeenCalled();
    } else {
      await res.json();
      expect(vi.mocked(searchByName)).toHaveBeenCalledWith("foo", { kind: "reposOnly" }, req.signal);
    }
  });

  it("forwards the request signal and an incomplete envelope", async () => {
    const controller = new AbortController();
    const req = new Request("http://127.0.0.1/api/files/search?q=foo", { signal: controller.signal });
    vi.mocked(searchByName).mockResolvedValueOnce({ hits: [], truncated: true });
    expect(await (await GET(req)).json()).toEqual({ ok: true, data: [], truncated: true });
    expect(searchByName).toHaveBeenCalledWith("foo", { kind: "all" }, req.signal);
  });

  it("returns a complete hit envelope", async () => {
    vi.mocked(searchByName).mockResolvedValueOnce({ hits: [hit], truncated: false });
    const req = new Request("http://127.0.0.1/api/files/search?q=a.ts&scope=vault");
    expect(await (await GET(req)).json()).toEqual({ ok: true, data: [hit], truncated: false });
    expect(searchByName).toHaveBeenCalledWith("a.ts", { kind: "vault" }, req.signal);
  });

  it("returns a complete empty envelope", async () => {
    vi.mocked(searchByName).mockResolvedValueOnce({ hits: [], truncated: false });
    const req = new Request("http://127.0.0.1/api/files/search?q=&scope=repo:jax-os");
    expect(await (await GET(req)).json()).toEqual({ ok: true, data: [], truncated: false });
    expect(searchByName).toHaveBeenCalledWith("", { kind: "repo", name: "jax-os" }, req.signal);
  });

  it("refuses a 257-byte query without calling the collector", async () => {
    const req = new Request(`http://127.0.0.1/api/files/search?q=${"a".repeat(257)}`);
    const res = await GET(req);
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: false, error: "invalid payload" });
    expect(searchByName).not.toHaveBeenCalled();
  });

  it("returns a source-error envelope on root failure", async () => {
    vi.mocked(searchByName).mockRejectedValueOnce(new Error("file search unavailable"));
    const res = await GET(new Request("http://127.0.0.1/api/files/search?q=foo"));
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: false, error: "file search unavailable" });
  });

  it("refuses an invalid scope without calling the collector", async () => {
    const req = new Request("http://127.0.0.1/api/files/search?q=foo&scope=repo:..");
    expect(await (await GET(req)).json()).toEqual({ ok: false, error: "invalid payload" });
    expect(searchByName).not.toHaveBeenCalled();
  });

  it("returns a cancellation envelope when the request is aborted", async () => {
    const controller = new AbortController();
    const req = new Request("http://127.0.0.1/api/files/search?q=foo", { signal: controller.signal });
    controller.abort(new Error("gone"));
    vi.mocked(searchByName).mockRejectedValueOnce(controller.signal.reason);
    const res = await GET(req);
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: false, error: "search cancelled" });
  });
});
