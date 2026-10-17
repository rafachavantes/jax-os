import { beforeEach, describe, expect, it, vi } from "vitest";

const settingsState = vi.hoisted(() => ({ vault: true, vaultPath: "/tmp/vault" as string | null }));
vi.mock("../../../../server/settings", () => ({
  readGeneralSettings: () => ({
    ok: true,
    data: { vaultPath: settingsState.vaultPath, integrations: { vault: settingsState.vault } },
  }),
}));

vi.mock("../../../../server/collectors/files", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../../server/collectors/files")>();
  return { ...actual, ROOTS: { repos: "/tmp/repos", vault: "/tmp/vault" }, planZip: vi.fn(), zipStream: vi.fn() };
});

import { planZip, zipStream } from "../../../../server/collectors/files";
import { GET } from "./route";

describe("GET /api/files/zip", () => {
  beforeEach(() => {
    vi.mocked(planZip).mockReset();
    vi.mocked(zipStream).mockReset();
    settingsState.vault = true;
    settingsState.vaultPath = "/tmp/vault";
  });

  it.each([
    { label: "root=vault refuses when vault is off, never planning a zip", vault: false, vaultPath: "/tmp/vault", root: "vault", refused: true },
    { label: "root=vault refuses when vault is enabled but unconfigured (null vaultPath)", vault: true, vaultPath: null, root: "vault", refused: true },
    { label: "root=repos still streams when vault is off", vault: false, vaultPath: "/tmp/vault", root: "repos", refused: false },
  ])("$label (spec per-integration table, decision 16)", async ({ vault, vaultPath, root, refused }) => {
    settingsState.vault = vault;
    settingsState.vaultPath = vaultPath;
    if (!refused) {
      vi.mocked(planZip).mockResolvedValueOnce({ ok: true, entries: [], secretsExcluded: 0 });
      async function* emptyGen() {}
      vi.mocked(zipStream).mockReturnValueOnce(emptyGen());
    }
    const res = await GET(new Request(`http://127.0.0.1/api/files/zip?root=${root}&rel=`));
    expect(res.status).toBe(200);
    if (refused) {
      expect(await res.json()).toEqual({ ok: false, error: "disabled" });
      expect(vi.mocked(planZip)).not.toHaveBeenCalled();
    } else {
      expect(res.headers.get("Content-Type")).toBe("application/zip");
      expect(vi.mocked(planZip)).toHaveBeenCalledWith("repos", "");
    }
  });

  it("refuses invalid input and surfaces zip-too-large, neither invoking/streaming the collector", async () => {
    const badRoot = await GET(new Request("http://127.0.0.1/api/files/zip?root=x&rel=a"));
    expect(await badRoot.json()).toEqual({ ok: false, error: "bad root" });
    expect(planZip).not.toHaveBeenCalled();
    // round 1 plan review F2: strictRel runs for real here (importOriginal above).
    for (const bad of ["a/../b", "/abs", "a\\b", "a//b", "."]) {
      const res0 = await GET(new Request(`http://127.0.0.1/api/files/zip?root=repos&rel=${encodeURIComponent(bad)}`));
      expect(await res0.json()).toEqual({ ok: false, error: "invalid payload" });
    }
    expect(planZip).not.toHaveBeenCalled();
    vi.mocked(planZip).mockResolvedValueOnce({ ok: false, error: "zip-too-large" });
    const res = await GET(new Request("http://127.0.0.1/api/files/zip?root=repos&rel=big"));
    expect(await res.json()).toEqual({ ok: false, error: "zip-too-large" });
    expect(zipStream).not.toHaveBeenCalled();
  });

  it("streams with secrets-excluded/entry-count headers and a folder name, or the root name when rel is empty", async () => {
    vi.mocked(planZip).mockResolvedValueOnce({
      ok: true,
      entries: [{ rel: "a.txt", absPath: "/tmp/repos/a.txt", size: 1 }],
      secretsExcluded: 2,
    });
    async function* fakeGen() { yield new Uint8Array([1]); }
    vi.mocked(zipStream).mockReturnValueOnce(fakeGen());
    const res = await GET(new Request("http://127.0.0.1/api/files/zip?root=repos&rel=docs"));
    expect(res.headers.get("Content-Type")).toBe("application/zip");
    expect(res.headers.get("X-Secrets-Excluded")).toBe("2");
    expect(res.headers.get("X-Zip-Entries")).toBe("1");
    expect(res.headers.get("Content-Disposition")).toContain("docs.zip");

    vi.mocked(planZip).mockResolvedValueOnce({ ok: true, entries: [], secretsExcluded: 0 });
    async function* emptyGen() {}
    vi.mocked(zipStream).mockReturnValueOnce(emptyGen());
    const rootRes = await GET(new Request("http://127.0.0.1/api/files/zip?root=repos&rel="));
    expect(rootRes.headers.get("Content-Disposition")).toContain("repos.zip");
  });
});
