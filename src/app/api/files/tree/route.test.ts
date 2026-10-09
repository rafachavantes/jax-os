import { beforeEach, describe, expect, it, vi } from "vitest";

const settingsState = vi.hoisted(() => ({ vault: true, vaultPath: "/tmp/vault" as string | null }));
vi.mock("../../../../server/settings", () => ({
  readGeneralSettings: () => ({
    ok: true,
    data: { vaultPath: settingsState.vaultPath, integrations: { vault: settingsState.vault } },
  }),
}));

vi.mock("../../../../server/collectors/files", () => ({
  ROOTS: { repos: "/tmp/repos", vault: "/tmp/vault" },
  listDir: vi.fn(),
  gateVaultRoot: (root: string, vaultUsable: boolean) => root === "vault" && !vaultUsable,
}));

import { listDir } from "../../../../server/collectors/files";
import { GET } from "./route";

describe("GET /api/files/tree", () => {
  beforeEach(() => {
    vi.mocked(listDir).mockReset();
    settingsState.vault = true;
    settingsState.vaultPath = "/tmp/vault";
  });

  it.each([
    { label: "root=vault refuses when vault is off, never calling listDir", vault: false, vaultPath: "/tmp/vault", root: "vault", refused: true },
    { label: "root=vault refuses when vault is enabled but unconfigured (null vaultPath)", vault: true, vaultPath: null, root: "vault", refused: true },
    { label: "root=repos still serves when vault is off", vault: false, vaultPath: "/tmp/vault", root: "repos", refused: false },
  ])("$label (spec per-integration table, decision 16)", async ({ vault, vaultPath, root, refused }) => {
    settingsState.vault = vault;
    settingsState.vaultPath = vaultPath;
    if (!refused) vi.mocked(listDir).mockResolvedValueOnce({ dirs: [], files: [] });
    const res = await GET(new Request(`http://127.0.0.1/api/files/tree?root=${root}&rel=`));
    expect(res.status).toBe(200);
    if (refused) {
      expect(await res.json()).toEqual({ ok: false, error: "disabled" });
      expect(vi.mocked(listDir)).not.toHaveBeenCalled();
    } else {
      await res.json();
      expect(vi.mocked(listDir)).toHaveBeenCalled();
    }
  });

  it("allows an empty root-relative directory and passes through mtime/size/gitStatus untouched", async () => {
    vi.mocked(listDir).mockResolvedValueOnce({
      dirs: [{ name: "docs", isDir: true, mtime: "2026-09-18T10:00:00.000Z", gitStatus: "modified" }],
      files: [{ name: "a.md", isDir: false, mtime: "2026-09-18T10:00:00.000Z", size: 1234, gitStatus: null }],
    });
    const res = await GET(new Request("http://127.0.0.1/api/files/tree?root=repos&rel="));
    expect(await res.json()).toEqual({
      ok: true,
      data: {
        dirs: [{ name: "docs", isDir: true, mtime: "2026-09-18T10:00:00.000Z", gitStatus: "modified" }],
        files: [{ name: "a.md", isDir: false, mtime: "2026-09-18T10:00:00.000Z", size: 1234, gitStatus: null }],
      },
    });
    expect(listDir).toHaveBeenCalledWith("repos", "");
  });

  it("refuses a NUL relative path without calling the collector", async () => {
    const res = await GET(new Request("http://127.0.0.1/api/files/tree?root=repos&rel=a%00b"));
    expect(await res.json()).toEqual({ ok: false, error: "invalid payload" });
    expect(listDir).not.toHaveBeenCalled();
  });
});
