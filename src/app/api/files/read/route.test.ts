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
  readFile: vi.fn(),
  gateVaultRoot: (root: string, vaultUsable: boolean) => root === "vault" && !vaultUsable,
}));

import { readFile } from "../../../../server/collectors/files";
import { GET } from "./route";

describe("GET /api/files/read", () => {
  beforeEach(() => {
    vi.mocked(readFile).mockReset();
    settingsState.vault = true;
    settingsState.vaultPath = "/tmp/vault";
  });

  it.each([
    { label: "root=vault refuses when vault is off, never calling readFile", vault: false, vaultPath: "/tmp/vault", root: "vault", refused: true },
    { label: "root=vault refuses when vault is enabled but unconfigured (null vaultPath)", vault: true, vaultPath: null, root: "vault", refused: true },
    { label: "root=repos still serves when vault is off", vault: false, vaultPath: "/tmp/vault", root: "repos", refused: false },
  ])("$label (spec per-integration table, decision 16)", async ({ vault, vaultPath, root, refused }) => {
    settingsState.vault = vault;
    settingsState.vaultPath = vaultPath;
    if (!refused) vi.mocked(readFile).mockReturnValueOnce({ binary: false, content: "x", hash: "h", size: 1 });
    const res = await GET(new Request(`http://127.0.0.1/api/files/read?root=${root}&rel=a.ts`));
    expect(res.status).toBe(200);
    if (refused) {
      expect(await res.json()).toEqual({ ok: false, error: "disabled" });
      expect(vi.mocked(readFile)).not.toHaveBeenCalled();
    } else {
      await res.json();
      expect(vi.mocked(readFile)).toHaveBeenCalled();
    }
  });

  it("refuses a bad root without calling the collector", async () => {
    const res = await GET(new Request("http://127.0.0.1/api/files/read?root=nope&rel=a.ts"));
    expect(await res.json()).toEqual({ ok: false, error: "bad root" });
    expect(readFile).not.toHaveBeenCalled();
  });

  it("refuses an invalid relative path without calling the collector", async () => {
    const res = await GET(new Request("http://127.0.0.1/api/files/read?root=repos&rel="));
    expect(await res.json()).toEqual({ ok: false, error: "invalid payload" });
    expect(readFile).not.toHaveBeenCalled();
  });

  it("returns the collector payload", async () => {
    vi.mocked(readFile).mockReturnValueOnce({ binary: false, content: "x", hash: "h", size: 1 });
    const res = await GET(new Request("http://127.0.0.1/api/files/read?root=repos&rel=a.ts"));
    expect(await res.json()).toEqual({ ok: true, data: { binary: false, content: "x", hash: "h", size: 1 } });
    expect(readFile).toHaveBeenCalledWith("repos", "a.ts");
  });
});
