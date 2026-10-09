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
  return {
    ...actual,
    ROOTS: { repos: "/tmp/repos", vault: "/tmp/vault" },
    renameEntry: vi.fn(),
  };
});
vi.mock("../../../../server/mutations", () => ({
  runMutation: vi.fn(),
  fileEffect: (fn: () => unknown) => fn(),
}));

import { renameEntry } from "../../../../server/collectors/files";
import { runMutation } from "../../../../server/mutations";
import { POST } from "./route";

function req(body: unknown) {
  return new Request("http://127.0.0.1/api/files/rename", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
}

describe("POST /api/files/rename", () => {
  beforeEach(() => {
    vi.mocked(renameEntry).mockClear();
    vi.mocked(runMutation).mockReset();
    settingsState.vault = true;
    settingsState.vaultPath = "/tmp/vault";
  });

  it.each([
    { label: "root=vault refuses when vault is off, never renaming", vault: false, vaultPath: "/tmp/vault", root: "vault", refused: true },
    { label: "root=vault refuses when vault is enabled but unconfigured (null vaultPath)", vault: true, vaultPath: null, root: "vault", refused: true },
    { label: "root=repos still renames when vault is off", vault: false, vaultPath: "/tmp/vault", root: "repos", refused: false },
  ])("$label (spec per-integration table, decision 16)", async ({ vault, vaultPath, root, refused }) => {
    settingsState.vault = vault;
    settingsState.vaultPath = vaultPath;
    if (!refused) vi.mocked(runMutation).mockImplementationOnce(async (_entry, effect) => ({ ok: true, value: await effect() }));
    const res = await POST(req({ root, relFrom: "x.txt", relTo: "y.txt" }));
    expect(res.status).toBe(200);
    if (refused) {
      expect(await res.json()).toEqual({ ok: false, error: "disabled" });
      expect(runMutation).not.toHaveBeenCalled();
      expect(renameEntry).not.toHaveBeenCalled();
    } else {
      expect((await res.json()).ok).toBe(true);
      expect(runMutation).toHaveBeenCalled();
      expect(renameEntry).toHaveBeenCalledWith("repos", "x.txt", "y.txt");
    }
  });
});
