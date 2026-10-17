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
    uploadFile: vi.fn(),
  };
});
vi.mock("../../../../server/mutations", () => ({
  runMutation: vi.fn(),
  fileEffect: (fn: () => unknown) => fn(),
}));

import { uploadFile } from "../../../../server/collectors/files";
import { runMutation } from "../../../../server/mutations";
import { POST } from "./route";

// The route requires a numeric content-length header; a FormData body does not set it
// automatically in the vitest/undici runtime, so serialize it and declare the length explicitly.
async function req(root: string) {
  const form = new FormData();
  form.set("root", root);
  form.set("relParentDir", "");
  form.set("file", new File(["x"], "a.txt"));
  const serialized = new Response(form);
  const bytes = new Uint8Array(await serialized.arrayBuffer());
  return new Request("http://127.0.0.1/api/files/upload", {
    method: "POST",
    headers: {
      "content-type": serialized.headers.get("content-type") ?? "",
      "content-length": String(bytes.byteLength),
    },
    body: bytes,
  });
}

describe("POST /api/files/upload", () => {
  beforeEach(() => {
    vi.mocked(uploadFile).mockClear();
    vi.mocked(runMutation).mockReset();
    settingsState.vault = true;
    settingsState.vaultPath = "/tmp/vault";
  });

  it.each([
    { label: "root=vault refuses when vault is off, never uploading", vault: false, vaultPath: "/tmp/vault", root: "vault", refused: true },
    { label: "root=vault refuses when vault is enabled but unconfigured (null vaultPath)", vault: true, vaultPath: null, root: "vault", refused: true },
    { label: "root=repos still uploads when vault is off", vault: false, vaultPath: "/tmp/vault", root: "repos", refused: false },
  ])("$label (spec per-integration table, decision 16)", async ({ vault, vaultPath, root, refused }) => {
    settingsState.vault = vault;
    settingsState.vaultPath = vaultPath;
    if (!refused) vi.mocked(runMutation).mockImplementationOnce(async (_entry, effect) => ({ ok: true, value: await effect() }));
    const res = await POST(await req(root));
    expect(res.status).toBe(200);
    if (refused) {
      expect(await res.json()).toEqual({ ok: false, error: "disabled" });
      expect(runMutation).not.toHaveBeenCalled();
      expect(uploadFile).not.toHaveBeenCalled();
    } else {
      expect((await res.json()).ok).toBe(true);
      expect(runMutation).toHaveBeenCalled();
      expect(uploadFile).toHaveBeenCalledWith("repos", "", "a.txt", Buffer.from("x"));
    }
  });
});
