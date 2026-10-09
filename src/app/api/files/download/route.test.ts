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
  gateVaultRoot: (root: string, vaultUsable: boolean) => root === "vault" && !vaultUsable,
}));

vi.mock("../../../../server/collectors/fileDownload", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../../server/collectors/fileDownload")>();
  return { ...actual, downloadFile: vi.fn() };
});

vi.mock("../../../../server/mutations", () => ({
  runMutation: vi.fn(),
  fileEffect: vi.fn(),
}));

import { contentDisposition, downloadFile } from "../../../../server/collectors/fileDownload";
import { runMutation } from "../../../../server/mutations";
import { GET } from "./route";

describe("GET /api/files/download", () => {
  beforeEach(() => {
    vi.mocked(downloadFile).mockReset();
    vi.mocked(runMutation).mockReset();
    settingsState.vault = true;
    settingsState.vaultPath = "/tmp/vault";
  });

  it.each([
    { label: "root=vault refuses when vault is off, never calling downloadFile", vault: false, vaultPath: "/tmp/vault", root: "vault", refused: true },
    { label: "root=vault refuses when vault is enabled but unconfigured (null vaultPath)", vault: true, vaultPath: null, root: "vault", refused: true },
    { label: "root=repos still serves when vault is off", vault: false, vaultPath: "/tmp/vault", root: "repos", refused: false },
  ])("$label (spec per-integration table, decision 16)", async ({ vault, vaultPath, root, refused }) => {
    settingsState.vault = vault;
    settingsState.vaultPath = vaultPath;
    if (!refused) {
      vi.mocked(downloadFile).mockResolvedValueOnce({
        body: new ReadableStream({ start(c) { c.close(); } }),
        size: 0,
        disposition: 'attachment; filename="a.txt"',
      });
    }
    const res = await GET(new Request(`http://127.0.0.1/api/files/download?root=${root}&rel=a.txt`));
    expect(res.status).toBe(200);
    if (refused) {
      expect(await res.json()).toEqual({ ok: false, error: "disabled" });
      expect(vi.mocked(downloadFile)).not.toHaveBeenCalled();
    } else {
      expect(vi.mocked(downloadFile)).toHaveBeenCalledWith("repos", "a.txt");
    }
  });

  it("refuses invalid input without invoking the collector", async () => {
    const badRoot = await GET(new Request("http://127.0.0.1/api/files/download?root=x&rel=a.bin"));
    expect(await badRoot.json()).toEqual({ ok: false, error: "bad root" });
    const empty = await GET(new Request("http://127.0.0.1/api/files/download?root=repos&rel="));
    expect(await empty.json()).toEqual({ ok: false, error: "invalid payload" });
    expect(downloadFile).not.toHaveBeenCalled();
  });

  it("returns octet-stream headers from a successful setup", async () => {
    vi.mocked(downloadFile).mockResolvedValueOnce({
      body: new ReadableStream({ start(c) { c.enqueue(new Uint8Array([1, 2])); c.close(); } }),
      size: 2,
      disposition: 'attachment; filename="a.bin"; filename*=UTF-8\'\'a.bin',
    });
    const res = await GET(new Request("http://127.0.0.1/api/files/download?root=repos&rel=a.bin"));
    expect(res.headers.get("Content-Type")).toBe("application/octet-stream");
    expect(res.headers.get("Content-Length")).toBe("2");
    expect(res.headers.get("Content-Disposition")).toContain("a.bin");
    expect(res.headers.get("Cache-Control")).toBe("no-store");
    expect(res.headers.get("X-Content-Type-Options")).toBe("nosniff");
    expect(runMutation).not.toHaveBeenCalled();
    expect(downloadFile).toHaveBeenCalledWith("repos", "a.bin");
  });

  it("constructs unicode Content-Disposition from the collector fallback", async () => {
    vi.mocked(downloadFile).mockResolvedValueOnce({
      body: new ReadableStream({ start(c) { c.close(); } }),
      size: 0,
      disposition: contentDisposition("文件.txt"),
    });
    const res = await GET(new Request("http://127.0.0.1/api/files/download?root=repos&rel=%E6%96%87%E4%BB%B6.txt"));
    expect(res.headers.get("Content-Disposition")).toContain("filename*=UTF-8''");
    expect(res.headers.get("Content-Disposition")).toContain(encodeURIComponent("文件.txt"));
  });

  it("cancels the collector body if response construction throws", async () => {
    const cancel = vi.fn(async () => undefined);
    vi.mocked(downloadFile).mockResolvedValueOnce({
      body: { cancel } as unknown as ReadableStream<Uint8Array>,
      size: 4,
      disposition: 'attachment; filename="文件.txt"; filename*=UTF-8\'\'x',
    });
    const res = await GET(new Request("http://127.0.0.1/api/files/download?root=repos&rel=x.txt"));
    expect(cancel).toHaveBeenCalledOnce();
    expect(await res.json()).toEqual({ ok: false, error: "read failed" });
  });
});
