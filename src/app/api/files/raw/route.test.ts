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
  imageMime: vi.fn((rel: string) => (rel.endsWith(".svg") ? "image/svg+xml" : rel.endsWith(".png") ? "image/png" : null)),
  mediaMime: vi.fn((rel: string) =>
    rel.endsWith(".pdf") ? { mime: "application/pdf", kind: "pdf" }
      : rel.endsWith(".mp3") ? { mime: "audio/mpeg", kind: "audio" }
        : rel.endsWith(".mp4") ? { mime: "video/mp4", kind: "video" }
          : null),
  readRawPreviewAt: vi.fn(),
  resolveExisting: vi.fn((root: string, rel: string) => `/abs/${root}/${rel}`),
  RAW_PDF_CAP: 20 * 1024 * 1024,
  RAW_MEDIA_CAP: 64 * 1024 * 1024,
}));

vi.mock("../../../../server/collectors/fileDownload", () => ({
  openRawStreamAt: vi.fn(),
  streamHandle: vi.fn(),
  parseRange: vi.fn(),
}));

import { imageMime, RAW_MEDIA_CAP, RAW_PDF_CAP, readRawPreviewAt, resolveExisting } from "../../../../server/collectors/files";
import { openRawStreamAt, parseRange, streamHandle } from "../../../../server/collectors/fileDownload";
import { GET } from "./route";

describe("GET /api/files/raw", () => {
  beforeEach(() => {
    vi.mocked(readRawPreviewAt).mockReset();
    vi.mocked(resolveExisting).mockClear();
    vi.mocked(openRawStreamAt).mockReset();
    vi.mocked(parseRange).mockReset();
    vi.mocked(streamHandle).mockReset();
    settingsState.vault = true;
    settingsState.vaultPath = "/tmp/vault";
  });

  it.each([
    { label: "root=vault refuses when vault is off, never resolving/serving", vault: false, vaultPath: "/tmp/vault", root: "vault", refused: true },
    { label: "root=vault refuses when vault is enabled but unconfigured (null vaultPath)", vault: true, vaultPath: null, root: "vault", refused: true },
    { label: "root=repos still serves when vault is off", vault: false, vaultPath: "/tmp/vault", root: "repos", refused: false },
  ])("$label (spec per-integration table, decision 16)", async ({ vault, vaultPath, root, refused }) => {
    settingsState.vault = vault;
    settingsState.vaultPath = vaultPath;
    if (!refused) vi.mocked(readRawPreviewAt).mockReturnValueOnce(Buffer.from("img"));
    const res = await GET(new Request(`http://127.0.0.1/api/files/raw?root=${root}&rel=x.png`));
    expect(res.status).toBe(200);
    if (refused) {
      expect(await res.json()).toEqual({ ok: false, error: "disabled" });
      expect(vi.mocked(resolveExisting)).not.toHaveBeenCalled();
      expect(vi.mocked(readRawPreviewAt)).not.toHaveBeenCalled();
    } else {
      expect(Buffer.from(await res.arrayBuffer()).toString()).toBe("img");
      expect(vi.mocked(resolveExisting)).toHaveBeenCalled();
      expect(vi.mocked(readRawPreviewAt)).toHaveBeenCalled();
    }
  });

  it("refuses invalid input without collector work", async () => {
    const badRoot = await GET(new Request("http://127.0.0.1/api/files/raw?root=x&rel=a.png"));
    expect(await badRoot.json()).toEqual({ ok: false, error: "bad root" });
    const empty = await GET(new Request("http://127.0.0.1/api/files/raw?root=repos&rel="));
    expect(await empty.json()).toEqual({ ok: false, error: "invalid payload" });
    expect(resolveExisting).not.toHaveBeenCalled();
    expect(readRawPreviewAt).not.toHaveBeenCalled();
  });

  it("keeps SVG sandbox headers, now with an inline disposition", async () => {
    vi.mocked(readRawPreviewAt).mockReturnValueOnce(Buffer.from("<svg></svg>"));
    const res = await GET(new Request("http://127.0.0.1/api/files/raw?root=repos&rel=icon.svg"));
    expect(res.headers.get("Content-Type")).toBe("image/svg+xml");
    expect(res.headers.get("Content-Security-Policy")).toBe("default-src 'none'; sandbox");
    expect(res.headers.get("X-Content-Type-Options")).toBe("nosniff");
    expect(res.headers.get("Content-Disposition")).toBe("inline");
    expect(res.headers.get("Cache-Control")).toBe("no-store");
    expect(Buffer.from(await res.arrayBuffer()).toString()).toBe("<svg></svg>");
  });

  it("streams a plain (no-Range) pdf request as a full 200 with Accept-Ranges advertised", async () => {
    const fakeHandle = { close: vi.fn(async () => {}) };
    vi.mocked(openRawStreamAt).mockResolvedValueOnce({ handle: fakeHandle as never, size: 100 });
    vi.mocked(parseRange).mockReturnValueOnce(null);
    vi.mocked(streamHandle).mockReturnValueOnce(new ReadableStream());
    const res = await GET(new Request("http://127.0.0.1/api/files/raw?root=repos&rel=doc.pdf"));
    expect(res.status).toBe(200);
    expect(res.headers.get("Content-Type")).toBe("application/pdf");
    expect(res.headers.get("Content-Disposition")).toBe("inline");
    expect(res.headers.get("Accept-Ranges")).toBe("bytes");
    expect(res.headers.get("Content-Length")).toBe("100");
    expect(streamHandle).toHaveBeenCalledWith(fakeHandle, 100, undefined);
  });

  it("streams a ranged audio request as 206 with Content-Range/Content-Length", async () => {
    const fakeHandle = { close: vi.fn(async () => {}) };
    vi.mocked(openRawStreamAt).mockResolvedValueOnce({ handle: fakeHandle as never, size: 1000 });
    vi.mocked(parseRange).mockReturnValueOnce({ start: 100, end: 199 });
    vi.mocked(streamHandle).mockReturnValueOnce(new ReadableStream());
    const res = await GET(new Request("http://127.0.0.1/api/files/raw?root=repos&rel=song.mp3", { headers: { Range: "bytes=100-199" } }));
    expect(res.status).toBe(206);
    expect(res.headers.get("Content-Range")).toBe("bytes 100-199/1000");
    expect(res.headers.get("Content-Length")).toBe("100");
    expect(streamHandle).toHaveBeenCalledWith(fakeHandle, 1000, { start: 100, end: 199 });
  });

  it("returns 416 with Content-Range on an unsatisfiable range, closing the handle and never streaming", async () => {
    const fakeHandle = { close: vi.fn(async () => {}) };
    vi.mocked(openRawStreamAt).mockResolvedValueOnce({ handle: fakeHandle as never, size: 50 });
    vi.mocked(parseRange).mockReturnValueOnce("unsatisfiable");
    const res = await GET(new Request("http://127.0.0.1/api/files/raw?root=repos&rel=song.mp3", { headers: { Range: "bytes=9999-" } }));
    expect(res.status).toBe(416);
    expect(res.headers.get("Content-Range")).toBe("bytes */50");
    expect(fakeHandle.close).toHaveBeenCalledOnce();
    expect(streamHandle).not.toHaveBeenCalled();
  });

  it("returns 413 for an over-cap source on both a plain and a ranged request, never opening a stream", async () => {
    vi.mocked(openRawStreamAt).mockRejectedValue(new Error("too large"));
    const plain = await GET(new Request("http://127.0.0.1/api/files/raw?root=repos&rel=big.mp3"));
    expect(plain.status).toBe(413);
    expect(await plain.json()).toEqual({ ok: false, error: "too large" });
    const ranged = await GET(new Request("http://127.0.0.1/api/files/raw?root=repos&rel=big.mp3", { headers: { Range: "bytes=0-10" } }));
    expect(ranged.status).toBe(413);
    expect(streamHandle).not.toHaveBeenCalled();
  });

  it("succeeds for a source exactly at the cap, on both a plain and a ranged request (cold review round 1 F5)", async () => {
    const fakeHandle = { close: vi.fn(async () => {}) };
    vi.mocked(openRawStreamAt).mockResolvedValueOnce({ handle: fakeHandle as never, size: RAW_MEDIA_CAP });
    vi.mocked(parseRange).mockReturnValueOnce(null);
    vi.mocked(streamHandle).mockReturnValueOnce(new ReadableStream());
    const plain = await GET(new Request("http://127.0.0.1/api/files/raw?root=repos&rel=big.mp3"));
    expect(plain.status).toBe(200);

    vi.mocked(openRawStreamAt).mockResolvedValueOnce({ handle: fakeHandle as never, size: RAW_MEDIA_CAP });
    vi.mocked(parseRange).mockReturnValueOnce({ start: 0, end: 99 });
    vi.mocked(streamHandle).mockReturnValueOnce(new ReadableStream());
    const ranged = await GET(new Request("http://127.0.0.1/api/files/raw?root=repos&rel=big.mp3", { headers: { Range: "bytes=0-99" } }));
    expect(ranged.status).toBe(206);
  });

  it("selects the PDF cap (not the media cap) for a pdf exactly at RAW_PDF_CAP, on both a plain and a ranged request", async () => {
    const fakeHandle = { close: vi.fn(async () => {}) };
    vi.mocked(openRawStreamAt).mockResolvedValueOnce({ handle: fakeHandle as never, size: RAW_PDF_CAP });
    vi.mocked(parseRange).mockReturnValueOnce(null);
    vi.mocked(streamHandle).mockReturnValueOnce(new ReadableStream());
    const plain = await GET(new Request("http://127.0.0.1/api/files/raw?root=repos&rel=big.pdf"));
    expect(plain.status).toBe(200);
    expect(openRawStreamAt).toHaveBeenLastCalledWith(expect.anything(), RAW_PDF_CAP);

    vi.mocked(openRawStreamAt).mockResolvedValueOnce({ handle: fakeHandle as never, size: RAW_PDF_CAP });
    vi.mocked(parseRange).mockReturnValueOnce({ start: 0, end: 99 });
    vi.mocked(streamHandle).mockReturnValueOnce(new ReadableStream());
    const ranged = await GET(new Request("http://127.0.0.1/api/files/raw?root=repos&rel=big.pdf", { headers: { Range: "bytes=0-99" } }));
    expect(ranged.status).toBe(206);
    expect(openRawStreamAt).toHaveBeenLastCalledWith(expect.anything(), RAW_PDF_CAP);
  });

  it("refuses a pdf one byte over RAW_PDF_CAP with the existing over-cap shape, using the PDF cap", async () => {
    vi.mocked(openRawStreamAt).mockRejectedValueOnce(new Error("too large"));
    const res = await GET(new Request("http://127.0.0.1/api/files/raw?root=repos&rel=big.pdf"));
    expect(res.status).toBe(413);
    expect(await res.json()).toEqual({ ok: false, error: "too large" });
    expect(openRawStreamAt).toHaveBeenCalledWith(expect.anything(), RAW_PDF_CAP);
  });

  it("streams an mp4 as video/mp4, plain 200 and ranged 206 (mirrors the mp3 fixtures)", async () => {
    const fakeHandle = { close: vi.fn(async () => {}) };
    vi.mocked(openRawStreamAt).mockResolvedValueOnce({ handle: fakeHandle as never, size: 1000 });
    vi.mocked(parseRange).mockReturnValueOnce(null);
    vi.mocked(streamHandle).mockReturnValueOnce(new ReadableStream());
    const plain = await GET(new Request("http://127.0.0.1/api/files/raw?root=repos&rel=movie.mp4"));
    expect(plain.status).toBe(200);
    expect(plain.headers.get("Content-Type")).toBe("video/mp4");
    expect(openRawStreamAt).toHaveBeenLastCalledWith(expect.anything(), RAW_MEDIA_CAP);

    vi.mocked(openRawStreamAt).mockResolvedValueOnce({ handle: fakeHandle as never, size: 1000 });
    vi.mocked(parseRange).mockReturnValueOnce({ start: 100, end: 199 });
    vi.mocked(streamHandle).mockReturnValueOnce(new ReadableStream());
    const ranged = await GET(new Request("http://127.0.0.1/api/files/raw?root=repos&rel=movie.mp4", { headers: { Range: "bytes=100-199" } }));
    expect(ranged.status).toBe(206);
    expect(ranged.headers.get("Content-Type")).toBe("video/mp4");
    expect(ranged.headers.get("Content-Range")).toBe("bytes 100-199/1000");
    expect(streamHandle).toHaveBeenCalledWith(fakeHandle, 1000, { start: 100, end: 199 });
  });

  it("falls through to unsupported type for an unrecognized extension", async () => {
    const res = await GET(new Request("http://127.0.0.1/api/files/raw?root=repos&rel=notes.xyz"));
    expect(await res.json()).toEqual({ ok: false, error: "unsupported type" });
  });
});
