import { QueryClient, QueryObserver } from "@tanstack/react-query";
import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchContentSearch, fetchFileSearch, fetchRepoNames, groupContentHits, repoNamesFromListing } from "./fileSearch";

const hit = { root: "repos" as const, rel: "b.ts", name: "b.ts" };

function jsonResponse(body: unknown, ok = true) {
  return { ok, json: async () => body };
}

describe("fetchFileSearch", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("encodes the query and forwards the signal", async () => {
    const signal = new AbortController().signal;
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ ok: true, data: [], truncated: false }));
    vi.stubGlobal("fetch", fetchMock);
    expect(await fetchFileSearch("a b", { kind: "repo", name: "jax-os" }, signal)).toEqual({ ok: true, data: [], truncated: false });
    expect(fetchMock).toHaveBeenCalledWith("/api/files/search?q=a%20b&scope=repo%3Ajax-os", { signal });
  });

  it("throws on a transport failure", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, json: async () => ({}) }));
    await expect(fetchFileSearch("x", { kind: "all" }, new AbortController().signal)).rejects.toThrow("file search unavailable");
  });

  it("returns complete and incomplete envelopes", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse({ ok: true, data: [hit], truncated: false }))
      .mockResolvedValueOnce(jsonResponse({ ok: true, data: [], truncated: true }));
    vi.stubGlobal("fetch", fetchMock);
    const signal = new AbortController().signal;
    expect(await fetchFileSearch("b", { kind: "all" }, signal)).toEqual({ ok: true, data: [hit], truncated: false });
    expect(await fetchFileSearch("b", { kind: "all" }, signal)).toEqual({ ok: true, data: [], truncated: true });
  });

  it("propagates abort instead of converting it to ok:false", async () => {
    const err = new DOMException("The operation was aborted.", "AbortError");
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(err));
    await expect(fetchFileSearch("x", { kind: "all" }, new AbortController().signal)).rejects.toBe(err);
  });

  it("keeps B after aborting A and a late A json completion", async () => {
    const aHit = { root: "repos" as const, rel: "a.ts", name: "a.ts" };
    const bResult = { ok: true as const, data: [hit], truncated: false };
    const pending = new Map<string, { resolve: (value: unknown) => void; signal?: AbortSignal }>();
    vi.stubGlobal("fetch", vi.fn((url: string, init?: RequestInit) => {
      const q = new URL(url, "http://127.0.0.1").searchParams.get("q") ?? "";
      return new Promise((resolve) => {
        pending.set(q, { resolve, signal: init?.signal ?? undefined });
      });
    }));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
    const observer = new QueryObserver(client, {
      queryKey: ["files", "search", "A"],
      queryFn: ({ signal }) => fetchFileSearch("A", { kind: "all" }, signal),
      retry: false,
      staleTime: 10000,
    });
    const unsub = observer.subscribe(() => {});
    try {
      await vi.waitFor(() => expect(pending.has("A")).toBe(true));
      observer.setOptions({
        queryKey: ["files", "search", "B"],
        queryFn: ({ signal }) => fetchFileSearch("B", { kind: "all" }, signal),
        retry: false,
        staleTime: 10000,
      });
      await vi.waitFor(() => expect(pending.has("B")).toBe(true));
      expect(pending.get("A")?.signal?.aborted).toBe(true);
      pending.get("B")!.resolve(jsonResponse(bResult));
      await vi.waitFor(() => {
        expect(observer.getCurrentResult().data).toEqual(bResult);
      });
      const lateBody = vi.fn(async () => ({ ok: true, data: [aHit], truncated: false }));
      pending.get("A")!.resolve({ ok: true, json: lateBody });
      await vi.waitFor(() => expect(lateBody).toHaveBeenCalledOnce());
      expect(observer.getCurrentResult().data).toEqual(bResult);
      expect(observer.getCurrentResult().isError).toBe(false);
    } finally {
      unsub();
      await client.cancelQueries();
      client.clear();
    }
  });
});

describe("fetchContentSearch", () => {
  afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });

  it("hits the content-search endpoint with the encoded query and serialized scope", async () => {
    const signal = new AbortController().signal;
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ ok: true, data: [], truncated: false }) });
    vi.stubGlobal("fetch", fetchMock);
    expect(await fetchContentSearch("needle", { kind: "vault" }, signal)).toEqual({ ok: true, data: [], truncated: false });
    expect(fetchMock).toHaveBeenCalledWith("/api/files/content-search?q=needle&scope=vault", { signal });
  });
  it("throws on a transport failure", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, json: async () => ({}) }));
    await expect(fetchContentSearch("x", { kind: "all" }, new AbortController().signal)).rejects.toThrow("content search unavailable");
  });
});

describe("fetchRepoNames / repoNamesFromListing", () => {
  afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });

  it("fetches the existing repos tree listing at its root (round-4 F1 — no new route)", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ ok: true, data: { dirs: [{ name: "jax-os" }], files: [] } }) });
    await fetchRepoNames(fetchMock as unknown as typeof fetch);
    expect(fetchMock).toHaveBeenCalledWith("/api/files/tree?root=repos&rel=");
  });
  it("repoNamesFromListing extracts just the directory names", () => {
    expect(repoNamesFromListing({ dirs: [{ name: "b" }, { name: "a" }], files: [{ name: "x.ts" }] } as never)).toEqual(["b", "a"]);
  });
});

describe("groupContentHits (review F2: group content hits by file)", () => {
  it("groups hits sharing root+rel into one entry, in first-appearance order", () => {
    const a1 = { root: "repos" as const, rel: "a.ts", line: 1, snippet: "needle one" };
    const b1 = { root: "vault" as const, rel: "b.md", line: 5, snippet: "needle two" };
    const a2 = { root: "repos" as const, rel: "a.ts", line: 9, snippet: "needle three" };
    expect(groupContentHits([a1, b1, a2])).toEqual([
      { root: "repos", rel: "a.ts", hits: [a1, a2] },
      { root: "vault", rel: "b.md", hits: [b1] },
    ]);
  });
  it("treats the same rel under different roots as separate files", () => {
    const repoHit = { root: "repos" as const, rel: "x.ts", line: 1, snippet: "needle" };
    const vaultHit = { root: "vault" as const, rel: "x.ts", line: 1, snippet: "needle" };
    expect(groupContentHits([repoHit, vaultHit])).toEqual([
      { root: "repos", rel: "x.ts", hits: [repoHit] },
      { root: "vault", rel: "x.ts", hits: [vaultHit] },
    ]);
  });
  it("returns an empty array for no hits", () => {
    expect(groupContentHits([])).toEqual([]);
  });
});

describe("scope URL encoding (round-2 F4: a repo name with reserved URL characters)", () => {
  afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });

  it("round-trips a repo name containing '#&?%' through both fetchers' scope param", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ ok: true, data: [], truncated: false }) });
    vi.stubGlobal("fetch", fetchMock);
    const scope = { kind: "repo" as const, name: "weird#&?%name" };
    const signal = new AbortController().signal;
    await fetchFileSearch("x", scope, signal);
    await fetchContentSearch("x", scope, signal);
    for (const [url] of fetchMock.mock.calls) {
      // searchParams.get already percent-decodes; the raw value must round-trip byte-for-byte.
      expect(new URL(url as string, "http://127.0.0.1").searchParams.get("scope")).toBe("repo:weird#&?%name");
    }
  });
});
