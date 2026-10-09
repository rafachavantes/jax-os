import { QueryClient, QueryObserver } from "@tanstack/react-query";
import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchEnvelope, retainOkMeta } from "./api";

function jsonResponse(body: unknown, ok = true) {
  return { ok, json: async () => body };
}

describe("fetchEnvelope", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("returns a successful envelope and forwards the signal", async () => {
    const signal = new AbortController().signal;
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ ok: true, data: [1] }));
    vi.stubGlobal("fetch", fetchMock);
    expect(await fetchEnvelope("/api/x", signal)).toEqual({ ok: true, data: [1] });
    expect(fetchMock).toHaveBeenCalledWith("/api/x", { signal });
  });

  it("throws on HTTP failure, invalid JSON, and ok:false", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, json: async () => ({}) }));
    await expect(fetchEnvelope("/api/x")).rejects.toThrow("unavailable");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => { throw new Error("bad json"); } }));
    await expect(fetchEnvelope("/api/x")).rejects.toThrow("unavailable");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ ok: false, error: "nope" })));
    await expect(fetchEnvelope("/api/x")).rejects.toThrow("nope");
  });

  it("keeps last data across transport failure and ok:false, replaces with empty success, and does not leak across keys", async () => {
    const pending: Array<(value: unknown) => void> = [];
    vi.stubGlobal("fetch", vi.fn((url: string) => new Promise((resolve) => {
      pending.push(resolve);
      void url;
    })));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
    const observer = new QueryObserver(client, {
      queryKey: ["poll", "A"],
      queryFn: ({ signal }) => fetchEnvelope<number[]>("/api/A", signal),
      retry: false,
    });
    const unsub = observer.subscribe(() => {});
    try {
      await vi.waitFor(() => expect(pending.length).toBe(1));
      pending.shift()!(jsonResponse({ ok: true, data: [1] }));
      await vi.waitFor(() => expect(observer.getCurrentResult().data).toEqual({ ok: true, data: [1] }));
      const firstAt = observer.getCurrentResult().dataUpdatedAt;

      observer.refetch();
      await vi.waitFor(() => expect(pending.length).toBe(1));
      pending.shift()!({ ok: false, json: async () => ({}) });
      await vi.waitFor(() => expect(observer.getCurrentResult().isError).toBe(true));
      expect(observer.getCurrentResult().data).toEqual({ ok: true, data: [1] });
      expect(observer.getCurrentResult().dataUpdatedAt).toBe(firstAt);

      observer.refetch();
      await vi.waitFor(() => expect(pending.length).toBe(1));
      pending.shift()!(jsonResponse({ ok: false, error: "down" }));
      await vi.waitFor(() => expect(observer.getCurrentResult().error?.message).toBe("down"));
      expect(observer.getCurrentResult().data).toEqual({ ok: true, data: [1] });

      observer.refetch();
      await vi.waitFor(() => expect(pending.length).toBe(1));
      pending.shift()!(jsonResponse({ ok: true, data: [] }));
      await vi.waitFor(() => expect(observer.getCurrentResult().data).toEqual({ ok: true, data: [] }));
      expect(observer.getCurrentResult().isError).toBe(false);

      observer.setOptions({
        queryKey: ["poll", "B"],
        queryFn: ({ signal }) => fetchEnvelope<number[]>("/api/B", signal),
        retry: false,
      });
      await vi.waitFor(() => expect(pending.length).toBe(1));
      expect(observer.getCurrentResult().data).toBeUndefined();
      pending.shift()!(jsonResponse({ ok: true, data: [9] }));
      await vi.waitFor(() => expect(observer.getCurrentResult().data).toEqual({ ok: true, data: [9] }));
    } finally {
      unsub();
      client.clear();
    }
  });
});

describe("retainOkMeta", () => {
  it("keeps last valid section data and timestamp when the incoming section fails", () => {
    const prev = { data: { n: 1 }, updatedAt: 10 };
    expect(retainOkMeta(prev, { ok: true, data: { n: 2 } }, 20)).toEqual({
      data: { n: 2 }, error: undefined, updatedAt: 20,
    });
    expect(retainOkMeta(prev, { ok: false, error: "timeout" }, 20)).toEqual({
      data: { n: 1 }, error: "timeout", updatedAt: 10,
    });
    expect(retainOkMeta(undefined, { ok: false, error: "timeout" }, 20)).toEqual({
      data: undefined, error: "timeout", updatedAt: undefined,
    });
  });
});
