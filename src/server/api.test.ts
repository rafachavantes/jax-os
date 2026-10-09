import { describe, expect, it, vi } from "vitest";
import { readBodyCapped, readJsonCapped, requireSameOrigin } from "./api";

const post = (body: string, headers?: Record<string, string>) =>
  new Request("http://127.0.0.1/x", { method: "POST", body, headers });

const streamed = (stream: ReadableStream<Uint8Array>, headers?: Record<string, string>) =>
  new Request("http://127.0.0.1/x", {
    method: "POST",
    body: stream,
    headers,
    duplex: "half",
  } as RequestInit);

function reqWith(header: string | null): Request {
  const headers = new Headers();
  if (header !== null) headers.set("sec-fetch-site", header);
  return new Request("http://127.0.0.1:3100/api/workflow/events", { method: "POST", headers });
}

describe("requireSameOrigin", () => {
  it("accepts same-origin, none, or a missing header (loopback script callers send none)", () => {
    expect(requireSameOrigin(reqWith("same-origin"))).toBeNull();
    expect(requireSameOrigin(reqWith("none"))).toBeNull();
    expect(requireSameOrigin(reqWith(null))).toBeNull();
  });

  it("rejects cross-site AND same-site — the live tightening (Finding 8e)", async () => {
    for (const site of ["cross-site", "same-site"]) {
      const res = requireSameOrigin(reqWith(site));
      expect(res).not.toBeNull();
      expect(await res!.json()).toEqual({ ok: false, error: "cross-site request blocked" });
    }
  });
});

describe("readJsonCapped", () => {
  it("parses a small JSON body", async () => {
    expect(await readJsonCapped(post('{"a":1}'), 1000)).toEqual({ ok: true, value: { a: 1 } });
  });
  it("treats an empty body as {}", async () => {
    expect(await readJsonCapped(post(""), 1000)).toEqual({ ok: true, value: {} });
  });
  it("rejects invalid JSON without throwing", async () => {
    expect(await readJsonCapped(post("not json"), 1000)).toEqual({ ok: false, error: "invalid JSON" });
  });
  it("rejects an oversized body declared by content-length", async () => {
    expect(await readJsonCapped(post("x".repeat(50), { "content-length": "50" }), 10)).toEqual({
      ok: false,
      error: "request too large",
    });
  });
  it("rejects an oversized body while reading, even without content-length", async () => {
    const stream = new ReadableStream<Uint8Array>({
      start(c) {
        c.enqueue(new TextEncoder().encode("x".repeat(64)));
        c.close();
      },
    });
    const req = new Request("http://127.0.0.1/x", { method: "POST", body: stream, duplex: "half" } as RequestInit);
    expect(await readJsonCapped(req, 10)).toEqual({ ok: false, error: "request too large" });
  });
  it("parses JSON when a multibyte character is split across chunks", async () => {
    const bytes = new TextEncoder().encode('"é"');
    const stream = new ReadableStream<Uint8Array>({
      start(c) {
        c.enqueue(bytes.slice(0, 2));
        c.enqueue(bytes.slice(2));
        c.close();
      },
    });
    expect(await readJsonCapped(streamed(stream), 100)).toEqual({ ok: true, value: "é" });
  });
});

describe("readBodyCapped", () => {
  it("counts UTF-8 bytes, not characters", async () => {
    const req = new Request("http://127.0.0.1/x", { method: "POST", body: "é" });
    expect(await readBodyCapped(req, 1)).toEqual({ ok: false, error: "request too large" });
  });
  it("accepts exact bytes", async () => {
    const req = new Request("http://127.0.0.1/x", { method: "POST", body: "é" });
    const result = await readBodyCapped(req, 2);
    expect(result.ok && new TextDecoder().decode(result.value)).toBe("é");
  });
  it("returns empty bytes when the request has no body", async () => {
    const req = new Request("http://127.0.0.1/x", { method: "POST" });
    expect(await readBodyCapped(req, 10)).toEqual({ ok: true, value: new Uint8Array(0) });
  });
  it("does not getReader when Content-Length already exceeds the cap", async () => {
    const req = post("x".repeat(50), { "content-length": "50" });
    const spy = vi.spyOn(ReadableStream.prototype, "getReader");
    try {
      expect(await readBodyCapped(req, 10)).toEqual({ ok: false, error: "request too large" });
      expect(spy).not.toHaveBeenCalled();
    } finally {
      spy.mockRestore();
    }
  });
  it("cancels on the first over-cap chunk and does not read later chunks", async () => {
    let pulls = 0;
    let cancelled = false;
    const stream = new ReadableStream<Uint8Array>(
      {
        pull(controller) {
          pulls += 1;
          if (pulls === 1) controller.enqueue(new Uint8Array(8).fill(97));
          else if (pulls === 2) controller.enqueue(new Uint8Array(8).fill(98));
          else if (pulls === 3) controller.enqueue(new Uint8Array(8).fill(99));
          else controller.close();
        },
        cancel() {
          cancelled = true;
        },
      },
      { highWaterMark: 0 },
    );
    expect(await readBodyCapped(streamed(stream), 10)).toEqual({ ok: false, error: "request too large" });
    expect(cancelled).toBe(true);
    expect(pulls).toBe(2);
  });
  it("still caps a multi-chunk body when Content-Length is dishonestly small", async () => {
    let pulls = 0;
    let cancelled = false;
    const stream = new ReadableStream<Uint8Array>(
      {
        pull(controller) {
          pulls += 1;
          if (pulls === 1) controller.enqueue(new Uint8Array(8).fill(97));
          else if (pulls === 2) controller.enqueue(new Uint8Array(8).fill(98));
          else if (pulls === 3) controller.enqueue(new Uint8Array(8).fill(99));
          else controller.close();
        },
        cancel() {
          cancelled = true;
        },
      },
      { highWaterMark: 0 },
    );
    expect(await readBodyCapped(streamed(stream, { "content-length": "3" }), 10)).toEqual({
      ok: false,
      error: "request too large",
    });
    expect(cancelled).toBe(true);
    expect(pulls).toBe(2);
  });
  it("returns a contract error when a read rejects", async () => {
    const stream = new ReadableStream<Uint8Array>({
      pull() {
        return Promise.reject(new Error("boom"));
      },
    });
    expect(await readBodyCapped(streamed(stream), 100)).toEqual({
      ok: false,
      error: "could not read request body",
    });
  });
  it("still reports over-cap when cancellation rejects", async () => {
    const cancel = vi.fn(() => Promise.reject(new Error("nope")));
    const stream = new ReadableStream<Uint8Array>(
      {
        pull(controller) { controller.enqueue(new Uint8Array(64).fill(97)); },
        cancel,
      },
      { highWaterMark: 0 },
    );
    expect(await readBodyCapped(streamed(stream), 10)).toEqual({
      ok: false,
      error: "request too large",
    });
    expect(cancel).toHaveBeenCalledOnce();
  });
  it("returns a contract error when the body stream is already locked", async () => {
    const stream = new ReadableStream<Uint8Array>({
      start(c) {
        c.enqueue(new Uint8Array([1]));
        c.close();
      },
    });
    const req = streamed(stream);
    req.body!.getReader();
    expect(await readBodyCapped(req, 100)).toEqual({ ok: false, error: "could not read request body" });
  });
});
