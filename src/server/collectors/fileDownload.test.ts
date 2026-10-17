import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { chmodSync, mkdtempSync, mkdirSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { tmpdir } from "node:os";
import { join } from "node:path";

const actualFs = await vi.importActual<typeof import("node:fs/promises")>("node:fs/promises");

vi.mock("node:fs/promises", async (importOriginal) => {
  const fs = await importOriginal<typeof import("node:fs/promises")>();
  return { ...fs, open: vi.fn(fs.open) };
});

import { open } from "node:fs/promises";
import { NextResponse } from "next/server";
import { contentDisposition, downloadFileAt, openRawStreamAt, parseRange, streamHandle } from "./fileDownload";

let base: string;

async function collect(stream: ReadableStream<Uint8Array>): Promise<Uint8Array> {
  const reader = stream.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    total += value.byteLength;
  }
  const out = new Uint8Array(total);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.byteLength; }
  return out;
}

beforeAll(() => {
  base = mkdtempSync(join(tmpdir(), "jax-dl-"));
  writeFileSync(join(base, "empty.bin"), "");
  writeFileSync(join(base, "hello.bin"), "hello");
  writeFileSync(join(base, "ro.bin"), "abc");
  chmodSync(join(base, "ro.bin"), 0o444);
  writeFileSync(join(base, "big.bin"), Buffer.alloc(70 * 1024, 9));
  writeFileSync(join(base, "文件.txt"), "utf8");
  writeFileSync(join(base, "🎉.txt"), "emoji");
  writeFileSync(join(base, 'a"b\\c.txt'), "quotes");
  writeFileSync(join(base, "a*.txt"), "star");
  writeFileSync(join(base, "bad\nname.txt"), "ctrl");
  mkdirSync(join(base, "dir"));
  writeFileSync(join(base, "target.bin"), "secret");
  symlinkSync(join(base, "target.bin"), join(base, "link.bin"));
  execFileSync("mkfifo", [join(base, "pipe")]);
});

afterAll(() => {
  chmodSync(join(base, "ro.bin"), 0o644);
  rmSync(base, { recursive: true, force: true });
});

beforeEach(() => {
  vi.mocked(open).mockImplementation(actualFs.open);
});

afterEach(() => {
  vi.mocked(open).mockImplementation(actualFs.open);
});

describe("downloadFileAt", () => {
  it("streams an empty regular file", async () => {
    const file = await downloadFileAt(join(base, "empty.bin"));
    expect(file.size).toBe(0);
    expect(await collect(file.body)).toEqual(new Uint8Array());
    expect(file.disposition).toContain("filename=\"empty.bin\"");
  });

  it("streams a read-only file's saved bytes", async () => {
    const file = await downloadFileAt(join(base, "ro.bin"));
    expect(Buffer.from(await collect(file.body)).toString()).toBe("abc");
    expect(file.size).toBe(3);
  });

  it("streams a large file in bounded chunks without extra growth", async () => {
    const file = await downloadFileAt(join(base, "big.bin"));
    expect(file.size).toBe(70 * 1024);
    expect((await collect(file.body)).byteLength).toBe(70 * 1024);
  });

  it("refuses missing, directory, FIFO, and symlink targets", async () => {
    await expect(downloadFileAt(join(base, "missing.bin"))).rejects.toThrow();
    await expect(downloadFileAt(join(base, "dir"))).rejects.toThrow();
    await expect(downloadFileAt(join(base, "pipe"))).rejects.toThrow();
    await expect(downloadFileAt(join(base, "link.bin"))).rejects.toThrow();
  });

  it("encodes unicode and special ASCII filenames", async () => {
    const unicode = await downloadFileAt(join(base, "文件.txt"));
    expect(unicode.disposition).toContain("filename*=UTF-8''");
    expect(unicode.disposition).toContain(encodeURIComponent("文件.txt"));
    expect(unicode.disposition).toMatch(/filename="[_]+.txt"/);
    expect(unicode.disposition).not.toContain("文件");
    const quoted = await downloadFileAt(join(base, 'a"b\\c.txt'));
    expect(quoted.disposition).toContain('filename="a_b_c.txt"');
    const star = await downloadFileAt(join(base, "a*.txt"));
    expect(star.disposition).toContain("%2A");
    await collect(unicode.body);
    await collect(quoted.body);
    await collect(star.body);
  });

  it("builds NextResponse headers for unicode and emoji names", async () => {
    const unicode = await downloadFileAt(join(base, "文件.txt"));
    const emoji = await downloadFileAt(join(base, "🎉.txt"));
    const unicodeRes = new NextResponse(unicode.body, { headers: { "Content-Disposition": unicode.disposition } });
    const emojiRes = new NextResponse(emoji.body, { headers: { "Content-Disposition": emoji.disposition } });
    expect(unicodeRes.headers.get("Content-Disposition")).toContain(encodeURIComponent("文件.txt"));
    expect(emojiRes.headers.get("Content-Disposition")).toContain(encodeURIComponent("🎉.txt"));
    expect(contentDisposition("文件.txt")).not.toMatch(/filename="[^"]*[^\x20-\x7e]/);
    await unicode.body.cancel();
    await emoji.body.cancel();
  });

  it("rejects a basename with control characters", async () => {
    await expect(downloadFileAt(join(base, "bad\nname.txt"))).rejects.toThrow();
  });

  it("closes the handle exactly once when the disposition throws (F1)", async () => {
    let closes = 0;
    vi.mocked(open).mockImplementation(async (path, flags) => {
      const handle = await actualFs.open(path, flags);
      const origClose = handle.close.bind(handle);
      handle.close = (async () => {
        closes++;
        return origClose();
      }) as typeof handle.close;
      return handle;
    });
    await expect(downloadFileAt(join(base, "bad\nname.txt"))).rejects.toThrow("invalid filename");
    expect(closes).toBe(1);
  });

  it("does not start reading until the stream is consumed, then closes", async () => {
    let reads = 0;
    vi.mocked(open).mockImplementation(async (path, flags) => {
      const handle = await actualFs.open(path, flags);
      const orig = handle.read.bind(handle);
      handle.read = ((...args: Parameters<typeof orig>) => {
        reads++;
        return orig(...args);
      }) as typeof handle.read;
      return handle;
    });
    const file = await downloadFileAt(join(base, "hello.bin"));
    expect(reads).toBe(0);
    expect(Buffer.from(await collect(file.body)).toString()).toBe("hello");
    expect(reads).toBeGreaterThan(0);
  });

  it("errors when a read returns zero before the captured size", async () => {
    vi.mocked(open).mockImplementation(async (path, flags) => {
      const handle = await actualFs.open(path, flags);
      handle.read = (async () => ({ bytesRead: 0, buffer: new Uint8Array() })) as typeof handle.read;
      return handle;
    });
    const file = await downloadFileAt(join(base, "hello.bin"));
    await expect(collect(file.body)).rejects.toThrow();
  });

  it("does not append bytes beyond the captured size", async () => {
    vi.mocked(open).mockImplementation(async (path, flags) => {
      const handle = await actualFs.open(path, flags);
      const origStat = handle.stat.bind(handle);
      handle.stat = (async () => {
        const s = await origStat();
        return Object.assign(Object.create(s), { size: 2 });
      }) as typeof handle.stat;
      return handle;
    });
    const file = await downloadFileAt(join(base, "hello.bin"));
    expect(file.size).toBe(2);
    expect(Buffer.from(await collect(file.body)).toString()).toBe("he");
  });

  it("cancels before pull and closes once", async () => {
    let closes = 0;
    vi.mocked(open).mockImplementation(async (path, flags) => {
      const handle = await actualFs.open(path, flags);
      const orig = handle.close.bind(handle);
      handle.close = (async () => {
        closes++;
        return orig();
      }) as typeof handle.close;
      return handle;
    });
    const file = await downloadFileAt(join(base, "hello.bin"));
    await file.body.cancel();
    expect(closes).toBe(1);
  });

  it("cancels during a pending read without enqueueing later bytes", async () => {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => { release = resolve; });
    let entered!: () => void;
    const started = new Promise<void>((resolve) => { entered = resolve; });
    let closes = 0;
    vi.mocked(open).mockImplementation(async (path, flags) => {
      const handle = await actualFs.open(path, flags);
      const origRead = handle.read.bind(handle);
      const origClose = handle.close.bind(handle);
      handle.read = (async (...args: Parameters<typeof origRead>) => {
        entered();
        await gate;
        return origRead(...args);
      }) as typeof handle.read;
      handle.close = (async () => {
        closes++;
        return origClose();
      }) as typeof handle.close;
      return handle;
    });
    const file = await downloadFileAt(join(base, "hello.bin"));
    const reader = file.body.getReader();
    const pending = reader.read();
    await started;
    const canceling = reader.cancel();
    release();
    await canceling;
    await pending.catch(() => {});
    expect(closes).toBe(1);
  });

  it("errors the stream on read failure and closes", async () => {
    let closes = 0;
    vi.mocked(open).mockImplementation(async (path, flags) => {
      const handle = await actualFs.open(path, flags);
      const origClose = handle.close.bind(handle);
      handle.read = (async () => { throw new Error("read failed"); }) as typeof handle.read;
      handle.close = (async () => {
        closes++;
        return origClose();
      }) as typeof handle.close;
      return handle;
    });
    const file = await downloadFileAt(join(base, "hello.bin"));
    await expect(collect(file.body)).rejects.toThrow("read failed");
    expect(closes).toBe(1);
  });

  it("closes once when close itself fails after a successful read", async () => {
    let closes = 0;
    vi.mocked(open).mockImplementation(async (path, flags) => {
      const handle = await actualFs.open(path, flags);
      const origClose = handle.close.bind(handle);
      handle.close = (async () => {
        closes++;
        await origClose();
        throw new Error("close failed");
      }) as typeof handle.close;
      return handle;
    });
    const file = await downloadFileAt(join(base, "hello.bin"));
    expect(Buffer.from(await collect(file.body)).toString()).toBe("hello");
    expect(closes).toBe(1);
  });
});

describe("parseRange", () => {
  it("returns null for no header (unchanged full response for every existing image caller)", () => {
    expect(parseRange(null, 1000)).toBeNull();
  });

  it("is unsatisfiable for an explicitly empty header, distinct from no header at all (cold review round 1 F3)", () => {
    expect(parseRange("", 1000)).toBe("unsatisfiable");
  });

  it("resolves a normal, an open-ended, and a suffix range", () => {
    expect(parseRange("bytes=0-499", 1000)).toEqual({ start: 0, end: 499 });
    expect(parseRange("bytes=500-", 1000)).toEqual({ start: 500, end: 999 });
    expect(parseRange("bytes=-100", 1000)).toEqual({ start: 900, end: 999 });
  });

  it("clamps an end past the file size to size-1", () => {
    expect(parseRange("bytes=0-999999", 1000)).toEqual({ start: 0, end: 999 });
  });

  it("is unsatisfiable for a malformed header, a start past size, start>end, or a non-numeric suffix", () => {
    for (const header of ["nonsense", "bytes=-", "bytes=1000-1", "bytes=2000-3000", "bytes=--5"]) {
      expect(parseRange(header, 1000)).toBe("unsatisfiable");
    }
  });

  it("allows a range that ends exactly at the last byte", () => {
    expect(parseRange("bytes=990-999", 1000)).toEqual({ start: 990, end: 999 });
  });
});

describe("streamHandle with a range", () => {
  it("streams only the requested byte span, byte-identical to a slice of the source", async () => {
    const p = join(base, "range-src.bin");
    writeFileSync(p, Buffer.from(Array.from({ length: 100 }, (_, i) => i)));
    const handle = await actualFs.open(p, "r");
    const bytes = await collect(streamHandle(handle, 100, { start: 10, end: 19 }));
    expect(Buffer.from(bytes)).toEqual(Buffer.from(Array.from({ length: 10 }, (_, i) => i + 10)));
  });

  it("still streams the whole file when no range is given (regression)", async () => {
    const handle = await actualFs.open(join(base, "hello.bin"), "r");
    expect(Buffer.from(await collect(streamHandle(handle, 5))).toString()).toBe("hello");
  });
});

describe("openRawStreamAt", () => {
  it("returns a handle and size within cap", async () => {
    const { handle, size } = await openRawStreamAt(join(base, "hello.bin"), 1024);
    expect(size).toBe(5);
    await handle.close();
  });

  it("succeeds when the source size is exactly equal to the cap, not just under it (cold review round 1 F5)", async () => {
    const exact = join(base, "exact-cap.bin");
    writeFileSync(exact, Buffer.alloc(5));
    const { handle, size } = await openRawStreamAt(exact, 5);
    expect(size).toBe(5);
    await handle.close();
  });

  it("closes the handle and throws 'too large' over cap, without the caller needing to close it", async () => {
    await expect(openRawStreamAt(join(base, "big.bin"), 1024)).rejects.toThrow("too large");
  });

  it("still refuses a symlink or a directory target exactly like downloadFileAt", async () => {
    await expect(openRawStreamAt(join(base, "link.bin"), 1024 * 1024)).rejects.toThrow();
    await expect(openRawStreamAt(join(base, "dir"), 1024 * 1024)).rejects.toThrow();
  });
});
