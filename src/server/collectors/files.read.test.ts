import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { chmodSync, mkdtempSync, mkdirSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { tmpdir } from "node:os";
import { join } from "node:path";

vi.mock("node:fs", async (importOriginal) => {
  const fs = await importOriginal<typeof import("node:fs")>();
  return {
    ...fs,
    openSync: vi.fn(fs.openSync),
    fstatSync: vi.fn(fs.fstatSync),
    readSync: vi.fn(fs.readSync),
    closeSync: vi.fn(fs.closeSync),
    statSync: vi.fn(fs.statSync),
    readFileSync: vi.fn(fs.readFileSync),
  };
});

import { closeSync, fstatSync, openSync, readFileSync, readSync, statSync } from "node:fs";
import {
  READ_CAP,
  UPLOAD_CAP,
  hashBytes,
  imageMime,
  readFileAt,
  readRawPreviewAt,
} from "./files";

let base: string;

beforeAll(() => {
  base = mkdtempSync(join(tmpdir(), "jax-read-"));
  writeFileSync(join(base, "note.md"), "# hi");
  writeFileSync(join(base, "bin"), Buffer.from([1, 0, 2]));
  writeFileSync(join(base, "empty.txt"), "");
  writeFileSync(join(base, "big.bin"), Buffer.alloc(READ_CAP + 1024, 7));
  writeFileSync(join(base, "preview.png"), Buffer.from("PNG"));
  writeFileSync(join(base, "huge.png"), Buffer.alloc(UPLOAD_CAP + 1, 9));
  mkdirSync(join(base, "dir"));
  writeFileSync(join(base, "target.txt"), "secret");
  symlinkSync(join(base, "target.txt"), join(base, "link.txt"));
  execFileSync("mkfifo", [join(base, "pipe")]);
  writeFileSync(join(base, "ro.txt"), "locked");
  chmodSync(join(base, "ro.txt"), 0o444);
});

afterAll(() => {
  chmodSync(join(base, "ro.txt"), 0o644);
  rmSync(base, { recursive: true, force: true });
});

beforeEach(() => {
  vi.mocked(openSync).mockClear();
  vi.mocked(fstatSync).mockClear();
  vi.mocked(readSync).mockClear();
  vi.mocked(closeSync).mockClear();
  vi.mocked(statSync).mockClear();
  vi.mocked(readFileSync).mockClear();
});

afterEach(async () => {
  const fs = await vi.importActual<typeof import("node:fs")>("node:fs");
  vi.mocked(openSync).mockImplementation(fs.openSync);
  vi.mocked(fstatSync).mockImplementation(fs.fstatSync);
  vi.mocked(readSync).mockImplementation(fs.readSync);
  vi.mocked(closeSync).mockImplementation(fs.closeSync);
  vi.mocked(statSync).mockImplementation(fs.statSync);
  vi.mocked(readFileSync).mockImplementation(fs.readFileSync);
});

describe("readFileAt", () => {
  it("reads text with a hash and reports size", () => {
    const r = readFileAt(join(base, "note.md"));
    expect(r).toEqual({
      binary: false,
      content: "# hi",
      hash: hashBytes(Buffer.from("# hi")),
      size: 4,
    });
  });

  it("flags binary content without exposing bytes", () => {
    const r = readFileAt(join(base, "bin"));
    expect(r.binary).toBe(true);
    expect(r.content).toBeNull();
    expect(r.hash).toBe(hashBytes(Buffer.from([1, 0, 2])));
  });

  it("returns binary metadata for an over-cap regular file without buffering it", () => {
    const r = readFileAt(join(base, "big.bin"));
    expect(r).toEqual({ binary: true, content: null, hash: "", size: READ_CAP + 1024 });
    expect(readFileSync).not.toHaveBeenCalled();
    expect(vi.mocked(readSync).mock.calls).toHaveLength(0);
  });

  it("reads an empty regular file", () => {
    expect(readFileAt(join(base, "empty.txt"))).toEqual({
      binary: false,
      content: "",
      hash: hashBytes(Buffer.alloc(0)),
      size: 0,
    });
  });

  it("reads a read-only regular file", () => {
    const r = readFileAt(join(base, "ro.txt"));
    expect(r.binary).toBe(false);
    expect(r.content).toBe("locked");
  });

  it("refuses a directory", () => {
    expect(() => readFileAt(join(base, "dir"))).toThrow();
  });

  it("refuses a symlink without following it", () => {
    expect(() => readFileAt(join(base, "link.txt"))).toThrow();
  });

  it("refuses a FIFO", () => {
    expect(() => readFileAt(join(base, "pipe"))).toThrow();
  });

  it("treats growth past the cap as binary metadata and still closes", () => {
    const fs = vi.mocked(fstatSync);
    const actual = fs.getMockImplementation()!;
    let checks = 0;
    fs.mockImplementation((fd) => {
      const stat = actual(fd);
      checks++;
      if (checks > 1) return Object.assign(Object.create(stat), { size: READ_CAP + 8 });
      return stat;
    });
    const r = readFileAt(join(base, "note.md"));
    expect(r).toEqual({ binary: true, content: null, hash: "", size: READ_CAP + 8 });
    expect(closeSync).toHaveBeenCalled();
  });

  it("closes the descriptor when open succeeds and fstat fails", () => {
    vi.mocked(fstatSync).mockImplementation(() => {
      throw new Error("fstat failed");
    });
    expect(() => readFileAt(join(base, "note.md"))).toThrow("fstat failed");
    expect(closeSync).toHaveBeenCalled();
  });
});

describe("readRawPreviewAt", () => {
  it("returns bounded image bytes", () => {
    expect(Buffer.from(readRawPreviewAt(join(base, "preview.png")))).toEqual(Buffer.from("PNG"));
  });

  it("refuses an over-cap image without using readFileSync", () => {
    expect(() => readRawPreviewAt(join(base, "huge.png"))).toThrow();
    expect(readFileSync).not.toHaveBeenCalled();
  });

  it("refuses directories, symlinks, and FIFOs", () => {
    expect(() => readRawPreviewAt(join(base, "dir"))).toThrow();
    expect(() => readRawPreviewAt(join(base, "link.txt"))).toThrow();
    expect(() => readRawPreviewAt(join(base, "pipe"))).toThrow();
  });

  it("refuses growth beyond the preview cap and closes", () => {
    const fs = vi.mocked(fstatSync);
    const actual = fs.getMockImplementation()!;
    let checks = 0;
    fs.mockImplementation((fd) => {
      const stat = actual(fd);
      checks++;
      if (checks > 1) return Object.assign(Object.create(stat), { size: UPLOAD_CAP + 2 });
      return stat;
    });
    expect(() => readRawPreviewAt(join(base, "preview.png"))).toThrow();
    expect(closeSync).toHaveBeenCalled();
  });
});

describe("imageMime", () => {
  it("still maps svg for the raw helper", () => {
    expect(imageMime("icon.SVG")).toBe("image/svg+xml");
  });
});
