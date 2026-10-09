import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { MutationRejected } from "../../lib/mutationOutcome";

const actual = await vi.importActual<typeof import("node:fs")>("node:fs");
const actualCrypto = await vi.importActual<typeof import("node:crypto")>("node:crypto");

vi.mock("node:fs", async (importOriginal) => {
  const fs = await importOriginal<typeof import("node:fs")>();
  return {
    ...fs,
    openSync: vi.fn(fs.openSync),
    closeSync: vi.fn(fs.closeSync),
    readSync: vi.fn(fs.readSync),
    writeFileSync: vi.fn(fs.writeFileSync),
    fsyncSync: vi.fn(fs.fsyncSync),
    fchmodSync: vi.fn(fs.fchmodSync),
    renameSync: vi.fn(fs.renameSync),
    unlinkSync: vi.fn(fs.unlinkSync),
  };
});

vi.mock("node:crypto", async (importOriginal) => {
  const crypto = await importOriginal<typeof import("node:crypto")>();
  return { ...crypto, randomUUID: vi.fn(crypto.randomUUID) };
});

import { closeSync, fchmodSync, fsyncSync, openSync, readSync, renameSync, unlinkSync, writeFileSync } from "node:fs";
import { randomUUID } from "node:crypto";
import { READ_CAP, hashBytes, writeFileAt } from "./files";

const liveFds = new Set<number>();

function expectRejected(fn: () => unknown, message: string) {
  let thrown: unknown;
  try { fn(); } catch (e) { thrown = e; }
  expect(thrown).toBeInstanceOf(MutationRejected);
  expect((thrown as Error).message).toBe(message);
  expect([...liveFds]).toEqual([]);
}

describe("writeFileAt atomic save", () => {
  let dir: string;
  let target: string;

  beforeEach(() => {
    dir = actual.realpathSync(actual.mkdtempSync(join(tmpdir(), "jax-atomic-")));
    target = join(dir, "target.txt");
    vi.mocked(writeFileSync).mockImplementation(actual.writeFileSync);
    vi.mocked(fsyncSync).mockImplementation(actual.fsyncSync);
    vi.mocked(fchmodSync).mockImplementation(actual.fchmodSync);
    vi.mocked(renameSync).mockImplementation(actual.renameSync);
    vi.mocked(unlinkSync).mockImplementation(actual.unlinkSync);
    vi.mocked(readSync).mockImplementation(actual.readSync);
    vi.mocked(randomUUID).mockImplementation(actualCrypto.randomUUID);
    liveFds.clear();
    vi.mocked(openSync).mockImplementation((path, flags, mode) => {
      const fd = mode === undefined ? actual.openSync(path, flags) : actual.openSync(path, flags, mode);
      liveFds.add(fd);
      return fd;
    });
    vi.mocked(closeSync).mockImplementation((fd) => {
      actual.closeSync(fd);
      liveFds.delete(fd);
    });
    vi.mocked(writeFileSync).mockClear();
    vi.mocked(fsyncSync).mockClear();
    vi.mocked(fchmodSync).mockClear();
    vi.mocked(renameSync).mockClear();
    vi.mocked(unlinkSync).mockClear();
    vi.mocked(readSync).mockClear();
    vi.mocked(randomUUID).mockClear();
    vi.mocked(openSync).mockClear();
    vi.mocked(closeSync).mockClear();
  });

  afterEach(() => {
    for (const fd of [...liveFds]) {
      try { actual.closeSync(fd); } catch { /* leftover from a failed assertion */ }
      liveFds.delete(fd);
    }
    actual.rmSync(dir, { recursive: true, force: true });
  });

  it("keeps the original complete when a temp write throws ENOSPC", () => {
    const before = Buffer.from("original content");
    actual.writeFileSync(target, before);
    vi.mocked(writeFileSync).mockImplementationOnce((file, _data, options) => {
      actual.writeFileSync(file, Buffer.from("partial"), options);
      throw Object.assign(new Error("fixture"), { code: "ENOSPC" });
    });
    expectRejected(
      () => writeFileAt(target, "replacement", hashBytes(before)),
      "atomic save failed; original file was not replaced",
    );
    expect(actual.readFileSync(target)).toEqual(before);
    expect(actual.readdirSync(dir)).toEqual(["target.txt"]);
  });

  it("changes destination inode while preserving bytes and mode", () => {
    const before = Buffer.from("original content");
    actual.writeFileSync(target, before);
    actual.chmodSync(target, 0o755);
    const beforeStat = actual.lstatSync(target);
    const replacement = "replacement";
    const result = writeFileAt(target, replacement, hashBytes(before));
    expect(result.hash).toBe(hashBytes(Buffer.from(replacement)));
    expect(actual.readFileSync(target).toString()).toBe(replacement);
    expect(actual.lstatSync(target).ino).not.toBe(beforeStat.ino);
    expect(actual.lstatSync(target).mode & 0o7777).toBe(0o755);
    expect(actual.readdirSync(dir)).toEqual(["target.txt"]);
    expect([...liveFds]).toEqual([]);
  });

  it("leaves original unchanged and no owned temp when fsync fails", () => {
    const before = Buffer.from("original content");
    actual.writeFileSync(target, before);
    vi.mocked(fsyncSync).mockImplementationOnce(() => {
      throw Object.assign(new Error("fixture"), { code: "EIO" });
    });
    expectRejected(
      () => writeFileAt(target, "replacement", hashBytes(before)),
      "atomic save failed; original file was not replaced",
    );
    expect(actual.readFileSync(target)).toEqual(before);
    expect(actual.readdirSync(dir)).toEqual(["target.txt"]);
  });

  it("saves a destination of exactly READ_CAP", () => {
    const before = Buffer.alloc(READ_CAP, 7);
    actual.writeFileSync(target, before);
    actual.chmodSync(target, 0o644);
    const result = writeFileAt(target, "ok", hashBytes(before));
    expect(result.hash).toBe(hashBytes(Buffer.from("ok")));
    expect(actual.readFileSync(target).toString()).toBe("ok");
    expect([...liveFds]).toEqual([]);
  });

  it("rejects READ_CAP+1 before snapshot buffering", () => {
    actual.writeFileSync(target, Buffer.alloc(READ_CAP + 1, 7));
    vi.mocked(readSync).mockClear();
    expectRejected(
      () => writeFileAt(target, "ok", "00".repeat(32)),
      "file too large",
    );
    expect(readSync).not.toHaveBeenCalled();
  });

  it("rejects growth beyond READ_CAP during read without unbounded requests", () => {
    const before = Buffer.from("small");
    actual.writeFileSync(target, before);
    let maxEnd = 0;
    const growRead = (
      fd: number,
      buffer: NodeJS.ArrayBufferView,
      offset: number,
      length: number,
      position: number,
    ): number => {
      maxEnd = Math.max(maxEnd, offset + length);
      actual.writeFileSync(target, Buffer.alloc(READ_CAP + 2, 1));
      return actual.readSync(fd, buffer, offset, length, position);
    };
    vi.mocked(readSync).mockImplementation(growRead as typeof readSync);
    expectRejected(
      () => writeFileAt(target, "n", hashBytes(before)),
      "file too large",
    );
    expect(maxEnd).toBeLessThanOrEqual(READ_CAP + 1);
  });

  it("refuses an initial directory", () => {
    expectRejected(() => writeFileAt(dir, "x", "00".repeat(32)), "not a regular file");
  });

  it("refuses a last-path symlink without changing its target", () => {
    const real = join(dir, "real.txt");
    actual.writeFileSync(real, "keep");
    actual.symlinkSync(real, target);
    expectRejected(
      () => writeFileAt(target, "x", hashBytes(Buffer.from("keep"))),
      "changed on disk",
    );
    expect(actual.readFileSync(real).toString()).toBe("keep");
  });

  it("replaces with empty content and matching hash", () => {
    const before = Buffer.from("original content");
    actual.writeFileSync(target, before);
    actual.chmodSync(target, 0o644);
    const result = writeFileAt(target, "", hashBytes(before));
    expect(result.hash).toBe(hashBytes(Buffer.alloc(0)));
    expect(actual.readFileSync(target).length).toBe(0);
    expect(actual.lstatSync(target).mode & 0o7777).toBe(0o644);
    expect(actual.readdirSync(dir)).toEqual(["target.txt"]);
    expect([...liveFds]).toEqual([]);
  });

  it("rejects a stale initial hash without writing a temp", () => {
    const before = Buffer.from("original content");
    actual.writeFileSync(target, before);
    expectRejected(() => writeFileAt(target, "n", "00".repeat(32)), "changed on disk");
    expect(actual.readFileSync(target)).toEqual(before);
    expect(actual.readdirSync(dir)).toEqual(["target.txt"]);
    expect(renameSync).not.toHaveBeenCalled();
  });

  it("preserves original when fchmod fails", () => {
    const before = Buffer.from("original content");
    actual.writeFileSync(target, before);
    vi.mocked(fchmodSync).mockImplementationOnce(() => {
      throw Object.assign(new Error("fixture"), { code: "EPERM" });
    });
    expectRejected(
      () => writeFileAt(target, "replacement", hashBytes(before)),
      "atomic save failed; original file was not replaced",
    );
    expect(actual.readFileSync(target)).toEqual(before);
    expect(actual.readdirSync(dir)).toEqual(["target.txt"]);
  });

  it("preserves original when rename fails", () => {
    const before = Buffer.from("original content");
    actual.writeFileSync(target, before);
    const ino = actual.lstatSync(target).ino;
    vi.mocked(renameSync).mockImplementationOnce(() => {
      throw Object.assign(new Error("fixture"), { code: "EXDEV" });
    });
    expectRejected(
      () => writeFileAt(target, "replacement", hashBytes(before)),
      "atomic save failed; original file was not replaced",
    );
    expect(actual.readFileSync(target)).toEqual(before);
    expect(actual.lstatSync(target).ino).toBe(ino);
    expect(actual.readdirSync(dir)).toEqual(["target.txt"]);
  });

  it("keeps externally edited bytes when recheck sees a change", () => {
    const before = Buffer.from("original content");
    actual.writeFileSync(target, before);
    vi.mocked(fsyncSync).mockImplementationOnce((fd) => {
      actual.fsyncSync(fd);
      actual.writeFileSync(target, "external");
    });
    expectRejected(
      () => writeFileAt(target, "replacement", hashBytes(before)),
      "changed on disk",
    );
    expect(actual.readFileSync(target).toString()).toBe("external");
    expect(actual.readdirSync(dir)).toEqual(["target.txt"]);
  });

  it("refuses a same-bytes replacement that changed inode", () => {
    const before = Buffer.from("original content");
    actual.writeFileSync(target, before);
    const ino = actual.lstatSync(target).ino;
    vi.mocked(fsyncSync).mockImplementationOnce((fd) => {
      actual.fsyncSync(fd);
      // Write the twin before the old file goes away: unlink-then-create lets ext4 reuse
      // the freed inode number, which made this pass for the wrong reason on CI.
      const twin = join(dir, "twin.tmp");
      actual.writeFileSync(twin, before);
      actual.renameSync(twin, target);
    });
    expectRejected(
      () => writeFileAt(target, "replacement", hashBytes(before)),
      "changed on disk",
    );
    expect(actual.readFileSync(target)).toEqual(before);
    expect(actual.lstatSync(target).ino).not.toBe(ino);
    expect(actual.readdirSync(dir)).toEqual(["target.txt"]);
  });

  it("refuses a replaced parent and leaves the outside sentinel untouched", () => {
    const sub = join(dir, "sub");
    actual.mkdirSync(sub);
    target = join(sub, "target.txt");
    const sentinel = join(dir, "sentinel.txt");
    const before = Buffer.from("original content");
    actual.writeFileSync(sentinel, "outside");
    actual.writeFileSync(target, before);
    vi.mocked(fsyncSync).mockImplementationOnce((fd) => {
      actual.fsyncSync(fd);
      actual.renameSync(sub, join(dir, "sub.moved"));
      actual.mkdirSync(sub);
    });
      expectRejected(
        () => writeFileAt(target, "replacement", hashBytes(before)),
        "changed on disk",
      );
      expect(actual.readFileSync(join(dir, "sub.moved", "target.txt"))).toEqual(before);
      expect(actual.readFileSync(sentinel).toString()).toBe("outside");
      expect(actual.readdirSync(join(dir, "sub.moved")).filter((n) => n.startsWith(".jax-save-"))).toEqual([]);
  });

  it("refuses a destination that became a symlink and leaves the sentinel untouched", () => {
    const sentinel = join(dir, "sentinel.txt");
    const before = Buffer.from("original content");
    actual.writeFileSync(sentinel, "outside");
    actual.writeFileSync(target, before);
    vi.mocked(fsyncSync).mockImplementationOnce((fd) => {
      actual.fsyncSync(fd);
      actual.unlinkSync(target);
      actual.symlinkSync(sentinel, target);
    });
    expectRejected(
      () => writeFileAt(target, "replacement", hashBytes(before)),
      "changed on disk",
    );
    expect(actual.readFileSync(sentinel).toString()).toBe("outside");
  });

  it("does not unlink an unowned temp UUID collision", () => {
    const before = Buffer.from("original content");
    actual.writeFileSync(target, before);
    const uuid = "11111111-1111-1111-1111-111111111111";
    const collision = join(dir, `.jax-save-${uuid}`);
    actual.writeFileSync(collision, "not ours");
    vi.mocked(randomUUID).mockReturnValue(uuid);
    expectRejected(
      () => writeFileAt(target, "replacement", hashBytes(before)),
      "atomic save failed; original file was not replaced",
    );
    expect(actual.readFileSync(collision).toString()).toBe("not ours");
    expect(actual.readFileSync(target)).toEqual(before);
    expect(unlinkSync).not.toHaveBeenCalled();
  });

  it("does not follow or unlink an unowned colliding symlink", () => {
    const before = Buffer.from("original content");
    actual.writeFileSync(target, before);
    const sentinel = join(dir, "sentinel.txt");
    actual.writeFileSync(sentinel, "outside");
    const uuid = "22222222-2222-2222-2222-222222222222";
    actual.symlinkSync(sentinel, join(dir, `.jax-save-${uuid}`));
    vi.mocked(randomUUID).mockReturnValue(uuid);
    expectRejected(
      () => writeFileAt(target, "replacement", hashBytes(before)),
      "atomic save failed; original file was not replaced",
    );
    expect(actual.readFileSync(sentinel).toString()).toBe("outside");
    expect(actual.readFileSync(target)).toEqual(before);
    expect(unlinkSync).not.toHaveBeenCalled();
  });

  it("surfaces owned-temp unlink failure without claiming success", () => {
    const before = Buffer.from("original content");
    actual.writeFileSync(target, before);
    vi.mocked(fsyncSync).mockImplementationOnce(() => {
      throw Object.assign(new Error("fixture"), { code: "EIO" });
    });
    vi.mocked(unlinkSync).mockImplementationOnce(() => {
      throw Object.assign(new Error("fixture"), { code: "EPERM" });
    });
    expectRejected(
      () => writeFileAt(target, "replacement", hashBytes(before)),
      "atomic save failed; original file was not replaced; temporary cleanup failed",
    );
    expect(actual.readFileSync(target)).toEqual(before);
  });

  it("refuses a read-only original without replacing it", () => {
    const before = Buffer.from("original content");
    actual.writeFileSync(target, before);
    actual.chmodSync(target, 0o444);
    expectRejected(
      () => writeFileAt(target, "replacement", hashBytes(before)),
      "atomic save failed; original file was not replaced",
    );
    expect(actual.readFileSync(target)).toEqual(before);
  });

  it("maps a destination that became over-cap during recheck to changed on disk", () => {
    const before = Buffer.from("original content");
    actual.writeFileSync(target, before);
    vi.mocked(fsyncSync).mockImplementationOnce((fd) => {
      actual.fsyncSync(fd);
      actual.writeFileSync(target, Buffer.alloc(READ_CAP + 1, 9));
    });
    expectRejected(
      () => writeFileAt(target, "replacement", hashBytes(before)),
      "changed on disk",
    );
  });

    it("maps a destination that became a directory during recheck to changed on disk", () => {
      const before = Buffer.from("original content");
      actual.writeFileSync(target, before);
      vi.mocked(fsyncSync).mockImplementationOnce((fd) => {
        actual.fsyncSync(fd);
        actual.unlinkSync(target);
        actual.mkdirSync(target);
      });
      expectRejected(
        () => writeFileAt(target, "replacement", hashBytes(before)),
        "changed on disk",
      );
    });

    it("does not unlink a same-named sentinel through a replaced parent symlink", () => {
      const parentDir = join(dir, "parent");
      const held = join(dir, "held");
      const other = join(dir, "other");
      actual.mkdirSync(parentDir);
      actual.mkdirSync(other);
      target = join(parentDir, "target.txt");
      const before = Buffer.from("old");
      actual.writeFileSync(target, before);
      const uuid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";
      const sentinel = join(other, `.jax-save-${uuid}`);
      vi.mocked(randomUUID).mockReturnValue(uuid);
      vi.mocked(fsyncSync).mockImplementationOnce((fd) => {
        actual.fsyncSync(fd);
        actual.renameSync(parentDir, held);
        actual.symlinkSync(other, parentDir);
        actual.writeFileSync(sentinel, "unrelated-sentinel");
      });
      expectRejected(
        () => writeFileAt(target, "replacement", hashBytes(before)),
        "changed on disk",
      );
      expect(actual.readFileSync(join(held, "target.txt"))).toEqual(before);
      expect(actual.readFileSync(sentinel).toString()).toBe("unrelated-sentinel");
      expect(actual.readdirSync(held).filter((n) => n.startsWith(".jax-save-"))).toEqual([]);
    });

    it("does not unlink a same-named sentinel through a replaced parent directory", () => {
      const parentDir = join(dir, "parent");
      const held = join(dir, "held");
      actual.mkdirSync(parentDir);
      target = join(parentDir, "target.txt");
      const before = Buffer.from("old");
      actual.writeFileSync(target, before);
      const uuid = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb";
      const sentinel = join(parentDir, `.jax-save-${uuid}`);
      vi.mocked(randomUUID).mockReturnValue(uuid);
      vi.mocked(fsyncSync).mockImplementationOnce((fd) => {
        actual.fsyncSync(fd);
        actual.renameSync(parentDir, held);
        actual.mkdirSync(parentDir);
        actual.writeFileSync(sentinel, "unrelated-sentinel");
      });
      expectRejected(
        () => writeFileAt(target, "replacement", hashBytes(before)),
        "changed on disk",
      );
      expect(actual.readFileSync(join(held, "target.txt"))).toEqual(before);
      expect(actual.readFileSync(sentinel).toString()).toBe("unrelated-sentinel");
      expect(actual.readdirSync(held).filter((n) => n.startsWith(".jax-save-"))).toEqual([]);
    });

    it("does not unlink an owned temp replaced by another inode before abort", () => {
      const before = Buffer.from("original content");
      actual.writeFileSync(target, before);
      const uuid = "cccccccc-cccc-cccc-cccc-cccccccccccc";
      const substitute = join(dir, `.jax-save-${uuid}`);
      vi.mocked(randomUUID).mockReturnValue(uuid);
      vi.mocked(fsyncSync).mockImplementationOnce((fd) => {
        actual.fsyncSync(fd);
        actual.unlinkSync(substitute);
        actual.writeFileSync(substitute, "not ours");
        throw Object.assign(new Error("fixture"), { code: "EIO" });
      });
      expectRejected(
        () => writeFileAt(target, "replacement", hashBytes(before)),
        "atomic save failed; original file was not replaced; temporary cleanup failed",
      );
      expect(actual.readFileSync(target)).toEqual(before);
      expect(actual.readFileSync(substitute).toString()).toBe("not ours");
    });

    it("does not rename an owned temp replaced before commit", () => {
      const before = Buffer.from("original content");
      actual.writeFileSync(target, before);
      const uuid = "dddddddd-dddd-dddd-dddd-dddddddddddd";
      const substitute = join(dir, `.jax-save-${uuid}`);
      vi.mocked(randomUUID).mockReturnValue(uuid);
      vi.mocked(fsyncSync).mockImplementationOnce((fd) => {
        actual.fsyncSync(fd);
        actual.unlinkSync(substitute);
        actual.writeFileSync(substitute, "not ours");
      });
      expectRejected(
        () => writeFileAt(target, "replacement", hashBytes(before)),
        "atomic save failed; original file was not replaced; temporary cleanup failed",
      );
      expect(actual.readFileSync(target)).toEqual(before);
      expect(actual.readFileSync(substitute).toString()).toBe("not ours");
      expect(renameSync).not.toHaveBeenCalled();
    });

    it("rejects when the proc descriptor anchor cannot be opened", () => {
      const before = Buffer.from("original content");
      actual.writeFileSync(target, before);
      vi.mocked(openSync).mockImplementation((path, flags, mode) => {
        if (typeof path === "string" && path.includes("/proc/self/fd/")) {
          throw Object.assign(new Error("fixture"), { code: "ENOENT" });
        }
        const fd = mode === undefined ? actual.openSync(path, flags) : actual.openSync(path, flags, mode);
        liveFds.add(fd);
        return fd;
      });
      expectRejected(
        () => writeFileAt(target, "replacement", hashBytes(before)),
        "atomic save failed; original file was not replaced",
      );
      expect(actual.readFileSync(target)).toEqual(before);
      expect(actual.readdirSync(dir)).toEqual(["target.txt"]);
    });

    it("rejects when the parent directory cannot be opened", () => {
      const before = Buffer.from("original content");
      actual.writeFileSync(target, before);
      vi.mocked(openSync).mockImplementation((path, flags, mode) => {
        if (typeof flags === "number" && (flags & actual.constants.O_DIRECTORY)) {
          throw Object.assign(new Error("fixture"), { code: "EACCES" });
        }
        const fd = mode === undefined ? actual.openSync(path, flags) : actual.openSync(path, flags, mode);
        liveFds.add(fd);
        return fd;
      });
      expectRejected(
        () => writeFileAt(target, "replacement", hashBytes(before)),
        "atomic save failed; original file was not replaced",
      );
      expect(actual.readFileSync(target)).toEqual(before);
      expect(actual.readdirSync(dir)).toEqual(["target.txt"]);
    });

    it("returns the committed hash if parent descriptor close fails after success", () => {
      const before = Buffer.from("original content");
      actual.writeFileSync(target, before);
      const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
      let parentFd: number | undefined;
      const parentCloses: number[] = [];
      vi.mocked(openSync).mockImplementation((path, flags, mode) => {
        const fd = mode === undefined ? actual.openSync(path, flags) : actual.openSync(path, flags, mode);
        liveFds.add(fd);
        if (typeof flags === "number" && (flags & actual.constants.O_DIRECTORY)) parentFd = fd;
        return fd;
      });
      vi.mocked(closeSync).mockImplementation((fd) => {
        if (parentFd !== undefined && fd === parentFd) {
          parentCloses.push(fd);
          liveFds.delete(fd);
          actual.closeSync(fd);
          throw Object.assign(new Error("fixture"), { code: "EIO" });
        }
        actual.closeSync(fd);
        liveFds.delete(fd);
      });
      const result = writeFileAt(target, "replacement", hashBytes(before));
      expect(result.hash).toBe(hashBytes(Buffer.from("replacement")));
      expect(actual.readFileSync(target).toString()).toBe("replacement");
      expect(parentCloses).toEqual([parentFd]);
      expect(errorSpy).toHaveBeenCalledWith("[files] atomic save completed; parent descriptor close failed");
      expect([...liveFds]).toEqual([]);
      errorSpy.mockRestore();
    });
  });
