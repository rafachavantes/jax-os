import { describe, expect, it, beforeAll, afterAll } from "vitest";
import { execFileSync } from "node:child_process";
import { mkdtempSync, mkdirSync, writeFileSync, symlinkSync, rmSync, readFileSync, truncateSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { isSecretName, planZipAt, zipStream, ZIP_ENTRY_CAP, ZIP_BYTES_CAP } from "./files";

async function collect(gen: AsyncGenerator<Uint8Array>): Promise<Buffer> {
  const chunks: Buffer[] = [];
  for await (const chunk of gen) chunks.push(Buffer.from(chunk));
  return Buffer.concat(chunks);
}

describe("isSecretName", () => {
  it("matches env/key/credential shapes case-insensitively and nothing else", () => {
    expect(isSecretName(".env")).toBe(true);
    expect(isSecretName(".env.local")).toBe(true);
    expect(isSecretName(".ENV")).toBe(true);
    expect(isSecretName("id_rsa")).toBe(true);
    expect(isSecretName("id_rsa.pub")).toBe(true);
    expect(isSecretName("id_ed25519")).toBe(true);
    expect(isSecretName("server.pem")).toBe(true);
    expect(isSecretName("a.key")).toBe(true);
    expect(isSecretName("box.p12")).toBe(true);
    expect(isSecretName("box.pfx")).toBe(true);
    expect(isSecretName("keystore.jks")).toBe(true);
    expect(isSecretName("credentials.json")).toBe(true);
    expect(isSecretName("secrets.yaml")).toBe(true);
    expect(isSecretName("keys")).toBe(false);
    expect(isSecretName("notes.md")).toBe(false);
  });
});

let base: string, outside: string;
beforeAll(() => {
  base = mkdtempSync(join(tmpdir(), "jax-zip-"));
  outside = mkdtempSync(join(tmpdir(), "jax-zip-outside-"));
  mkdirSync(join(base, "sub"));
  writeFileSync(join(base, "sub", "a.txt"), "hello");
  writeFileSync(join(base, "b.txt"), "world");
  mkdirSync(join(base, "secrets"));
  writeFileSync(join(base, "secrets", "x.txt"), "shh");
  writeFileSync(join(base, ".env"), "SECRET=1");
  writeFileSync(join(base, ".ENV"), "SECRET=2");
  mkdirSync(join(base, "keys"));
  writeFileSync(join(base, "keys", "ok.txt"), "not a secret dir");
  writeFileSync(join(outside, "escaped.txt"), "nope");
  symlinkSync(join(outside, "escaped.txt"), join(base, "escape.txt"));
});
afterAll(() => {
  rmSync(base, { recursive: true, force: true });
  rmSync(outside, { recursive: true, force: true });
});

describe("planZipAt", () => {
  it("walks a folder, skips a symlink escape, prunes secret names, and lazily counts them", async () => {
    const plan = await planZipAt(base, "");
    expect(plan.ok).toBe(true);
    if (!plan.ok) return;
    expect(plan.entries.map((e) => e.rel).sort()).toEqual(["b.txt", "keys/ok.txt", "sub/a.txt"]);
    // .env + .ENV (2 individually-matched files) + secrets/x.txt (1, counted via the pruned dir's
    // lazy metadata-only walk) = 3; escape.txt is skipped as a symlink, never counted either way.
    expect(plan.secretsExcluded).toBe(3);
  });

  it("refuses cleanly with zip-too-large over the entry cap, exactly-at-cap allowed", async () => {
    const big = mkdtempSync(join(tmpdir(), "jax-zip-cap-"));
    try {
      for (let i = 0; i < ZIP_ENTRY_CAP + 1; i++) writeFileSync(join(big, `f${i}`), "");
      expect(await planZipAt(big, "")).toEqual({ ok: false, error: "zip-too-large" });
      rmSync(join(big, `f${ZIP_ENTRY_CAP}`));
      const atCap = await planZipAt(big, "");
      expect(atCap.ok).toBe(true);
      if (atCap.ok) expect(atCap.entries.length).toBe(ZIP_ENTRY_CAP);
    } finally { rmSync(big, { recursive: true, force: true }); }
  }, 20_000);

  it("refuses cleanly with zip-too-large over the byte cap, exactly-at-cap allowed", async () => {
    const big = mkdtempSync(join(tmpdir(), "jax-zip-bytes-"));
    try {
      const p = join(big, "huge.bin");
      writeFileSync(p, "");
      truncateSync(p, ZIP_BYTES_CAP + 1);
      expect(await planZipAt(big, "")).toEqual({ ok: false, error: "zip-too-large" });
      truncateSync(p, ZIP_BYTES_CAP);
      expect((await planZipAt(big, "")).ok).toBe(true);
    } finally { rmSync(big, { recursive: true, force: true }); }
  });

  it("rejects a target that resolves to a file, not a directory", async () => {
    await expect(planZipAt(base, "b.txt")).rejects.toThrow("not a directory");
  });

  it("rejects a/../b, /abs, a\\b, a//b, and . before any fs access, but keeps allowing empty (round 1 plan review F2)", async () => {
    for (const bad of ["a/../b", "/abs", "a\\b", "a//b", "."]) {
      await expect(planZipAt(base, bad)).rejects.toThrow("outside allowlist");
    }
    await expect(planZipAt(base, "")).resolves.toMatchObject({ ok: true });
  });
});

// ponytail: no separate "zero entries" case — the round-trip test below already exercises
// zipStream's central-directory/EOCD writing at n=3; n=0 is the same code path, one less iteration.

describe("zipStream", () => {
  it("produces a structurally valid zip that unzip accepts, byte-identical per entry, excluding secrets", async () => {
    const plan = await planZipAt(base, "");
    if (!plan.ok) throw new Error("plan should have succeeded");
    const buf = await collect(zipStream(plan.entries));
    const outDir = mkdtempSync(join(tmpdir(), "jax-zip-check-"));
    const zipPath = join(outDir, "out.zip");
    writeFileSync(zipPath, buf);
    try {
      expect(() => execFileSync("unzip", ["-t", zipPath])).not.toThrow();
      const listing = execFileSync("unzip", ["-l", zipPath], { encoding: "utf8" });
      expect(listing).toContain("sub/a.txt");
      expect(listing).toContain("b.txt");
      expect(listing).toContain("keys/ok.txt");
      expect(listing).not.toContain(".env");
      expect(listing).not.toContain("secrets/");
      expect(execFileSync("unzip", ["-p", zipPath, "sub/a.txt"])).toEqual(readFileSync(join(base, "sub", "a.txt")));
      expect(execFileSync("unzip", ["-p", zipPath, "b.txt"])).toEqual(readFileSync(join(base, "b.txt")));
    } finally { rmSync(outDir, { recursive: true, force: true }); }
  });
});
