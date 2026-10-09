import { readdirSync, readFileSync, mkdtempSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type Database from "better-sqlite3";

let testDb: Database.Database;
vi.mock("../../../server/db", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../server/db")>();
  return { ...actual, getDb: () => testDb };
});

import { openDb } from "../../../server/db";
import { getRuleDocs, seedRuleDocsIfNeeded } from "../../../server/db/rules";
import { applyRuleFile, hashOrAbsent, sourceRevision } from "../../../server/collectors/rules";
import { GET } from "./route";
import { POST as canonicalPost } from "./canonical/route";
import { POST as exceptionPost } from "./exception/route";
import { POST as applyPost } from "./apply/route";
import type { RawRead } from "../../../server/collectors/rules";
import type { RuleDocs } from "../../../lib/rules";

const CLAUDE_RAW = "<!-- BEGIN JAX RULES -->\n## Jax OS\ncontent\n<!-- END JAX RULES -->\n";
const SEED: RuleDocs = { global: CLAUDE_RAW, claude: "", codex: "prefix\n", opencode: "" };

function seed() {
  seedRuleDocsIfNeeded(testDb, () => ({ starter: SEED, existing: { claudeRaw: null, codexRaw: null } }));
}

const request = (payload: unknown) =>
  new Request("http://127.0.0.1/api/rules", {
    method: "POST",
    body: JSON.stringify(payload),
    headers: { "content-type": "application/json" },
  });

const okRead: RawRead = { raw: "disk\n", buf: Buffer.from("disk\n") };

function getRequest(deps: { getDocs: () => RuleDocs; readRuleFile: (app: "claude" | "codex" | "opencode") => RawRead }) {
  const req = new Request("http://127.0.0.1/api/rules");
  Object.defineProperty(req, "jaxDeps", { value: deps });
  return req;
}

describe("rules GET sourceRevisions", () => {
  beforeEach(() => {
    testDb = openDb(":memory:");
    seed();
  });
  afterEach(() => testDb.close());

  it("exposes a SHA-256 per stored slot and keeps per-app disk isolation", async () => {
    const res = await GET(getRequest({ getDocs: () => SEED, readRuleFile: () => okRead }));
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.ok).toBe(true);
    expect(body.data.sourceRevisions).toEqual({
      global: sourceRevision(CLAUDE_RAW),
      claude: sourceRevision(""),
      codex: sourceRevision("prefix\n"),
      opencode: sourceRevision(""),
    });
    expect(body.data.apps.claude.diskHash).toBeTruthy();
  });
});

describe("rules canonical/exception revisions", () => {
  beforeEach(() => {
    testDb = openDb(":memory:");
    seed();
  });
  afterEach(() => testDb.close());

  it("canonical save with the matching revision returns new content + revision", async () => {
    const rev = sourceRevision(CLAUDE_RAW);
    const res = await canonicalPost(request({ content: "new canon\n\n", expectedRevision: rev }));
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body).toEqual({ ok: true, data: { content: "new canon\n", revision: sourceRevision("new canon\n") } });
  });

  it("a stale expectedRevision is a typed 409 and leaves the slot untouched", async () => {
    const rev = sourceRevision(CLAUDE_RAW);
    await canonicalPost(request({ content: "first\n", expectedRevision: rev }));
    const stale = await canonicalPost(request({ content: "second\n", expectedRevision: rev }));
    expect(stale.status).toBe(409);
    expect(await stale.json()).toEqual({
      ok: false,
      code: "stale-revision",
      error: "rule source changed since it was read",
    });
    expect(testDb.prepare("SELECT content FROM rule_docs WHERE slot = 'global'").get()).toEqual({ content: "first\n" });
  });

  it("exception save keeps the empty sentinel and returns the new revision", async () => {
    const emptyRev = sourceRevision("prefix\n");
    const set = await exceptionPost(request({ app: "codex", content: "exc\n\n", expectedRevision: emptyRev }));
    expect(set.status).toBe(200);
    expect((await set.json()).data).toEqual({ content: "exc\n", revision: sourceRevision("exc\n") });

    const clear = await exceptionPost(request({ app: "codex", content: "", expectedRevision: sourceRevision("exc\n") }));
    expect(clear.status).toBe(200);
    expect((await clear.json()).data).toEqual({ content: "", revision: sourceRevision("") });
  });

  it("rejects a malformed expectedRevision with 400 and writes nothing", async () => {
    for (const bad of ["", "abc", "A".repeat(64), "g".repeat(64), 42, undefined]) {
      const res = await canonicalPost(request({ content: "x\n", expectedRevision: bad }));
      expect(res.status).toBe(400);
      expect(await res.json()).toEqual({ ok: false, error: "invalid expectedRevision" });
    }
    expect(testDb.prepare("SELECT content FROM rule_docs WHERE slot = 'global'").get()).toEqual({ content: CLAUDE_RAW });
  });

  it("fails closed with 500 when the rule_docs table was never seeded", async () => {
    testDb.close();
    testDb = openDb(":memory:"); // fresh, unseeded
    const res = await canonicalPost(request({ content: "x\n", expectedRevision: "a".repeat(64) }));
    expect(res.status).toBe(500);
    expect((await res.json()).ok).toBe(false);
  });
});

describe("rules apply — temporary-root source/disk boundaries", () => {
  let root: string;
  let testRoot: string;

  beforeEach(() => {
    testDb = openDb(":memory:");
    seed();
    root = mkdtempSync(join(tmpdir(), "jax-rules-route-"));
  });
  afterEach(() => {
    testDb.close();
    rmSync(root, { recursive: true, force: true });
  });

  const AGENTS_ON = { ok: true, data: { integrations: { agents: { claude: true, codex: true, opencode: true } } } };

  function applyRequest(payload: unknown, opts: { settings?: unknown; apply?: unknown } = {}) {
    const req = new Request("http://127.0.0.1/api/rules/apply", {
      method: "POST",
      body: JSON.stringify(payload),
      headers: { "content-type": "application/json" },
    });
    Object.defineProperty(req, "jaxDeps", {
      value: {
        getDocs: (db: Database.Database) => getRuleDocs(db),
        apply: opts.apply ?? applyRuleFile,
        path: (app: string) => join(root, `${app}.md`),
        readSettings: () => opts.settings ?? AGENTS_ON,
      },
    });
    return req;
  }

  const target = () => join(root, "claude.md");
  const revisions = () => ({ global: sourceRevision(CLAUDE_RAW), exception: sourceRevision("") });

  it("applies the composed saved source and preserves a restrictive target mode", async () => {
    writeFileSync(target(), "old\n", { mode: 0o600 });
    const res = await applyPost(applyRequest({ app: "claude", diskHash: hashOrAbsent("old\n"), applySourceRevisions: revisions() }));
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.ok).toBe(true);
    expect(readFileSync(target(), "utf8")).toBe(CLAUDE_RAW);
    expect(statSync(target()).mode & 0o777).toBe(0o600);
    const backups = readdirSync(root).filter((n) => n.startsWith(".bak-jaxrules-"));
    expect(backups).toHaveLength(1);
    expect(statSync(join(root, backups[0])).mode & 0o777).toBe(0o600);
  });

  it("rejects a stale disk hash with 409 and leaves the file untouched", async () => {
    writeFileSync(target(), "old\n");
    const res = await applyPost(applyRequest({ app: "claude", diskHash: hashOrAbsent("different\n"), applySourceRevisions: revisions() }));
    expect(res.status).toBe(409);
    expect(await res.json()).toMatchObject({ ok: false, code: "mutation-rejected", error: "file changed on disk since diskHash was read" });
    expect(readFileSync(target(), "utf8")).toBe("old\n");
  });

  it("checks the selected exception slot: another app's exception revision is refused", async () => {
    writeFileSync(target(), "old\n");
    const res = await applyPost(applyRequest({
      app: "claude",
      diskHash: hashOrAbsent("old\n"),
      applySourceRevisions: { global: sourceRevision(CLAUDE_RAW), exception: sourceRevision("prefix\n") },
    }));
    expect(res.status).toBe(409);
    expect(await res.json()).toMatchObject({ error: "exception source changed since preview" });
    expect(readFileSync(target(), "utf8")).toBe("old\n");
  });

  it("source-save then apply with the pre-save revisions is a stale-source 409", async () => {
    const captured = revisions();
    writeFileSync(target(), "old\n");
    const saved = await canonicalPost(request({ content: "changed\n", expectedRevision: captured.global }));
    expect(saved.status).toBe(200);
    const res = await applyPost(applyRequest({ app: "claude", diskHash: hashOrAbsent("old\n"), applySourceRevisions: captured }));
    expect(res.status).toBe(409);
    expect(await res.json()).toMatchObject({ error: "canonical source changed since preview" });
    expect(readFileSync(target(), "utf8")).toBe("old\n");
  });

  it("an audit finalization failure reports applied + audit pending with the disk effect", async () => {
    writeFileSync(target(), "old\n");
    testDb.exec("CREATE TRIGGER fail_upd BEFORE UPDATE ON mutations BEGIN SELECT RAISE(FAIL, 'fixture'); END");
    const res = await applyPost(applyRequest({ app: "claude", diskHash: hashOrAbsent("old\n"), applySourceRevisions: revisions() }));
    expect(res.status).toBe(500);
    expect(await res.json()).toMatchObject({
      ok: false, code: "audit-finalization-failed", effect: "applied", audit: "pending",
    });
    expect(readFileSync(target(), "utf8")).toBe(CLAUDE_RAW);
  });

  it.each([
    ["a disabled app", { ok: true, data: { integrations: { agents: { claude: false, codex: true, opencode: true } } } }, "agent-disabled"],
    ["malformed settings", { ok: false, error: "settings-malformed" }, "settings-malformed"],
    ["unreadable settings", { ok: false, error: "settings-unreadable" }, "settings-unreadable"],
  ])("%s: refused before any mutation; the apply function is never called and nothing is written", async (_n, settings, error) => {
    writeFileSync(target(), "old\n");
    const apply = vi.fn();
    const rows = () => (testDb.prepare("SELECT COUNT(*) AS n FROM mutations").get() as { n: number }).n;
    const before = rows();
    const res = await applyPost(applyRequest({ app: "claude", diskHash: hashOrAbsent("old\n"), applySourceRevisions: revisions() }, { settings, apply }));
    expect(res.status).toBe(409);
    expect(await res.json()).toEqual({ ok: false, error });
    expect(apply).not.toHaveBeenCalled();
    expect(readFileSync(target(), "utf8")).toBe("old\n");
    expect(readdirSync(root).filter((n) => n.startsWith(".bak-jaxrules-"))).toHaveLength(0);
    expect(rows()).toBe(before);
  });

  it("an enabled app is applied even when another agent is off", async () => {
    writeFileSync(target(), "old\n");
    const settings = { ok: true, data: { integrations: { agents: { claude: true, codex: false, opencode: false } } } };
    const res = await applyPost(applyRequest({ app: "claude", diskHash: hashOrAbsent("old\n"), applySourceRevisions: revisions() }, { settings }));
    expect(res.status).toBe(200);
  });
});
