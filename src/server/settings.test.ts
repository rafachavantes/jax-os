import { afterEach, describe, expect, it, vi } from "vitest";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { tmpdir } from "node:os";
import { join } from "node:path";
import RawDatabase from "better-sqlite3";
import { openDb } from "./db";
import {
  parseSettings,
  readGeneralSettings,
  seedAgentsIfFirstRun,
  writeGeneralSettings,
  SettingsAuditUnavailableError,
  SettingsAuditFailedError,
  type GeneralSettings,
} from "./settings";

const FIXTURES_DIR = join(fileURLToPath(new URL(".", import.meta.url)), "../../workflow/fixtures");
const FIXTURE = join(FIXTURES_DIR, "general-settings-v1.json");
const FIXTURE_V2 = join(FIXTURES_DIR, "general-settings-v2.json");
const NULL_FIXTURE = join(FIXTURES_DIR, "general-settings-null.json");
const DEFAULTS = (parseSettings(undefined) as { ok: true; data: GeneralSettings }).data;

const EXPECTED: GeneralSettings = {
  ownerName: "Owner", locale: "pt-BR", reposRoot: "/home/rafa/repos",
  vaultPath: "/home/rafa/obsidian-vault",
  monitoredUnits: { user: ["hermes-gateway.service"], system: ["tailscaled.service"] },
  integrations: {
    linear: true, ttyd: false, webhook: true, hermesTokens: false,
    agents: { claude: true, codex: false, opencode: false },
    classifier: true, github: false, vault: false,
  },
};

const EXPECTED_V2: GeneralSettings = {
  ...EXPECTED,
  integrations: { ...EXPECTED.integrations, agents: { claude: true, codex: true, opencode: false } },
};

describe("parseSettings (spec schema table + Object handling + Error behaviour)", () => {
  it("the shared fixture parses to the exact expected object (parity with Python's own assertion)", () => {
    expect(parseSettings(JSON.parse(readFileSync(FIXTURE, "utf8")))).toEqual({ ok: true, data: EXPECTED });
  });

  it("file absent (undefined input) -> ok:true with every default", () => {
    expect(DEFAULTS).toEqual({
      ownerName: "", locale: "en-US", reposRoot: expect.stringContaining("repos"), vaultPath: null,
      monitoredUnits: { user: [], system: [] },
      integrations: {
        linear: false, ttyd: false, webhook: false, hermesTokens: false,
        agents: { claude: false, codex: false, opencode: false },
        classifier: false, github: false, vault: false,
      },
    });
  });

  it("a partial nested object defaults its missing child independently (not malformed)", () => {
    const result = parseSettings({ monitoredUnits: { user: ["a.service"] } });
    expect(result).toEqual({ ok: true, data: { ...DEFAULTS, monitoredUnits: { user: ["a.service"], system: [] } } });
  });

  it.each<[string, unknown]>([
    ["not an object", "nope"],
    ["an array", [1, 2]],
    ["wrong-shape locale", { locale: "fr-FR" }],
    ["unknown top-level key", { foo: 1 }],
    ["unknown key inside monitoredUnits", { monitoredUnits: { user: [], system: [], extra: [] } }],
    ["unknown key inside integrations.subscriptions", { integrations: { subscriptions: { claude: true, codex: false, opencodeGo: false, extra: true } } }],
    ["null for ownerName (only vaultPath allows null)", { ownerName: null }],
    ["null for locale (only vaultPath allows null)", { locale: null }],
    ["explicit null for monitoredUnits (round 2 F1 parity)", { monitoredUnits: null }],
    ["explicit null for integrations (round 2 F1 parity)", { integrations: null }],
    ["explicit null for integrations.subscriptions (round 2 F1 parity)", { integrations: { subscriptions: null } }],
    ["unknown key inside integrations.agents", { integrations: { agents: { claude: true, extra: true } } }],
    ["explicit null for integrations.agents", { integrations: { agents: null } }],
    ["non-boolean agent value", { integrations: { agents: { claude: "yes" } } }],
  ])("%s -> settings-malformed", (_name, raw) => {
    expect(parseSettings(raw)).toEqual({ ok: false, error: "settings-malformed" });
  });

  it("null IS valid for vaultPath", () => {
    expect(parseSettings({ vaultPath: null }).ok).toBe(true);
  });

  it("v1 fixture (legacy `subscriptions`) migrates into agents; the parsed result never carries `subscriptions` (MOA-504 D2/D3)", () => {
    const parsed = parseSettings(JSON.parse(readFileSync(FIXTURE, "utf8")));
    expect(parsed).toEqual({ ok: true, data: EXPECTED });
    expect(JSON.stringify(parsed)).not.toContain("subscriptions");
  });

  it("v2 fixture (`agents`) parses to the exact expected object (parity with Python)", () => {
    expect(parseSettings(JSON.parse(readFileSync(FIXTURE_V2, "utf8")))).toEqual({ ok: true, data: EXPECTED_V2 });
  });

  it("legacy opencodeGo maps to agents.opencode; claude/codex map by name", () => {
    const r = parseSettings({ integrations: { subscriptions: { opencodeGo: true } } });
    expect(r.ok && r.data.integrations.agents).toEqual({ claude: false, codex: false, opencode: true });
  });

  it("both keys present: agents wins and subscriptions is ignored", () => {
    const r = parseSettings({ integrations: {
      agents: { codex: true },
      subscriptions: { claude: true, codex: false, opencodeGo: true },
    } });
    expect(r.ok && r.data.integrations.agents).toEqual({ claude: false, codex: true, opencode: false });
  });

  it("each agent key defaults to false independently", () => {
    const r = parseSettings({ integrations: { agents: { claude: true } } });
    expect(r.ok && r.data.integrations.agents).toEqual({ claude: true, codex: false, opencode: false });
  });
});

describe("readGeneralSettings", () => {
  afterEach(() => vi.restoreAllMocks());

  it("a missing file returns ok:true with defaults", () => {
    expect(readGeneralSettings("/tmp/moa-497-does-not-exist/settings.json")).toEqual({ ok: true, data: DEFAULTS });
  });

  it("malformed JSON on disk -> settings-malformed", () => {
    const dir = mkdtempSync(join(tmpdir(), "gs-"));
    const path = join(dir, "settings.json");
    writeFileSync(path, "{not valid json");
    expect(readGeneralSettings(path)).toEqual({ ok: false, error: "settings-malformed" });
  });

  it("a path that is a directory, not a file -> settings-unreadable", () => {
    const dir = mkdtempSync(join(tmpdir(), "gs-"));
    const dirPath = join(dir, "settings.json");
    mkdirSync(dirPath);
    expect(readGeneralSettings(dirPath)).toEqual({ ok: false, error: "settings-unreadable" });
  });

  it("F1: a non-ENOENT stat failure (ENOTDIR: a path component is a file, not a directory) -> settings-unreadable, NOT defaults", () => {
    const dir = mkdtempSync(join(tmpdir(), "gs-"));
    const notADir = join(dir, "notadir");
    writeFileSync(notADir, "");
    const path = join(notADir, "settings.json"); // statSync throws ENOTDIR, not ENOENT
    expect(readGeneralSettings(path)).toEqual({ ok: false, error: "settings-unreadable" });
  });

  it("the fixture round-trips to the same expected object", () => {
    expect(readGeneralSettings(FIXTURE)).toEqual({ ok: true, data: EXPECTED });
  });

  it("round 1 F3: a file whose JSON is top-level null -> settings-malformed, NOT defaults (only an absent file yields defaults)", () => {
    expect(readGeneralSettings(NULL_FIXTURE)).toEqual({ ok: false, error: "settings-malformed" });
  });
});

describe("writeGeneralSettings — atomic write + mutations audit (spec Write path)", () => {
  it("writes a unique-per-call tmp file then renames it over the target (mirrors workflow-callbacks.ts's writeAnswerLine; round 1 F4)", () => {
    const calls: string[] = [];
    const fs = {
      mkdirSync: vi.fn(),
      writeFileSync: vi.fn((p: string) => calls.push(`write ${p}`)),
      renameSync: vi.fn((from: string, to: string) => calls.push(`rename ${from} -> ${to}`)),
    };
    const db = openDb(":memory:");
    writeGeneralSettings(db, EXPECTED, "/tmp/x/settings.json", fs as never);
    db.close();
    expect(calls).toHaveLength(2);
    expect(calls[0]).toMatch(/^write \/tmp\/x\/settings\.json\.[0-9a-f]{12}\.tmp$/);
    expect(calls[1]).toMatch(/^rename \/tmp\/x\/settings\.json\.[0-9a-f]{12}\.tmp -> \/tmp\/x\/settings\.json$/);
  });

  it("round-trip: write then read returns the same object; exactly one mutations row; no leftover .tmp file", () => {
    const dir = mkdtempSync(join(tmpdir(), "gs-"));
    const path = join(dir, "settings.json");
    const db = openDb(":memory:");
    writeGeneralSettings(db, EXPECTED, path);
    expect(readGeneralSettings(path)).toEqual({ ok: true, data: EXPECTED });
    expect(readdirSync(dir).some((f) => f.endsWith(".tmp"))).toBe(false);
    const rows = db.prepare("SELECT kind, ok, payload FROM mutations WHERE kind = 'general-settings-write'").all() as { kind: string; ok: number; payload: string }[];
    expect(rows).toHaveLength(1);
    expect(rows[0].ok).toBe(1);
    expect(JSON.parse(rows[0].payload).changed).toEqual(Object.keys(EXPECTED));
    db.close();
  });

  it("round 1 F1: an unavailable audit store (no mutations table) fails the write with a named error, BEFORE settings.json is touched", () => {
    const dir = mkdtempSync(join(tmpdir(), "gs-"));
    const path = join(dir, "settings.json");
    const rawDb = new RawDatabase(":memory:"); // no migrate() -- no mutations table, standing
    // in for a missing/unmigrated $JAXOS_HOME/jaxos.db
    expect(() => writeGeneralSettings(rawDb, EXPECTED, path)).toThrow(SettingsAuditUnavailableError);
    rawDb.close();
    expect(existsSync(path)).toBe(false);
    expect(readdirSync(dir)).toEqual([]);
  });

  it("round 2 F2: an audit insert failure AFTER the file replace throws a distinct named error, and the file change stands (accepted as LOW, not a HIGH — see round 2 triage)", () => {
    const dir = mkdtempSync(join(tmpdir(), "gs-"));
    const path = join(dir, "settings.json");
    const db = openDb(":memory:");
    const fs = {
      mkdirSync,
      writeFileSync,
      renameSync: (from: string, to: string) => {
        renameSync(from, to);
        db.close(); // audit store becomes unusable exactly after the replace, before insertMutation runs
      },
    };
    expect(() => writeGeneralSettings(db, EXPECTED, path, fs)).toThrow(SettingsAuditFailedError);
    expect(readGeneralSettings(path)).toEqual({ ok: true, data: EXPECTED });
  });

  it("round 1 F4: two interleaved writes never corrupt each other — the final file is one complete payload or the other", () => {
    const dir = mkdtempSync(join(tmpdir(), "gs-"));
    const path = join(dir, "settings.json");
    const db = openDb(":memory:");
    const dataA: GeneralSettings = { ...EXPECTED, ownerName: "A" };
    const dataB: GeneralSettings = { ...EXPECTED, ownerName: "B" };
    let interleaved = false;
    const fs = {
      mkdirSync,
      writeFileSync,
      renameSync: (from: string, to: string) => {
        if (!interleaved) {
          // B's whole write (unique tmp file, its own rename) completes while A's own
          // rename is still in flight -- A's unique tmp name means B's completion cannot
          // touch A's still-unwritten tmp file.
          interleaved = true;
          writeGeneralSettings(db, dataB, path);
        }
        renameSync(from, to);
      },
    };
    writeGeneralSettings(db, dataA, path, fs);
    db.close();
    const result = readGeneralSettings(path);
    expect(result.ok).toBe(true);
    expect(result.ok && (result.data.ownerName === "A" || result.data.ownerName === "B")).toBe(true);
    expect(readdirSync(dir).some((f) => f.endsWith(".tmp"))).toBe(false);
  });
});

describe("seedAgentsIfFirstRun (MOA-504 D4)", () => {
  const has = (on: string[]) => (name: string) => on.includes(name);
  const auditRows = (db: ReturnType<typeof openDb>) =>
    db.prepare("SELECT kind FROM mutations WHERE kind = 'general-settings-write'").all();

  it("absent file: writes exactly the agents found, one audit row, returns true", () => {
    const path = join(mkdtempSync(join(tmpdir(), "gs-")), "settings.json");
    const db = openDb(":memory:");
    expect(seedAgentsIfFirstRun(db, has(["claude", "opencode"]), path)).toBe(true);
    const r = readGeneralSettings(path);
    expect(r.ok && r.data.integrations.agents).toEqual({ claude: true, codex: false, opencode: true });
    expect(auditRows(db)).toHaveLength(1);
    db.close();
  });

  it("is once: a second call (and any existing file) is a no-op and leaves the file byte-identical", () => {
    const path = join(mkdtempSync(join(tmpdir(), "gs-")), "settings.json");
    const db = openDb(":memory:");
    seedAgentsIfFirstRun(db, has(["claude"]), path);
    const before = readFileSync(path, "utf8");
    expect(seedAgentsIfFirstRun(db, has(["claude", "codex", "opencode"]), path)).toBe(false);
    expect(readFileSync(path, "utf8")).toBe(before);
    expect(auditRows(db)).toHaveLength(1);
    db.close();
  });

  it("a non-ENOENT stat error (ENOTDIR) is NOT first run: no write", () => {
    const dir = mkdtempSync(join(tmpdir(), "gs-"));
    writeFileSync(join(dir, "notadir"), "");
    const db = openDb(":memory:");
    expect(seedAgentsIfFirstRun(db, has(["claude"]), join(dir, "notadir", "settings.json"))).toBe(false);
    expect(auditRows(db)).toHaveLength(0);
    db.close();
  });

  it("a failure BEFORE the file is replaced leaves no file (defaults apply)", () => {
    const dir = mkdtempSync(join(tmpdir(), "gs-"));
    const path = join(dir, "settings.json");
    const db = openDb(":memory:");
    const fs = { mkdirSync, writeFileSync: () => { throw new Error("disk full"); }, renameSync };
    expect(() => seedAgentsIfFirstRun(db, has(["claude"]), path, fs)).toThrow("disk full");
    expect(existsSync(path)).toBe(false);
    expect(readGeneralSettings(path)).toEqual({ ok: true, data: DEFAULTS });
    db.close();
  });

  it("an audit-insert failure AFTER the replacement keeps the seeded file and throws SettingsAuditFailedError", () => {
    const path = join(mkdtempSync(join(tmpdir(), "gs-")), "settings.json");
    const db = openDb(":memory:");
    const fs = { mkdirSync, writeFileSync, renameSync: (a: string, b: string) => { renameSync(a, b); db.close(); } };
    expect(() => seedAgentsIfFirstRun(db, has(["codex"]), path, fs)).toThrow(SettingsAuditFailedError);
    const r = readGeneralSettings(path);
    expect(r.ok && r.data.integrations.agents).toEqual({ claude: false, codex: true, opencode: false });
  });
});
