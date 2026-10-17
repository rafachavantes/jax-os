// MOA-497 Build A: the $JAXOS_HOME/settings.json schema, reader, and writer. All fields
// optional on disk (a partial file is not malformed); every object is CLOSED (an unknown
// key at any level is settings-malformed), and a nested object's missing child fields
// default independently of the rest of the file. See scripts/general_settings.py for the
// Python twin -- the two MUST parse the exact same fixture to the exact same shape.
import { randomBytes } from "node:crypto";
import { mkdirSync, readFileSync, renameSync, statSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import type Database from "better-sqlite3";
import { insertMutation } from "./db/mutations";
import { jaxosHome } from "./env";

// Round 1 F1: thrown by writeGeneralSettings when the mutations audit store is
// unavailable -- a named error, checked BEFORE any write to settings.json, never a
// silently skipped audit row.
export class SettingsAuditUnavailableError extends Error {
  constructor() {
    super("settings-audit-unavailable");
    this.name = "SettingsAuditUnavailableError";
  }
}

// Round 2 F2 (downgraded HIGH->LOW, accepted at that severity): thrown when the audit
// insert itself fails AFTER settings.json has already been replaced -- no transaction
// spans the two, so the file change stands; this error is the caller's only signal that
// the mutation row never made it in.
export class SettingsAuditFailedError extends Error {
  constructor(cause: unknown) {
    super("settings-audit-failed");
    this.name = "SettingsAuditFailedError";
    this.cause = cause;
  }
}

export type Locale = "en-US" | "pt-BR";
export type GeneralSettings = {
  ownerName: string;
  locale: Locale;
  reposRoot: string;
  vaultPath: string | null;
  monitoredUnits: { user: string[]; system: string[] };
  integrations: {
    linear: boolean; ttyd: boolean; webhook: boolean; hermesTokens: boolean;
    agents: { claude: boolean; codex: boolean; opencode: boolean };
    classifier: boolean; github: boolean; vault: boolean;
  };
};
export type SettingsResult =
  | { ok: true; data: GeneralSettings }
  | { ok: false; error: "settings-malformed" | "settings-unreadable" };

const ROOT_KEYS = ["ownerName", "locale", "reposRoot", "vaultPath", "monitoredUnits", "integrations"] as const;
const MONITORED_KEYS = ["user", "system"] as const;
const AGENT_KEYS = ["claude", "codex", "opencode"] as const;
// MOA-504 D2: the old key is accepted on READ only and mapped into `agents`; it is never written back.
const LEGACY_SUBSCRIPTION_KEYS = ["claude", "codex", "opencodeGo"] as const;
const BOOL_INTEGRATION_KEYS = ["linear", "ttyd", "webhook", "hermesTokens", "classifier", "github", "vault"] as const;
const INTEGRATION_KEYS = [...BOOL_INTEGRATION_KEYS, "agents", "subscriptions"] as const;

type Obj = Record<string, unknown>;
const isObj = (v: unknown): v is Obj => typeof v === "object" && v !== null && !Array.isArray(v);
const isStrArray = (v: unknown): v is string[] => Array.isArray(v) && v.every((x) => typeof x === "string");
const closed = (o: Obj, allowed: readonly string[]) => Object.keys(o).every((k) => (allowed as readonly string[]).includes(k));

function defaultSettings(): GeneralSettings {
  return {
    ownerName: "", locale: "en-US", reposRoot: join(homedir(), "repos"), vaultPath: null,
    monitoredUnits: { user: [], system: [] },
    integrations: {
      linear: false, ttyd: false, webhook: false, hermesTokens: false,
      agents: { claude: false, codex: false, opencode: false },
      classifier: false, github: false, vault: false,
    },
  };
}

const MALFORMED: SettingsResult = { ok: false, error: "settings-malformed" };

function parseBools<K extends string>(v: unknown, keys: readonly K[]): Record<K, boolean> | null {
  if (!isObj(v) || !closed(v, keys)) return null;
  const out = {} as Record<K, boolean>;
  for (const k of keys) {
    const val = k in v ? v[k] : false;
    if (typeof val !== "boolean") return null;
    out[k] = val;
  }
  return out;
}

// MOA-504 D2: `agents` present -> strict; else legacy `subscriptions` mapped; else all off.
function parseAgents(v: Obj): GeneralSettings["integrations"]["agents"] | null {
  if ("agents" in v) return parseBools(v.agents, AGENT_KEYS);
  if ("subscriptions" in v) {
    const s = parseBools(v.subscriptions, LEGACY_SUBSCRIPTION_KEYS);
    return s && { claude: s.claude, codex: s.codex, opencode: s.opencodeGo };
  }
  return { claude: false, codex: false, opencode: false };
}

function parseIntegrations(v: unknown): GeneralSettings["integrations"] | null {
  if (v === undefined) return defaultSettings().integrations;
  if (!isObj(v) || !closed(v, INTEGRATION_KEYS)) return null;
  const out = {} as Record<string, unknown>;
  for (const k of BOOL_INTEGRATION_KEYS) {
    const val = k in v ? v[k] : false;
    if (typeof val !== "boolean") return null;
    out[k] = val;
  }
  const agents = parseAgents(v);
  if (agents === null) return null;
  out.agents = agents;
  return out as GeneralSettings["integrations"];
}

function parseMonitoredUnits(v: unknown): GeneralSettings["monitoredUnits"] | null {
  if (v === undefined) return { user: [], system: [] };
  if (!isObj(v) || !closed(v, MONITORED_KEYS)) return null;
  const user = "user" in v ? v.user : [];
  const system = "system" in v ? v.system : [];
  if (!isStrArray(user) || !isStrArray(system)) return null;
  return { user, system };
}

export function parseSettings(raw: unknown): SettingsResult {
  const def = defaultSettings();
  if (raw === undefined) return { ok: true, data: def };
  if (!isObj(raw) || !closed(raw, ROOT_KEYS)) return MALFORMED;
  const ownerName = "ownerName" in raw ? raw.ownerName : def.ownerName;
  if (typeof ownerName !== "string") return MALFORMED;
  const locale = "locale" in raw ? raw.locale : def.locale;
  if (locale !== "en-US" && locale !== "pt-BR") return MALFORMED;
  const reposRoot = "reposRoot" in raw ? raw.reposRoot : def.reposRoot;
  if (typeof reposRoot !== "string") return MALFORMED;
  let vaultPath: string | null;
  if (!("vaultPath" in raw)) vaultPath = def.vaultPath;
  else if (raw.vaultPath === null || typeof raw.vaultPath === "string") vaultPath = raw.vaultPath;
  else return MALFORMED;
  const monitoredUnits = parseMonitoredUnits(raw.monitoredUnits);
  if (monitoredUnits === null) return MALFORMED;
  const integrations = parseIntegrations(raw.integrations);
  if (integrations === null) return MALFORMED;
  return { ok: true, data: { ownerName, locale, reposRoot, vaultPath, monitoredUnits, integrations } };
}

function defaultPath(): string {
  return join(jaxosHome(), "settings.json");
}

export function readGeneralSettings(path: string = defaultPath()): SettingsResult {
  let stat: ReturnType<typeof statSync>;
  try {
    stat = statSync(path);
  } catch (err) {
    // F1: only ENOENT means absent -> defaults. Any other stat failure (EACCES, ENOTDIR,
    // ...) is a real problem and must be reported, not silently defaulted.
    if ((err as NodeJS.ErrnoException).code === "ENOENT") return parseSettings(undefined);
    return { ok: false, error: "settings-unreadable" };
  }
  if (!stat.isFile()) return { ok: false, error: "settings-unreadable" };
  let text: string;
  try {
    text = readFileSync(path, "utf8");
  } catch {
    return { ok: false, error: "settings-unreadable" };
  }
  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch {
    return { ok: false, error: "settings-malformed" };
  }
  return parseSettings(json);
}

type WriteFs = {
  mkdirSync: (path: string, options: { recursive: true }) => void;
  writeFileSync: (path: string, data: string, encoding: "utf8") => void;
  renameSync: (from: string, to: string) => void;
};
const systemWriteFs: WriteFs = { mkdirSync, writeFileSync, renameSync };

// Atomic unique-tmp+rename, mirroring workflow-callbacks.ts's writeAnswerLine. `db` is
// dependency-injected (never `getDb()` internally) so this module never touches the real
// ~/.jax-os database from a test. Records the full set of top-level field names as
// "changed" -- the whole file is written every call, so this is an honest audit, not a
// diff; a future PATCH-shaped consumer (MOA-496) can narrow it at the call site.
//
// Round 1 F1: the audit store is checked BEFORE any write to settings.json -- an
// unavailable mutations table throws SettingsAuditUnavailableError and settings.json is
// left completely untouched (no tmp file, no rename), never a silent skip.
// Round 1 F4: the tmp file gets a random suffix, unique per call, in the same directory
// as `path` -- two concurrent writers never share a temp file, so neither can rename the
// other's still-in-flight write.
export function writeGeneralSettings(
  db: Database.Database,
  data: GeneralSettings,
  path: string = defaultPath(),
  fs: WriteFs = systemWriteFs,
): void {
  try {
    db.prepare("SELECT 1 FROM mutations LIMIT 1").get();
  } catch {
    throw new SettingsAuditUnavailableError();
  }
  fs.mkdirSync(dirname(path), { recursive: true });
  const tmp = `${path}.${randomBytes(6).toString("hex")}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(data, null, 2) + "\n", "utf8");
  fs.renameSync(tmp, path);
  // Round 2 F2: no transaction spans the replace above and this insert -- the file change
  // already stands by the time we get here. A failure here is reported as a distinct named
  // error, never swallowed, but it is NOT rolled back (KISS; see round 2 triage).
  try {
    insertMutation(db, { kind: "general-settings-write", ok: true, changed: Object.keys(data) });
  } catch (cause) {
    throw new SettingsAuditFailedError(cause);
  }
}

// MOA-504 D4: seeds integrations.agents ONCE, from the CLIs found at server start. "First run" is
// settings.json absent (ENOENT only; any other stat error is not first run). `has` is injected so
// this module never touches PATH (collectors boundary). Throws what writeGeneralSettings throws.
export function seedAgentsIfFirstRun(
  db: Database.Database,
  has: (cli: string) => boolean,
  path: string = defaultPath(),
  fs: WriteFs = systemWriteFs,
): boolean {
  try {
    statSync(path);
    return false;
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== "ENOENT") return false;
  }
  // ponytail: parseSettings of a literal this module controls cannot be malformed.
  const seeded = parseSettings({ integrations: { agents: { claude: has("claude"), codex: has("codex"), opencode: has("opencode") } } }) as { ok: true; data: GeneralSettings };
  writeGeneralSettings(db, seeded.data, path, fs);
  return true;
}
