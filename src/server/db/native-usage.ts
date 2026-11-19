import type Database from "better-sqlite3";
import {
  mergeNativeItem,
  utcDay,
  type Harness,
  type NativeUsageItem,
  type SourceCoverage,
  type SourceState,
} from "../collectors/native-usage";

export type UsagePeriod = 7 | 30 | 90;
export type UsageSlice = {
  name: string;
  allInput: number;
  allOutput: number;
  tokens: number;
  cost: number | null;
  currency: string | null;
  items: number;
};
export type NativeContextRow = {
  fileIdentity: string;
  turnId: string;
  rootTurnId: string;
  model: string;
};
export type NativeUsageSnapshot = {
  period: UsagePeriod;
  startDay: string;
  endDay: string;
  totalInput: number;
  totalOutput: number;
  totalTokens: number;
  totalCacheRead: number | null;
  totalCacheWrite: number | null;
  totalReasoning: number | null;
  totalCost: number | null;
  costCurrency: string | null;
  costCoverage: { measured: number; total: number; unknownUnit: number };
  daily: { date: string; input: number; output: number; tokens: number }[];
  byModel: UsageSlice[];
  byAgent: UsageSlice[];
  sessions: { lead: number; child: number };
  sources: SourceCoverage[];
  filters: { agents: string[]; models: string[] };
  mixedProjectionSessions: number;
  excludedV1Items: number;
  history: { cutoffDay: string; indexing: boolean; includesCurrentPartialDay: true };
  warnings: string[];
};
export type FileCheckpoint = {
  source: Harness;
  identity: string;
  generation: string;
  dev: string;
  ino: string;
  size: number;
  mtime: number;
  offset: number;
  done: boolean;
  processedAt: number;
  child?: boolean;
  discarding?: boolean;
};

export const UNKNOWN_MODEL_FILTER = "__unknown__";

type ItemRow = {
  harness: Harness;
  item_id: string;
  session_id: string;
  child_id: string;
  parent_id: string | null;
  is_child: number;
  provenance: string;
  agent: string;
  model: string | null;
  model_conflict: number;
  identity_conflict: number;
  ts: number;
  updated_ts: number;
  day: string;
  all_input: number;
  all_output: number;
  input_tokens: number | null;
  output_tokens: number | null;
  cache_read: number | null;
  cache_write: number | null;
  reasoning: number | null;
  cost_amount: number | null;
  cost_provenance: string | null;
  cost_currency: string | null;
  projection: string | null;
  file_identity: string | null;
  turn_id: string | null;
  root_turn_id: string | null;
};

function rowToItem(row: ItemRow): NativeUsageItem {
  return {
    harness: row.harness,
    itemId: row.item_id,
    sessionId: row.session_id,
    childId: row.child_id === "" ? null : row.child_id,
    parentId: row.parent_id,
    isChild: row.is_child === 1,
    provenance: row.provenance,
    agent: row.agent,
    model: row.model,
    modelConflict: row.model_conflict === 1,
    identityConflict: row.identity_conflict === 1,
    ts: row.ts,
    updatedTs: row.updated_ts,
    day: row.day,
    allInput: row.all_input,
    allOutput: row.all_output,
    details: {
      input: row.input_tokens,
      output: row.output_tokens,
      cacheRead: row.cache_read,
      cacheWrite: row.cache_write,
      reasoning: row.reasoning,
    },
    cost:
      row.cost_amount === null
        ? null
        : {
            amount: row.cost_amount,
            provenance: "source-reported",
            currency: row.cost_currency,
          },
    projection: row.projection === "v1" || row.projection === "v2" ? row.projection : null,
    fileIdentity: row.file_identity,
    turnId: row.turn_id,
    rootTurnId: row.root_turn_id,
  };
}

function getItemRow(db: Database.Database, harness: Harness, itemId: string): ItemRow | undefined {
  return db
    .prepare("SELECT * FROM native_usage_items WHERE harness = ? AND item_id = ?")
    .get(harness, itemId) as ItemRow | undefined;
}

function writeItem(db: Database.Database, item: NativeUsageItem): void {
  db.prepare(
    `INSERT INTO native_usage_items (
       harness, item_id, session_id, child_id, parent_id, is_child, provenance, agent,
       model, model_conflict, identity_conflict, ts, updated_ts, day, all_input, all_output,
       input_tokens, output_tokens, cache_read, cache_write, reasoning,
       cost_amount, cost_provenance, cost_currency, projection,
       file_identity, turn_id, root_turn_id
     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT (harness, item_id) DO UPDATE SET
       session_id = excluded.session_id, child_id = excluded.child_id,
       parent_id = excluded.parent_id, is_child = excluded.is_child,
       provenance = excluded.provenance, agent = excluded.agent,
       model = excluded.model, model_conflict = excluded.model_conflict,
       identity_conflict = excluded.identity_conflict, ts = excluded.ts, updated_ts = excluded.updated_ts, day = excluded.day,
       all_input = excluded.all_input, all_output = excluded.all_output,
       input_tokens = excluded.input_tokens, output_tokens = excluded.output_tokens,
       cache_read = excluded.cache_read, cache_write = excluded.cache_write,
       reasoning = excluded.reasoning, cost_amount = excluded.cost_amount,
       cost_provenance = excluded.cost_provenance, cost_currency = excluded.cost_currency,
       projection = excluded.projection, file_identity = excluded.file_identity,
       turn_id = excluded.turn_id, root_turn_id = excluded.root_turn_id`,
  ).run(
    item.harness,
    item.itemId,
    item.sessionId,
    item.childId ?? "",
    item.parentId,
    item.isChild ? 1 : 0,
    item.provenance,
    item.agent,
    item.model,
    item.modelConflict ? 1 : 0,
    item.identityConflict ? 1 : 0,
    item.ts,
    item.updatedTs,
    item.day,
    item.allInput,
    item.allOutput,
    item.details.input,
    item.details.output,
    item.details.cacheRead,
    item.details.cacheWrite,
    item.details.reasoning,
    item.cost?.amount ?? null,
    item.cost?.provenance ?? null,
    item.cost?.currency ?? null,
    item.projection,
    item.fileIdentity,
    item.turnId,
    item.rootTurnId,
  );
}

function recordSessionFact(db: Database.Database, item: NativeUsageItem): void {
  if (item.harness !== "opencode") return;
  db.prepare(
    `INSERT INTO native_usage_session_facts (harness, session_id, projection, time_updated)
     VALUES ('opencode', ?, ?, ?)
     ON CONFLICT (harness, session_id, projection) DO UPDATE SET
       time_updated = MAX(native_usage_session_facts.time_updated, excluded.time_updated)`,
  ).run(item.sessionId, item.projection ?? "", item.updatedTs);
}

function resolveCodexModel(db: Database.Database, item: NativeUsageItem): NativeUsageItem {
  if (
    item.harness !== "codex" || item.model !== null ||
    !item.fileIdentity || !item.turnId || !item.rootTurnId
  ) return item;
  const context = db
    .prepare(
      "SELECT model, conflict FROM native_usage_contexts WHERE file_identity = ? AND turn_id = ? AND root_turn_id = ?",
    )
    .get(item.fileIdentity, item.turnId, item.rootTurnId) as
    | { model: string | null; conflict: number }
    | undefined;
  if (!context) return item;
  if (context.conflict === 1) return { ...item, modelConflict: true };
  return context.model === null ? item : { ...item, model: context.model };
}

export function upsertNativeUsageItem(db: Database.Database, item: NativeUsageItem): "inserted" | "updated" {
  const resolved = resolveCodexModel(db, item);
  const existing = getItemRow(db, resolved.harness, resolved.itemId);
  const merged = existing ? mergeNativeItem(rowToItem(existing), resolved) : resolved;
  writeItem(db, merged);
  recordSessionFact(db, merged);
  return existing ? "updated" : "inserted";
}

export function insertNativeContext(
  db: Database.Database,
  context: NativeContextRow,
  now: number,
): "inserted" | "conflict" {
  const existing = db
    .prepare(
      "SELECT model, conflict FROM native_usage_contexts WHERE file_identity = ? AND turn_id = ? AND root_turn_id = ?",
    )
    .get(context.fileIdentity, context.turnId, context.rootTurnId) as
    | { model: string | null; conflict: number }
    | undefined;
  if (existing && (existing.conflict === 1 || existing.model !== context.model)) {
    db.prepare(
      "UPDATE native_usage_contexts SET model = NULL, conflict = 1 WHERE file_identity = ? AND turn_id = ? AND root_turn_id = ?",
    ).run(context.fileIdentity, context.turnId, context.rootTurnId);
    db.prepare(
      `UPDATE native_usage_items SET model = NULL, model_conflict = 1
        WHERE harness = 'codex' AND file_identity = ? AND turn_id = ? AND root_turn_id = ?`,
    ).run(context.fileIdentity, context.turnId, context.rootTurnId);
    return "conflict";
  }
  db.prepare(
    `INSERT INTO native_usage_contexts (file_identity, turn_id, root_turn_id, model, conflict, created_at)
     VALUES (?, ?, ?, ?, 0, ?)
     ON CONFLICT (file_identity, turn_id, root_turn_id) DO UPDATE SET model = excluded.model`,
  ).run(context.fileIdentity, context.turnId, context.rootTurnId, context.model, now);
  db.prepare(
    `UPDATE native_usage_items SET model = ?
      WHERE harness = 'codex' AND model IS NULL AND model_conflict = 0
        AND file_identity = ? AND turn_id = ? AND root_turn_id = ?`,
  ).run(context.model, context.fileIdentity, context.turnId, context.rootTurnId);
  return "inserted";
}

export function getFileCheckpoint(
  db: Database.Database,
  source: Harness,
  identity: string,
): FileCheckpoint | undefined {
  const row = db
    .prepare("SELECT * FROM native_usage_files WHERE source = ? AND identity = ?")
    .get(source, identity) as
    | {
        source: Harness;
        identity: string;
        child: number;
        generation: string;
        dev: string | null;
        ino: string | null;
        size: number | null;
        mtime: number | null;
        offset: number;
        discarding: number;
        done: number;
        processed_at: number | null;
      }
    | undefined;
  if (!row) return undefined;
  return {
    source: row.source,
    identity: row.identity,
    generation: row.generation,
    dev: row.dev ?? "",
    ino: row.ino ?? "",
    size: row.size ?? 0,
    mtime: row.mtime ?? 0,
    offset: row.offset,
    done: row.done === 1,
    processedAt: row.processed_at ?? 0,
    child: row.child === 1,
    discarding: row.discarding === 1,
  };
}

function writeFileCheckpoint(db: Database.Database, cp: FileCheckpoint, now: number): void {
  db.prepare(
    `INSERT INTO native_usage_files
       (source, identity, child, generation, dev, ino, size, mtime, offset, discarding, done, processed_at, updated_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT (source, identity) DO UPDATE SET
       child = excluded.child,
       generation = excluded.generation, dev = excluded.dev, ino = excluded.ino,
       size = excluded.size, mtime = excluded.mtime, offset = excluded.offset,
       discarding = excluded.discarding, done = excluded.done,
       processed_at = excluded.processed_at, updated_at = excluded.updated_at,
       failed_until = 0, last_warning = NULL`,
  ).run(
    cp.source,
    cp.identity,
    cp.child ? 1 : 0,
    cp.generation,
    cp.dev,
    cp.ino,
    cp.size,
    cp.mtime,
    cp.offset,
    cp.discarding ? 1 : 0,
    cp.done ? 1 : 0,
    cp.processedAt,
    now,
  );
}

export function listFileCheckpoints(db: Database.Database, source: Harness): FileCheckpoint[] {
  const rows = db
    .prepare("SELECT identity FROM native_usage_files WHERE source = ? ORDER BY identity")
    .all(source) as { identity: string }[];
  return rows
    .map((r) => getFileCheckpoint(db, source, r.identity))
    .filter((r): r is FileCheckpoint => r !== undefined);
}

// ---------------------------------------------------------------------------
// Durable discovery registry and fair ingestion selection.

export type DiscoveryStateRow = {
  source: Harness;
  cycle: number;
  state: "active" | "complete";
  currentDir: string | null;
  visitSeq: number;
  lastComplete: number | null;
};

function readDiscovery(db: Database.Database, source: Harness): DiscoveryStateRow | undefined {
  const row = db
    .prepare("SELECT * FROM native_usage_discovery WHERE source = ?")
    .get(source) as
    | {
        source: Harness;
        cycle: number;
        state: "active" | "complete";
        current_dir: string | null;
        visit_seq: number;
        last_complete: number | null;
      }
    | undefined;
  if (!row) return undefined;
  return {
    source: row.source,
    cycle: row.cycle,
    state: row.state,
    currentDir: row.current_dir,
    visitSeq: row.visit_seq,
    lastComplete: row.last_complete,
  };
}

// Re-pending a known directory keeps its failure fact: a rediscovered retained row is
// queued again without clearing last_warning. Success clears it through markDirDone.
function upsertPendingDir(db: Database.Database, source: Harness, path: string, child: boolean, cycle: number): void {
  db.prepare(
    `INSERT INTO native_usage_dirs (source, path, child, state, cycle) VALUES (?, ?, ?, 'pending', ?)
     ON CONFLICT (source, path) DO UPDATE SET
       child = excluded.child, state = 'pending', cycle = excluded.cycle`,
  ).run(source, path, child ? 1 : 0, cycle);
}

// Starts a cycle when none exists, or when the previous complete cycle is older
// than the refresh cadence. Completed directory state is retained within a cycle; a
// new cycle keeps warning-bearing rows until a successful enumeration reconfirms them.
export function ensureDiscoveryCycle(
  db: Database.Database,
  source: Harness,
  now: number,
  cadenceSeconds: number,
): DiscoveryStateRow {
  return db.transaction(() => {
    const existing = readDiscovery(db, source);
    if (!existing) {
      db.prepare(
        `INSERT INTO native_usage_discovery (source, cycle, state, current_dir, visit_seq, last_complete, updated_at)
         VALUES (?, 1, 'active', NULL, 0, NULL, ?)`,
      ).run(source, now);
      upsertPendingDir(db, source, ".", false, 1);
      return readDiscovery(db, source)!;
    }
    if (existing.state === "complete" && now - (existing.lastComplete ?? 0) >= cadenceSeconds) {
      const cycle = existing.cycle + 1;
      db.prepare("DELETE FROM native_usage_dirs WHERE source = ? AND last_warning IS NULL").run(source);
      upsertPendingDir(db, source, ".", false, cycle);
      db.prepare(
        `UPDATE native_usage_discovery SET cycle = ?, state = 'active', current_dir = NULL, updated_at = ? WHERE source = ?`,
      ).run(cycle, now, source);
    }
    return readDiscovery(db, source)!;
  }).immediate();
}

export function addPendingDir(
  db: Database.Database,
  source: Harness,
  path: string,
  child: boolean,
  cycle: number,
  now: number,
): void {
  upsertPendingDir(db, source, path, child, cycle);
  db.prepare("UPDATE native_usage_discovery SET updated_at = ? WHERE source = ?").run(now, source);
}

export function nextPendingDir(db: Database.Database, source: Harness): string | null {
  // Prefix paths sort before their extensions, so plain path order keeps each directory
  // ahead of the children it registers. Codex walks the newest layout first.
  const order = source === "codex" ? "DESC" : "ASC";
  const row = db
    .prepare(
      `SELECT path FROM native_usage_dirs WHERE source = ? AND state = 'pending'
        ORDER BY path ${order} LIMIT 1`,
    )
    .get(source) as { path: string } | undefined;
  return row?.path ?? null;
}

export function setDiscoveryCurrentDir(
  db: Database.Database,
  source: Harness,
  currentDir: string | null,
  now: number,
): void {
  db.prepare("UPDATE native_usage_discovery SET current_dir = ?, updated_at = ? WHERE source = ?").run(
    currentDir,
    now,
    source,
  );
}

export function markDirDone(
  db: Database.Database,
  source: Harness,
  path: string,
  now: number,
  warning: string | null = null,
): void {
  db.transaction(() => {
    db.prepare("UPDATE native_usage_dirs SET state = 'done', last_warning = ? WHERE source = ? AND path = ?").run(
      warning,
      source,
      path,
    );
    db.prepare("UPDATE native_usage_discovery SET current_dir = NULL, updated_at = ? WHERE source = ?").run(now, source);
  }).immediate();
}

export function hasPendingDirs(db: Database.Database, source: Harness): boolean {
  const row = db
    .prepare("SELECT 1 AS n FROM native_usage_dirs WHERE source = ? AND state = 'pending' LIMIT 1")
    .get(source) as { n: number } | undefined;
  return row !== undefined;
}

// A completed enumeration can prove that an unseen old failed directory is gone; an
// unreadable current-cycle ancestor cannot. Unseen retained rows from older cycles are
// removed only when the completed cycle itself has no directory failure left.
export function finishDiscoveryCycle(
  db: Database.Database,
  source: Harness,
  now: number,
  completeEnumeration = true,
): void {
  db.transaction(() => {
    const cycle = readDiscovery(db, source)?.cycle ?? 0;
    if (completeEnumeration) {
      const failed = db
        .prepare(
          "SELECT 1 AS n FROM native_usage_dirs WHERE source = ? AND cycle = ? AND last_warning IS NOT NULL LIMIT 1",
        )
        .get(source, cycle) as { n: number } | undefined;
      if (!failed) {
        db.prepare("DELETE FROM native_usage_dirs WHERE source = ? AND cycle < ?").run(source, cycle);
      }
    }
    db.prepare(
      `UPDATE native_usage_discovery SET state = 'complete', last_complete = ?, current_dir = NULL, updated_at = ? WHERE source = ?`,
    ).run(now, now, source);
  }).immediate();
}

export function registerNativeFile(
  db: Database.Database,
  source: Harness,
  identity: string,
  child: boolean,
  now: number,
): boolean {
  return db.transaction(() => {
    const result = db
      .prepare(
        `INSERT OR IGNORE INTO native_usage_files
           (source, identity, child, generation, offset, discarding, done, failed_until, visit_ord, updated_at)
         VALUES (?, ?, ?, '', 0, 0, 0, 0, 0, ?)`,
      )
      .run(source, identity, child ? 1 : 0, now);
    if (result.changes === 0) {
      db.prepare("UPDATE native_usage_files SET child = ?, updated_at = ? WHERE source = ? AND identity = ?").run(
        child ? 1 : 0,
        now,
        source,
        identity,
      );
    }
    return result.changes > 0;
  }).immediate();
}

export type IngestionCandidate = {
  identity: string;
  child: boolean;
  mtime: number | null;
};

// Fixed indexed selection: the two persisted-newest native tails plus a fair rotation
// over every other readable registered file. visit_ord advances even for already-done
// files, so a frozen wall clock cannot pin the same selection forever.
export function selectIngestionCandidates(
  db: Database.Database,
  source: Harness,
  now: number,
  max = 6,
): IngestionCandidate[] {
  return db.transaction(() => {
    const hotLimit = Math.min(2, max);
    const hot = db
      .prepare(
        `SELECT identity FROM native_usage_files
          WHERE source = ? AND failed_until <= ?
          ORDER BY mtime DESC, identity ASC LIMIT ?`,
      )
      .all(source, now, hotLimit) as { identity: string }[];
    const hotSet = new Set(hot.map((r) => r.identity));
    const hotArgs = [...hotSet];
    const fairSql =
      `SELECT identity FROM native_usage_files
        WHERE source = ? AND failed_until <= ?
          ${hotSet.size > 0 ? `AND identity NOT IN (${hotArgs.map(() => "?").join(", ")})` : ""}
        ORDER BY visit_ord ASC, identity ASC LIMIT ?`;
    const fair = db
      .prepare(fairSql)
      .all(source, now, ...hotArgs, max) as { identity: string }[];
    const chosen: string[] = [...hotSet];
    for (const row of fair) {
      if (chosen.length >= max) break;
      if (!chosen.includes(row.identity)) chosen.push(row.identity);
    }
    const state = readDiscovery(db, source);
    let seq = state?.visitSeq ?? 0;
    const bump = db.prepare("UPDATE native_usage_files SET visit_ord = ? WHERE source = ? AND identity = ?");
    for (const identity of chosen) {
      seq += 1;
      bump.run(seq, source, identity);
    }
    if (state) {
      db.prepare("UPDATE native_usage_discovery SET visit_seq = ?, updated_at = ? WHERE source = ?").run(seq, now, source);
    }
    return chosen.map((identity) => {
      const row = db
        .prepare("SELECT child, mtime FROM native_usage_files WHERE source = ? AND identity = ?")
        .get(source, identity) as { child: number; mtime: number | null } | undefined;
      return { identity, child: row?.child === 1, mtime: row?.mtime ?? null };
    });
  }).immediate();
}

export function markFileFailure(
  db: Database.Database,
  source: Harness,
  identity: string,
  retryAt: number,
  now: number,
  warning: string,
): void {
  db.prepare(
    "UPDATE native_usage_files SET failed_until = ?, last_warning = ?, updated_at = ? WHERE source = ? AND identity = ?",
  ).run(retryAt, warning, now, source, identity);
}

export function markFileCurrent(
  db: Database.Database,
  source: Harness,
  identity: string,
  size: number,
  mtime: number,
  now: number,
): void {
  db.prepare(
    `UPDATE native_usage_files SET size = ?, mtime = ?, failed_until = 0, last_warning = NULL, updated_at = ?
      WHERE source = ? AND identity = ?`,
  ).run(size, mtime, now, source, identity);
}

// Safe persisted failure facts survive the poll that skipped the failed entry; only a
// successful retry or a fresh discovery cycle clears them.
export function getPersistentWarnings(db: Database.Database, source: Harness): string[] {
  const rows = db
    .prepare(
      `SELECT last_warning AS warning FROM native_usage_files
        WHERE source = ? AND last_warning IS NOT NULL
       UNION
       SELECT last_warning AS warning FROM native_usage_dirs
        WHERE source = ? AND last_warning IS NOT NULL`,
    )
    .all(source, source) as { warning: string }[];
  return rows.map((r) => r.warning);
}

export function getRegistryCounts(
  db: Database.Database,
  source: Harness,
  now: number,
): { total: number; done: number; pending: number } {
  const row = db
    .prepare(
      `SELECT COUNT(*) AS total,
              SUM(CASE WHEN done = 1 THEN 1 ELSE 0 END) AS done,
              SUM(CASE WHEN done = 0 AND failed_until <= ? THEN 1 ELSE 0 END) AS pending
         FROM native_usage_files WHERE source = ?`,
    )
    .get(now, source) as { total: number; done: number | null; pending: number | null };
  return { total: row.total, done: row.done ?? 0, pending: row.pending ?? 0 };
}

export function countRegisteredFiles(db: Database.Database, source: Harness): number {
  return (db.prepare("SELECT COUNT(*) AS n FROM native_usage_files WHERE source = ?").get(source) as { n: number }).n;
}

// ---------------------------------------------------------------------------
// Atomic ingestion.

export type FileBatch = {
  checkpoint: FileCheckpoint;
  // The exact prior registry state observed by this writer: null means "no row", a
  // placeholder row is passed explicitly. A stale generation/offset is never a wildcard.
  expectedPrior: FileCheckpoint | null;
  items: NativeUsageItem[];
  contexts?: NativeContextRow[];
  now: number;
};

// Item/projection/context upserts, the retention filter and the consumed offset
// advance share one immediate transaction: a crash or retry can never advance the
// checkpoint without its items, and an old item never re-enters the retained set.
export function applyFileBatch(
  db: Database.Database,
  batch: FileBatch,
): { inserted: number; updated: number; rejected: number; checkpointAdvanced: boolean } {
  const cutoffDay = utcDay(batch.now - 90 * 86400);
  return db.transaction(() => {
    const existing = getFileCheckpoint(db, batch.checkpoint.source, batch.checkpoint.identity);
    const prior = batch.expectedPrior;
    if (prior === null) {
      if (existing) throw new Error("checkpoint-conflict");
    } else if (!existing || existing.generation !== prior.generation || existing.offset !== prior.offset) {
      throw new Error("checkpoint-conflict");
    }
    for (const context of batch.contexts ?? []) insertNativeContext(db, context, batch.now);
    let inserted = 0;
    let updated = 0;
    let rejected = 0;
    for (const item of batch.items) {
      if (item.day < cutoffDay) {
        rejected += 1;
        continue;
      }
      if (upsertNativeUsageItem(db, item) === "inserted") inserted += 1;
      else updated += 1;
    }
    writeFileCheckpoint(db, batch.checkpoint, batch.now);
    return { inserted, updated, rejected, checkpointAdvanced: true };
  }).immediate();
}

// ---------------------------------------------------------------------------
// OpenCode cursor state: canonical decimal rowid strings, never JS numbers.

export type OpenCodeTableCursor = { after: string | null; ceiling: string | null; done: boolean };
export type OpenCodeCursorState = { v: 1; v1: OpenCodeTableCursor; v2: OpenCodeTableCursor };

const ROWID_RE = /^-?\d{1,20}$/;

export function canonicalRowId(value: unknown): string | null {
  if (typeof value !== "string" || !ROWID_RE.test(value)) return null;
  try {
    const numeric = BigInt(value);
    if (numeric < BigInt("-9223372036854775808") || numeric > BigInt("9223372036854775807")) return null;
    // Canonical spelling only: no "-0", no leading zeroes.
    if (numeric.toString() !== value) return null;
  } catch {
    return null;
  }
  return value;
}

export function freshOpenCodeCursor(): OpenCodeCursorState {
  return {
    v: 1,
    v1: { after: null, ceiling: null, done: false },
    v2: { after: null, ceiling: null, done: false },
  };
}

function sanitizeCursor(raw: unknown): OpenCodeTableCursor | null {
  if (typeof raw !== "object" || raw === null) return null;
  const t = raw as Record<string, unknown>;
  const after = t.after === null || t.after === undefined ? null : canonicalRowId(t.after);
  if (after === null && t.after !== null && t.after !== undefined) return null;
  const ceiling = t.ceiling === null || t.ceiling === undefined ? null : canonicalRowId(t.ceiling);
  if (ceiling === null && t.ceiling !== null && t.ceiling !== undefined) return null;
  if (after !== null && ceiling !== null && BigInt(after) > BigInt(ceiling)) return null;
  return { after, ceiling, done: t.done === true };
}

function parseCursorState(raw: unknown): OpenCodeCursorState {
  const out = freshOpenCodeCursor();
  if (typeof raw !== "object" || raw === null) return out;
  const state = raw as Record<string, unknown>;
  if (state.v !== 1) return out;
  for (const key of ["v1", "v2"] as const) {
    const parsed = sanitizeCursor(state[key]);
    if (parsed) out[key] = parsed;
  }
  return out;
}

export function getOpenCodeCursor(db: Database.Database): OpenCodeCursorState {
  const row = db
    .prepare("SELECT progress FROM native_usage_sources WHERE source = 'opencode'")
    .get() as { progress: string } | undefined;
  if (!row) return freshOpenCodeCursor();
  try {
    return parseCursorState(JSON.parse(row.progress));
  } catch {
    return freshOpenCodeCursor();
  }
}

export function setOpenCodeCursor(
  db: Database.Database,
  state: OpenCodeCursorState,
  now: number,
): void {
  db.prepare(
    `INSERT INTO native_usage_sources (source, state, last_refresh, warnings, files_done, files_total, progress, updated_at)
     VALUES ('opencode', 'indexing', NULL, '[]', 0, 0, ?, ?)
     ON CONFLICT (source) DO UPDATE SET progress = excluded.progress, updated_at = excluded.updated_at`,
  ).run(JSON.stringify(state), now);
}

export function persistOpenCodeBatch(
  db: Database.Database,
  items: NativeUsageItem[],
  state: OpenCodeCursorState,
  now: number,
): { inserted: number; updated: number; rejected: number } {
  const cutoffDay = utcDay(now - 90 * 86400);
  return db.transaction(() => {
    let inserted = 0;
    let updated = 0;
    let rejected = 0;
    for (const item of items) {
      if (item.day < cutoffDay) {
        rejected += 1;
        continue;
      }
      if (upsertNativeUsageItem(db, item) === "inserted") inserted += 1;
      else updated += 1;
    }
    setOpenCodeCursor(db, state, now);
    return { inserted, updated, rejected };
  }).immediate();
}

// ---------------------------------------------------------------------------
// Source coverage.

export function setSourceStatus(
  db: Database.Database,
  status: SourceCoverage,
  now: number,
): void {
  db.prepare(
    `INSERT INTO native_usage_sources
       (source, state, last_refresh, warnings, files_done, files_total, updated_at)
     VALUES (?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT (source) DO UPDATE SET
       state = excluded.state, last_refresh = excluded.last_refresh,
       warnings = excluded.warnings, files_done = excluded.files_done,
       files_total = excluded.files_total, updated_at = excluded.updated_at`,
  ).run(
    status.source,
    status.state,
    status.lastRefresh,
    JSON.stringify(status.warnings),
    status.progress?.filesDone ?? 0,
    status.progress?.filesTotal ?? 0,
    now,
  );
}

export function getSourceStatuses(db: Database.Database): SourceCoverage[] {
  const rows = db
    .prepare("SELECT * FROM native_usage_sources ORDER BY source")
    .all() as {
    source: Harness;
    state: SourceState;
    last_refresh: number | null;
    warnings: string;
    files_done: number;
    files_total: number;
  }[];
  return rows.map((row) => {
    let warnings: string[] = [];
    try {
      const parsed = JSON.parse(row.warnings);
      if (Array.isArray(parsed)) warnings = parsed.filter((w): w is string => typeof w === "string");
    } catch {
      warnings = [];
    }
    return {
      source: row.source,
      state: row.state,
      lastRefresh: row.last_refresh,
      progress:
        row.files_total > 0 ? { filesDone: row.files_done, filesTotal: row.files_total } : null,
      warnings,
    };
  });
}

export function pruneNativeUsage(db: Database.Database, now: number): { itemsDeleted: number } {
  const cutoffDay = utcDay(now - 90 * 86400);
  return db.transaction(() => {
    const deleted = db.prepare("DELETE FROM native_usage_items WHERE day < ?").run(cutoffDay);
    db.prepare(
      `DELETE FROM native_usage_contexts
        WHERE created_at < ? AND NOT EXISTS (
          SELECT 1 FROM native_usage_items i
           WHERE i.file_identity = native_usage_contexts.file_identity
             AND i.turn_id = native_usage_contexts.turn_id
             AND i.root_turn_id = native_usage_contexts.root_turn_id
        )`,
    ).run(now - 90 * 86400);
    db.prepare(
      `DELETE FROM native_usage_session_facts
        WHERE NOT EXISTS (
          SELECT 1 FROM native_usage_items i
           WHERE i.harness = native_usage_session_facts.harness
             AND i.session_id = native_usage_session_facts.session_id
        )`,
    ).run();
    return { itemsDeleted: deleted.changes };
  }).immediate();
}

// ---------------------------------------------------------------------------
// Honest aggregation. Every addition is checked: an unrepresentable sum fails the
// safe envelope instead of rounding, wrapping or silently saturating.

const CANONICAL = `(i.harness <> 'opencode' OR i.projection = 'v2' OR NOT EXISTS (
  SELECT 1 FROM native_usage_items v2
   WHERE v2.harness = 'opencode' AND v2.projection = 'v2' AND v2.session_id = i.session_id
))`;

type TotalRow = {
  items: number;
  all_input: number | null;
  all_output: number | null;
  cache_read: number | null;
  cache_write: number | null;
  reasoning: number | null;
  identity_conflict: number | null;
};

type DailyRow = { day: string; all_input: number | null; all_output: number | null };

type FacetRow = {
  agent: string;
  model: string | null;
  cost_currency: string | null;
  cost_present: number;
  items: number;
  all_input: number | null;
  all_output: number | null;
  cost_sum: number | null;
};

type SliceAcc = {
  items: number;
  allInput: number;
  allOutput: number;
  costByCurrency: Map<string, number>;
  costItemsByCurrency: Map<string, number>;
};

function checkedAdd(total: number, value: number): number {
  if (!Number.isSafeInteger(value) || value < 0) throw new Error("aggregate-overflow");
  const next = total + value;
  if (!Number.isSafeInteger(next)) throw new Error("aggregate-overflow");
  return next;
}

export function sumChecked(values: number[]): number {
  let total = 0;
  for (const value of values) total = checkedAdd(total, value);
  return total;
}

function checkedMoney(total: number, value: number): number {
  if (!Number.isFinite(value) || value < 0) throw new Error("aggregate-overflow");
  const next = total + value;
  if (!Number.isFinite(next)) throw new Error("aggregate-overflow");
  return next;
}

function emptySlice(name: string): SliceAcc & { name: string } {
  return {
    name,
    items: 0,
    allInput: 0,
    allOutput: 0,
    costByCurrency: new Map(),
    costItemsByCurrency: new Map(),
  };
}

function addToSlice(acc: SliceAcc, row: FacetRow): void {
  const items = checkedAdd(0, row.items);
  const input = checkedAdd(0, row.all_input ?? 0);
  const output = checkedAdd(0, row.all_output ?? 0);
  acc.items = checkedAdd(acc.items, items);
  acc.allInput = checkedAdd(acc.allInput, input);
  acc.allOutput = checkedAdd(acc.allOutput, output);
  if (row.cost_present !== 1 || row.cost_currency === null) return;
  const currency = row.cost_currency;
  acc.costByCurrency.set(currency, checkedMoney(acc.costByCurrency.get(currency) ?? 0, row.cost_sum ?? 0));
  acc.costItemsByCurrency.set(currency, checkedAdd(acc.costItemsByCurrency.get(currency) ?? 0, items));
}

function sliceFrom(name: string, acc: SliceAcc): UsageSlice {
  const currencies = [...acc.costByCurrency.keys()];
  const compatible = currencies.length === 1;
  return {
    name,
    allInput: acc.allInput,
    allOutput: acc.allOutput,
    tokens: checkedAdd(acc.allInput, acc.allOutput),
    cost: compatible ? acc.costByCurrency.get(currencies[0]) as number : null,
    currency: compatible ? currencies[0] : null,
    items: acc.items,
  };
}

function daysBetween(startDay: string, endDay: string): string[] {
  const out: string[] = [];
  const start = Date.parse(`${startDay}T00:00:00Z`);
  const end = Date.parse(`${endDay}T00:00:00Z`);
  for (let ms = start; ms <= end; ms += 86400_000) out.push(new Date(ms).toISOString().slice(0, 10));
  return out;
}

export function queryNativeUsage(
  db: Database.Database,
  opts: { period: UsagePeriod; agent?: string | null; model?: string | null; now: number },
): NativeUsageSnapshot {
  const days = opts.period;
  const endDay = utcDay(opts.now);
  const startDay = utcDay(opts.now - (days - 1) * 86400);
  let filterSql = "";
  const filterParams: unknown[] = [];
  if (opts.agent) {
    filterSql += " AND i.agent = ?";
    filterParams.push(opts.agent);
  }
  if (opts.model !== undefined && opts.model !== null) {
    if (opts.model === UNKNOWN_MODEL_FILTER) filterSql += " AND i.model IS NULL";
    else {
      filterSql += " AND i.model = ?";
      filterParams.push(opts.model);
    }
  }
  const where = `i.day >= ? AND i.day <= ? AND ${CANONICAL}${filterSql}`;
  const params = [startDay, endDay, ...filterParams];

  const totalRow = db
    .prepare(
      `SELECT COUNT(*) AS items, SUM(i.all_input) AS all_input, SUM(i.all_output) AS all_output,
              SUM(i.cache_read) AS cache_read, SUM(i.cache_write) AS cache_write,
              SUM(i.reasoning) AS reasoning, MAX(i.identity_conflict) AS identity_conflict
         FROM native_usage_items i
        WHERE ${where}`,
    )
    .get(...params) as TotalRow;

  const dailyRows = db
    .prepare(
      `SELECT i.day AS day, SUM(i.all_input) AS all_input, SUM(i.all_output) AS all_output
         FROM native_usage_items i
        WHERE ${where}
        GROUP BY i.day`,
    )
    .all(...params) as DailyRow[];

  // One reduced projection feeds every agent/model slice and the total cost coverage:
  // the row count follows groups, never items.
  const facetRows = db
    .prepare(
      `SELECT i.agent AS agent, i.model AS model, i.cost_currency AS cost_currency,
              CASE WHEN i.cost_amount IS NULL THEN 0 ELSE 1 END AS cost_present,
              COUNT(*) AS items, SUM(i.all_input) AS all_input, SUM(i.all_output) AS all_output,
              SUM(i.cost_amount) AS cost_sum
         FROM native_usage_items i
        WHERE ${where}
        GROUP BY i.agent, i.model, i.cost_currency, CASE WHEN i.cost_amount IS NULL THEN 0 ELSE 1 END`,
    )
    .all(...params) as FacetRow[];

  const daily = daysBetween(startDay, endDay).map((date) => ({ date, input: 0, output: 0, tokens: 0 }));
  const byDate = new Map(daily.map((d) => [d.date, d]));
  for (const row of dailyRows) {
    const bucket = byDate.get(row.day);
    if (!bucket) continue;
    bucket.input = checkedAdd(bucket.input, row.all_input ?? 0);
    bucket.output = checkedAdd(bucket.output, row.all_output ?? 0);
    bucket.tokens = checkedAdd(bucket.input, bucket.output);
  }

  const agents = new Map<string, SliceAcc>();
  const models = new Map<string, SliceAcc>();
  const totalCostByCurrency = new Map<string, number>();
  const totalCostItems = new Map<string, number>();
  let unknownUnit = 0;
  for (const row of facetRows) {
    const items = checkedAdd(0, row.items);
    const agentAcc = agents.get(row.agent) ?? emptySlice(row.agent);
    addToSlice(agentAcc, row);
    agents.set(row.agent, agentAcc);
    const modelName = row.model ?? UNKNOWN_MODEL_FILTER;
    const modelAcc = models.get(modelName) ?? emptySlice(modelName);
    addToSlice(modelAcc, row);
    models.set(modelName, modelAcc);
    if (row.cost_present !== 1) continue;
    if (row.cost_currency === null) {
      unknownUnit = checkedAdd(unknownUnit, items);
      continue;
    }
    totalCostByCurrency.set(
      row.cost_currency,
      checkedMoney(totalCostByCurrency.get(row.cost_currency) ?? 0, row.cost_sum ?? 0),
    );
    totalCostItems.set(
      row.cost_currency,
      checkedAdd(totalCostItems.get(row.cost_currency) ?? 0, items),
    );
  }

  const sessionRow = db
    .prepare(
      `SELECT
         (SELECT COUNT(DISTINCT i.session_id) FROM native_usage_items i WHERE ${where} AND i.is_child = 0) AS lead,
         (SELECT COUNT(*) FROM (
            SELECT 1 FROM native_usage_items i WHERE ${where} AND i.is_child = 1
             GROUP BY i.session_id, i.child_id
          )) AS child`,
    )
    .get(...params, ...params) as { lead: number; child: number };

  // F1: only a selected V1 with no same-session mirror of its own native message is a
  // coverage gap. The candidate keeps the selected range/agent/model; both lookups are
  // global and the canonical predicate never applies to the candidate itself.
  const diagnosticRow = db
    .prepare(
      `SELECT COUNT(*) AS items, COUNT(DISTINCT i.session_id) AS sessions
         FROM native_usage_items i
        WHERE i.harness = 'opencode' AND i.projection = 'v1'
          AND substr(i.item_id, -3) = '|v1'
          AND i.day >= ? AND i.day <= ?${filterSql}
          AND EXISTS (
            SELECT 1 FROM native_usage_items v2
             WHERE v2.harness = 'opencode' AND v2.projection = 'v2' AND v2.session_id = i.session_id
          )
          AND NOT EXISTS (
            SELECT 1 FROM native_usage_items mirror
             WHERE mirror.harness = 'opencode' AND mirror.projection = 'v2'
               AND mirror.session_id = i.session_id
               AND mirror.item_id = substr(i.item_id, 1, length(i.item_id) - 3) || '|v2'
          )`,
    )
    .get(startDay, endDay, ...filterParams) as { items: number; sessions: number };

  const filterEnd = utcDay(opts.now);
  const filterStart = utcDay(opts.now - 90 * 86400);
  const agentNames = db
    .prepare(
      `SELECT DISTINCT i.agent AS name FROM native_usage_items i
        WHERE i.day >= ? AND i.day <= ? AND ${CANONICAL} ORDER BY i.agent`,
    )
    .all(filterStart, filterEnd) as { name: string }[];
  const modelNames = db
    .prepare(
      `SELECT DISTINCT i.model AS name FROM native_usage_items i
        WHERE i.day >= ? AND i.day <= ? AND ${CANONICAL} ORDER BY i.model`,
    )
    .all(filterStart, filterEnd) as { name: string | null }[];

  const sourceRows = getSourceStatuses(db);
  const sources: SourceCoverage[] = (["claude", "codex", "opencode"] as const).map((source) =>
    sourceRows.find((s) => s.source === source) ?? {
      source,
      state: "indexing" as const,
      lastRefresh: null,
      progress: null,
      warnings: [],
    },
  );
  const warnings = new Set(sources.flatMap((s) => s.warnings));
  if (diagnosticRow.sessions > 0) warnings.add("mixed-projection");
  if ((totalRow.identity_conflict ?? 0) === 1) warnings.add("identity-conflict");

  const totalInput = checkedAdd(0, totalRow.all_input ?? 0);
  const totalOutput = checkedAdd(0, totalRow.all_output ?? 0);
  const detail = (value: number | null) => (value === null ? null : checkedAdd(0, value));
  const costCurrencies = [...totalCostByCurrency.keys()];
  const compatibleCurrency = costCurrencies.length === 1;
  const totalCost = compatibleCurrency ? (totalCostByCurrency.get(costCurrencies[0]) as number) : null;

  return {
    period: opts.period,
    startDay,
    endDay,
    totalInput,
    totalOutput,
    totalTokens: checkedAdd(totalInput, totalOutput),
    totalCacheRead: detail(totalRow.cache_read),
    totalCacheWrite: detail(totalRow.cache_write),
    totalReasoning: detail(totalRow.reasoning),
    totalCost,
    costCurrency: compatibleCurrency ? costCurrencies[0] : null,
    costCoverage: {
      measured: compatibleCurrency ? (totalCostItems.get(costCurrencies[0]) as number) : 0,
      total: checkedAdd(0, totalRow.items),
      unknownUnit,
    },
    daily,
    byModel: [...models.entries()]
      .map(([name, acc]) => sliceFrom(name, acc))
      .sort((a, b) => b.tokens - a.tokens || (a.name === UNKNOWN_MODEL_FILTER ? 1 : b.name === UNKNOWN_MODEL_FILTER ? -1 : 0)),
    byAgent: [...agents.entries()].map(([name, acc]) => sliceFrom(name, acc)).sort((a, b) => b.tokens - a.tokens),
    sessions: { lead: sessionRow.lead, child: sessionRow.child },
    sources,
    filters: {
      agents: agentNames.map((r) => r.name),
      models: modelNames.map((r) => r.name ?? UNKNOWN_MODEL_FILTER),
    },
    mixedProjectionSessions: diagnosticRow.sessions,
    excludedV1Items: diagnosticRow.items,
    history: {
      cutoffDay: utcDay(opts.now - 90 * 86400),
      indexing: sources.some((s) => s.state === "indexing"),
      includesCurrentPartialDay: true,
    },
    warnings: [...warnings].sort(),
  };
}
