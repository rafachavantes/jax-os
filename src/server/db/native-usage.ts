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
  totalCost: number | null;
  costCoverage: { measured: number; total: number };
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
       model, model_conflict, ts, updated_ts, day, all_input, all_output,
       input_tokens, output_tokens, cache_read, cache_write, reasoning,
       cost_amount, cost_provenance, cost_currency, projection,
       file_identity, turn_id, root_turn_id
     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT (harness, item_id) DO UPDATE SET
       session_id = excluded.session_id, child_id = excluded.child_id,
       parent_id = excluded.parent_id, is_child = excluded.is_child,
       provenance = excluded.provenance, agent = excluded.agent,
       model = excluded.model, model_conflict = excluded.model_conflict,
       ts = excluded.ts, updated_ts = excluded.updated_ts, day = excluded.day,
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

export function upsertNativeUsageItem(db: Database.Database, item: NativeUsageItem): "inserted" | "updated" {
  const existing = getItemRow(db, item.harness, item.itemId);
  const merged = existing ? mergeNativeItem(rowToItem(existing), item) : item;
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
        generation: string;
        dev: string | null;
        ino: string | null;
        size: number | null;
        mtime: number | null;
        offset: number;
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
  };
}

function writeFileCheckpoint(db: Database.Database, cp: FileCheckpoint, now: number): void {
  db.prepare(
    `INSERT INTO native_usage_files
       (source, identity, generation, dev, ino, size, mtime, offset, done, processed_at, updated_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT (source, identity) DO UPDATE SET
       generation = excluded.generation, dev = excluded.dev, ino = excluded.ino,
       size = excluded.size, mtime = excluded.mtime, offset = excluded.offset,
       done = excluded.done, processed_at = excluded.processed_at,
       updated_at = excluded.updated_at`,
  ).run(
    cp.source,
    cp.identity,
    cp.generation,
    cp.dev,
    cp.ino,
    cp.size,
    cp.mtime,
    cp.offset,
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

export type FileBatch = {
  checkpoint: FileCheckpoint;
  expectedOffset: number | null;
  items: NativeUsageItem[];
  contexts?: NativeContextRow[];
  now: number;
};

// Item/projection upserts and the consumed offset advance share one immediate
// transaction: a crash or retry can never advance the offset without the items.
export function applyFileBatch(
  db: Database.Database,
  batch: FileBatch,
): { inserted: number; updated: number; checkpointAdvanced: boolean } {
  return db.transaction(() => {
    if (batch.expectedOffset !== null) {
      const existing = getFileCheckpoint(db, batch.checkpoint.source, batch.checkpoint.identity);
      if (!existing) {
        if (batch.expectedOffset !== 0) throw new Error("checkpoint-conflict");
      } else if (
        existing.generation !== batch.checkpoint.generation ||
        existing.offset !== batch.expectedOffset
      ) {
        throw new Error("checkpoint-conflict");
      }
    }
    for (const context of batch.contexts ?? []) insertNativeContext(db, context, batch.now);
    let inserted = 0;
    let updated = 0;
    for (const item of batch.items) {
      if (upsertNativeUsageItem(db, item) === "inserted") inserted += 1;
      else updated += 1;
    }
    writeFileCheckpoint(db, batch.checkpoint, batch.now);
    return { inserted, updated, checkpointAdvanced: true };
  })();
}

export type SourceStatus = SourceCoverage;

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
  })();
}

const CANONICAL = `(i.harness <> 'opencode' OR i.projection = 'v2' OR NOT EXISTS (
  SELECT 1 FROM native_usage_items v2
   WHERE v2.harness = 'opencode' AND v2.projection = 'v2' AND v2.session_id = i.session_id
))`;

type AggRow = {
  harness: Harness;
  session_id: string;
  child_id: string;
  is_child: number;
  agent: string;
  model: string | null;
  day: string;
  all_input: number;
  all_output: number;
  cost_amount: number | null;
  cost_currency: string | null;
};

function sliceCost(rows: AggRow[]): number | null {
  const costed = rows.filter((r) => r.cost_amount !== null);
  if (costed.length === 0) return null;
  const currencies = new Set(costed.map((r) => r.cost_currency));
  if (currencies.size !== 1 || currencies.has(null)) return null;
  return costed.reduce((sum, r) => sum + (r.cost_amount ?? 0), 0);
}

function toSlice(name: string, rows: AggRow[]): UsageSlice {
  const allInput = rows.reduce((s, r) => s + r.all_input, 0);
  const allOutput = rows.reduce((s, r) => s + r.all_output, 0);
  return { name, allInput, allOutput, tokens: allInput + allOutput, cost: sliceCost(rows), items: rows.length };
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
  const params: unknown[] = [startDay, endDay];
  let where = `i.day >= ? AND i.day <= ? AND ${CANONICAL}`;
  if (opts.agent) {
    where += " AND i.agent = ?";
    params.push(opts.agent);
  }
  if (opts.model !== undefined && opts.model !== null) {
    if (opts.model === UNKNOWN_MODEL_FILTER) where += " AND i.model IS NULL";
    else {
      where += " AND i.model = ?";
      params.push(opts.model);
    }
  }
  const rows = db
    .prepare(
      `SELECT i.harness, i.session_id, i.child_id, i.is_child, i.agent, i.model, i.day,
              i.all_input, i.all_output, i.cost_amount, i.cost_currency
         FROM native_usage_items i
        WHERE ${where}`,
    )
    .all(...params) as AggRow[];

  const daily = daysBetween(startDay, endDay).map((date) => ({ date, input: 0, output: 0, tokens: 0 }));
  const byDate = new Map(daily.map((d) => [d.date, d]));
  const agents = new Map<string, AggRow[]>();
  const models = new Map<string, AggRow[]>();
  const lead = new Set<string>();
  const child = new Set<string>();
  for (const row of rows) {
    const bucket = byDate.get(row.day);
    if (bucket) {
      bucket.input += row.all_input;
      bucket.output += row.all_output;
      bucket.tokens += row.all_input + row.all_output;
    }
    const agentRows = agents.get(row.agent) ?? [];
    agentRows.push(row);
    agents.set(row.agent, agentRows);
    const modelName = row.model ?? UNKNOWN_MODEL_FILTER;
    const modelRows = models.get(modelName) ?? [];
    modelRows.push(row);
    models.set(modelName, modelRows);
    if (row.is_child === 1) child.add(`${row.session_id}|${row.child_id}`);
    else lead.add(row.session_id);
  }

  const costed = rows.filter((r) => r.cost_amount !== null);
  const totalCost = sliceCost(rows);

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

  const mixed = db
    .prepare(
      `SELECT COUNT(*) AS n FROM (
         SELECT session_id FROM native_usage_items
          WHERE harness = 'opencode' AND day >= ? AND day <= ?
          GROUP BY session_id
         HAVING SUM(CASE WHEN projection = 'v2' THEN 1 ELSE 0 END) > 0
            AND SUM(CASE WHEN projection = 'v1' THEN 1 ELSE 0 END) > 0
       )`,
    )
    .get(startDay, endDay) as { n: number };
  const excludedV1 = db
    .prepare(
      `SELECT COUNT(*) AS n FROM native_usage_items i
        WHERE i.harness = 'opencode' AND i.projection = 'v1' AND i.day >= ? AND i.day <= ?
          AND EXISTS (
            SELECT 1 FROM native_usage_items v2
             WHERE v2.harness = 'opencode' AND v2.projection = 'v2' AND v2.session_id = i.session_id
          )`,
    )
    .get(startDay, endDay) as { n: number };

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
  if (mixed.n > 0) warnings.add("mixed-projection");

  return {
    period: opts.period,
    startDay,
    endDay,
    totalInput: rows.reduce((s, r) => s + r.all_input, 0),
    totalOutput: rows.reduce((s, r) => s + r.all_output, 0),
    totalTokens: rows.reduce((s, r) => s + r.all_input + r.all_output, 0),
    totalCost,
    costCoverage: { measured: costed.length, total: rows.length },
    daily,
    byModel: [...models.entries()]
      .map(([name, grouped]) => toSlice(name, grouped))
      .sort((a, b) => b.tokens - a.tokens || (a.name === UNKNOWN_MODEL_FILTER ? 1 : b.name === UNKNOWN_MODEL_FILTER ? -1 : 0)),
    byAgent: [...agents.entries()].map(([name, grouped]) => toSlice(name, grouped)).sort((a, b) => b.tokens - a.tokens),
    sessions: { lead: lead.size, child: child.size },
    sources,
    filters: {
      agents: agentNames.map((r) => r.name),
      models: modelNames.map((r) => r.name ?? UNKNOWN_MODEL_FILTER),
    },
    mixedProjectionSessions: mixed.n,
    excludedV1Items: excludedV1.n,
    history: {
      cutoffDay: utcDay(opts.now - 90 * 86400),
      indexing: sources.some((s) => s.state === "indexing"),
      includesCurrentPartialDay: true,
    },
    warnings: [...warnings].sort(),
  };
}
