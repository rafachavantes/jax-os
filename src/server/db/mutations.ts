import type Database from "better-sqlite3";

export type MutationEntry = Record<string, unknown>;
export type MutationRow = {
  id: number;
  ts: string;
  kind: string;
  ok: boolean | null;
  error: string | null;
  payload: unknown;
};
export type MutationCursor = { ts: string; id: number };
export type MutationQuery = { kind?: string; from?: string; to?: string; limit: number; cursor?: MutationCursor };
export type MutationPage = { rows: MutationRow[]; kinds: string[]; total: number; nextCursor: string | null };

const CURSOR_RAW_CAP = 1024;
const CURSOR_TS_CAP = 256;

export function encodeMutationCursor(ts: string, id: number): string {
  return Buffer.from(JSON.stringify([ts, id]), "utf8").toString("base64url");
}

export function decodeMutationCursor(raw: string): MutationCursor | null {
  if (typeof raw !== "string" || Buffer.byteLength(raw, "utf8") > CURSOR_RAW_CAP) return null;
  let bytes: Buffer;
  try {
    bytes = Buffer.from(raw, "base64url");
  } catch {
    return null;
  }
  if (bytes.length === 0 || bytes.toString("base64url") !== raw) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(bytes.toString("utf8"));
  } catch {
    return null;
  }
  if (!Array.isArray(parsed) || parsed.length !== 2) return null;
  const [ts, id] = parsed;
  if (typeof ts !== "string" || ts.length === 0 || Buffer.byteLength(ts, "utf8") > CURSOR_TS_CAP) return null;
  if (/[\u0000-\u001f\u007f]/.test(ts)) return null;
  if (typeof id !== "number" || !Number.isSafeInteger(id) || id <= 0) return null;
  return { ts, id };
}

// Common columns extracted; the full entry survives intact as JSON payload —
// real entries are heterogeneous (take-control has no ok at all), so fixed
// per-kind columns would lie. kind falls back instead of throwing so a
// malformed entry still records.
export function extractMutation(entry: MutationEntry): {
  ts: string;
  kind: string;
  ok: number | null;
  error: string | null;
  payload: string;
} {
  return {
    ts: typeof entry.ts === "string" && entry.ts ? entry.ts : new Date().toISOString(),
    kind: typeof entry.kind === "string" && entry.kind ? entry.kind : "unknown",
    ok: typeof entry.ok === "boolean" ? (entry.ok ? 1 : 0) : null,
    error: typeof entry.error === "string" ? entry.error : null,
    payload: JSON.stringify(entry),
  };
}

export function insertMutation(db: Database.Database, entry: MutationEntry): number {
  const m = extractMutation(entry);
  const result = db.prepare("INSERT INTO mutations (ts, kind, ok, error, payload) VALUES (?, ?, ?, ?, ?)").run(
    m.ts,
    m.kind,
    m.ok,
    m.error,
    m.payload,
  );
  return Number(result.lastInsertRowid);
}

export function finishMutation(
  db: Database.Database,
  id: number,
  ok: boolean,
  error: string | null,
  outcome: "done" | "failed" | "abandoned",
  details: Record<string, unknown> = {},
): void {
  const row = db.prepare("SELECT payload FROM mutations WHERE id = ?").get(id) as { payload: string } | undefined;
  if (!row) throw new Error("mutation not found");
  const payload = safeParse(row.payload);
  const base = payload && typeof payload === "object" && !Array.isArray(payload)
    ? payload as Record<string, unknown>
    : {};
  db.prepare("UPDATE mutations SET ok = ?, error = ?, payload = ? WHERE id = ?").run(
    ok ? 1 : 0,
    error,
    JSON.stringify({ ...base, ...details, ok, error, outcome }),
    id,
  );
}

type RawRow = { id: number; ts: string; kind: string; ok: number | null; error: string | null; payload: string };

function mapRow(r: RawRow): MutationRow {
  return {
    id: r.id,
    ts: r.ts,
    kind: r.kind,
    ok: r.ok === null ? null : r.ok === 1,
    error: r.error,
    payload: safeParse(r.payload),
  };
}

export function queryMutations(db: Database.Database, q: MutationQuery): MutationPage {
  const filter: string[] = [];
  const filterArgs: unknown[] = [];
  if (q.kind) { filter.push("kind = ?"); filterArgs.push(q.kind); }
  if (q.from) { filter.push("ts >= ?"); filterArgs.push(q.from); }
  if (q.to) { filter.push("ts <= ?"); filterArgs.push(q.to); }
  const page = [...filter];
  const pageArgs: unknown[] = [...filterArgs];
  if (q.cursor) {
    page.push("(ts < ? OR (ts = ? AND id < ?))");
    pageArgs.push(q.cursor.ts, q.cursor.ts, q.cursor.id);
  }
  const pageWhere = page.length > 0 ? ` WHERE ${page.join(" AND ")}` : "";
  const filterWhere = filter.length > 0 ? ` WHERE ${filter.join(" AND ")}` : "";

  const fetched = db
    .prepare(`SELECT id, ts, kind, ok, error, payload FROM mutations${pageWhere} ORDER BY ts DESC, id DESC LIMIT ?`)
    .all(...pageArgs, q.limit + 1) as RawRow[];
  const hasMore = fetched.length > q.limit;
  const sliced = hasMore ? fetched.slice(0, q.limit) : fetched;
  const rows = sliced.map(mapRow);
  const last = rows[rows.length - 1];
  const nextCursor = hasMore && last ? encodeMutationCursor(last.ts, last.id) : null;
  const total = (db.prepare(`SELECT COUNT(*) AS c FROM mutations${filterWhere}`).get(...filterArgs) as { c: number }).c;
  const kinds = (db.prepare("SELECT DISTINCT kind FROM mutations ORDER BY kind").all() as { kind: string }[]).map(
    (r) => r.kind,
  );
  return { rows, kinds, total, nextCursor };
}

function safeParse(s: string): unknown {
  try {
    return JSON.parse(s);
  } catch {
    return s; // corrupt payload → show the raw string rather than crash
  }
}
