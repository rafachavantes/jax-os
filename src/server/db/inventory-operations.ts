// Inventory operation lookups over the EXISTING mutations table (no new
// table/schema, no change to the shared mutation runner). A reservation row is
// inserted by runMutation with a generated reservationId and the client
// operationId; `prepareInventoryOperation` then attaches bounded restoration
// provenance, the recorded pre-operation revision and the candidate digest
// before any native effect.

import type Database from "better-sqlite3";
import { MutationRejected } from "../../lib/mutationOutcome";
import type { InventoryData, InventoryExecutor, InventoryItem, InventoryRecovery, InventoryRevisionPair } from "../collectors/inventory";

export type { InventoryRevisionPair };

export const INVENTORY_KIND = "inventory-action";
// Owner-only, bounded restoration data (spec §7.2): never whole native files.
export const PROVENANCE_CAP = 32 * 1024;
// Stages that still need a human/readback settlement: reload recovery joins
// them onto items (or a recovery row) instead of losing them.
export const UNRESOLVED_STAGES = ["pending", "unconfirmed", "activation-pending", "audit-pending", "conflict"] as const;

export type InventoryOperationView = {
  operationId: string;
  itemId: string;
  change: string;
  ok: boolean | null;
  outcome: string | null;
  error: string | null;
  stage: string;
  effect: string | null;
  executor: string | null;
};

type RawRow = { id: number; ok: number | null; error: string | null; payload: string };

function columns(): string {
  return "id, ok, error, payload";
}

function byReservationId(db: Database.Database, reservationId: string): RawRow | undefined {
  return db
    .prepare(
      `SELECT ${columns()} FROM mutations
       WHERE kind = ? AND json_valid(payload) AND json_extract(payload, '$.reservationId') = ?
       ORDER BY id DESC LIMIT 1`,
    )
    .get(INVENTORY_KIND, reservationId) as RawRow | undefined;
}

function byOperationId(db: Database.Database, operationId: string): RawRow | undefined {
  return db
    .prepare(
      `SELECT ${columns()} FROM mutations
       WHERE kind = ? AND json_valid(payload) AND json_extract(payload, '$.operationId') = ?
       ORDER BY id DESC LIMIT 1`,
    )
    .get(INVENTORY_KIND, operationId) as RawRow | undefined;
}

function parse(payload: string): Record<string, unknown> {
  try {
    const value = JSON.parse(payload);
    return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
  } catch {
    return {};
  }
}

function view(row: RawRow, operationId: string): InventoryOperationView {
  const payload = parse(row.payload);
  return {
    operationId,
    itemId: typeof payload.itemId === "string" ? payload.itemId : "",
    change: typeof payload.change === "string" ? payload.change : "",
    ok: row.ok === null ? null : row.ok === 1,
    outcome: typeof payload.outcome === "string" ? payload.outcome : null,
    error: row.error,
    stage: typeof payload.stage === "string" ? payload.stage : row.ok === null ? "pending" : "settled",
    effect: typeof payload.effect === "string" ? payload.effect : null,
    executor: typeof payload.executor === "string" ? payload.executor : null,
  };
}

export function readInventoryOperation(db: Database.Database, operationId: string): InventoryOperationView | null {
  const row = byOperationId(db, operationId);
  return row ? view(row, operationId) : null;
}

export function readInventoryOperationPayload(db: Database.Database, operationId: string): Record<string, unknown> | null {
  const row = byOperationId(db, operationId);
  return row ? parse(row.payload) : null;
}

export function boundedProvenance(provenance: unknown): unknown {
  if (provenance === undefined || provenance === null) return null;
  let text: string;
  try {
    text = JSON.stringify(provenance);
  } catch {
    throw new MutationRejected("invalid provenance");
  }
  if (Buffer.byteLength(text, "utf8") > PROVENANCE_CAP) throw new MutationRejected("provenance too large");
  return JSON.parse(text);
}

// Runs inside the effect callback, before any native write. Locates the EXACT
// reservation row by its server-generated reservationId (never by operation ID
// alone), refuses a duplicate operation ID, a second pending operation for the
// item, and attaches bounded restoration data, the recorded pre-operation
// revision and the expected candidate digest in one transaction.
export function prepareInventoryOperation(
  db: Database.Database,
  params: {
    reservationId: string;
    operationId: string;
    itemId: string;
    change: string;
    beforeDigest: string;
    candidateDigest?: string | null;
    provenance?: unknown;
    settingsExpected?: InventoryRevisionPair | null;
    settingsCandidate?: InventoryRevisionPair | null;
    executor?: string | null;
    selector?: string | null;
    target?: string | null;
  },
): number {
  const bounded = boundedProvenance(params.provenance);
  return db.transaction((): number => {
    const row = byReservationId(db, params.reservationId);
    if (!row) throw new MutationRejected("inventory reservation missing");
    const payload = parse(row.payload);
    if (typeof payload.operationId === "string" && payload.operationId !== params.operationId) {
      throw new MutationRejected("inventory reservation mismatch", 409);
    }
    if (row.ok !== null) throw new MutationRejected("inventory operation already settled", 409);
    if (payload.prepared === true) throw new MutationRejected("inventory operation already prepared", 409);
    const duplicate = db
      .prepare(
        `SELECT id FROM mutations
         WHERE kind = ? AND id <> ? AND json_valid(payload)
           AND json_extract(payload, '$.operationId') = ?
         LIMIT 1`,
      )
      .get(INVENTORY_KIND, row.id, params.operationId) as { id: number } | undefined;
    if (duplicate) throw new MutationRejected("inventory operation already exists", 409);
    const other = db
      .prepare(
        `SELECT id FROM mutations
         WHERE kind = ? AND id <> ?
           AND json_valid(payload) AND json_extract(payload, '$.itemId') = ?
           AND (
             ok IS NULL
             OR COALESCE(json_extract(payload, '$.stage'), '') IN (${UNRESOLVED_STAGES.map(() => "?").join(", ")})
           )
         LIMIT 1`,
      )
      .get(INVENTORY_KIND, row.id, params.itemId, ...UNRESOLVED_STAGES) as { id: number } | undefined;
    if (other) throw new MutationRejected("inventory operation already pending for item", 409);
    db.prepare("UPDATE mutations SET payload = ? WHERE id = ?").run(
      JSON.stringify({
        ...payload,
        prepared: true,
        itemId: params.itemId,
        change: params.change,
        expectedRevision: params.beforeDigest,
        beforeDigest: params.beforeDigest,
        candidateDigest: params.candidateDigest ?? null,
        provenance: bounded,
        settingsExpected: params.settingsExpected ?? null,
        settingsCandidate: params.settingsCandidate ?? null,
        executor: params.executor ?? null,
        selector: params.selector ?? null,
        target: params.target ?? null,
      }),
      row.id,
    );
    return row.id;
  })();
}

// Persist the post-effect stage/effect (and the helper's actual candidate
// digest). The shared runMutation details callback only carries bytes/backup,
// so inventory-specific outcomes live here, in this domain helper.
export function recordInventoryStage(
  db: Database.Database,
  operationId: string,
  stage: string,
  extra: Record<string, unknown> = {},
): void {
  const row = byOperationId(db, operationId);
  if (!row) return;
  const payload = parse(row.payload);
  db.prepare("UPDATE mutations SET payload = ? WHERE id = ?").run(JSON.stringify({ ...payload, ...extra, stage }), row.id);
}

// Recheck finalization: settle an EXISTING row after a confirmed native
// readback — never a new mutation. A row whose recorded stage is still
// unresolved (unconfirmed / activation-pending / audit-pending / conflict) or
// that was never finalized is settled too; a FINAL settlement is idempotent.
export function finalizeInventoryOperation(
  db: Database.Database,
  operationId: string,
  ok: boolean,
  outcome: "done" | "failed" | "abandoned",
  error: string | null,
  extra: Record<string, unknown> = {},
): void {
  const row = byOperationId(db, operationId);
  if (!row) throw new MutationRejected("inventory operation unknown", 404);
  const payload = parse(row.payload);
  if (payload.settled === true) return; // a final settlement is idempotent
  const finalStage = typeof extra.stage === "string" ? extra.stage : null;
  const final = finalStage !== null && !(UNRESOLVED_STAGES as readonly string[]).includes(finalStage);
  db.prepare("UPDATE mutations SET ok = ?, error = ?, payload = ? WHERE id = ?").run(
    ok ? 1 : 0,
    error,
    JSON.stringify({ ...payload, ...extra, ok, error, outcome, ...(final ? { settled: true } : {}) }),
    row.id,
  );
}

// Attach the bounded restoration provenance the helper returned, after the
// native effect settled. Never rewrites the settled ok/error columns.
export function attachInventoryProvenance(db: Database.Database, operationId: string, provenance: unknown): void {
  const bounded = boundedProvenance(provenance);
  if (bounded === null) return;
  const row = byOperationId(db, operationId);
  if (!row) return;
  const payload = parse(row.payload);
  db.prepare("UPDATE mutations SET payload = ? WHERE id = ?").run(JSON.stringify({ ...payload, provenance: bounded }), row.id);
}

// Latest settled Jax-managed disable for an item, so a re-enable can restore
// the exact prior selector/policy instead of an unconditional allow.
export function findRestorationProvenance(db: Database.Database, itemId: string): unknown {
  const row = db
    .prepare(
      `SELECT payload FROM mutations
       WHERE kind = ? AND ok = 1 AND json_valid(payload)
         AND json_extract(payload, '$.itemId') = ?
         AND json_extract(payload, '$.change') = 'disable'
         AND json_extract(payload, '$.provenance') IS NOT NULL
       ORDER BY id DESC LIMIT 1`,
    )
    .get(INVENTORY_KIND, itemId) as { payload: string } | undefined;
  return row ? parse(row.payload).provenance : null;
}

// Every operation that is not yet settled AND still relevant to a human:
// pending rows plus recorded uncertain/partial outcomes (unconfirmed,
// activation-pending, audit-pending, conflict). A settled `applied` row is
// omitted.
export function listUnresolvedInventoryOperations(db: Database.Database): InventoryOperationView[] {
  const placeholders = UNRESOLVED_STAGES.map(() => "?").join(", ");
  const rows = db
    .prepare(
      `SELECT ${columns()} FROM mutations
       WHERE kind = ? AND json_valid(payload)
         AND json_extract(payload, '$.operationId') IS NOT NULL
         AND (
           ok IS NULL
           OR COALESCE(json_extract(payload, '$.stage'), '') IN (${placeholders})
         )
       ORDER BY id DESC`,
    )
    .all(INVENTORY_KIND, ...UNRESOLVED_STAGES) as RawRow[];
  return rows.map((row) => {
    const payload = parse(row.payload);
    return view(row, typeof payload.operationId === "string" ? payload.operationId : "");
  });
}

// Capability join (Task 1/F1): a disabled skill whose Jax-managed disable
// provenance is recorded offers the supported inverse restore. A native
// disabled skill WITHOUT provenance never receives a fabricated Enable.
export function restoreManagedCapabilities(db: Database.Database, data: InventoryData): InventoryData {
  for (const bucket of Object.values(data.executors)) {
    if (!bucket.ok) continue;
    for (const item of bucket.items as InventoryItem[]) {
      if (item.kind !== "skill" || item.state !== "disabled") continue;
      const enable = item.capabilities.enable;
      if (enable && enable.available) continue;
      if (!findRestorationProvenance(db, item.id)) continue;
      item.capabilities.enable = { available: true, reason: null };
    }
  }
  return data;
}

// Reload recovery (Task 3/F5): join every unresolved operation onto its item,
// and keep a SEPARATE recovery row when the item is absent from discovery. The
// recorded executor decides: its own failed bucket means the item is
// UNAVAILABLE (never "removed"); a legacy record with no provable owner stays
// uncertain; only absence from the op's healthy source is authoritative.
export function joinInventoryOperations(db: Database.Database, data: InventoryData): InventoryData {
  const unresolved = listUnresolvedInventoryOperations(db);
  const byItem = new Map<string, InventoryOperationView>();
  for (const op of unresolved) {
    if (!op.itemId) continue;
    if (!byItem.has(op.itemId)) byItem.set(op.itemId, op);
  }
  const recovery: InventoryRecovery[] = [];
  for (const op of byItem.values()) {
    const executor = op.executor;
    const record = {
      operationId: op.operationId,
      itemId: op.itemId,
      change: op.change,
      stage: op.stage,
      effect: op.effect,
    };
    if (!executor) {
      // no proven owner: uncertain, never presented as a settled removal
      recovery.push({ ...record, unavailable: true });
      continue;
    }
    const bucket = data.executors[executor as InventoryExecutor];
    if (!bucket || !bucket.ok) {
      recovery.push({ ...record, executor, unavailable: true });
      continue;
    }
    const item = (bucket.items as InventoryItem[]).find((candidate) => candidate.id === op.itemId);
    if (item) {
      item.operation = { operationId: op.operationId, change: op.change, stage: op.stage, effect: op.effect, executor, unavailable: false };
      continue;
    }
    recovery.push({ ...record, executor, unavailable: false });
  }
  data.recovery = recovery;
  return data;
}
