import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type Database from "better-sqlite3";
import { openDb } from "./index";
import { insertMutation } from "./mutations";
import {
  INVENTORY_KIND,
  attachInventoryProvenance,
  finalizeInventoryOperation,
  findRestorationProvenance,
  joinInventoryOperations,
  listUnresolvedInventoryOperations,
  prepareInventoryOperation,
  readInventoryOperation,
  recordInventoryStage,
  restoreManagedCapabilities,
} from "./inventory-operations";
import { MutationRejected } from "../../lib/mutationOutcome";
import type { InventoryData, InventoryItem } from "../collectors/inventory";

const OP = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee";
const OP2 = "bbbbbbbb-cccc-4ddd-8eee-ffffffffffff";
const RES = "11111111-2222-4333-8444-555555555555";
const RES2 = "99999999-8888-4777-8666-555555555555";
const ITEM = "a".repeat(64);
const REV = "b".repeat(64);
const CAND = "c".repeat(64);

let db: Database.Database;
beforeEach(() => {
  db = openDb(":memory:");
});
afterEach(() => db.close());

function reserve(operationId = OP, itemId = ITEM, reservationId = RES) {
  return insertMutation(db, {
    ts: "t",
    kind: INVENTORY_KIND,
    operationId,
    itemId,
    change: "disable",
    reservationId,
    outcome: "pending",
  });
}

function prepare(overrides: Record<string, unknown> = {}) {
  return prepareInventoryOperation(db, {
    reservationId: RES,
    operationId: OP,
    itemId: ITEM,
    change: "disable",
    beforeDigest: REV,
    candidateDigest: CAND,
    provenance: { selector: "solo", prior: "allow" },
    executor: "claude",
    ...overrides,
  });
}

function invItem(overrides: Partial<InventoryItem> = {}): InventoryItem {
  return {
    id: ITEM,
    executor: "claude",
    kind: "skill",
    registration: "solo",
    name: "solo",
    description: "",
    path: "/p/solo",
    scope: "user",
    origin: "independent",
    parent: null,
    canonicalTarget: null,
    state: "disabled",
    capabilities: {},
    ...overrides,
  };
}

function invData(items: InventoryItem[]): InventoryData {
  return {
    executors: {
      claude: { ok: true, items },
      codex: { ok: true, items: [] },
      opencode: { ok: true, items: [] },
    },
  };
}

describe("inventory operations over the mutations table", () => {
  it("prepare locates the exact reservation and attaches digests + provenance", () => {
    reserve();
    prepare();
    const view = readInventoryOperation(db, OP);
    expect(view).toMatchObject({ itemId: ITEM, change: "disable", ok: null, stage: "pending" });
    const payload = JSON.parse((db.prepare("SELECT payload FROM mutations").get() as { payload: string }).payload);
    expect(payload).toMatchObject({
      prepared: true,
      reservationId: RES,
      beforeDigest: REV,
      candidateDigest: CAND,
      provenance: { selector: "solo", prior: "allow" },
    });
  });

  it("refuses an unknown reservationId and a mismatched operationId", () => {
    expect(() => prepare({ reservationId: "00000000-0000-4000-8000-000000000000" })).toThrow(MutationRejected);
    reserve();
    expect(() => prepare({ operationId: OP2 })).toThrow(MutationRejected);
  });

  it("refuses a second prepare, a duplicate operation ID and a second pending operation for the item", () => {
    reserve();
    prepare();
    expect(() => prepare()).toThrow(MutationRejected);
    // a different reservation but the SAME operationId
    reserve(OP, ITEM, RES2);
    expect(() => prepare({ reservationId: RES2 })).toThrow(MutationRejected);
    // a different operation for the same item while the first is unresolved
    reserve(OP2, ITEM, "22222222-3333-4444-8555-666666666666");
    expect(() => prepare({ reservationId: "22222222-3333-4444-8555-666666666666", operationId: OP2 })).toThrow(MutationRejected);
  });

  it("finalize settles an existing row idempotently and findRestorationProvenance reads it", () => {
    reserve();
    prepare();
    attachInventoryProvenance(db, OP, { selector: "solo", prior: "allow" });
    finalizeInventoryOperation(db, OP, true, "done", null, { stage: "applied" });
    expect(readInventoryOperation(db, OP)).toMatchObject({ ok: true, outcome: "done", stage: "applied" });
    finalizeInventoryOperation(db, OP, false, "failed", "later", { stage: "conflict" });
    expect(readInventoryOperation(db, OP)).toMatchObject({ ok: true, outcome: "done" });
    expect(findRestorationProvenance(db, ITEM)).toEqual({ selector: "solo", prior: "allow" });
  });

  it("rejects an oversized provenance payload", () => {
    reserve();
    expect(() => prepare({ provenance: { blob: "x".repeat(40 * 1024) } })).toThrow(MutationRejected);
  });

  it("lists unresolved pending AND recorded uncertain outcomes, not settled ones", () => {
    reserve();
    prepare();
    recordInventoryStage(db, OP, "activation-pending", { effect: "activation-pending" });
    finalizeInventoryOperation(db, OP, false, "abandoned", null, { stage: "activation-pending" });
    reserve(OP2, "e".repeat(64), RES2);
    prepare({ reservationId: RES2, operationId: OP2, itemId: "e".repeat(64) });
    finalizeInventoryOperation(db, OP2, true, "done", null, { stage: "applied" });
    const unresolved = listUnresolvedInventoryOperations(db);
    expect(unresolved.map((o) => o.operationId)).toContain(OP);
    expect(unresolved.map((o) => o.operationId)).not.toContain(OP2);
  });

  it("joins an unresolved operation onto its item and keeps an absent item as a recovery row", () => {
    reserve();
    prepare();
    const joined = joinInventoryOperations(db, invData([invItem()]));
    const claude = joined.executors.claude;
    expect(claude.ok).toBe(true);
    if (claude.ok) expect(claude.items[0].operation).toMatchObject({ operationId: OP, stage: "pending" });

    const joinedAbsent = joinInventoryOperations(db, invData([]));
    expect(joinedAbsent.recovery?.[0]).toMatchObject({ operationId: OP, itemId: ITEM, executor: "claude", unavailable: false });
  });

  it("keeps an unavailable source separate from an authoritative absence", () => {
    reserve();
    prepare();
    // the op's OWN bucket failed: an unavailable recovery row, never "removed"
    const failed: InventoryData = {
      executors: {
        claude: { ok: false, error: "inventory-format-unsupported", items: [] },
        codex: { ok: true, items: [] },
        opencode: { ok: true, items: [] },
      },
    };
    expect(joinInventoryOperations(db, failed).recovery?.[0]).toMatchObject({ operationId: OP, executor: "claude", unavailable: true });

    // a legacy record with no recorded executor stays uncertain, never settled
    db.prepare("UPDATE mutations SET payload = ? WHERE json_extract(payload, '$.operationId') = ?").run(
      JSON.stringify({ operationId: OP, itemId: ITEM, change: "disable", stage: "pending" }),
      OP,
    );
    expect(joinInventoryOperations(db, invData([])).recovery?.[0]).toMatchObject({ operationId: OP, unavailable: true });
  });

  it("offers the managed inverse only when a Jax disable provenance exists", () => {
    reserve();
    prepare();
    finalizeInventoryOperation(db, OP, true, "done", null, { stage: "applied" });
    const item = invItem({ capabilities: { enable: { available: false, reason: "no-managed-state" } } });
    restoreManagedCapabilities(db, invData([item]));
    expect(item.capabilities.enable).toEqual({ available: true, reason: null });

    const native = invItem({ id: "f".repeat(64), capabilities: { enable: { available: false, reason: "no-managed-state" } } });
    restoreManagedCapabilities(db, invData([native]));
    expect(native.capabilities.enable).toEqual({ available: false, reason: "no-managed-state" });
  });
});
