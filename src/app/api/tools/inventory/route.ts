import { createHash, randomUUID } from "node:crypto";
import { NextResponse } from "next/server";
import { readJsonCapped, requireSameOrigin } from "@/server/api";
import { getDb } from "@/server/db";
import {
  applyInventory,
  getInventoryData,
  inventoryRevision,
  prepareInventory,
  previewInventory,
  reconcileInventory,
  systemInventoryDeps,
  type InventoryChange,
  type InventoryData,
  type InventoryPreparation,
  type InventoryPreview,
  type InventoryReadback,
} from "@/server/collectors/inventory";
import {
  INVENTORY_KIND,
  attachInventoryProvenance,
  finalizeInventoryOperation,
  findRestorationProvenance,
  joinInventoryOperations,
  prepareInventoryOperation,
  readInventoryOperation,
  readInventoryOperationPayload,
  recordInventoryStage,
  restoreManagedCapabilities,
  type InventoryRevisionPair,
} from "@/server/db/inventory-operations";
import { runMutation } from "@/server/mutations";
import { MutationRejected } from "@/lib/mutationOutcome";

export const dynamic = "force-dynamic";

const HEX64 = /^[0-9a-f]{64}$/;
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
const CHANGES: readonly InventoryChange[] = ["disable", "enable", "remove-registration", "remove-installation"];

export type InventoryRouteDeps = {
  getData: () => Promise<InventoryData>;
  preview: (itemId: string, change: InventoryChange, provenance?: unknown) => Promise<InventoryPreview>;
  prepare: (itemId: string, change: InventoryChange, expectedRevision: string, provenance?: unknown, operationId?: string) => Promise<InventoryPreparation>;
  apply: (itemId: string, change: InventoryChange, revision: string, operationId: string, provenance?: unknown) => Promise<unknown>;
  reconcile: (expected: unknown, operationId: string) => Promise<unknown>;
  revision: (itemId: string, executor?: string) => Promise<InventoryReadback>;
};

const systemDeps: InventoryRouteDeps = {
  getData: () => getInventoryData(),
  preview: (itemId, change, provenance) => previewInventory(systemInventoryDeps(), itemId, change, provenance),
  prepare: (itemId, change, expectedRevision, provenance, operationId) =>
    prepareInventory(systemInventoryDeps(), itemId, change, expectedRevision, provenance, operationId),
  apply: (itemId, change, revision, operationId, provenance) =>
    applyInventory(systemInventoryDeps(), itemId, change, revision, operationId, provenance),
  // The reconcile expected-state comes from the RECORDED operation (route), not
  // a fresh live snapshot.
  reconcile: (expected, operationId) => reconcileInventory(systemInventoryDeps(), expected, operationId),
  revision: (itemId, executor) => inventoryRevision(systemInventoryDeps(), itemId, executor),
};

function isObj(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function keysOk(obj: Record<string, unknown>, required: string[], optional: string[] = []): boolean {
  const allow = new Set([...required, ...optional]);
  for (const key of Object.keys(obj)) if (!allow.has(key)) return false;
  return required.every((key) => Object.hasOwn(obj, key));
}

function depsOf(req: Request): InventoryRouteDeps {
  return (req as Request & { jaxDeps?: InventoryRouteDeps }).jaxDeps ?? systemDeps;
}

function asRevisionPair(value: unknown): InventoryRevisionPair | null {
  if (!isObj(value)) return null;
  const { settings, sources } = value as { settings?: unknown; sources?: unknown };
  if (typeof settings !== "string" || !isObj(sources)) return null;
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(sources)) if (typeof v === "string") out[k] = v;
  return { settings, sources: out };
}

export async function GET(req: Request) {
  const deps = depsOf(req);
  if (req.method === "GET") {
    const url = new URL(req.url);
    const operation = url.searchParams.get("operation");
    if (operation !== null) {
      if (!UUID_RE.test(operation)) return NextResponse.json({ ok: false, error: "invalid operation" }, { status: 400 });
      const view = readInventoryOperation(getDb(), operation);
      if (!view) return NextResponse.json({ ok: false, error: "operation unknown" }, { status: 404 });
      return NextResponse.json({ ok: true, data: view });
    }
  }
  try {
    const data = await deps.getData();
    const db = getDb();
    // Reload recovery + managed-provenance capabilities are joined server-side,
    // so the client never carries operation identity in a ref.
    return NextResponse.json({ ok: true, data: joinInventoryOperations(db, restoreManagedCapabilities(db, data)) });
  } catch (e) {
    return NextResponse.json({ ok: false, error: e instanceof Error ? e.message : String(e) });
  }
}

export async function POST(req: Request) {
  const csrf = requireSameOrigin(req);
  if (csrf) return csrf;
  const body = await readJsonCapped(req, 16 * 1024);
  if (!body.ok) return NextResponse.json({ ok: false, error: body.error }, { status: 400 });
  if (!isObj(body.value)) return NextResponse.json({ ok: false, error: "invalid payload" }, { status: 400 });
  const action = body.value.action;
  const deps = depsOf(req);

  if (action === "preview") {
    if (!keysOk(body.value, ["action", "itemId", "change"])) return invalid();
    const { itemId, change } = body.value as { itemId: unknown; change: unknown };
    if (typeof itemId !== "string" || !HEX64.test(itemId)) return invalid();
    if (typeof change !== "string" || !(CHANGES as readonly string[]).includes(change)) return invalid();
    try {
      // The restoration record is looked up and forwarded server-side; the
      // browser can never supply it, and never receives it back.
      const provenance = change === "enable" ? findRestorationProvenance(getDb(), itemId) : undefined;
      const preview = await deps.preview(itemId, change as InventoryChange, provenance);
      const { restoration: _restoration, ...safe } = preview as InventoryPreview & { restoration?: unknown };
      return NextResponse.json({ ok: true, data: safe });
    } catch (e) {
      return NextResponse.json({ ok: false, error: e instanceof Error ? e.message : String(e) });
    }
  }

  if (action === "apply") {
    if (!keysOk(body.value, ["action", "itemId", "change", "expectedRevision", "operationId"])) return invalid();
    const { itemId, change, expectedRevision, operationId } = body.value as Record<string, unknown>;
    if (typeof itemId !== "string" || !HEX64.test(itemId)) return invalid();
    if (typeof change !== "string" || !(CHANGES as readonly string[]).includes(change)) return invalid();
    if (typeof expectedRevision !== "string" || !HEX64.test(expectedRevision)) return invalid();
    if (typeof operationId !== "string" || !UUID_RE.test(operationId)) return invalid();
    // An existing operation ID returns that operation's status, never a relisten.
    const existing = readInventoryOperation(getDb(), operationId);
    if (existing) return NextResponse.json({ ok: true, data: existing });
    const provenance = change === "enable" ? findRestorationProvenance(getDb(), itemId) : undefined;
    // Finite internal preparation from actual current native state, BEFORE any
    // reservation/effect. A changed preview is rejected here.
    let preparation: InventoryPreparation;
    try {
      preparation = await deps.prepare(itemId, change as InventoryChange, expectedRevision, provenance, operationId);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      if (e instanceof MutationRejected) return NextResponse.json({ ok: false, error: message, effect: "not-applied" }, { status: e.status });
      return NextResponse.json({ ok: false, error: message, effect: "not-applied" }, { status: 409 });
    }
    const reservationId = randomUUID();
    const result = await runMutation(
      { ts: new Date().toISOString(), kind: INVENTORY_KIND, operationId, itemId, change, revision: expectedRevision, reservationId },
      async () => {
        // Bind the exact server-generated reservation ID and persist the
        // complete preparation in one transaction before native apply.
        prepareInventoryOperation(getDb(), {
          reservationId,
          operationId,
          itemId,
          change,
          beforeDigest: preparation.beforeDigest,
          candidateDigest: preparation.candidateDigest,
          provenance: preparation.provenance ?? provenance,
          settingsExpected: preparation.settingsExpected,
          settingsCandidate: preparation.settingsCandidate,
          executor: preparation.executor,
          selector: preparation.selector,
          target: preparation.target,
        });
        return deps.apply(itemId, change as InventoryChange, expectedRevision, operationId, provenance);
      },
      () => ({}),
    );
    if (result.ok) {
      const value = isObj(result.value) ? result.value : {};
      const effect = typeof value.effect === "string" ? value.effect : null;
      const candidateDigest = typeof value.candidateDigest === "string" ? value.candidateDigest : undefined;
      // A partial publisher response has no editor_revision: never overwrite the
      // prepared candidate facts with an absent post-effect value.
      const settingsCandidate = asRevisionPair(value.editor_revision) ?? preparation.settingsCandidate ?? undefined;
      // A published native change whose settings write failed is
      // activation-pending, not "applied".
      const stage = effect === "activation-pending" ? "activation-pending" : "applied";
      recordInventoryStage(getDb(), operationId, stage, {
        effect,
        ...(candidateDigest !== undefined ? { candidateDigest } : {}),
        ...(settingsCandidate != null ? { settingsCandidate } : {}),
        requiresNewSession: typeof value.requiresNewSession === "boolean" ? value.requiresNewSession : null,
      });
      if (value.provenance !== undefined) attachInventoryProvenance(getDb(), operationId, value.provenance);
      return NextResponse.json({ ok: true, data: { operationId, ...value } });
    }
    // Preserve the native result fields even when audit finalization failed.
    const { value, ...failure } = result;
    const failEffect = failure.effect as string;
    const appliedEffect = failEffect === "applied" || failEffect === "published";
    const stage = failEffect === "unconfirmed" ? "unconfirmed" : appliedEffect ? "audit-pending" : "not-applied";
    recordInventoryStage(getDb(), operationId, stage, { effect: failEffect ?? null });
    return NextResponse.json(value === undefined ? failure : { ...failure, data: value }, { status: failure.status ?? 500 });
  }

  if (action === "recheck") {
    if (!keysOk(body.value, ["action", "operationId"])) return invalid();
    const { operationId } = body.value as { operationId: unknown };
    if (typeof operationId !== "string" || !UUID_RE.test(operationId)) return invalid();
    const row = readInventoryOperation(getDb(), operationId);
    if (!row) return NextResponse.json({ ok: false, error: "operation unknown" }, { status: 404 });
    const payload = readInventoryOperationPayload(getDb(), operationId) ?? {};
    const before = typeof payload.beforeDigest === "string" ? payload.beforeDigest : null;
    const candidate = typeof payload.candidateDigest === "string" ? payload.candidateDigest : null;
    const settingsCandidate = asRevisionPair(payload.settingsCandidate);
    const executor = typeof payload.executor === "string" ? payload.executor : undefined;
    if (payload.settled === true) {
      return NextResponse.json({ ok: true, data: readInventoryOperation(getDb(), operationId) ?? row });
    }

    // Read the CURRENT revision directly — never preview eligibility — scoped to
    // the recorded executor so a failed unrelated bucket is never "absent".
    let current: InventoryReadback;
    try {
      current = await deps.revision(row.itemId, executor);
    } catch {
      // a failed source read never proves the effect
      recordInventoryStage(getDb(), operationId, "unconfirmed", { effect: "unconfirmed" });
      return NextResponse.json({ ok: true, data: readInventoryOperation(getDb(), operationId) ?? { ...row, ok: false, stage: "unconfirmed" } });
    }

    const persistedOr = (fallback: typeof row) => readInventoryOperation(getDb(), operationId) ?? fallback;
    if (candidate !== null && current.revision === candidate) {
      // Native state matches the candidate, but an OpenCode publication is not
      // fully applied until the settings file carries the candidate sources.
      if (settingsCandidate && current.settings && !sameSources(current.settings.sources, settingsCandidate.sources)) {
        recordInventoryStage(getDb(), operationId, "activation-pending", { effect: "activation-pending" });
        return NextResponse.json({ ok: true, data: persistedOr({ ...row, ok: null, stage: "activation-pending" }) });
      }
      finalizeInventoryOperation(getDb(), operationId, true, "done", null, { stage: "applied" });
      return NextResponse.json({ ok: true, data: persistedOr({ ...row, ok: true, outcome: "done", stage: "applied" }) });
    }
    if (before !== null && current.revision === before) {
      finalizeInventoryOperation(getDb(), operationId, false, "abandoned", "native state unchanged since the attempt", { stage: "not-applied" });
      return NextResponse.json({ ok: true, data: persistedOr({ ...row, ok: false, outcome: "abandoned", stage: "not-applied" }) });
    }
    finalizeInventoryOperation(getDb(), operationId, false, "failed", "native state does not match the recorded candidate", { stage: "conflict" });
    return NextResponse.json({ ok: true, data: persistedOr({ ...row, ok: false, outcome: "failed", stage: "conflict" }) });
  }

  // Settings-only reconciliation is confirmed by its OWN bounded revision, not a
  // fresh live item preview (which cannot exist for an already-removed item).
  if (action === "preview-reconcile") {
    if (!keysOk(body.value, ["action", "operationId"])) return invalid();
    const { operationId } = body.value as { operationId: unknown };
    if (typeof operationId !== "string" || !UUID_RE.test(operationId)) return invalid();
    const row = readInventoryOperation(getDb(), operationId);
    if (!row) return NextResponse.json({ ok: false, error: "operation unknown" }, { status: 404 });
    if (row.stage !== "activation-pending") {
      return NextResponse.json({ ok: false, error: "operation is not a partial publication" }, { status: 409 });
    }
    const payload = readInventoryOperationPayload(getDb(), operationId) ?? {};
    const candidate = asRevisionPair(payload.settingsCandidate);
    const recorded = asRevisionPair(payload.settingsExpected);
    const candidateDigest = typeof payload.candidateDigest === "string" ? payload.candidateDigest : null;
    if (!candidate || !recorded || candidateDigest === null) {
      return NextResponse.json({ ok: false, error: "operation has no recorded publication revision" }, { status: 409 });
    }
    const executor = typeof payload.executor === "string" ? payload.executor : undefined;
    let current: InventoryReadback;
    try {
      current = await deps.revision(row.itemId, executor);
    } catch {
      return NextResponse.json({ ok: false, error: "operation source is unavailable" }, { status: 409 });
    }
    const nativeOk = candidateDigest === "absent" ? current.revision === "absent" : current.revision === candidateDigest;
    if (!nativeOk) {
      return NextResponse.json({ ok: false, error: "native state changed since the operation" }, { status: 409 });
    }
    return NextResponse.json({
      ok: true,
      data: {
        operationId,
        itemId: row.itemId,
        change: row.change,
        stage: row.stage,
        revision: reconcileConfirmation(candidateDigest, candidate),
        effect: "settings-only",
        settingsOnly: true,
        requiresNewSession: false,
      },
    });
  }

  if (action === "reconcile") {
    if (!keysOk(body.value, ["action", "operationId", "expectedRevision"])) return invalid();
    const { operationId, expectedRevision } = body.value as Record<string, unknown>;
    if (typeof operationId !== "string" || !UUID_RE.test(operationId)) return invalid();
    if (typeof expectedRevision !== "string" || !HEX64.test(expectedRevision)) return invalid();
    const row = readInventoryOperation(getDb(), operationId);
    if (!row) return NextResponse.json({ ok: false, error: "operation unknown" }, { status: 404 });
    const payload = readInventoryOperationPayload(getDb(), operationId) ?? {};
    if (row.stage !== "activation-pending") {
      return NextResponse.json({ ok: false, error: "operation is not a partial publication" }, { status: 409 });
    }
    const recorded = asRevisionPair(payload.settingsExpected);
    const candidate = asRevisionPair(payload.settingsCandidate);
    const candidateDigest = typeof payload.candidateDigest === "string" ? payload.candidateDigest : null;
    if (!recorded || !candidate) {
      return NextResponse.json({ ok: false, error: "operation has no recorded publication revision" }, { status: 409 });
    }
    // The client's confirmed revision must be the one captured for THIS recorded
    // native candidate and pre-operation settings revision.
    if (expectedRevision !== reconcileConfirmation(candidateDigest, candidate)) {
      return NextResponse.json({ ok: false, error: "native state changed since the operation" }, { status: 409 });
    }
    // Compare against the RECORDED pre-operation settings revision and the
    // RECORDED candidate native sources — never a fresh live snapshot.
    const expected = { settings: recorded.settings, sources: candidate.sources };
    try {
      const data = await deps.reconcile(expected, operationId);
      recordInventoryStage(getDb(), operationId, "applied", { effect: "published" });
      return NextResponse.json({ ok: true, data });
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      if (e instanceof MutationRejected) return NextResponse.json({ ok: false, error: message }, { status: e.status });
      return NextResponse.json({ ok: false, error: message });
    }
  }

  return invalid();
}

function sameSources(a: Record<string, string>, b: Record<string, string>): boolean {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  for (const key of keys) if (a[key] !== b[key]) return false;
  return true;
}

function reconcileConfirmation(candidateDigest: string | null, candidate: InventoryRevisionPair): string {
  return createHash("sha256").update(JSON.stringify({ candidateDigest, sources: candidate.sources })).digest("hex");
}

function invalid() {
  return NextResponse.json({ ok: false, error: "invalid payload" }, { status: 400 });
}
