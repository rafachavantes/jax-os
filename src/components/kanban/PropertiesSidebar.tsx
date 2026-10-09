"use client";

import { useQuery, useQueryClient, type QueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import type { Envelope } from "@/lib/api";
import type { MutationFailure } from "@/lib/mutationOutcome";
import type { KanbanIssue, KanbanLabel, KanbanState, LinearMember } from "@/server/collectors/linear";

// Linear priority: 0 none, 1 urgent, 2 high, 3 medium, 4 low (Linear's own dropdown order)
export const PRIORITY_LABEL: Record<number, string> = {
  0: "priorityNone",
  1: "priorityUrgent",
  2: "priorityHigh",
  3: "priorityMedium",
  4: "priorityLow",
};
export const PRIORITY_ORDER = [0, 1, 2, 3, 4];

const SELECT_CLASS =
  "w-full rounded-md border border-line bg-surface-2 px-2 py-1.5 text-[13px] text-ink outline-none focus:border-line-strong";

// Result of a sanctioned mutation POST, preserving the server's effect/audit
// metadata (src/server/mutations.ts): a bare {ok:false,error} means the route
// rejected the request outright, MutationFailure carries the uncertainty
// classification the UI must not flatten into a plain "failed".
export type WriteOutcome =
  | { ok: true }
  | (MutationFailure & { ok: false })
  | { ok: false; code?: undefined; effect?: undefined; audit?: undefined; error: string };

export type WriteGuidance =
  | { kind: "refused"; error: string }
  | { kind: "unconfirmed" }
  | { kind: "applied-unrecorded" };

const UNCONFIRMED: WriteOutcome = {
  ok: false,
  code: "mutation-unconfirmed",
  effect: "unconfirmed",
  audit: "unavailable",
  error: "unavailable",
};

const FAILURE_CODES = new Set<string>([
  "audit-unavailable",
  "mutation-rejected",
  "mutation-unconfirmed",
  "audit-finalization-failed",
]);
const FAILURE_EFFECTS = new Set<string>(["not-applied", "unconfirmed", "applied"]);
const FAILURE_AUDITS = new Set<string>(["unavailable", "pending", "recorded"]);

// shared write helper (also used by the comment composer, rule apply and board
// moves) — POST + classify. Transport/JSON failures AND syntactically valid
// but malformed responses (null, arrays, unknown shapes, invalid ok value or
// metadata) are unconfirmed, never a proven refusal or success. Only the
// explicit legacy {ok:false,error:string} rejection and fully validated
// MutationFailure combinations are classified refused/uncertain/applied.
export async function write(url: string, body: unknown): Promise<WriteOutcome> {
  let res: unknown;
  try {
    res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => r.json());
  } catch {
    return UNCONFIRMED;
  }
  if (typeof res !== "object" || res === null || Array.isArray(res)) return UNCONFIRMED;
  const env = res as Record<string, unknown>;
  if (env.ok === true) return { ok: true };
  if (env.ok === false && typeof env.error === "string") {
    if (env.code === undefined && env.effect === undefined && env.audit === undefined) {
      return { ok: false, error: env.error };
    }
    if (
      typeof env.code === "string" &&
      FAILURE_CODES.has(env.code) &&
      typeof env.effect === "string" &&
      FAILURE_EFFECTS.has(env.effect) &&
      typeof env.audit === "string" &&
      FAILURE_AUDITS.has(env.audit)
    ) {
      return {
        ok: false,
        code: env.code as MutationFailure["code"],
        effect: env.effect as MutationFailure["effect"],
        audit: env.audit as MutationFailure["audit"],
        error: env.error,
      };
    }
  }
  return UNCONFIRMED;
}

// synchronous per-field reservation: admits the first call for a field and
// rejects a duplicate BEFORE any request promise exists; independent fields
// stay usable. The component guards each write through this same function.
export function reserveField(inflight: Set<string>, key: string): boolean {
  if (inflight.has(key)) return false;
  inflight.add(key);
  return true;
}

// production write runner used by the sidebar fields: reserve synchronously,
// run the lazy request, always release the reservation. A duplicate same-field
// call resolves null and never creates a request.
export async function reserveAndWrite(
  inflight: Set<string>,
  key: string,
  post: () => Promise<WriteOutcome>,
): Promise<WriteOutcome | null> {
  if (!reserveField(inflight, key)) return null;
  try {
    return await post();
  } finally {
    inflight.delete(key);
  }
}

// toggling a label replaces the whole set; derive the next set from the labels
// the issue currently shows, never from a stale copy.
export function nextLabelSet(current: string[], labelId: string): string[] {
  return current.includes(labelId) ? current.filter((x) => x !== labelId) : [...current, labelId];
}

// post-write label-set reservation decision: only a confirmed refusal may
// release immediately (nothing changed). A successful POST keeps the lock
// until a refetch shows the returned set converged; uncertain and
// applied-unrecorded results must also reconverge before another set is
// derived from stale labels (U2: no clobber, no write from stale labels).
export function releaseLabelLock(outcome: WriteOutcome): boolean {
  return writeGuidance(outcome)?.kind === "refused";
}

export function writeGuidance(outcome: WriteOutcome): WriteGuidance | null {
  if (outcome.ok) return null;
  if (outcome.code === undefined) return { kind: "refused", error: outcome.error };
  if (outcome.effect === "applied") return { kind: "applied-unrecorded" };
  if (outcome.effect === "unconfirmed") return { kind: "unconfirmed" };
  return { kind: "refused", error: outcome.error };
}

// Apply/rollback decision shared by move's board snapshot and title/
// description's detail snapshot (phase 6 spec §6.3, Decision 11) — generic
// over the snapshot's shape. NOT used for comments: a full-snapshot rollback
// would also discard a second comment or draft typed in the meantime: see
// reconcileComment in src/lib/commentSubmit.ts instead.
export function reconcileWrite<T>(snapshot: T, applied: T, outcome: WriteGuidance | null): T {
  return outcome?.kind === "refused" ? snapshot : applied;
}

export type CreateIssueOutcome =
  | { ok: true; id: string; identifier: string }
  | (MutationFailure & { ok: false })
  | { ok: false; code?: undefined; effect?: undefined; audit?: undefined; error: string };

// Round-2 F2: same value as UNCONFIRMED, but typed CreateIssueOutcome. UNCONFIRMED is
// typed WriteOutcome, whose {ok:true} variant has no id/identifier and is not
// assignable to CreateIssueOutcome's success case — reusing it here is a
// `tsc` build error, not just a lint nit.
const UNCONFIRMED_CREATE: CreateIssueOutcome = { ok: false, code: "mutation-unconfirmed", effect: "unconfirmed", audit: "unavailable", error: "unavailable" };

// Pure mapper: turns the /api/kanban/create envelope into a typed outcome —
// same three-shape classification `write` already uses, but preserving
// id/identifier on success so the caller can open the new issue.
export function parseCreateResponse(res: unknown): CreateIssueOutcome {
  if (typeof res !== "object" || res === null || Array.isArray(res)) return UNCONFIRMED_CREATE;
  const env = res as Record<string, unknown>;
  if (env.ok === true && typeof env.id === "string" && typeof env.identifier === "string") {
    return { ok: true, id: env.id, identifier: env.identifier };
  }
  if (env.ok === false && typeof env.error === "string") {
    if (env.code === undefined && env.effect === undefined && env.audit === undefined) {
      return { ok: false, error: env.error };
    }
    if (
      typeof env.code === "string" && FAILURE_CODES.has(env.code) &&
      typeof env.effect === "string" && FAILURE_EFFECTS.has(env.effect) &&
      typeof env.audit === "string" && FAILURE_AUDITS.has(env.audit)
    ) {
      return {
        ok: false,
        code: env.code as MutationFailure["code"],
        effect: env.effect as MutationFailure["effect"],
        audit: env.audit as MutationFailure["audit"],
        error: env.error,
      };
    }
  }
  return UNCONFIRMED_CREATE;
}

export async function postCreateIssue(body: unknown): Promise<CreateIssueOutcome> {
  let res: unknown;
  try {
    res = await fetch("/api/kanban/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => r.json());
  } catch {
    return UNCONFIRMED_CREATE;
  }
  return parseCreateResponse(res);
}

// Reserve/POST/invalidate mechanics shared by every property field. Exported
// so IssueDetailOverlay's title/description edits call the exact same runner
// PropertiesSidebar's own runField (below) uses internally. Returns null for
// a duplicate same-field call rejected before any request was sent (nothing
// to classify or reconcile); otherwise the raw outcome, left for the caller
// to classify via writeGuidance.
export async function runWriteField(
  inflight: Set<string>,
  key: string,
  start: () => Promise<WriteOutcome>,
  queryClient: QueryClient,
  issueId: string,
): Promise<WriteOutcome | null> {
  const outcome = await reserveAndWrite(inflight, key, start);
  if (!outcome) return null;
  queryClient.invalidateQueries({ queryKey: ["kanban", "board"] });
  queryClient.invalidateQueries({ queryKey: ["kanban", "issue", issueId] });
  return outcome;
}

export type Translator = (key: string, params?: Record<string, string | number | Date>) => string;

// kanban-namespace guidance; rules keeps its own keys via writeGuidance.
export function guidanceText(t: Translator, g: WriteGuidance): string {
  if (g.kind === "refused") return t("writeError", { error: g.error });
  if (g.kind === "unconfirmed") return t("writeUnconfirmed");
  return t("writeAppliedUnrecorded");
}

// order-insensitive id-set equality — used to detect when a refetch reflects a
// pending label write (the convergence seam the toggle sequence tests drive)
export function sameLabelSet(a: string[], b: string[]) {
  return a.length === b.length && a.every((x) => b.includes(x));
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-[11px] font-semibold uppercase tracking-wide text-muted">{label}</span>
      {children}
    </div>
  );
}

export function PropertiesSidebar({
  issue,
  states,
  teamId,
}: {
  issue: KanbanIssue;
  states: KanbanState[];
  teamId: string;
}) {
  const t = useTranslations("kanban");
  const queryClient = useQueryClient();
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  // synchronous per-issue/field reservation: one in-flight request per field,
  // reserved before any request promise exists; independent fields stay usable
  const inflight = useRef(new Set<string>());

  const membersQuery = useQuery<Envelope<LinearMember[]>>({
    queryKey: ["kanban", "members", teamId],
    queryFn: async () => (await fetch(`/api/kanban/members?team=${teamId}`)).json(),
  });
  const members = membersQuery.data?.ok ? membersQuery.data.data : [];

  const labelsQuery = useQuery<Envelope<KanbanLabel[]>>({
    queryKey: ["kanban", "labels", teamId],
    queryFn: async () => (await fetch(`/api/kanban/labels?team=${teamId}`)).json(),
  });
  const labels = labelsQuery.data?.ok ? labelsQuery.data.data : [];

  // every write reconverges on Linear truth: refetch board + this issue's detail, success or fail
  async function runField(key: string, start: () => Promise<WriteOutcome>): Promise<WriteOutcome | null> {
    setError(null);
    setPending(true);
    const outcome = await runWriteField(inflight.current, key, start, queryClient, issue.id);
    setPending(inflight.current.size > 0);
    if (outcome) {
      const g = writeGuidance(outcome);
      if (g) setError(guidanceText(t, g));
    }
    return outcome;
  }

  const selectedLabelIds = issue.labels.map((l) => l.id);

  // in-flight lock for label writes. `labelIds` replaces the whole set, so two rapid toggles that
  // both derive from a stale `issue.labels` clobber each other (last-write-wins). We hold the lock —
  // and keep the buttons disabled — until a refetch makes `issue.labels` reflect the set we sent,
  // so the next toggle always derives from settled truth. On error we release at once (nothing changed).
  // ponytail: lifts when a refetch shows the exact set we sent; assumes Linear returns the applied
  // labels on the next read (its normal read-after-write). A refetch racing ahead of Linear's
  // consistency just keeps the buttons locked until the 10s poll catches up — still no clobber.
  const [pendingLabels, setPendingLabels] = useState<string[] | null>(null);
  const savingLabels = pendingLabels !== null;
  useEffect(() => {
    if (pendingLabels && sameLabelSet(pendingLabels, selectedLabelIds)) setPendingLabels(null);
  });

  async function toggleLabel(id: string) {
    if (savingLabels) return;
    if (!reserveField(inflight.current, "labels")) return; // synchronous per-field guard
    const next = nextLabelSet(selectedLabelIds, id);
    setPendingLabels(next);
    setError(null);
    try {
      const outcome = await write("/api/kanban/update", { issueId: issue.id, patch: { labelIds: next } });
      const g = writeGuidance(outcome);
      if (g) setError(guidanceText(t, g));
      if (releaseLabelLock(outcome)) setPendingLabels(null);
      return outcome;
    } finally {
      inflight.current.delete("labels");
      setPending(inflight.current.size > 0);
      queryClient.invalidateQueries({ queryKey: ["kanban", "board"] });
      queryClient.invalidateQueries({ queryKey: ["kanban", "issue", issue.id] });
    }
  }

  return (
    <div className="flex flex-col gap-5">
      <Row label={t("propStatus")}>
        <select
          className={SELECT_CLASS}
          aria-label={t("propStatus")}
          value={issue.stateId}
          onChange={(e) => void runField("status", () => write("/api/kanban/move", { issueId: issue.id, stateId: e.target.value }))}
        >
          {states.map((s) => (
            <option key={s.id} value={s.id}>
              {s.name}
            </option>
          ))}
        </select>
      </Row>

      <Row label={t("propPriority")}>
        <select
          className={SELECT_CLASS}
          aria-label={t("propPriority")}
          value={issue.priority}
          onChange={(e) =>
            void runField("priority", () => write("/api/kanban/update", { issueId: issue.id, patch: { priority: Number(e.target.value) } }))
          }
        >
          {PRIORITY_ORDER.map((p) => (
            <option key={p} value={p}>
              {t(PRIORITY_LABEL[p])}
            </option>
          ))}
        </select>
      </Row>

      <Row label={t("propAssignee")}>
        <select
          className={SELECT_CLASS}
          aria-label={t("propAssignee")}
          value={issue.assigneeId ?? ""}
          onChange={(e) =>
            void runField("assignee", () => write("/api/kanban/update", { issueId: issue.id, patch: { assigneeId: e.target.value || null } }))
          }
        >
          <option value="">{t("unassigned")}</option>
          {members.map((m) => (
            <option key={m.id} value={m.id}>
              {m.displayName}
            </option>
          ))}
        </select>
      </Row>

      {issue.project ? (
        <Row label={t("propProject")}>
          <span className="text-[13px] text-ink">{issue.project}</span>
        </Row>
      ) : null}

      {labels.length > 0 ? (
        <Row label={t("propLabels")}>
          <div className="flex flex-wrap items-center gap-1.5">
            {labels.map((l) => {
              const on = selectedLabelIds.includes(l.id);
              return (
                <button
                  key={l.id}
                  onClick={() => void toggleLabel(l.id)}
                  disabled={savingLabels}
                  className="rounded-full border px-2 py-0.5 text-[10px] font-semibold transition-opacity disabled:cursor-not-allowed"
                  // Linear label color is runtime data, not a design token
                  style={{ color: l.color, borderColor: l.color, opacity: on ? 1 : 0.4 }}
                >
                  {l.name}
                </button>
              );
            })}
          </div>
        </Row>
      ) : null}

      {pending || error ? (
        <div role="status" aria-live="polite" className={`rounded-md border px-2.5 py-1.5 text-[11px] ${error ? "border-danger bg-danger-soft text-danger" : "border-line bg-surface-2 text-muted"}`}>
          {error ?? t("saving")}
        </div>
      ) : null}
    </div>
  );
}