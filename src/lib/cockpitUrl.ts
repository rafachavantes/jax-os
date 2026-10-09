import type { Filters, SortKey } from "./boardView";
import type { HealthRange } from "@/server/db/metrics";

const ID_CAP = 256;
const VIEWS = ["board", "list"] as const;
const SORTS: SortKey[] = ["priority", "updated", "created", "number", "status"];
const RANGES: HealthRange[] = ["1h", "24h", "7d", "90d"];
const PRESETS = ["7d", "30d", "all"] as const;

export type KanbanUrlState = {
  team: string | null;
  issue: string | null;
  view: "board" | "list";
  sort: SortKey;
  q: string;
  project: string | null;
  priority: number | null;
  label: string | null;
  assignee: string | null;
};

export type HealthUrlState = { range: HealthRange };
export type AuditUrlState = { kind: string; preset: (typeof PRESETS)[number] };

function utf8Len(s: string): number {
  return new TextEncoder().encode(s).length;
}

function cap(s: string | null, max: number): string | null {
  if (s === null || s === "") return null;
  return utf8Len(s) <= max ? s : null;
}

function pick<T extends string>(raw: string | null, allowed: readonly T[], fallback: T): T {
  return raw !== null && (allowed as readonly string[]).includes(raw) ? (raw as T) : fallback;
}

export function parseKanbanSearch(params: URLSearchParams): KanbanUrlState {
  const priorityRaw = params.get("priority");
  const priority = priorityRaw !== null && /^[0-4]$/.test(priorityRaw) ? Number(priorityRaw) : null;
  return {
    team: cap(params.get("team"), ID_CAP),
    issue: cap(params.get("issue"), ID_CAP),
    view: pick(params.get("view"), VIEWS, "board"),
    sort: pick(params.get("sort"), SORTS, "priority"),
    q: cap(params.get("q") ?? "", ID_CAP) ?? "",
    project: cap(params.get("project"), ID_CAP),
    priority,
    label: cap(params.get("label"), ID_CAP),
    assignee: cap(params.get("assignee"), ID_CAP),
  };
}

export function serializeKanbanSearch(params: URLSearchParams, state: KanbanUrlState): URLSearchParams {
  const next = new URLSearchParams(params);
  const set = (key: string, value: string | null, fallback = "") => {
    if (value === null || value === fallback) next.delete(key);
    else next.set(key, value);
  };
  set("team", state.team);
  set("issue", state.issue);
  set("view", state.view, "board");
  set("sort", state.sort, "priority");
  set("q", state.q, "");
  set("project", state.project);
  if (state.priority === null) next.delete("priority");
  else next.set("priority", String(state.priority));
  set("label", state.label);
  set("assignee", state.assignee);
  return next;
}

export function kanbanFilters(state: KanbanUrlState): Filters {
  return { query: state.q, project: state.project, priority: state.priority, label: state.label, assignee: state.assignee };
}

// Merges a patch onto the given PENDING state, not necessarily the committed
// URL yet. Callers must thread the previous result back in as `pending` so
// two patches issued before router.replace() commits both survive, instead
// of the second one overwriting the first's fields against a stale base.
export function mergeKanbanPatch(pending: KanbanUrlState, patch: Partial<KanbanUrlState>): KanbanUrlState {
  return { ...pending, ...patch };
}

export const STORAGE_KEY = "jax-os.kanban.v1";

export type KanbanSelection = {
  team: string | null; project: string | null; priority: number | null;
  label: string | null; assignee: string | null; sort: SortKey; view: "board" | "list";
};

const KANBAN_DEFAULTS: KanbanSelection = {
  team: null, project: null, priority: null, label: null, assignee: null, sort: "priority", view: "board",
};

// Defensive load of persisted UI state, mirroring filesState.ts's normalizeState:
// junk/corrupt/hand-edited localStorage yields the same hard defaults
// parseKanbanSearch already uses, never a thrown error.
export function normalizeKanbanSelection(raw: unknown): KanbanSelection {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) return { ...KANBAN_DEFAULTS };
  const o = raw as Record<string, unknown>;
  const str = (v: unknown): string | null => (typeof v === "string" ? cap(v, ID_CAP) : null);
  const priority = typeof o.priority === "number" && /^[0-4]$/.test(String(o.priority)) ? o.priority : null;
  return {
    team: str(o.team),
    project: str(o.project),
    priority,
    label: str(o.label),
    assignee: str(o.assignee),
    sort: typeof o.sort === "string" && (SORTS as string[]).includes(o.sort) ? (o.sort as SortKey) : "priority",
    view: o.view === "list" ? "list" : "board",
  };
}

// Per-field precedence (phase 6 spec §7): present-and-valid → URL; present-
// but-invalid → hard default, never storage; absent → storage if valid, else
// default. Takes the RAW params (not a KanbanUrlState) because
// parseKanbanSearch alone can't tell "absent" from "present but invalid" —
// both already collapse to the same hard default there.
export function hydrateFilters(rawParams: URLSearchParams | Record<string, string>, storage: unknown): KanbanUrlState {
  const raw = rawParams instanceof URLSearchParams ? rawParams : new URLSearchParams(rawParams);
  const url = parseKanbanSearch(raw);
  const stored = normalizeKanbanSelection(storage);
  const pick = <K extends keyof KanbanSelection>(key: K): KanbanSelection[K] =>
    (raw.has(key) ? url[key] : stored[key]) as KanbanSelection[K];
  return {
    ...url,
    team: pick("team"),
    project: pick("project"),
    priority: pick("priority"),
    label: pick("label"),
    assignee: pick("assignee"),
    sort: pick("sort"),
    view: pick("view"),
  };
}

// Board query enabled predicate (phase 6 spec §6.1 step 4, F5): a URL-supplied
// team enables same-tick, exactly as before. An absent URL team must wait for
// `hydrated` — otherwise a warm teams cache resolves the fallback team and
// fires the board query before the hydration effect had a chance to seed a
// stored team, flashing the wrong board.
export function boardEnabled(urlTeam: string | null, hydrated: boolean, teamId: string | null): boolean {
  return (urlTeam !== null || hydrated) && teamId !== null;
}

export function parseHealthSearch(params: URLSearchParams): HealthUrlState {
  return { range: pick(params.get("range"), RANGES, "24h") };
}

export function serializeHealthSearch(params: URLSearchParams, state: HealthUrlState): URLSearchParams {
  const next = new URLSearchParams(params);
  if (state.range === "24h") next.delete("range");
  else next.set("range", state.range);
  return next;
}

export function parseAuditSearch(params: URLSearchParams): AuditUrlState {
  return {
    kind: cap(params.get("kind"), ID_CAP) ?? "",
    preset: pick(params.get("preset"), PRESETS, "30d"),
  };
}

export function serializeAuditSearch(params: URLSearchParams, state: AuditUrlState): URLSearchParams {
  const next = new URLSearchParams(params);
  if (!state.kind) next.delete("kind");
  else next.set("kind", state.kind);
  if (state.preset === "30d") next.delete("preset");
  else next.set("preset", state.preset);
  return next;
}

export function searchHref(pathname: string, params: URLSearchParams): string {
  const qs = params.toString();
  return qs ? `${pathname}?${qs}` : pathname;
}
