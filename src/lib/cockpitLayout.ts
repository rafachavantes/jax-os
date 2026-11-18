// M2 saved project layout (spec "Saved project layout", plan Task 1). Pure and client-safe:
// the identity is MissionCard.dir — the same key project_prefs.project already uses — never a
// filesystem path. Every validation lives here so the route and the DB seam share one rule.

export type ProjectWidth = "third" | "half" | "full";
export type PrScope = "mine" | "all";

export type LayoutEntry = { project: string; width: ProjectWidth };
export type CockpitSnapshot = { revision: number; layout: LayoutEntry[] | null; prScope: PrScope };

export const DEFAULT_WIDTH: ProjectWidth = "third";
export const WIDTHS: readonly ProjectWidth[] = ["third", "half", "full"];
export const PR_SCOPES: readonly PrScope[] = ["mine", "all"];
export const MAX_LAYOUT_PROJECTS = 500;
export const MAX_PROJECT_KEY_CHARS = 256;

const CONTROL_RE = /[\u0000-\u001f\u007f]/;

export function isProjectWidth(value: unknown): value is ProjectWidth {
  return typeof value === "string" && (WIDTHS as readonly string[]).includes(value);
}

export function isPrScope(value: unknown): value is PrScope {
  return typeof value === "string" && (PR_SCOPES as readonly string[]).includes(value);
}

// Bounded, normalized RELATIVE key: no absolute prefix, no backslashes, no control/NUL, no
// empty/dot/dot-dot segments. Nested `a/b` keys are ordinary values, never paths to resolve.
export function normalizeProjectKey(value: unknown): string | null {
  if (typeof value !== "string" || value.length === 0 || value.length > MAX_PROJECT_KEY_CHARS) return null;
  if (CONTROL_RE.test(value) || value.includes("\\")) return null;
  if (value.startsWith("/")) return null;
  const segments = value.split("/");
  for (const segment of segments) {
    if (segment === "" || segment === "." || segment === "..") return null;
  }
  return value;
}

export function parseLayoutEntries(value: unknown): LayoutEntry[] | null {
  if (!Array.isArray(value) || value.length > MAX_LAYOUT_PROJECTS) return null;
  const seen = new Set<string>();
  const out: LayoutEntry[] = [];
  for (const raw of value) {
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
    const entry = raw as Record<string, unknown>;
    if (Object.keys(entry).length !== 2) return null;
    const project = normalizeProjectKey(entry.project);
    if (project === null || !isProjectWidth(entry.width) || seen.has(project)) return null;
    seen.add(project);
    out.push({ project, width: entry.width });
  }
  return out;
}

// Save merges the submitted visible order with every saved identity absent from it. Each absent
// entry keeps its prior width and relative order, anchored after the nearest preceding submitted
// identity in the previous order (spec: "each stays anchored after the nearest previously
// preceding submitted identity, or at the end when none preceded it"). An absent head blocks the
// absent entries behind it, so they can never overtake it.
export function mergeSavedLayout(prior: LayoutEntry[] | null, submitted: LayoutEntry[]): LayoutEntry[] {
  if (prior === null) return submitted;
  const submittedSet = new Set(submitted.map((e) => e.project));
  const anchors = new Map<string, string | null>();
  const absent: LayoutEntry[] = [];
  let lastSubmittedSeen: string | null = null;
  for (const entry of prior) {
    if (submittedSet.has(entry.project)) {
      lastSubmittedSeen = entry.project;
    } else {
      absent.push(entry);
      anchors.set(entry.project, lastSubmittedSeen);
    }
  }
  const emitted = new Set<string>();
  const out: LayoutEntry[] = [];
  const flushAbsent = () => {
    while (absent.length > 0) {
      const head = absent[0];
      const anchor = anchors.get(head.project) ?? null;
      if (anchor === null || !emitted.has(anchor)) break;
      emitted.add(head.project);
      out.push(head);
      absent.shift();
    }
  };
  for (const entry of submitted) {
    out.push(entry);
    emitted.add(entry.project);
    flushAbsent();
  }
  out.push(...absent);
  return out;
}

export function layoutMap(layout: LayoutEntry[] | null): Map<string, ProjectWidth> {
  return new Map((layout ?? []).map((e) => [e.project, e.width]));
}

// Saved order applies before display filters. Unknown cards follow the saved portion in stable
// dir-ascending order; without an override the existing automatic ranking is untouched.
export function applySavedOrder<T extends { dir: string }>(cards: T[], layout: LayoutEntry[] | null): T[] {
  if (layout === null) return cards;
  const index = new Map(layout.map((e, i) => [e.project, i]));
  const saved = cards.filter((c) => index.has(c.dir)).sort((a, b) => index.get(a.dir)! - index.get(b.dir)!);
  const fresh = cards.filter((c) => !index.has(c.dir)).sort((a, b) => a.dir.localeCompare(b.dir));
  return [...saved, ...fresh];
}

// The Organize draft's visible snapshot: current order, saved width where one exists.
export function visibleLayout(cards: { dir: string }[], layout: LayoutEntry[] | null): LayoutEntry[] {
  const map = layoutMap(layout);
  return cards.map((c) => ({ project: c.dir, width: map.get(c.dir) ?? DEFAULT_WIDTH }));
}

export function moveEntry(draft: LayoutEntry[], project: string, delta: -1 | 1): LayoutEntry[] {
  const index = draft.findIndex((e) => e.project === project);
  if (index < 0) return draft;
  const next = index + delta;
  if (next < 0 || next >= draft.length) return draft;
  const copy = [...draft];
  const [entry] = copy.splice(index, 1);
  copy.splice(next, 0, entry);
  return copy;
}

export function dropEntry(draft: LayoutEntry[], dragged: string, target: string): LayoutEntry[] {
  if (dragged === target) return draft;
  const from = draft.findIndex((e) => e.project === dragged);
  if (from < 0 || !draft.some((e) => e.project === target)) return draft;
  const copy = [...draft];
  const [entry] = copy.splice(from, 1);
  const to = copy.findIndex((e) => e.project === target);
  if (to < 0) return draft;
  copy.splice(to, 0, entry);
  return copy;
}

export function setEntryWidth(draft: LayoutEntry[], project: string, width: ProjectWidth): LayoutEntry[] {
  return draft.map((e) => (e.project === project ? { ...e, width } : e));
}
