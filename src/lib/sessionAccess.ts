// Exact existing tmux target access (M3b KISS): the shared, dependency-free contracts a Jax
// focused link carries and the ONE place a ttyd frame URL is assembled. The selector binds
// identity only — a stable project/pane/incarnation/harness tuple or a bounded tmux session
// name — never a shell command, upstream URL, raw tmux flag or trusted target.

export type SessionAccessSelector =
  | { kind: "card"; project: string; harnessSession: string | null; pane: string; incarnation: string }
  | { kind: "tmux"; session: string };

export type ExpectedSessionTarget = { target: string; incarnation: string };

export type ResolvedSessionAccess = {
  selector: SessionAccessSelector;
  expected: ExpectedSessionTarget;
  sessionName: string;
  runtime: "claude" | "codex" | "unknown";
  observedAt: string;
  roUrl: string;
};

export type TakenSessionAccess = ResolvedSessionAccess & { rwUrl: string };

export type SessionAccessCode =
  | "invalid-selector"
  | "unconfigured"
  | "unavailable"
  | "lost-target"
  | "source-failure";

export class SessionAccessError extends Error {
  constructor(readonly code: Exclude<SessionAccessCode, "invalid-selector">) {
    super(code);
    this.name = "SessionAccessError";
  }
}

export const SESSION_QUERY_CAP = 2048;
export const SESSION_NAME_CAP = 256;
export const HARNESS_SESSION_CAP = 512;
export const PANE_RE = /^%[0-9]{1,10}$/;
export const INCARNATION_RE = /^[0-9]{1,10}:[0-9]{1,20}$/;
// $<session>:@<window>.%<pane> — at most 35 characters, built from native IDs only.
export const TARGET_RE = /^\$[0-9]{1,10}:@[0-9]{1,10}\.%[0-9]{1,10}$/;

function byteLength(value: string): number {
  return new TextEncoder().encode(value).length;
}

function boundedText(value: unknown, cap: number): value is string {
  return typeof value === "string" && value.length > 0
    && byteLength(value) <= cap && !/[\u0000-\u001f]/.test(value);
}

function entriesOf(value: unknown): Map<string, string> | null {
  if (value instanceof URLSearchParams) {
    if (value.toString().length > SESSION_QUERY_CAP) return null;
    const out = new Map<string, string>();
    for (const key of new Set(value.keys())) {
      const all = value.getAll(key);
      if (all.length !== 1) return null; // duplicate query fields are never a selector
      out.set(key, all[0]);
    }
    return out;
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const out = new Map<string, string>();
  for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
    if (entry === null) { out.set(key, ""); continue; } // an explicit null harness is the unidentified seat
    if (typeof entry !== "string") return null;
    out.set(key, entry);
  }
  return out;
}

export function parseSessionAccessSelector(value: unknown): SessionAccessSelector | null {
  const fields = entriesOf(value);
  if (!fields) return null;
  const kind = fields.get("kind");
  if (kind === "card") {
    const project = fields.get("project");
    const pane = fields.get("pane");
    const incarnation = fields.get("incarnation");
    if (!boundedText(project, SESSION_NAME_CAP)) return null;
    if (typeof pane !== "string" || !PANE_RE.test(pane)) return null;
    if (typeof incarnation !== "string" || !INCARNATION_RE.test(incarnation)) return null;
    const rawHarness = fields.get("harnessSession");
    const harness = rawHarness === undefined || rawHarness === "" ? null : rawHarness;
    if (harness !== null && !boundedText(harness, HARNESS_SESSION_CAP)) return null;
    return { kind: "card", project, harnessSession: harness, pane, incarnation };
  }
  if (kind === "tmux") {
    const session = fields.get("session");
    if (!boundedText(session, SESSION_NAME_CAP)) return null;
    return { kind: "tmux", session };
  }
  return null;
}

export function parseSessionAccessOrigin(value: unknown): "home" | "mosaic" | null {
  const fields = entriesOf(value);
  const from = fields?.get("from");
  return from === "home" || from === "mosaic" ? from : null;
}

export type ExpectedBindingParse =
  | { kind: "absent" }
  | { kind: "valid"; expected: ExpectedSessionTarget }
  | { kind: "invalid" };

export function parseExpectedBinding(value: unknown): ExpectedBindingParse {
  const fields = entriesOf(value);
  if (!fields) return { kind: "invalid" };
  const target = fields.get("expectedTarget");
  const incarnation = fields.get("expectedIncarnation");
  if (target === undefined && incarnation === undefined) {
    for (const key of fields.keys()) if (key.startsWith("expected")) return { kind: "invalid" };
    return { kind: "absent" };
  }
  if (target === undefined || incarnation === undefined) return { kind: "invalid" };
  for (const key of fields.keys()) if (key.startsWith("expected") && key !== "expectedTarget" && key !== "expectedIncarnation") {
    return { kind: "invalid" };
  }
  if (!TARGET_RE.test(target) || target.length > 35 || !INCARNATION_RE.test(incarnation)) return { kind: "invalid" };
  return { kind: "valid", expected: { target, incarnation } };
}

export function sessionAccessHref(selector: SessionAccessSelector, from: "home" | "mosaic"): string {
  const params = new URLSearchParams();
  if (selector.kind === "card") {
    params.set("kind", "card");
    params.set("project", selector.project);
    params.set("harnessSession", selector.harnessSession ?? "");
    params.set("pane", selector.pane);
    params.set("incarnation", selector.incarnation);
  } else {
    params.set("kind", "tmux");
    params.set("session", selector.session);
  }
  params.set("from", from);
  return `/tmux?${params.toString()}`;
}

export function buildTtydFrameUrl(base: string, target: string): string {
  if (!TARGET_RE.test(target) || target.length > 35) throw new Error("invalid target");
  let url: URL;
  try {
    url = new URL(base);
  } catch {
    throw new Error("invalid base");
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") throw new Error("invalid base");
  if (url.username || url.password || url.search || url.hash) throw new Error("invalid base");
  url.pathname = url.pathname.endsWith("/") ? url.pathname : `${url.pathname}/`;
  url.searchParams.set("arg", target);
  url.searchParams.set("disableReconnect", "true");
  return url.toString();
}
