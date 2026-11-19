// Shared server-side resolution of an exact existing tmux target (M3b KISS). It combines ONE
// bounded native snapshot with the existing project-pane/seat projection and the configured ttyd
// bases. No persisted state, no target fallback: ambiguity or a changed binding fails visibly.

import { getDb } from "./db";
import { getProjectPanes, sessionSeats, type HubPane, type LiveSnapshot } from "./db/workflows";
import { readGeneralSettings } from "./settings";
import { collectSessionAccessSnapshot, type SessionAccessSnapshot } from "./collectors/session-access";
import {
  PANE_RE,
  SessionAccessError,
  buildTtydFrameUrl,
  type ExpectedSessionTarget,
  type ResolvedSessionAccess,
  type SessionAccessSelector,
} from "../lib/sessionAccess";

export type SessionAccessConfig = { enabled: boolean; roUrl?: string; rwUrl?: string };

export type SessionAccessDeps = {
  collect: () => Promise<SessionAccessSnapshot>;
  projectPanes: (project: string, live: LiveSnapshot, nowMs: number) => HubPane[];
  config: () => SessionAccessConfig;
  now: () => number;
};

export function readSessionAccessConfig(): SessionAccessConfig {
  const settings = readGeneralSettings();
  if (!(settings.ok && settings.data.integrations.ttyd)) return { enabled: false };
  return { enabled: true, roUrl: process.env.TTYD_RO_URL, rwUrl: process.env.TTYD_RW_URL };
}

const systemDeps: SessionAccessDeps = {
  collect: () => collectSessionAccessSnapshot(),
  projectPanes: (project, live, nowMs) => getProjectPanes(getDb(), project, live, nowMs, () => false),
  config: readSessionAccessConfig,
  now: () => Date.now(),
};

function liveSnapshot(snapshot: SessionAccessSnapshot): LiveSnapshot {
  return {
    paneIds: new Set(snapshot.rows.filter((r) => PANE_RE.test(r.pane) && !r.dead).map((r) => r.pane)),
    incarnation: snapshot.serverIncarnation,
    paneCommands: new Map(),
  };
}

function pof(code: ConstructorParameters<typeof SessionAccessError>[0]): never {
  throw new SessionAccessError(code);
}

// Configuration is validated with the SAME URL grammar that builds the frame, before any native
// read — a malformed base is unconfigured, never a source failure.
function usableBase(base: unknown): base is string {
  if (typeof base !== "string" || base.length === 0) return false;
  try {
    buildTtydFrameUrl(base, "$0:@0.%0");
    return true;
  } catch {
    return false;
  }
}

function requireExpectedBinding(
  expected: ExpectedSessionTarget | undefined,
  target: string,
  incarnation: string,
): void {
  if (expected && (expected.target !== target || expected.incarnation !== incarnation)) pof("lost-target");
}

export async function resolveSessionAccess(
  selector: SessionAccessSelector,
  expected?: ExpectedSessionTarget,
  deps: SessionAccessDeps = systemDeps,
): Promise<ResolvedSessionAccess> {
  const config = deps.config();
  const roBase = config.roUrl;
  if (!config.enabled || !usableBase(roBase)) pof("unconfigured");

  const snapshot = await deps.collect().catch(() => pof("source-failure"));
  const live = liveSnapshot(snapshot);

  let target: string;
  let incarnation: string;
  let sessionName: string;
  let runtime: ResolvedSessionAccess["runtime"] = "unknown";

  if (selector.kind === "card") {
    const panes = deps.projectPanes(selector.project, live, deps.now());
    const hub = panes.find(
      (p) => p.pane === selector.pane && p.tmuxIncarnation === selector.incarnation && p.live,
    );
    if (!hub) pof("unavailable");
    // Strict identity: an unrecorded harness (undefined) is not the explicit unidentified seat
    // (null), and neither is a wildcard for the other.
    if (hub.harnessSession !== selector.harnessSession) pof("unavailable");
    const seat = sessionSeats(panes, []).find(
      (s) => s.transport === "tmux"
        && s.panes.some((r) => r.pane === selector.pane && r.tmuxIncarnation === selector.incarnation),
    );
    if (!seat || seat.sessionId !== selector.harnessSession) pof("unavailable");

    const paneRows = snapshot.rows.filter((r) => r.eligible && r.pane === selector.pane);
    const row = paneRows.find((r) => r.incarnation === selector.incarnation && !r.dead);
    if (!row) {
      // A missing initial target is unavailable; a moved/dead one is a lost target.
      if (paneRows.length === 0) pof("unavailable");
      pof("lost-target");
    }
    requireExpectedBinding(expected, row.target, row.incarnation);
    target = row.target;
    incarnation = row.incarnation;
    sessionName = row.sessionName;
    runtime = hub.runtime;
  } else if (expected) {
    // With a pinned binding, native IDs and incarnation are the identity — a rename keeps it.
    const row = snapshot.rows.find(
      (r) => r.eligible && !r.dead && r.target === expected.target && r.incarnation === expected.incarnation,
    );
    if (!row) pof("lost-target");
    target = row.target;
    incarnation = row.incarnation;
    sessionName = row.sessionName;
  } else {
    const named = snapshot.rows.filter((r) => r.eligible && r.sessionName === selector.session);
    if (named.length === 0 || new Set(named.map((r) => r.sessionId)).size > 1) pof("unavailable");
    const active = named.filter((r) => !r.dead && r.windowActive && r.paneActive);
    if (active.length !== 1) pof("unavailable");
    target = active[0].target;
    incarnation = active[0].incarnation;
    sessionName = active[0].sessionName;
  }

  let roUrl: string;
  try {
    roUrl = buildTtydFrameUrl(roBase, target);
  } catch {
    pof("unconfigured");
  }
  return {
    selector,
    expected: { target, incarnation },
    sessionName,
    runtime,
    observedAt: new Date(deps.now()).toISOString(),
    roUrl,
  };
}

// The ONLY RW URL builder: after a successful fresh resolution + audit insert. A client never
// derives it from config or a mode boolean.
export function takenSessionAccess(resolved: ResolvedSessionAccess, config: SessionAccessConfig): ResolvedSessionAccess & { rwUrl: string } {
  if (!config.enabled || typeof config.rwUrl !== "string" || config.rwUrl.length === 0) throw new SessionAccessError("unconfigured");
  try {
    return { ...resolved, rwUrl: buildTtydFrameUrl(config.rwUrl, resolved.expected.target) };
  } catch {
    throw new SessionAccessError("unconfigured");
  }
}
