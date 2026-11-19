// Shared server-side resolution of an exact existing tmux target (M3b KISS). It combines ONE
// bounded native snapshot with the existing project-pane/seat projection and the configured ttyd
// bases. No persisted state, no target fallback: ambiguity or a changed binding fails visibly.

import { getDb } from "./db";
import { getCodexSessions, getProjectPanes, sessionSeats, type HubPane, type LiveSnapshot } from "./db/workflows";
import { readGeneralSettings } from "./settings";
import { collectSessionAccessSnapshot, type SessionAccessSnapshot } from "./collectors/session-access";
import { collectCodexSnapshot, createCodexConnector, type CodexSnapshot } from "./collectors/workflow-codex";
import { codexTerminalBindings } from "./collectors/codex-tmux";
import { isCodexCommand } from "./collectors/workflow-tmux";
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
  // Fresh native Codex identity proof for a card selector only. Optional so ordinary Claude/tmux
  // access never depends on the Codex App Server; omitted means "unavailable" for a Codex command.
  codexEnabled?: () => boolean;
  codexSnapshot?: (timeoutMs: number) => Promise<CodexSnapshot>;
  codexSessions?: (project: string, snapshot: CodexSnapshot, nowMs: number) => ReturnType<typeof getCodexSessions>;
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
  codexEnabled: () => { const s = readGeneralSettings(); return s.ok && s.data.integrations.agents.codex; },
  codexSnapshot: (timeoutMs) => collectCodexSnapshot(createCodexConnector, { timeoutMs }),
  codexSessions: (project, snapshot, nowMs) => getCodexSessions(getDb(), project, snapshot, nowMs),
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

  const requestStartedAt = deps.now();
  const snapshot = await deps.collect().catch(() => pof("source-failure"));
  const live = liveSnapshot(snapshot);

  let target: string;
  let incarnation: string;
  let sessionName: string;
  let runtime: ResolvedSessionAccess["runtime"] = "unknown";

  if (selector.kind === "card") {
    // A positive Codex command on the selected pane takes the native path BEFORE any historical
    // Claude pane/seat fallback: an unprovable Codex identity never resolves as Claude.
    const freshRow = snapshot.rows.find((r) => r.eligible && r.pane === selector.pane);
    if (freshRow && isCodexCommand(freshRow.command ?? "")) {
      const readCodex = deps.codexSnapshot;
      const readMembership = deps.codexSessions;
      if (!deps.codexEnabled?.() || !readCodex || !readMembership) pof("unavailable");
      const timeoutMs = Math.min(3_000, 4_500 - (deps.now() - requestStartedAt));
      if (timeoutMs <= 0) pof("source-failure");
      const codex = await readCodex(timeoutMs).catch(() => pof("source-failure"));
      if (!codex.ok || codex.truncated || !codex.loadedThreadIds) pof("source-failure");
      const roots = codex.threads.filter((s) =>
        s.threadId === selector.harnessSession && ["active", "idle"].includes(s.status),
      );
      if (roots.length !== 1 || roots[0].owner !== selector.project) pof("unavailable");
      const membership = readMembership(selector.project, codex, deps.now());
      if (!membership.sessions.some((s) => s.threadId === selector.harnessSession)) pof("unavailable");
      const observations = snapshot.rows.map((r) => ({
        pane: r.pane, tmuxIncarnation: r.incarnation, session: r.sessionName,
        command: r.command ?? "", title: r.title ?? "", dead: r.dead,
        eligible: r.eligible && r.title !== undefined,
      }));
      const binding = selector.harnessSession === null
        ? undefined
        : codexTerminalBindings(codex, observations).get(selector.harnessSession);
      if (!binding) pof(expected ? "lost-target" : "unavailable");
      if (binding.project !== selector.project) pof("unavailable");
      if (binding.terminal.pane !== selector.pane || binding.terminal.tmuxIncarnation !== selector.incarnation) pof("lost-target");
      const row = snapshot.rows.find((r) => r.eligible && !r.dead && r.pane === selector.pane && r.incarnation === selector.incarnation);
      if (!row) pof(expected ? "lost-target" : "unavailable");
      requireExpectedBinding(expected, row.target, row.incarnation);
      target = row.target;
      incarnation = row.incarnation;
      sessionName = row.sessionName;
      runtime = "codex";
    } else {
      const panes = deps.projectPanes(selector.project, live, deps.now());
      const hub = panes.find(
        (p) => p.pane === selector.pane && p.tmuxIncarnation === selector.incarnation && p.live,
      );
      // A pinned target whose live hub row disappeared is a lost target; an initial miss is merely
      // unavailable. Harness/seat ownership mismatches below stay unavailable either way.
      if (!hub) pof(expected ? "lost-target" : "unavailable");
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
        // A pinned binding that is no longer live is always lost. Without one, a missing initial
        // target is unavailable while a moved/dead row is a lost target.
        if (expected || paneRows.length > 0) pof("lost-target");
        pof("unavailable");
      }
      requireExpectedBinding(expected, row.target, row.incarnation);
      target = row.target;
      incarnation = row.incarnation;
      sessionName = row.sessionName;
      runtime = hub.runtime;
    }
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
