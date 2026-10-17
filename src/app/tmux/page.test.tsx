import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

const harness = vi.hoisted(() => ({
  projects: { data: { ok: true, data: { projects: [], skipped: 0 } }, isError: false },
  // Plan correction: the plan's harness used `{ panes: [], projects: {} }`, but buildMission reads
  // `hub.data.byProject` (and the other HubEnvelopeData fields) — the malformed envelope crashed
  // buildModel. Same empty outcome, real shape.
  hub: { data: { ok: true, data: { byProject: {}, historicalTruncated: false, codexSource: "ok", tmuxSource: "ok", prefs: {}, settings: { archiveAfterDays: 30 } } }, isError: false },
  agents: { data: undefined as unknown, isError: false, dataUpdatedAt: 0, error: undefined as unknown },
  panesTail: { data: undefined as unknown, isError: false },
  config: { data: undefined as unknown, isError: false, isLoading: false },
  useMissionCalls: [] as { endpoint: string; enabled: boolean }[],
  useQueryCalls: [] as { key: string; enabled: boolean }[],
}));

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en-US",
}));
vi.mock("@/lib/useMission", () => ({
  useMission: (endpoint: "projects" | "hub" | "agents", _interval: number, enabled = true) => {
    harness.useMissionCalls.push({ endpoint, enabled });
    return harness[endpoint];
  },
}));
vi.mock("@tanstack/react-query", () => ({
  useQuery: (opts: { queryKey: [string, string]; enabled?: boolean }) => {
    harness.useQueryCalls.push({ key: opts.queryKey[1], enabled: opts.enabled ?? true });
    return opts.queryKey[1] === "panes-tail" ? harness.panesTail : harness.config;
  },
}));
vi.mock("@/components/sessions/SessionCard", () => ({
  SessionCard: ({ agent }: { agent: { session: string } }) => createElement("div", null, agent.session),
}));
vi.mock("@/components/sessions/Mosaic", () => ({
  Mosaic: (props: { readiness: { kind: string }; cards: unknown[] }) =>
    createElement("div", { "data-testid": "mosaic", "data-readiness": props.readiness.kind, "data-cards": props.cards.length }),
}));
vi.mock("@/components/sessions/MosaicCellDetail", () => ({
  MosaicCellDetail: () => createElement("div", { "data-testid": "mosaic-cell-detail" }),
}));

import SessionsPage from "./page";
import {
  captureControlTicket, controlModeAfter, controlStateReducer, controlTicketCurrent, TerminalPanel, type ControlState,
} from "@/components/sessions/TerminalPanel";

const ttyd = { configured: true, reachable: true, roUrl: "http://ro", rwUrl: "http://rw" };

const idle: ControlState = { mode: "ro", busy: false, error: null };
const writable: ControlState = { mode: "rw", busy: false, error: null };

describe("controlModeAfter (I1 tmux semantics)", () => {
  it("takes control only after a confirmed success on the same session", () => {
    expect(controlModeAfter("take", true, true)).toBe("rw");
    expect(controlModeAfter("take", true, false)).toBe("ro"); // session switched mid-flight
    expect(controlModeAfter("take", false, true)).toBe("ro"); // delayed failure
  });

  it("releases to read-only immediately, even when recording fails", () => {
    expect(controlModeAfter("release", false, true)).toBe("ro");
    expect(controlModeAfter("release", true, true)).toBe("ro");
  });
});

describe("controlStateReducer (production Take/Release seam)", () => {
  it("pending Release leaves the state ro immediately while the POST is unresolved", () => {
    const requested = controlStateReducer(writable, { type: "request", action: "release" });
    expect(requested).toEqual({ mode: "ro", busy: true, error: null });
  });

  it("keeps ro through a delayed release failure and records the error", () => {
    let state = controlStateReducer(writable, { type: "request", action: "release" });
    state = controlStateReducer(state, {
      type: "settle",
      action: "release",
      ok: false,
      error: "recording failed",
      current: true,
    });
    expect(state.mode).toBe("ro");
    expect(state.busy).toBe(false);
    expect(state.error).toBe("recording failed");
  });

  it("takes control only after confirmed success; a delayed failure stays ro", () => {
    let state = controlStateReducer(idle, { type: "request", action: "take" });
    expect(state).toEqual({ mode: "ro", busy: true, error: null });
    state = controlStateReducer(state, { type: "settle", action: "take", ok: true, error: null, current: true });
    expect(state.mode).toBe("rw");
    state = controlStateReducer(idle, { type: "request", action: "take" });
    state = controlStateReducer(state, { type: "settle", action: "take", ok: false, error: "down", current: true });
    expect(state.mode).toBe("ro");
    expect(state.error).toBe("down");
  });

  it("is a synchronous duplicate guard: a second request while busy is ignored", () => {
    const first = controlStateReducer(idle, { type: "request", action: "take" });
    expect(controlStateReducer(first, { type: "request", action: "take" })).toEqual(first);
  });

  it("a late prior-session settle neither enables control nor overwrites current feedback", () => {
    let state = controlStateReducer(writable, { type: "request", action: "release" });
    state = controlStateReducer(state, { type: "reset" }); // user switched sessions
    state = controlStateReducer(state, {
      type: "settle",
      action: "release",
      ok: false,
      error: "stale failure",
      current: false,
    });
    expect(state.mode).toBe("ro");
    expect(state.error).toBeNull();
    expect(state.busy).toBe(false);
    // a stale take success must never flip the current session writable
    const staleTake = controlStateReducer(idle, { type: "request", action: "take" });
    const afterReset = controlStateReducer(staleTake, { type: "reset" });
    expect(controlStateReducer(afterReset, { type: "settle", action: "take", ok: true, error: null, current: false }).mode).toBe("ro");
  });

  it("reset drops control without hiding an in-flight busy request", () => {
    const requested = controlStateReducer(writable, { type: "request", action: "release" });
    expect(controlStateReducer(requested, { type: "reset" })).toEqual({ mode: "ro", busy: true, error: null });
  });
});

describe("control ticket lifetime (A -> B -> A)", () => {
  it("a delayed Take success for A after selecting B and A again stays read-only", () => {
    let token = 0;
    const ticket = captureControlTicket("A", token)!;
    expect(ticket).toEqual({ target: "A", token: 0 });
    // page invalidation points: explicit select() bumps the token
    // synchronously (twice here: select B, then select A)
    token += 1;
    expect(controlTicketCurrent("B", token, ticket)).toBe(false);
    token += 1;
    expect(controlTicketCurrent("A", token, ticket)).toBe(false);
    // the production dispatch order: Take request, resets on each select,
    // then the delayed settlement with the ticket check the page computes
    let state = controlStateReducer(idle, { type: "request", action: "take" });
    state = controlStateReducer(state, { type: "reset" });
    state = controlStateReducer(state, { type: "reset" });
    state = controlStateReducer(state, {
      type: "settle",
      action: "take",
      ok: true,
      error: null,
      current: controlTicketCurrent("A", token, ticket),
    });
    expect(state.mode).toBe("ro");
    expect(state.busy).toBe(false);
    // a fresh Take on the now-current A still works
    const fresh = captureControlTicket("A", token)!;
    expect(controlTicketCurrent("A", token, fresh)).toBe(true);
    state = controlStateReducer(state, { type: "request", action: "take" });
    state = controlStateReducer(state, {
      type: "settle",
      action: "take",
      ok: true,
      error: null,
      current: controlTicketCurrent("A", token, fresh),
    });
    expect(state.mode).toBe("rw");
  });

  it("polling fallback to another active session also invalidates an in-flight ticket", () => {
    let token = 0;
    const ticket = captureControlTicket("A", token)!;
    token += 1; // the [active] reset effect fires after the poll fell back to B
    expect(controlTicketCurrent("B", token, ticket)).toBe(false);
  });

  it("a stale failure leaves the current feedback alone and release still drops ro immediately", () => {
    let token = 0;
    const ticket = captureControlTicket("A", token)!;
    // Release request flips the client ro synchronously, POST still pending
    let state = controlStateReducer(writable, { type: "request", action: "release" });
    expect(state.mode).toBe("ro");
    token += 1; // user switched sessions while the release POST was in flight
    state = controlStateReducer(state, { type: "reset" });
    state = controlStateReducer(state, {
      type: "settle",
      action: "release",
      ok: false,
      error: "stale failure",
      current: controlTicketCurrent("B", token, ticket),
    });
    expect(state.mode).toBe("ro");
    expect(state.error).toBeNull(); // stale feedback never overwrites current state
    expect(state.busy).toBe(false);
  });
});

describe("TerminalPanel live feedback", () => {
  it("disables the control button and announces while a take is pending", () => {
    const html = renderToStaticMarkup(
      createElement(TerminalPanel, {
        session: "s1",
        config: ttyd,
        configLoading: false,
        mode: "ro",
        busy: true,
        error: null,
        onTakeControl: () => {},
        onReleaseControl: () => {},
      }),
    );
    expect(html).toContain("takingControl");
    expect(html).toContain('disabled=""');
    expect(html).toContain('aria-busy="true"');
    expect(html).toContain('role="status"');
  });

  it("announces release while busy and keeps the error visible", () => {
    const html = renderToStaticMarkup(
      createElement(TerminalPanel, {
        session: "s1",
        config: ttyd,
        configLoading: false,
        mode: "rw",
        busy: true,
        error: "recording failed",
        onTakeControl: () => {},
        onReleaseControl: () => {},
      }),
    );
    expect(html).toContain("releasingControl");
    expect(html).toContain("recording failed");
  });

  it("enables the button when idle with a session", () => {
    const html = renderToStaticMarkup(
      createElement(TerminalPanel, {
        session: "s1",
        config: ttyd,
        configLoading: false,
        mode: "ro",
        busy: false,
        error: null,
        onTakeControl: () => {},
        onReleaseControl: () => {},
      }),
    );
    expect(html).not.toContain('disabled=""');
    expect(html).toContain("takeControl");
  });
});

describe("SessionsPage — mosaic-first view (spec §8 D13, round-1 F4)", () => {
  it("renders the Mosaic by default, not the detail view", () => {
    const html = renderToStaticMarkup(createElement(SessionsPage));
    expect(html).toContain('data-testid="mosaic"');
    expect(html).not.toContain('data-testid="mosaic-cell-detail"');
  });

  it("passes buildMission's readiness/cards through to Mosaic", () => {
    harness.projects = { data: { ok: true, data: { projects: [{ dir: "p1", name: "Alpha", stage: "build", branch: "main", builder: "claude-code", gate: null, legacyGateField: false, now: "", residuals: [], updated: "2026-09-19T10:00:00.000Z", archived: false }] as unknown as never, skipped: 0 } }, isError: false };
    const html = renderToStaticMarkup(createElement(SessionsPage));
    expect(html).toContain('data-readiness="ready"');
    expect(html).toContain('data-cards="1"');
  });

  it("keeps projects/hub/panes-tail enabled and agents/config disabled in the default mosaic view (round-3 F6)", () => {
    harness.useMissionCalls.length = 0;
    harness.useQueryCalls.length = 0;
    renderToStaticMarkup(createElement(SessionsPage));
    expect(harness.useMissionCalls).toContainEqual({ endpoint: "projects", enabled: true });
    expect(harness.useMissionCalls).toContainEqual({ endpoint: "hub", enabled: true });
    expect(harness.useMissionCalls).toContainEqual({ endpoint: "agents", enabled: false });
    expect(harness.useQueryCalls).toContainEqual({ key: "panes-tail", enabled: true });
    expect(harness.useQueryCalls).toContainEqual({ key: "config", enabled: false });
  });
});