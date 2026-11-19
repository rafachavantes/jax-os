import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

const harness = vi.hoisted(() => ({
  projects: { data: { ok: true, data: { projects: [], skipped: 0 } }, isError: false },
  hub: { data: { ok: true, data: { byProject: {}, historicalTruncated: false, codexSource: "ok", tmuxSource: "ok", prefs: {}, settings: { archiveAfterDays: 30 } } }, isError: false },
  agents: { data: undefined as unknown, isError: false, dataUpdatedAt: 0, error: undefined as unknown },
  panesTail: { data: undefined as unknown, isError: false },
  config: { data: undefined as unknown, isError: false, isLoading: false },
  target: { data: undefined as unknown, dataUpdatedAt: 0, refetch: () => {} },
  useMissionCalls: [] as { endpoint: string; enabled: boolean }[],
  useQueryCalls: [] as { key: string; enabled: boolean }[],
}));

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en-US",
}));
// Client navigation is a platform detail here: static rendering just needs the anchor.
vi.mock("next/link", () => ({
  default: ({ href, children }: { href: string; children: React.ReactNode }) =>
    createElement("a", { href }, children),
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
    if (opts.queryKey[1] === "panes-tail") return harness.panesTail;
    if (opts.queryKey[1] === "target") return harness.target;
    return harness.config;
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

import SessionsPage, {
  backHref, largerWindowHref, parseFocusedLocation, selectorIdentity, targetFreshness, targetQueryString,
} from "./page";
import {
  captureControlTicket, controlModeAfter, controlStateReducer, controlTicketCurrent, type ControlState,
} from "@/components/sessions/TerminalPanel";
import { sessionAccessHref, type SessionAccessSelector } from "@/lib/sessionAccess";

const idle: ControlState = { mode: "ro", busy: false, error: null };
const writable: ControlState = { mode: "rw", busy: false, error: null };

const cardSelector: SessionAccessSelector = {
  kind: "card", project: "jax-os", harnessSession: "sess-a", pane: "%7", incarnation: "1234:1700000000",
};

describe("controlModeAfter (I1 tmux semantics)", () => {
  it("takes control only after a confirmed success on the same session", () => {
    expect(controlModeAfter("take", true, true)).toBe("rw");
    expect(controlModeAfter("take", true, false)).toBe("ro");
    expect(controlModeAfter("take", false, true)).toBe("ro");
  });

  it("releases to read-only immediately, even when recording fails", () => {
    expect(controlModeAfter("release", false, true)).toBe("ro");
    expect(controlModeAfter("release", true, true)).toBe("ro");
  });
});

describe("controlStateReducer (production Take/Release seam)", () => {
  it("pending Release leaves the state ro immediately while the POST is unresolved", () => {
    expect(controlStateReducer(writable, { type: "request", action: "release" }))
      .toEqual({ mode: "ro", busy: true, error: null });
  });

  it("keeps ro through a delayed release failure and records the error", () => {
    let state = controlStateReducer(writable, { type: "request", action: "release" });
    state = controlStateReducer(state, { type: "settle", action: "release", ok: false, error: "recording failed", current: true });
    expect(state.mode).toBe("ro");
    expect(state.busy).toBe(false);
    expect(state.error).toBe("recording failed");
  });

  it("is a synchronous duplicate guard and a late prior-session settle changes nothing", () => {
    const first = controlStateReducer(idle, { type: "request", action: "take" });
    expect(controlStateReducer(first, { type: "request", action: "take" })).toEqual(first);

    let state = controlStateReducer(writable, { type: "request", action: "release" });
    state = controlStateReducer(state, { type: "reset" });
    state = controlStateReducer(state, { type: "settle", action: "release", ok: false, error: "stale failure", current: false });
    expect(state.mode).toBe("ro");
    expect(state.error).toBeNull();
    expect(state.busy).toBe(false);
  });
});

describe("control ticket lifetime (A -> B -> A)", () => {
  it("a delayed Take success for A after selecting B and A again stays read-only", () => {
    let token = 0;
    const ticket = captureControlTicket("A", token)!;
    token += 1;
    expect(controlTicketCurrent("B", token, ticket)).toBe(false);
    token += 1;
    expect(controlTicketCurrent("A", token, ticket)).toBe(false);
  });
});

describe("validation freshness (client receipt clock)", () => {
  it("negative and >=10s receipt ages are expired; fresh receipts are fresh", () => {
    expect(targetFreshness(1_000, 1_000 + 9_999)).toBe("fresh");
    expect(targetFreshness(1_000, 1_000 + 10_000)).toBe("expired");
    expect(targetFreshness(1_000, 999)).toBe("expired");
  });
});

describe("focused deep links", () => {
  it("opens the exact card on desktop (home) and phone (mosaic) without a second selection", () => {
    for (const from of ["home", "mosaic"] as const) {
      const parsed = parseFocusedLocation(sessionAccessHref(cardSelector, from).slice("/tmux".length));
      expect(parsed).toEqual({ kind: "focus", selector: cardSelector, origin: from });
    }
  });

  it("a missing selector never becomes a focus, and an invalid one fails instead of falling back", () => {
    expect(parseFocusedLocation("")).toEqual({ kind: "none" });
    expect(parseFocusedLocation("?view=activity")).toEqual({ kind: "none" });
    expect(parseFocusedLocation("?kind=card&project=p")).toEqual({ kind: "invalid" });
    expect(parseFocusedLocation("?kind=tmux&session=a&session=b")).toEqual({ kind: "invalid" });
  });

  it("Back follows only the fixed origin enum; a larger window falls back to Home", () => {
    expect(backHref("mosaic")).toBe("/tmux");
    expect(backHref("home")).toBe("/");
    expect(backHref(null)).toBe("/");
    const larger = largerWindowHref(cardSelector);
    expect(larger.startsWith("/tmux?")).toBe(true);
    expect(larger).not.toContain("from=");
    expect(parseFocusedLocation(larger.slice("/tmux".length))).toMatchObject({ kind: "focus", origin: null });
  });

  it("builds bounded selector/expected query strings with URLSearchParams", () => {
    const query = new URLSearchParams(targetQueryString(cardSelector, { target: "$2:@3.%7", incarnation: "1234:1700000000" }));
    expect(query.get("kind")).toBe("card");
    expect(query.get("pane")).toBe("%7");
    expect(query.get("expectedTarget")).toBe("$2:@3.%7");
    expect(selectorIdentity(cardSelector)).toBe("card|jax-os|sess-a|%7|1234:1700000000");
    expect(selectorIdentity({ kind: "tmux", session: "a" })).toBe("tmux|a");
  });
});

describe("SessionsPage — mosaic-first view", () => {
  it("renders the Mosaic by default, not the focused view", () => {
    const html = renderToStaticMarkup(createElement(SessionsPage));
    expect(html).toContain('data-testid="mosaic"');
    expect(html).not.toContain('data-testid="focused-view"');
  });

  it("keeps projects/hub/panes-tail enabled and agents/config disabled in the default mosaic view", () => {
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
