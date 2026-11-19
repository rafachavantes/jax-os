import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";

const nav = vi.hoisted(() => ({
  search: "",
  push: [] as string[],
  replace: [] as string[],
}));

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
// Mounted tests of the real router are not possible in this Node/SSR environment; the location
// contract itself is asserted here (a URL-driven open must never navigate on its own).
vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(nav.search),
  useRouter: () => ({
    push: (href: string) => { nav.push.push(href); },
    replace: (href: string) => { nav.replace.push(href); },
  }),
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
  FRESHNESS_MS, REQUEST_TIMEOUT_MS, TARGET_POLL_MS, backHref, fetchTarget, largerWindowHref, parseFocusedLocation,
  parseTargetResult, selectorIdentity, targetApplication, targetAppliesTo, targetFreshness, targetQueryKey, targetQueryString,
  type ValidationLifetime,
} from "./page";
import {
  accessLifecycle, captureControlTicket, controlModeAfter, controlStateReducer, controlTicketCurrent,
  initialAccessLifecycle, type ControlState,
} from "@/components/sessions/TerminalPanel";
import { sessionAccessHref, type ExpectedSessionTarget, type SessionAccessSelector } from "@/lib/sessionAccess";

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

  it("a fresh Take on the now-current A after A -> B -> A succeeds", () => {
    let token = 0;
    const firstA = captureControlTicket("A", token)!;
    token += 1; // select B
    token += 1; // select A again
    let state = controlStateReducer(idle, { type: "request", action: "take" });
    state = controlStateReducer(state, { type: "reset" });
    state = controlStateReducer(state, { type: "reset" });
    state = controlStateReducer(state, {
      type: "settle", action: "take", ok: true, error: null, current: controlTicketCurrent("A", token, firstA),
    });
    expect(state.mode).toBe("ro");

    const fresh = captureControlTicket("A", token)!;
    state = controlStateReducer(state, { type: "request", action: "take" });
    state = controlStateReducer(state, {
      type: "settle", action: "take", ok: true, error: null, current: controlTicketCurrent("A", token, fresh),
    });
    expect(state.mode).toBe("rw");
  });

  it("a polling fallback to another active session also invalidates an in-flight ticket", () => {
    let token = 0;
    const ticket = captureControlTicket("A", token)!;
    token += 1;
    expect(controlTicketCurrent("B", token, ticket)).toBe(false);
  });

  it("an observed lost-target invalidates a pending Take", () => {
    let token = 0;
    const ticket = captureControlTicket("A", token)!;
    let state = controlStateReducer(idle, { type: "request", action: "take" });
    // The validation effect bumps the token and resets control when the target is lost.
    token += 1;
    state = controlStateReducer(state, { type: "reset" });
    state = controlStateReducer(state, {
      type: "settle", action: "take", ok: true, error: null, current: controlTicketCurrent("A", token, ticket),
    });
    expect(state.mode).toBe("ro");
    expect(state.busy).toBe(false);
  });
});

describe("control reducer restoration (M3b final corrections)", () => {
  it("takes control only after a confirmed success; a delayed failure stays read-only", () => {
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

  it("reset drops control without hiding an in-flight busy request", () => {
    const requested = controlStateReducer(writable, { type: "request", action: "release" });
    expect(controlStateReducer(requested, { type: "reset" })).toEqual({ mode: "ro", busy: true, error: null });
  });

  it("a stale success cannot alter current control or feedback", () => {
    const requested = controlStateReducer(idle, { type: "request", action: "take" });
    const afterReset = controlStateReducer(requested, { type: "reset" });
    const stale = controlStateReducer(afterReset, {
      type: "settle", action: "take", ok: true, error: null, current: false,
    });
    expect(stale.mode).toBe("ro");
    expect(stale.error).toBeNull();
    expect(stale.busy).toBe(false);
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

const expected: ExpectedSessionTarget = { target: "$2:@3.%7", incarnation: "1234:1700000000" };
const resolvedAccess = {
  selector: cardSelector, expected, sessionName: "project agent", runtime: "claude" as const,
  observedAt: "2023-11-14T22:15:00.000Z", roUrl: "http://127.0.0.1:10000/ro/?arg=%242%3A%403.%257",
};
const lifetime = (token: number): ValidationLifetime => ({ token, selector: selectorIdentity(cardSelector), expected: "" });

function response(body: unknown, status = 200): Response {
  return { ok: status >= 200 && status < 300, status, json: async () => body } as Response;
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

const abortingFetch = (_url: RequestInfo | URL, init?: RequestInit) =>
  new Promise<Response>((_resolve, reject) => {
    init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")), { once: true });
  });

describe("bounded target validation results (M3b corrections)", () => {
  it("maps network rejection, non-2xx, malformed envelopes and wrong success shapes to unknown", async () => {
    const signal = new AbortController().signal;
    const rejection = await fetchTarget(cardSelector, null, signal, async () => { throw new Error("down"); });
    expect(rejection).toMatchObject({ ok: false, code: "source-failure" });

    expect(await fetchTarget(cardSelector, null, signal, async () => response({ ok: true, data: resolvedAccess }, 500)))
      .toMatchObject({ ok: false, code: "source-failure" });
    expect(await fetchTarget(cardSelector, null, signal, async () => response("not json")))
      .toMatchObject({ ok: false, code: "source-failure" });
    expect(await fetchTarget(cardSelector, null, signal, async () => response({ ok: true })))
      .toMatchObject({ ok: false, code: "source-failure" });
    expect(await fetchTarget(cardSelector, null, signal, async () => response({ ok: true, data: { ...resolvedAccess, expected: { target: "bad" } } })))
      .toMatchObject({ ok: false, code: "source-failure" });
    expect(await fetchTarget(cardSelector, null, signal, async () => response({ ok: true, data: { ...resolvedAccess, roUrl: 7 } })))
      .toMatchObject({ ok: false, code: "source-failure" });
    expect(await fetchTarget(cardSelector, null, signal, async () => response({ ok: true, data: resolvedAccess })))
      .toMatchObject({ ok: true, data: { expected } });
  });

  it("times out through the injected fetch signal as a bounded unknown failure", async () => {
    const started = Date.now();
    const result = await fetchTarget(cardSelector, null, new AbortController().signal, abortingFetch, 5);
    expect(result).toMatchObject({ ok: false, code: "source-failure" });
    expect(Date.now() - started).toBeLessThan(1_000);
  });

  it("keeps the fixed polling, timeout and freshness bounds and never shares a cache entry across lifetimes", () => {
    expect(targetQueryKey("card|a", "", 1)).not.toEqual(targetQueryKey("card|a", "", 2));
    expect(targetQueryKey("card|a", "", 1)).toEqual(targetQueryKey("card|a", "", 1));
    expect(TARGET_POLL_MS).toBe(5_000);
    expect(REQUEST_TIMEOUT_MS).toBe(5_000);
    expect(FRESHNESS_MS).toBe(10_000);
    expect(parseTargetResult({ ok: false, code: "lost-target", error: "x" })).toEqual({ ok: false, code: "lost-target", error: "x" });
    expect(parseTargetResult({ ok: false, code: "made-up" })).toMatchObject({ ok: false, code: "source-failure" });
  });
});

describe("validation lifetimes (deferred old success, failure and rearm)", () => {
  it("tags each request with the lifetime captured at begin and ignores a stale result", async () => {
    const first = deferred<Response>();
    const pending = fetchTarget(cardSelector, null, new AbortController().signal, () => first.promise);
    const responseLifetime = lifetime(1);
    first.resolve(response({ ok: true, data: resolvedAccess }));
    const responseA = { lifetime: responseLifetime, result: await pending };

    const currentB = lifetime(2);
    expect(targetAppliesTo(currentB, responseA.lifetime)).toBe(false);
    expect(targetApplication(responseA, currentB, 1_000, 1_001, 0)).toEqual({ kind: "ignore" });
    expect(targetApplication(responseA, currentB, 1_000, 1_001, 1_500)).toEqual({ kind: "ignore" });

    const currentA = lifetime(1);
    expect(targetApplication(responseA, currentA, 1_000, 1_001, 0)).toMatchObject({ kind: "ok" });
    expect(targetApplication(responseA, currentA, 1_000, 1_000 + FRESHNESS_MS, 0)).toEqual({ kind: "expired" });
    expect(targetApplication({ ...responseA, result: { ok: false, code: "source-failure", error: "x" } }, currentA, 1_000, 1_001, 0))
      .toEqual({ kind: "failure", status: "unknown" });
  });

  it("a held old GET completing after hide/show cannot attach; only the new lifetime's success can", async () => {
    let state = accessLifecycle(initialAccessLifecycle, { type: "open" });
    state = accessLifecycle(state, { type: "validation", roUrl: "http://ro/A" });
    state = accessLifecycle(state, { type: "hide" });
    state = accessLifecycle(state, { type: "show" });

    const oldApplication = targetApplication(
      { lifetime: lifetime(1), result: { ok: true, data: resolvedAccess } }, lifetime(2), 1_000, 1_001, 0,
    );
    expect(oldApplication.kind).toBe("ignore");
    expect(state.frame).toBeNull();

    const newApplication = targetApplication(
      { lifetime: lifetime(2), result: { ok: true, data: resolvedAccess } }, lifetime(2), 1_000, 1_001, 0,
    );
    if (newApplication.kind !== "ok") throw new Error("expected a current validation");
    state = accessLifecycle(state, { type: "validation", roUrl: newApplication.data.roUrl });
    expect(state.frame).toMatchObject({ mode: "ro" });
  });

  it("a current failure removes RW and latches the frame off; a later success cannot reattach without Reconnect", () => {
    let state = accessLifecycle(initialAccessLifecycle, { type: "open" });
    state = accessLifecycle(state, { type: "takeStart" });
    state = accessLifecycle(state, { type: "takeOk", rwUrl: "http://rw/A" });
    expect(state.frame?.mode).toBe("rw");

    state = accessLifecycle(state, { type: "failure", status: "unknown" });
    expect(state.frame).toBeNull();
    expect(controlStateReducer(writable, { type: "reset" }).mode).toBe("ro");

    const laterPoll = accessLifecycle(state, { type: "validation", roUrl: "http://ro/later" });
    expect(laterPoll.frame).toBeNull();

    state = accessLifecycle(state, { type: "reconnect" });
    state = accessLifecycle(state, { type: "validation", roUrl: "http://ro/fresh" });
    expect(state.frame).toMatchObject({ mode: "ro" });
  });
});

beforeEach(() => {
  nav.search = "";
  nav.push.length = 0;
  nav.replace.length = 0;
  harness.projects = { data: { ok: true, data: { projects: [], skipped: 0 } }, isError: false };
});

describe("live same-route location (M3b final corrections)", () => {
  it("a focused query renders the exact focused view and never navigates on its own", () => {
    nav.search = sessionAccessHref(cardSelector, "mosaic").split("?")[1];
    const html = renderToStaticMarkup(createElement(SessionsPage));
    expect(html).toContain('data-testid="focused-view"');
    expect(html).not.toContain('data-testid="mosaic"');
    expect(html).toContain("openedFromMosaic");
    expect(nav.push).toHaveLength(0);
    expect(nav.replace).toHaveLength(0);
  });

  it("a query without a selector renders the generic Mosaic and never navigates", () => {
    nav.search = "view=activity";
    const html = renderToStaticMarkup(createElement(SessionsPage));
    expect(html).toContain('data-testid="mosaic"');
    expect(html).not.toContain('data-testid="focused-view"');
    expect(nav.push).toHaveLength(0);
    expect(nav.replace).toHaveLength(0);
  });

  it("an invalid focused query fails visibly and never navigates", () => {
    nav.search = "kind=card&project=p";
    const html = renderToStaticMarkup(createElement(SessionsPage));
    expect(html).toContain('data-testid="focused-view"');
    expect(html).toContain("targetUnavailable");
    expect(nav.push).toHaveLength(0);
    expect(nav.replace).toHaveLength(0);
  });
});

describe("SessionsPage — mosaic-first view", () => {
  it("renders the Mosaic by default, not the focused view", () => {
    const html = renderToStaticMarkup(createElement(SessionsPage));
    expect(html).toContain('data-testid="mosaic"');
    expect(html).not.toContain('data-testid="focused-view"');
  });

  it("passes buildMission's readiness/cards through to Mosaic", () => {
    harness.projects = {
      data: {
        ok: true,
        data: {
          projects: [{
            dir: "p1", name: "Alpha", stage: "build", branch: "main", builder: "claude-code",
            gate: null, legacyGateField: false, now: "", residuals: [],
            updated: "2026-09-19T10:00:00.000Z", archived: false,
          }] as unknown as never,
          skipped: 0,
        },
      },
      isError: false,
    };
    const html = renderToStaticMarkup(createElement(SessionsPage));
    expect(html).toContain('data-readiness="ready"');
    expect(html).toContain('data-cards="1"');
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
