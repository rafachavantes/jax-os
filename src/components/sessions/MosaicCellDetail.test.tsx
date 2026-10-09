// src/components/sessions/MosaicCellDetail.test.tsx
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string, values?: Record<string, string>) => (values ? `${key}:${JSON.stringify(values)}` : key),
  useLocale: () => "en-US",
}));
// Plan correction: the plan's spy inferred `{ ok: true; data: {} }` as its resolved type, so the
// failure-case `mockResolvedValueOnce({ ok: false, error })` below failed tsc (TS2353, a 19th
// error over the plan's 18 baseline). The widened return type keeps the same runtime behavior.
const postSpy = vi.hoisted(() => vi.fn(async (): Promise<{ ok: boolean; data?: unknown; error?: string }> => ({ ok: true, data: {} })));
vi.mock("@/lib/mission", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/mission")>();
  return { ...actual, postWorkflowAction: postSpy };
});

import type { MissionCard, MissionPane } from "@/lib/mission";
import { MosaicCellDetail, takeControl, releaseControl } from "./MosaicCellDetail";

const card: MissionCard = {
  dir: "p1", name: "Alpha", stage: "build", branch: "main", builder: "claude-code",
  gate: null, legacyGateField: false, now: "", residuals: [], updated: "2026-09-19T10:00:00.000Z",
  headline: "needs-you", hubReady: true, lastEventAgo: null, freshnessCount: null, freshnessWarning: false,
  sessions: [], pendingQuestions: [], timeline: [], pr: { status: "unknown" },
  codexSource: "ok", codexUntracked: false,
  heartbeat: new Array(60).fill(0), lastAction: null, prefs: { pinned: false, hiddenAt: null }, visibility: "visible",
};
function pane(overrides: Partial<MissionPane>): MissionPane {
  return {
    pane: "%1", tmuxIncarnation: "1:1", session: "s1", role: "lead", state: "needs-you",
    lastEventTs: null, pendingQuestion: null, subagentCount: 0, capsuleEventId: null, capsuleStatus: null,
    capsuleMinutes: null, capsuleDeclaredAt: null, capsuleMergeAsk: null, capsuleQuestion: null,
    live: true, tmuxSession: "s1",
    ...overrides,
  };
}

describe("MosaicCellDetail — four-way branch (spec §9 rows 22-24, round-1 F2/F3)", () => {
  const single = { pane: "%1", eventId: 9, questions: [{ question: "Env?", multiSelect: false, options: [{ label: "dev" }] }] };
  it("22: simple single-select renders numbered options at the 44px mobile size (round-3 F5), no text input", () => {
    const html = renderToStaticMarkup(createElement(MosaicCellDetail, {
      card, pane: pane({ pendingQuestion: single }), session: "s1", lines: ["a"], busy: false, onAction: () => {}, onClose: () => {},
    }));
    expect(html).toContain("dev");
    expect(html).toContain("min-h-11");
    expect(html).toContain("min-w-11");
    expect(html).toContain("Env?");
    expect(html).toContain("1 · dev");
    expect(html).not.toContain('type="text"');
  });
  it("23: a freeform capsule renders a disabled input before Take Control", () => {
    const html = renderToStaticMarkup(createElement(MosaicCellDetail, {
      card, pane: pane({ capsuleStatus: "needs_input", capsuleEventId: 5 }), session: "s1", lines: null, busy: false, onAction: () => {}, onClose: () => {},
    }));
    expect(html).toContain('type="text"');
    expect(html).toContain('disabled=""');
    expect(html).toContain("takeControl");
  });
  it("23a: blocked with no question renders status text only, no controls", () => {
    const html = renderToStaticMarkup(createElement(MosaicCellDetail, {
      card, pane: pane({ capsuleStatus: "blocked" }), session: "s1", lines: null, busy: false, onAction: () => {}, onClose: () => {},
    }));
    expect(html).toContain("inbox.blocked");
    expect(html).not.toContain('type="text"');
    expect(html).not.toContain("takeControl");
  });
  it("24: nothing pending renders lines only, no controls at all", () => {
    const html = renderToStaticMarkup(createElement(MosaicCellDetail, {
      card, pane: pane({ state: "working", capsuleStatus: null }), session: "s1", lines: ["a"], busy: false, onAction: () => {}, onClose: () => {},
    }));
    expect(html).not.toContain('type="text"');
    expect(html).not.toContain("inbox.blocked");
  });
  it("renders up to 8 lines unsliced, and the muted noOutput line when lines is null", () => {
    const eight = ["1", "2", "3", "4", "5", "6", "7", "8"];
    const html = renderToStaticMarkup(createElement(MosaicCellDetail, { card, pane: pane({}), session: "s1", lines: eight, busy: false, onAction: () => {}, onClose: () => {} }));
    expect(html).toContain("8");
    const empty = renderToStaticMarkup(createElement(MosaicCellDetail, { card, pane: pane({}), session: "s1", lines: null, busy: false, onAction: () => {}, onClose: () => {} }));
    expect(empty).toContain("noOutput");
  });
});

describe("MosaicCellDetail — direct handler tests, no jsdom (round-2 F4 pattern)", () => {
  it("takeControl posts {session}, sets mode rw on success", async () => {
    postSpy.mockClear();
    const setBusy = vi.fn(); const setMode = vi.fn(); const setError = vi.fn();
    await takeControl("s1", setBusy, setMode, setError);
    expect(postSpy).toHaveBeenCalledExactlyOnceWith("/api/tmux/take-control", { session: "s1" });
    expect(setMode).toHaveBeenCalledExactlyOnceWith("rw");
    expect(setBusy).toHaveBeenNthCalledWith(1, true);
    expect(setBusy).toHaveBeenNthCalledWith(2, false);
  });
  it("takeControl records the error and never sets mode rw on failure", async () => {
    postSpy.mockClear();
    postSpy.mockResolvedValueOnce({ ok: false, error: "session not found" });
    const setBusy = vi.fn(); const setMode = vi.fn(); const setError = vi.fn();
    await takeControl("s1", setBusy, setMode, setError);
    expect(setMode).not.toHaveBeenCalled();
    // Plan correction: takeControl's own contract resets setError(null) before the POST, so on
    // failure setError is called twice — the plan's ExactlyOnceWith cannot hold. LastCalledWith
    // still proves the failure's message is what was recorded.
    expect(setError).toHaveBeenLastCalledWith("session not found");
  });
  it("releaseControl always sets mode ro and clears the error", async () => {
    postSpy.mockClear();
    const setBusy = vi.fn(); const setMode = vi.fn(); const setError = vi.fn();
    await releaseControl("s1", setBusy, setMode, setError);
    expect(postSpy).toHaveBeenCalledExactlyOnceWith("/api/tmux/release-control", { session: "s1" });
    expect(setMode).toHaveBeenCalledExactlyOnceWith("ro");
    expect(setError).toHaveBeenCalledExactlyOnceWith(null);
  });
});
