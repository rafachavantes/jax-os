import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { UserMission } from "@/server/db/missions";

// No QueryClientProvider exists in a renderToStaticMarkup test — @tanstack/react-query
// itself is mocked, keyed by queryKey, exactly as ServerPanel.test.tsx:8-19 already does.
const harness = vi.hoisted(() => ({
  results: new Map<string, { data: unknown; isError: boolean }>(),
  lastOptions: undefined as { queryKey: unknown[]; queryFn: (ctx: { signal?: AbortSignal }) => unknown; refetchInterval: number } | undefined,
}));
vi.mock("next-intl", () => ({ useTranslations: () => (key: string) => key }));
vi.mock("@tanstack/react-query", () => ({
  useQuery: (opts: { queryKey: unknown[]; queryFn: (ctx: { signal?: AbortSignal }) => unknown; refetchInterval: number }) => {
    harness.lastOptions = opts;
    return harness.results.get(opts.queryKey.join(":")) ?? { data: undefined, isError: false };
  },
}));

import { MissionBlock } from "./MissionBlock";

beforeEach(() => {
  harness.results.clear();
  harness.lastOptions = undefined;
});

const MISSION: UserMission = {
  id: 1, name: "Ship 6 phases tonight", goal: "Merge and review all six phases before sleep",
  statusLine: "Fases 1 e 2 merged, rodando review da fase 3", state: "active",
  startedAt: "2026-09-19T10:00:00.000Z", endedAt: null,
  milestones: [
    { title: "Fase 1", state: "done" },
    { title: "Fase 2", state: "done" },
    { title: "Fase 3", state: "in-progress" },
    { title: "Fase 4", state: "pending" },
  ],
};

describe("MissionBlock (spec §10)", () => {
  it("renders nothing while loading (no query result yet)", () => {
    expect(renderToStaticMarkup(createElement(MissionBlock))).toBe("");
  });
  it("renders nothing when the query errored (data stays undefined)", () => {
    harness.results.set("mission-block:current", { data: undefined, isError: true });
    expect(renderToStaticMarkup(createElement(MissionBlock))).toBe("");
  });
  it("F2: renders nothing when the query errored even though stale active-mission data is still cached", () => {
    harness.results.set("mission-block:current", { data: { ok: true, data: MISSION }, isError: true });
    expect(renderToStaticMarkup(createElement(MissionBlock))).toBe("");
  });
  it("renders nothing when there is no active mission ({ok:true, data:null})", () => {
    harness.results.set("mission-block:current", { data: { ok: true, data: null }, isError: false });
    expect(renderToStaticMarkup(createElement(MissionBlock))).toBe("");
  });
  it("renders name, goal, status line and one dot per milestone when active", () => {
    harness.results.set("mission-block:current", { data: { ok: true, data: MISSION }, isError: false });
    const html = renderToStaticMarkup(createElement(MissionBlock));
    expect(html).toContain("Ship 6 phases tonight");
    expect(html).toContain("Merge and review all six phases before sleep");
    expect(html).toContain("Fases 1 e 2 merged, rodando review da fase 3");
    expect((html.match(/role="listitem"/g) ?? []).length).toBe(4);
  });
  it("F1: statusLine === '' fills the status slot with the goal, and the goal line renders only once", () => {
    harness.results.set("mission-block:current", { data: { ok: true, data: { ...MISSION, statusLine: "" } }, isError: false });
    const html = renderToStaticMarkup(createElement(MissionBlock));
    expect((html.match(/Merge and review all six phases before sleep/g) ?? []).length).toBe(1);
  });
  it("F2: reads the mission-block's own query contract — key, 10s poll, and the /api/mission/current fetch (diff review a10bbf2c7307)", async () => {
    renderToStaticMarkup(createElement(MissionBlock));
    expect(harness.lastOptions?.queryKey).toEqual(["mission-block", "current"]);
    expect(harness.lastOptions?.refetchInterval).toBe(10_000);
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ ok: true, data: null }) });
    vi.stubGlobal("fetch", fetchMock);
    await harness.lastOptions?.queryFn({ signal: undefined });
    expect(fetchMock).toHaveBeenCalledWith("/api/mission/current", { signal: undefined });
    vi.unstubAllGlobals();
  });
  it("F1: keeps min-w-0 and wrap-anywhere classes on the name and status-line slots for a long unbroken value (390px overflow, diff review a10bbf2c7307)", () => {
    const longName = "a".repeat(80);
    const longStatusLine = "b".repeat(280);
    harness.results.set("mission-block:current", {
      data: { ok: true, data: { ...MISSION, name: longName, statusLine: longStatusLine } },
      isError: false,
    });
    const html = renderToStaticMarkup(createElement(MissionBlock));
    const nameMatch = html.match(new RegExp(`<span class="([^"]*)">${longName}</span>`));
    const statusMatch = html.match(new RegExp(`<p class="([^"]*)">${longStatusLine}</p>`));
    expect(nameMatch?.[1]).toContain("min-w-0");
    expect(nameMatch?.[1]).toContain("[overflow-wrap:anywhere]");
    expect(statusMatch?.[1]).toContain("min-w-0");
    expect(statusMatch?.[1]).toContain("[overflow-wrap:anywhere]");
  });
  it("dot classes match each milestone's own state (done/in-progress/pending)", () => {
    harness.results.set("mission-block:current", { data: { ok: true, data: MISSION }, isError: false });
    const html = renderToStaticMarkup(createElement(MissionBlock));
    expect(html).toContain('rounded-full bg-brand"');
    expect(html).toContain("bg-brand/50");
    expect(html).toContain("border-line-strong");
  });
  it("supports the 12-milestone cap with no crash, rail wraps only on overflow (spec §6 Decision 6, F1)", () => {
    const twelve: UserMission = {
      ...MISSION,
      milestones: Array.from({ length: 12 }, (_, i) => ({ title: `Fase ${i + 1}`, state: "pending" as const })),
    };
    harness.results.set("mission-block:current", { data: { ok: true, data: twelve }, isError: false });
    const html = renderToStaticMarkup(createElement(MissionBlock));
    expect((html.match(/role="listitem"/g) ?? []).length).toBe(12);
    // F1: the spec guarantees no exact line-count, only natural wrapping when the
    // rail overflows its row — `flex-wrap` is the mechanism, not a forced wrap.
    expect(html).toContain("flex flex-wrap");
  });
});
