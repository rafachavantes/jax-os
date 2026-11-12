import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en-US",
}));
vi.mock("@/components/RelativeTime", () => ({ RelativeTime: () => null }));
vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({}),
  useQuery: () => ({ data: { ok: true, data: { integrations: { github: true } } } }),
}));
vi.mock("@/lib/mission", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/mission")>();
  return { ...actual, postWorkflowAction: vi.fn() };
});

import {
  boardEntries, cancelAction, cardKey, type Mission, type MissionCard,
} from "@/lib/mission";
import { ProjectsColumn } from "./ProjectsColumn";
import { AnswerDraftsProvider, createAnswerDraftStore, draftKeys, draftUpdater } from "./AnswerDrafts";
import { missionOf, mkActivity, mkHubProject, mkPane, mkSeat, mkSplit, runSummary, U } from "@/lib/multiSessionFixtures";
import type { HubActiveRun, HubSessionSplit } from "@/server/db/workflows";

const A = U("a");
const B = U("b");
const paneRef = (pane: string) => ({ pane, tmuxIncarnation: "1:1" });

function run(runId: string): HubActiveRun {
  return { runId, startedAt: null, kind: "build", runtime: "claude", target: "feat/x", count: 1, lastStep: null };
}

const capsule = (eventId: number): NonNullable<ReturnType<typeof mkPane>["capsule"]> => ({
  status: "needs_input", minutes: null, declaredAt: "2026-11-12T11:00:00.000Z", eventId, mergeAsk: null, question: null,
});

function modelAt(step: "one" | "two" | "zero", eventId = 5): Mission {
  const aPane = mkPane({ pane: "%1", tmuxIncarnation: "1:1", harnessSession: A, capsule: capsule(eventId) });
  const bPane = mkPane({ pane: "%2", tmuxIncarnation: "1:1", harnessSession: B });
  const aSeat = mkSeat({ sessionId: A, panes: [paneRef("%1")] });
  const bSeat = mkSeat({ sessionId: B, panes: [paneRef("%2")], activity: mkActivity({ activeRuns: [runSummary("rB")], activeRun: run("rB") }) });
  const unlinked = mkActivity({ activeRuns: [runSummary("rB")], activeRun: run("rB") });
  let split: HubSessionSplit;
  if (step === "zero") split = mkSplit({ seats: [], unlinked });
  else if (step === "one") split = mkSplit({ seats: [aSeat], unlinked });
  else split = mkSplit({ seats: [aSeat, bSeat] });
  const panes = step === "zero" ? [aPane, bPane] : step === "one" ? [aPane] : [aPane, bPane];
  return missionOf({ Alpha: mkHubProject({ panes, split }) });
}

function htmlOf(m: Mission, store: ReturnType<typeof createAnswerDraftStore>, expandedDir: string | null = `Alpha#session:${A}`): string {
  const column = createElement(ProjectsColumn, {
    mission: m, reposRoot: "/repos", counts: null, filter: null, expandedDir, onToggleExpand: () => {},
  });
  return renderToStaticMarkup(createElement(AnswerDraftsProvider, { store }, column));
}

function segment(html: string, key: string): string {
  const start = html.indexOf(`data-card-key="${key}"`);
  if (start < 0) return "";
  const rest = html.slice(start);
  const next = rest.indexOf('data-card-key="', 10);
  return next < 0 ? rest : rest.slice(0, next);
}

function cardsShown(m: Mission): MissionCard[] {
  return boardEntries(m.model.cards, null).flatMap((e) => (e.kind === "group" ? e.cards : [e.card]));
}

describe("session lifecycle keeps expansion and drafts (spec §11, §15)", () => {
  it("1→2→1 keeps A's key, expansion and draft at every step", () => {
    const store = createAnswerDraftStore();
    draftUpdater(store, draftKeys.freeform(5), () => {})("half typed");
    for (const step of ["one", "two", "one"] as const) {
      const html = htmlOf(modelAt(step), store);
      const a = segment(html, `Alpha#session:${A}`);
      expect(a).not.toBe("");
      expect(a).toContain('aria-expanded="true"');
      expect(a).toContain("half typed");
    }
  });

  it("2→1→0→2 keeps A's draft through the zero-seat step and restores the card expanded", () => {
    const store = createAnswerDraftStore();
    draftUpdater(store, draftKeys.freeform(5), () => {})("half typed");
    for (const step of ["two", "one"] as const) {
      const a = segment(htmlOf(modelAt(step), store), `Alpha#session:${A}`);
      expect(a).toContain('aria-expanded="true"');
      expect(a).toContain("half typed");
    }
    const zeroModel = modelAt("zero");
    const zeroHtml = htmlOf(zeroModel, store);
    expect(zeroHtml).not.toContain(`data-card-key="Alpha#session:${A}"`);
    expect(zeroHtml).toContain('data-card-key="Alpha#unlinked"');
    expect(store.get("freeform:5")).toBe("half typed");

    const reopened = segment(htmlOf(modelAt("two"), store), `Alpha#session:${A}`);
    expect(reopened).toContain('aria-expanded="true"');
    expect(reopened).toContain("half typed");
  });

  it("the Cancel target follows the card that holds the run at every step", () => {
    const targets = (["two", "one", "zero", "two"] as const).map((step) => {
      const m = modelAt(step);
      const holder = cardsShown(m).find((c) => (c.activeRuns ?? []).some((r) => r.runId === "rB") || c.activeRun?.runId === "rB");
      return { step, key: holder ? cardKey(holder) : null, runId: holder ? cancelAction(holder)?.body.run_id ?? null : null };
    });
    expect(targets.map((t) => t.runId)).toEqual(["rB", "rB", "rB", "rB"]);
    expect(targets[0].key).toBe(`Alpha#session:${B}`);
    expect(targets[1].key).toBe("Alpha#unlinked");
    expect(targets[2].key).toBe("Alpha#unlinked");
    expect(targets[3].key).toBe(`Alpha#session:${B}`);

    for (const step of ["two", "one", "zero", "two"] as const) {
      const m = modelAt(step);
      const working = boardEntries(m.model.cards, "working").flatMap((e) => (e.kind === "group" ? e.cards : [e.card])).map(cardKey);
      expect(working).toContain(targets.find((t) => t.step === step)!.key);
    }
  });

  it("a different question event on the reopened session does not inherit the draft, and B never shows it", () => {
    const store = createAnswerDraftStore();
    draftUpdater(store, draftKeys.freeform(5), () => {})("half typed");
    const html = htmlOf(modelAt("two", 7), store);
    const a = segment(html, `Alpha#session:${A}`);
    expect(a).toContain("<textarea");
    expect(a).not.toContain("half typed");
    const b = segment(html, `Alpha#session:${B}`);
    expect(b).not.toContain("half typed");
  });
});
