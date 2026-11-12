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
const postSpy = vi.hoisted(() => vi.fn(async () => ({ ok: true, data: {} })));
vi.mock("@/lib/mission", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/mission")>();
  return { ...actual, postWorkflowAction: postSpy };
});

import {
  cancelAction, cardKey, confirmCancel, type Mission, type MissionActiveRun, type MissionActiveRunSummary, type MissionCard,
} from "@/lib/mission";
import type { MissionPane } from "@/lib/mission";
import { onConfirmCancel } from "./ProjectCard";
import { InboxStrip, runStuckCancel } from "./InboxStrip";
import { missionOf, mkActivity, mkHubProject, mkPane, mkSeat, mkSplit, runSummary, U } from "@/lib/multiSessionFixtures";

const A = U("a");
const B = U("b");

function active(runId: string): MissionActiveRun {
  return { runId, startedAt: null, kind: "build", runtime: "claude", target: "feat/x", count: 1, lastStep: null, stage: "build", descriptionKey: "building", targetKind: "branch" };
}
function summary(runId: string): MissionActiveRunSummary {
  return { runId, role: "builder", kind: "build", runtime: "claude", target: null, startedAt: null, stage: "build", descriptionKey: "building", targetKind: "branch" };
}
function card(over: Partial<MissionCard> & { dir: string }): MissionCard {
  return {
    name: over.dir, stage: "build", branch: "main", builder: "claude-code",
    gate: null, legacyGateField: false, now: "", residuals: [], updated: "2026-11-10T10:00:00-03:00",
    headline: "idle", hubReady: true, lastEventAgo: null, freshnessCount: null, freshnessWarning: false,
    sessions: [], pendingQuestions: [], timeline: [], pr: { status: "unknown" },
    codexSource: "ok", codexUntracked: false,
    heartbeat: new Array(60).fill(0), lastAction: null, prefs: { pinned: false, hiddenAt: null }, visibility: "visible",
    ...over,
  };
}
const scope = { kind: "session" as const, sessionId: A, transport: "tmux" as const, runtime: "claude" as const, name: null, tmuxSessions: ["s1"], panes: ["%1"] };

describe("cancelAction (spec §10)", () => {
  it("a working session card targets its own newest run", () => {
    expect(cancelAction(card({ dir: "p", headline: "working", scope, activeRun: active("r2") }))).toEqual({ url: "/api/workflow/cancel", body: { run_id: "r2" } });
  });
  it("a waiting plain card targets its own run only when it stands in for a session", () => {
    expect(cancelAction(card({ dir: "p", headline: "waiting", key: "p#session:A", activeRun: active("r1") }))?.body.run_id).toBe("r1");
    expect(cancelAction(card({ dir: "p", headline: "waiting", activeRun: active("r1") }))).toBeNull();
  });
  it("a grouped project card never cancels; idle never cancels; no run never cancels", () => {
    expect(cancelAction(card({ dir: "p", headline: "working", activeRun: active("r1"), group: { cards: [], footer: null } }))).toBeNull();
    expect(cancelAction(card({ dir: "p", headline: "idle", activeRun: active("r1") }))).toBeNull();
    expect(cancelAction(card({ dir: "p", headline: "working" }))).toBeNull();
  });
  it("the unlinked card targets its newest in-flight run", () => {
    const unlinked = card({ dir: "p", key: "p#unlinked", headline: "working", scope: { kind: "unlinked" }, activeRun: active("r9"), activeRuns: [summary("r9"), summary("r8")] });
    expect(cancelAction(unlinked)?.body.run_id).toBe("r9");
  });

  it("over the 2→1→0→2 transition the card holding the build always cancels that build", () => {
    const panes = [mkPane({ pane: "%1", tmuxIncarnation: "1:1", harnessSession: A }), mkPane({ pane: "%2", tmuxIncarnation: "1:1", harnessSession: B })];
    const bRun = { ...runSummary("rB"), startedAt: null };
    const two = mkSplit({ seats: [mkSeat({ sessionId: A, panes: [{ pane: "%1", tmuxIncarnation: "1:1" }] }), mkSeat({ sessionId: B, panes: [{ pane: "%2", tmuxIncarnation: "1:1" }], activity: mkActivity({ activeRuns: [bRun], activeRun: active("rB") }) })] });
    const one = mkSplit({ seats: [mkSeat({ sessionId: A, panes: [{ pane: "%1", tmuxIncarnation: "1:1" }] })], unlinked: mkActivity({ activeRuns: [bRun], activeRun: active("rB") }) });
    const zero = mkSplit({ seats: [], unlinked: mkActivity({ activeRuns: [bRun], activeRun: active("rB") }) });
    const mk = (split: ReturnType<typeof mkSplit>, ps = panes) => missionOf({ p: mkHubProject({ panes: ps, split }) });
    const holder = (m: Mission) => m.model.cards.flatMap((c) => c.group?.cards ?? [c]).find((c) => c.activeRuns?.length === 1 || c.activeRun?.runId === "rB")!;
    for (const m of [mk(two), mk(one, [panes[0]]), mk(zero, [panes[0]]), mk(two)]) {
      expect(cancelAction(holder(m))?.body.run_id).toBe("rB");
    }
  });
});

describe("confirmCancel pins the run id", () => {
  it("posts the pinned id while it is in flight", () => {
    const c = card({ dir: "p", activeRun: null, activeRuns: [summary("r1")] });
    expect(confirmCancel("r1", c)?.body.run_id).toBe("r1");
  });
  it("does not post after the run finished", () => {
    expect(confirmCancel("r1", card({ dir: "p", activeRun: null, activeRuns: [] }))).toBeNull();
  });
  it("never posts the replacement when the pinned run was replaced", () => {
    const c = card({ dir: "p", activeRun: active("r2"), activeRuns: [summary("r2")] });
    expect(confirmCancel("r1", c)).toBeNull();
  });
  it("still posts the pinned run when it survived alongside a newer one", () => {
    const c = card({ dir: "p", activeRun: active("r2"), activeRuns: [summary("r2"), summary("r1")] });
    expect(confirmCancel("r1", c)?.body.run_id).toBe("r1");
  });
});

describe("call-site wiring", () => {
  it("onConfirmCancel closes without posting when the run finished, and posts the pinned id otherwise", () => {
    const post = vi.fn();
    const close = vi.fn();
    onConfirmCancel("r1", card({ dir: "p", activeRun: null, activeRuns: [] }), post, close);
    expect(post).not.toHaveBeenCalled();
    expect(close).toHaveBeenCalledOnce();

    post.mockClear(); close.mockClear();
    onConfirmCancel("r1", card({ dir: "p", activeRun: null, activeRuns: [summary("r1")] }), post, close);
    expect(post).toHaveBeenCalledWith("/api/workflow/cancel", { run_id: "r1" });
    expect(close).toHaveBeenCalledOnce();
  });

  it("onConfirmCancel never posts a replacement run", () => {
    const post = vi.fn();
    onConfirmCancel("r1", card({ dir: "p", activeRun: active("r2"), activeRuns: [summary("r2")] }), post, () => {});
    expect(post).not.toHaveBeenCalled();
  });

  it("runStuckCancel posts the pinned id, and only closes when the run is gone", async () => {
    postSpy.mockClear();
    const setBusy = vi.fn();
    const setConfirming = vi.fn();
    await runStuckCancel("r1", card({ dir: "p", activeRun: null, activeRuns: [] }), setBusy, setConfirming);
    expect(postSpy).not.toHaveBeenCalled();
    expect(setConfirming).toHaveBeenCalledWith(false);
    expect(setBusy).not.toHaveBeenCalled();

    postSpy.mockClear(); setBusy.mockClear(); setConfirming.mockClear();
    await runStuckCancel("r1", card({ dir: "p", activeRun: null, activeRuns: [summary("r1"), summary("r2")] }), setBusy, setConfirming);
    expect(postSpy).toHaveBeenCalledWith("/api/workflow/cancel", { run_id: "r1" });
    expect(setConfirming).toHaveBeenCalledWith(false);
  });
});

describe("stuck rail lists board cards with their own cancel target", () => {
  it("a stuck session card and a stuck plain project are two rows, each cancelling its own run", () => {
    const himalaya: MissionPane = {
      pane: "%1", tmuxIncarnation: "1:1", session: "s1", role: "lead", live: true, state: "stuck",
      lastEventTs: null, pendingQuestion: null, capsuleStatus: null, capsuleMinutes: 1, capsuleDeclaredAt: null,
      capsuleMergeAsk: null, capsuleQuestion: null, subagentCount: 0, capsuleEventId: null, tmuxSession: "s1",
    };
    const idlePane: MissionPane = { ...himalaya, pane: "%2", state: "idle" };
    const sessionCard = card({
      dir: "p", key: `p#session:${A}`, headline: "stuck",
      scope: { ...scope, name: "multi-sessao" },
      sessions: [{ transport: "tmux", session: "s1", panes: [himalaya] }],
      activeRun: active("r5"),
    });
    const idleCard = card({ dir: "p", key: `p#session:${B}`, headline: "idle", sessions: [{ transport: "tmux", session: "s1", panes: [idlePane] }] });
    const project = card({ dir: "p", headline: "needs-you", group: { cards: [sessionCard, idleCard], footer: null } });
    const plain = card({ dir: "q", headline: "stuck", sessions: [{ transport: "tmux", session: "s1", panes: [himalaya] }], activeRun: active("r6") });

    const mission: Mission = {
      readiness: { kind: "ready" },
      model: {
        cards: [project, plain], uncardedProjectNames: [], historicalTruncated: false, skipped: 0,
        inbox: [], inboxTotalCount: 0, inboxPrRemainder: 0, prTruncatedAny: false,
        codexSource: "ok", tmuxSource: "ok", archiveAfterDays: 14, hiddenCount: 0, autoHiddenCount: 0, reposRoot: "",
      },
    };
    const html = renderToStaticMarkup(createElement(InboxStrip, { mission, onExpand: () => {} }));
    expect(html).toContain("multi-sessao");
    expect(cancelAction(sessionCard)?.body.run_id).toBe("r5");
    expect(cancelAction(plain)?.body.run_id).toBe("r6");
    expect(cardKey(sessionCard)).toBe(`p#session:${A}`);
  });
});
