import { afterEach, describe, expect, it, vi } from "vitest";
import { answerEventId, bucketOf, buildMission, buildTicker, composeIndexedReply, dispatchBodyFor, elapsedMinutesSince, formatClock, gateWaitingLine, hasQuestionText, heartbeatPath, isMergeAskEligible, liveSessions, mosaicPaneAction, mosaicPaneState, newestPr, pageSlice, paneState, PIPELINE_STAGES, postWorkflowAction, selectPendingAction, stageDotIndex, statusPillSuffix, stripPresentation, stuckLine, summarizeBuckets, tickerSentence, worstFinding, worstPaneState, type DispatchFields, type TickerItem } from "./mission";
import { parseIndexedReply } from "./workflow";
import type { Envelope } from "./api";
import type { Project, ProjectScan } from "../server/collectors/projects";
import type { Capsule, HubActiveRun, HubEnvelopeData, HubLastRun, HubPane, HubProject, PendingQuestion } from "../server/db/workflows";
import type { PrResult } from "../server/collectors/github";
import type { MissionModel, MissionReadiness, LastRunLabelKey, MissionState, MissionCard, MissionPane, MissionPendingQuestion, MissionSession, MissionTimelineEntry } from "./mission";
import en from "../../messages/en-US.json";
import pt from "../../messages/pt-BR.json";

// Fixture for stripPresentation's own tests (Finding 8) — a fully-resolved, empty model; each
// test overrides only the field(s) it needs.
const EMPTY_MODEL: MissionModel = {
  cards: [],
  uncardedProjectNames: [],
  historicalTruncated: false,
  skipped: 0,
  inbox: [],
  inboxTotalCount: 0,
  inboxPrRemainder: 0,
  prTruncatedAny: false,
  codexSource: "ok",
  tmuxSource: "ok",
  archiveAfterDays: 14,
  hiddenCount: 0,
  autoHiddenCount: 0,
  reposRoot: "",
};

const NOW = Date.parse("2026-08-27T12:00:00-03:00");
const NOW_ISO = new Date(NOW).toISOString();

function mkProject(overrides: Partial<Project> & { dir: string }): Project {
  return {
    dir: overrides.dir,
    name: overrides.name ?? overrides.dir,
    stage: overrides.stage ?? "build",
    gate: overrides.gate ?? null,
    legacyGateField: overrides.legacyGateField ?? false,
    builder: overrides.builder ?? "claude-code",
    branch: overrides.branch ?? "main",
    tmux: overrides.tmux,
    flag: overrides.flag,
    updated: overrides.updated ?? "2026-08-20T10:00:00-03:00",
    now: overrides.now ?? "",
    residuals: overrides.residuals ?? [],
    statusMtime: overrides.statusMtime ?? "2026-08-20T10:00:00-03:00",
    archived: overrides.archived ?? false,
  };
}
function mkPane(overrides: Partial<HubPane> & { pane: string }): HubPane {
  return {
    pane: overrides.pane,
    tmuxIncarnation: overrides.tmuxIncarnation ?? "100:1000",
    session: overrides.session ?? "s1",
    role: overrides.role ?? "lead",
    live: overrides.live ?? true,
    working: overrides.working ?? false,
    capsule: overrides.capsule ?? null,
    pendingQuestion: overrides.pendingQuestion ?? null,
    lastEventTs: overrides.lastEventTs ?? new Date(NOW - 1_000).toISOString(),
    subagentCount: overrides.subagentCount ?? 0,
    runtime: overrides.runtime ?? "claude",
  };
}

function mkHubProject(overrides: Partial<HubProject> = {}): HubProject {
  return {
    panes: overrides.panes ?? [],
    codexSessions: overrides.codexSessions ?? [],
    codexUntracked: overrides.codexUntracked ?? false,
    pendingQuestion: overrides.pendingQuestion ?? null,
    freshnessCount: overrides.freshnessCount ?? 0,
    newestEventTs: overrides.newestEventTs ?? null,
    timeline: overrides.timeline ?? [],
    activeRun: overrides.activeRun,
    activeRuns: overrides.activeRuns ?? [],
    lastRun: overrides.lastRun ?? null,
    heartbeat: overrides.heartbeat ?? [],
    lastAction: overrides.lastAction ?? null,
  };
}

function mkHub(byProject: Record<string, HubProject>, historicalTruncated = false, codexSource: "ok" | "failed" | "truncated" = "ok", tmuxSource: "ok" | "failed" = "ok"): Envelope<HubEnvelopeData> {
  return { ok: true, data: { byProject, historicalTruncated, codexSource, tmuxSource, prefs: {}, settings: { archiveAfterDays: 14 } } };
}

function scanOf(projects: Project[]): Envelope<ProjectScan> {
  return { ok: true, data: { projects, skipped: 0, reposRoot: "" } };
}

describe("buildMission — sort and worst-pane resolution (spec §4.1, §2.7 — test 4)", () => {
  it("orders headline states needs-you > stuck > working > waiting > idle, and a project's panes resolve to the worst", () => {
    const scan = scanOf([
      mkProject({ dir: "p-idle" }),
      mkProject({ dir: "p-waiting" }),
      mkProject({ dir: "p-working" }),
      mkProject({ dir: "p-stuck" }),
      mkProject({ dir: "p-mixed" }), // disagreeing panes: one working, one needs-you
    ]);
    const waitingCapsule: Capsule = { status: "waiting", eventId: 1, minutes: 30, declaredAt: new Date(NOW - 1_000).toISOString(), mergeAsk: null, question: null };
    const stuckCapsule: Capsule = { status: "waiting", eventId: 1, minutes: 1, declaredAt: new Date(NOW - 60 * 60_000).toISOString(), mergeAsk: null, question: null };
    const blockedCapsule: Capsule = { status: "blocked", eventId: 1, minutes: null, declaredAt: new Date(NOW - 1_000).toISOString(), mergeAsk: null, question: null };
    const hub = mkHub({
      "p-idle": mkHubProject(),
      "p-waiting": mkHubProject({ panes: [mkPane({ pane: "%1", capsule: waitingCapsule })] }),
      "p-working": mkHubProject({ panes: [mkPane({ pane: "%2", working: true })] }),
      "p-stuck": mkHubProject({ panes: [mkPane({ pane: "%3", capsule: stuckCapsule })] }),
      "p-mixed": mkHubProject({
        panes: [mkPane({ pane: "%10", working: true }), mkPane({ pane: "%11", capsule: blockedCapsule })],
      }),
    });

    const { model } = buildMission(scan, hub, undefined, NOW);
    const headlineOf = (dir: string) => model.cards.find((c) => c.dir === dir)!.headline;
    expect(headlineOf("p-mixed")).toBe("needs-you"); // worst of its two disagreeing panes wins
    expect(headlineOf("p-stuck")).toBe("stuck");
    expect(headlineOf("p-working")).toBe("working");
    expect(headlineOf("p-waiting")).toBe("waiting");
    expect(headlineOf("p-idle")).toBe("idle");

    const order = model.cards.map((c) => c.dir);
    const rank = (dir: string) => order.indexOf(dir);
    expect(rank("p-mixed")).toBeLessThan(rank("p-stuck"));
    expect(rank("p-stuck")).toBeLessThan(rank("p-working"));
    expect(rank("p-working")).toBeLessThan(rank("p-waiting"));
    expect(rank("p-waiting")).toBeLessThan(rank("p-idle"));
  });

  it("keeps duration-less waiting as waiting without a stuck deadline", () => {
    const hub = mkHub({ p1: mkHubProject({ panes: [mkPane({
      pane: "%1", capsule: { status: "waiting", eventId: 1, minutes: null,
        declaredAt: new Date(NOW - 30 * 24 * 60 * 60_000).toISOString(), mergeAsk: null, question: null },
    })] }) });
    const { model } = buildMission(scanOf([mkProject({ dir: "p1" })]), hub, undefined, NOW);
    expect(model.cards[0].headline).toBe("waiting");
    expect(model.inbox).toHaveLength(0);
  });

  it("breaks ties oldest-first, then by directory name", () => {
    const scan = scanOf([mkProject({ dir: "zebra" }), mkProject({ dir: "alpha" })]);
    const sameCapsule: Capsule = { status: "waiting", eventId: 1, minutes: 30, declaredAt: new Date(NOW - 1_000).toISOString(), mergeAsk: null, question: null };
    const sameNewestEventTs = new Date(NOW - 5_000).toISOString();
    const hub = mkHub({
      zebra: mkHubProject({ panes: [mkPane({ pane: "%1", capsule: sameCapsule })], newestEventTs: sameNewestEventTs }),
      alpha: mkHubProject({ panes: [mkPane({ pane: "%2", capsule: sameCapsule })], newestEventTs: sameNewestEventTs }),
    });
    const { model } = buildMission(scan, hub, undefined, NOW);
    expect(model.cards.map((c) => c.dir)).toEqual(["alpha", "zebra"]);
  });
});

describe("buildMission — uncarded remainder (spec §4.5, test 8)", () => {
  it("reports hub projects absent from the card list, and none when the hub is a subset", () => {
    const scan = scanOf([mkProject({ dir: "jax-os" })]);
    const { model: withExtra } = buildMission(
      scan,
      mkHub({ "jax-os": mkHubProject(), "ghost-repo": mkHubProject({ panes: [mkPane({ pane: "%9" })] }) }),
      undefined,
      NOW,
    );
    expect(withExtra.uncardedProjectNames).toEqual(["ghost-repo"]);

    const { model: subset } = buildMission(scan, mkHub({ "jax-os": mkHubProject() }), undefined, NOW);
    expect(subset.uncardedProjectNames).toEqual([]);
  });

  it("reports the hub's own historicalTruncated flag on the model, not just the byProject subset (post-merge cold review, Finding 2)", () => {
    const scan = scanOf([mkProject({ dir: "jax-os" })]);
    const { model } = buildMission(scan, mkHub({ "jax-os": mkHubProject() }, true), undefined, NOW);
    expect(model.historicalTruncated).toBe(true);

    const { model: untruncated } = buildMission(scan, mkHub({ "jax-os": mkHubProject() }, false), undefined, NOW);
    expect(untruncated.historicalTruncated).toBe(false);
  });

  it("never reports an uncarded project while `projects` itself is still pending, even with a non-empty hub (post-merge cold review, Finding 3)", () => {
    // projects undefined — buildModel falls back to EMPTY_SCAN, so `cardedDirs` is empty and
    // every hub project would otherwise look "uncarded" next to the column's own skeleton.
    const hub = mkHub({ "ghost-repo": mkHubProject({ panes: [mkPane({ pane: "%9" })] }) });
    const { model } = buildMission(undefined, hub, undefined, NOW);
    expect(model.uncardedProjectNames).toEqual([]);
  });

  it("never reports an uncarded project while `projects` has FAILED, even with a non-empty hub (post-merge cold review, Finding 3)", () => {
    const hub = mkHub({ "ghost-repo": mkHubProject({ panes: [mkPane({ pane: "%9" })] }) });
    const { model } = buildMission({ ok: false, error: "boom" }, hub, undefined, NOW);
    expect(model.uncardedProjectNames).toEqual([]);
  });
});

describe("buildMission — purity (test 12)", () => {
  it("produces cards from literal envelopes with no DB, tmux, or gh involved", () => {
    // Every import above this line is `import type` except `buildMission` itself — nothing in
    // this file opens a database, calls tmux, or shells out to gh. The fixtures are literals.
    const scan = scanOf([mkProject({ dir: "solo" })]);
    const { model } = buildMission(scan, mkHub({ solo: mkHubProject() }), { ok: true, data: {} }, NOW);
    expect(model.cards).toHaveLength(1);
    expect(model.cards[0]).toMatchObject({ dir: "solo", headline: "idle" });
  });
});

describe("buildMission — loading is not idle (spec §4.1b, test 13)", () => {
  it("reports unknown, not idle, while mission/hub is still pending — and the model is still built (design change: one page-level join)", () => {
    const scan = scanOf([mkProject({ dir: "jax-os" })]);
    const { readiness, model } = buildMission(scan, undefined, undefined, NOW);
    expect(model.cards[0].headline).toBe("unknown");
    expect(model.cards[0].freshnessCount).toBeNull();
    expect(model.cards[0].lastEventAgo).toBeNull();
    expect(model.cards[0].sessions).toEqual([]);
    expect(model.inboxTotalCount).toBe(0);
    expect(readiness).toEqual({ kind: "loading", pending: ["hub", "prs"] });
  });

  it("also reports unknown when mission/hub has failed, never idle", () => {
    const scan = scanOf([mkProject({ dir: "jax-os" })]);
    const { readiness, model } = buildMission(scan, { ok: false, error: "boom" }, undefined, NOW);
    expect(model.cards[0].headline).toBe("unknown");
    expect(readiness).toEqual({ kind: "failed", failed: ["hub"], pending: ["prs"] });
  });
});

describe("buildMission — github integration off resolves pr.status (diff review e52d9e6dc555 F1/F3)", () => {
  it("github disabled: every card's pr resolves to {status:'disabled'}, never 'unknown' (F1)", () => {
    const scan = scanOf([mkProject({ dir: "p1" })]);
    const { model } = buildMission(scan, mkHub({ p1: mkHubProject() }), { ok: true, data: {} }, NOW, undefined, false);
    expect(model.cards[0].pr).toEqual({ status: "disabled" });
  });

  it("github disabled: readiness reaches 'ready', never stuck pending on 'prs' (F1)", () => {
    const scan = scanOf([mkProject({ dir: "p1" })]);
    const { readiness } = buildMission(scan, mkHub({ p1: mkHubProject() }), { ok: true, data: {} }, NOW, undefined, false);
    expect(readiness).toEqual({ kind: "ready" });
  });

  it("github enabled (default/omitted arg): unchanged regression — a missing map entry is still 'unknown' and still pending", () => {
    const scan = scanOf([mkProject({ dir: "p1" })]);
    const { model, readiness } = buildMission(scan, mkHub({ p1: mkHubProject() }), { ok: true, data: {} }, NOW);
    expect(model.cards[0].pr).toEqual({ status: "unknown" });
    expect(readiness).toEqual({ kind: "loading", pending: ["prs"] });
  });
});

describe("buildMission — freshness warning threshold (spec §4.4, test 6)", () => {
  it("warns at freshnessCount 20 but not at 19", () => {
    const scan = scanOf([mkProject({ dir: "p1" }), mkProject({ dir: "p2" })]);
    const hub = mkHub({
      p1: mkHubProject({ freshnessCount: 19 }),
      p2: mkHubProject({ freshnessCount: 20 }),
    });
    const { model } = buildMission(scan, hub, undefined, NOW);
    const warningOf = (dir: string) => model.cards.find((c) => c.dir === dir)!.freshnessWarning;
    expect(warningOf("p1")).toBe(false);
    expect(warningOf("p2")).toBe(true);
  });
});

describe("buildMission — inbox PR cap and remainder (spec §7.4, test 11)", () => {
  it("caps merge-ready PR rows at 5 and reports the remainder", () => {
    const scan = scanOf([mkProject({ dir: "p1" })]);
    const hub = mkHub({ p1: mkHubProject() });
    const mkPr = (n: number) => ({ number: n, title: `pr ${n}`, isDraft: false, url: `u${n}`, ciState: "green" as const, mergeReady: true, updatedAt: "2026-09-01T00:00:00.000Z" });
    const { model } = buildMission(
      scan,
      hub,
      { ok: true, data: { p1: { ok: true, prs: [1, 2, 3, 4, 5, 6, 7].map(mkPr), truncated: false } } },
      NOW,
    );
    expect(model.inbox.filter((r) => r.kind === "pr")).toHaveLength(5);
    expect(model.inboxPrRemainder).toBe(2);
    expect(model.prTruncatedAny).toBe(false);
  });
});

describe("buildMission — prTruncatedAny (branch review, Finding 5)", () => {
  it("is true when any card's PR fetch was truncated, false when none were", () => {
    const scan = scanOf([mkProject({ dir: "p1" }), mkProject({ dir: "p2" })]);
    const hub = mkHub({ p1: mkHubProject(), p2: mkHubProject() });
    const { model: truncated } = buildMission(
      scan, hub,
      { ok: true, data: { p1: { ok: true, prs: [], truncated: true }, p2: { ok: true, prs: [], truncated: false } } },
      NOW,
    );
    expect(truncated.prTruncatedAny).toBe(true);

    const { model: untruncated } = buildMission(
      scan, hub,
      { ok: true, data: { p1: { ok: true, prs: [], truncated: false }, p2: { ok: true, prs: [], truncated: false } } },
      NOW,
    );
    expect(untruncated.prTruncatedAny).toBe(false);
  });
});

// Simplification (recurrence ledger R3, threshold crossed a third time — cold review round 3
// Findings 3/5/8): one page-level join, one contract. `readiness` is a small discriminated union
// about CONFIDENCE (loading/failed/degraded/ready) with no "empty" variant in it at all — the empty title
// is a two-clause check the CALLER makes against ordinary data (readiness.kind === "ready" &&
// model.inboxTotalCount === 0, InboxStrip's job, Task 8), not a type-level guarantee this module
// claims to enforce (TypeScript is structural; it cannot make a variant "unconstructible").
describe("buildMission — readiness (design change: one join, Findings 3/5/8)", () => {
  it("a still-pending source is reported as kind:'loading', and the model is still built from what IS known", () => {
    // Zero gates, zero pending questions, prs still pending — the model must still carry the one
    // known project's card; readiness alone says confidence isn't there yet.
    const scan = scanOf([mkProject({ dir: "jax-os" })]);
    const hub = mkHub({ "jax-os": mkHubProject() });
    const { readiness, model } = buildMission(scan, hub, undefined, NOW);
    expect(readiness).toEqual({ kind: "loading", pending: ["prs"] });
    expect(model.cards).toHaveLength(1); // the model never disappears just because one source is pending
  });

  it("reports every still-pending source by name, not just the first", () => {
    const { readiness } = buildMission(undefined, undefined, undefined, NOW);
    expect(readiness).toEqual({ kind: "loading", pending: ["projects", "hub", "prs"] });
  });

  it("a failed BACKBONE source (projects or hub) renders as kind:'failed', never silently as loading or a confident ready (hard rule 3)", () => {
    const scan = scanOf([mkProject({ dir: "jax-os" })]);
    const { readiness } = buildMission(scan, { ok: false, error: "hub down" }, undefined, NOW);
    expect(readiness).toEqual({ kind: "failed", failed: ["hub"], pending: ["prs"] });
  });

  it("a failed source wins over a merely-pending one — failure is reported, not swallowed by loading", () => {
    const { readiness } = buildMission({ ok: false, error: "boom" }, undefined, undefined, NOW);
    expect(readiness).toEqual({ kind: "failed", failed: ["projects"], pending: ["hub", "prs"] });
  });

  it("a failed BACKBONE source alongside a still-pending `projects` reports both (post-merge cold review, Finding 4) — ProjectsColumn needs this to keep its skeleton instead of a false empty state", () => {
    // projects undefined (still in flight), hub failed, prs undefined — the mixed-state fixture
    // Finding 4 asks for. Before this fix, `pending` did not exist on the `failed` variant at
    // all, so ProjectsColumn (Task 9) had no way to distinguish "projects still loading" from
    // "projects resolved to zero cards" once ANY backbone source had failed.
    const { readiness, model } = buildMission(undefined, { ok: false, error: "hub down" }, undefined, NOW);
    expect(readiness).toEqual({ kind: "failed", failed: ["hub"], pending: ["projects", "prs"] });
    expect(model.cards).toEqual([]); // buildModel falls back to EMPTY_SCAN — cards are unknown, not "confirmed empty"
  });

  it("a failed prs source NEVER elevates readiness to kind:'failed', and known gate/question rows survive it — it degrades instead of going ready (cold review round 3, Finding 5; round 4, Finding 2)", () => {
    const scan = scanOf([mkProject({ dir: "jax-os", gate: "blocked" })]);
    const hub = mkHub({ "jax-os": mkHubProject() });
    const { readiness, model } = buildMission(scan, hub, { ok: false, error: "gh: rate limited" }, NOW);
    expect(readiness).toEqual({ kind: "degraded", degraded: ["prs"] }); // projects + hub resolved clean, prs failed — not "failed", and not a confident "ready" either
    expect(model.inbox.some((r) => r.kind === "gate")).toBe(true); // the known gate row survives the prs failure
    expect(model.cards[0].pr).toEqual({ status: "error", error: "gh: rate limited" }); // scoped to the card's own PR line
  });

  it("a failed prs source with an EMPTY inbox never blesses a confident ready — titleEmpty (InboxStrip, Task 8) stays unreachable (cold review round 4, Finding 2 — R3's fifth recurrence: the blocked-gate test above can't catch this, since its inboxTotalCount is non-zero regardless of readiness.kind)", () => {
    const scan = scanOf([mkProject({ dir: "jax-os" })]);
    const hub = mkHub({ "jax-os": mkHubProject() });
    const { readiness, model } = buildMission(scan, hub, { ok: false, error: "gh: rate limited" }, NOW);
    expect(readiness).toEqual({ kind: "degraded", degraded: ["prs"] });
    expect(model.inboxTotalCount).toBe(0); // empty inbox AND readiness.kind !== "ready" — titleEmpty can never render here
  });

  it("a per-project PR failure inside an otherwise-ok prs envelope also degrades, never a confident ready (post-merge cold review, Finding 1)", () => {
    // prs envelope itself is {ok:true} — only this ONE project's own PrResult is {ok:false}. The
    // old code only checked `!prs.ok` (the whole-envelope case), so this degraded to a confident
    // titleEmpty even though the card's own pr line reads an error.
    const scan = scanOf([mkProject({ dir: "jax-os" })]);
    const hub = mkHub({ "jax-os": mkHubProject() });
    const { readiness, model } = buildMission(
      scan, hub, { ok: true, data: { "jax-os": { ok: false, error: "gh: rate limited" } } }, NOW,
    );
    expect(readiness).toEqual({ kind: "degraded", degraded: ["prs"] });
    expect(model.inboxTotalCount).toBe(0); // no gate/question/attention/PR rows — titleEmpty must still be unreachable
    expect(model.cards[0].pr).toEqual({ status: "error", error: "gh: rate limited" });
  });

  it("all three resolved clean with nothing in the inbox is kind:'ready' with inboxTotalCount 0 — there is no separate 'empty' variant to keep in sync (Finding 8)", () => {
    const scan = scanOf([mkProject({ dir: "jax-os" })]);
    const hub = mkHub({ "jax-os": mkHubProject() });
    const { readiness, model } = buildMission(
      scan, hub, { ok: true, data: { "jax-os": { ok: true, prs: [], truncated: false } } }, NOW,
    );
    expect(readiness).toEqual({ kind: "ready" });
    expect(model.inboxTotalCount).toBe(0);
  });

  it("all three resolved with a gate open is kind:'ready' and the model carries it", () => {
    const scan = scanOf([mkProject({ dir: "jax-os", gate: "blocked" })]);
    const hub = mkHub({ "jax-os": mkHubProject() });
    const { readiness, model } = buildMission(
      scan, hub, { ok: true, data: { "jax-os": { ok: true, prs: [], truncated: false } } }, NOW,
    );
    expect(readiness).toEqual({ kind: "ready" });
    expect(model.inboxTotalCount).toBe(1);
  });
});

describe("buildMission — a resolved blocked/needs_input capsule produces an inbox row (cold review round 3, Finding 3)", () => {
  it("a pane whose last capsule is blocked, with no pending question, is an 'attention' row, and readiness cannot be a confident empty", () => {
    const scan = scanOf([mkProject({ dir: "jax-os" })]);
    const blockedCapsule: Capsule = { status: "blocked", eventId: 1, minutes: null, declaredAt: new Date(NOW - 1_000).toISOString(), mergeAsk: null, question: null };
    const hub = mkHub({ "jax-os": mkHubProject({ panes: [mkPane({ pane: "%1", capsule: blockedCapsule })] }) });
    const { readiness, model } = buildMission(
      scan, hub, { ok: true, data: { "jax-os": { ok: true, prs: [], truncated: false } } }, NOW,
    );
    expect(model.cards[0].headline).toBe("needs-you");
    expect(model.inbox).toContainEqual({ kind: "attention", dir: "jax-os", name: "jax-os", status: "blocked", transport: "tmux", pane: "%1", tmuxIncarnation: "100:1000" });
    expect(model.inboxTotalCount).toBe(1);
    expect(readiness).toEqual({ kind: "ready" }); // a needs-you card is never swallowed by the empty predicate
  });

  it("needs_input behaves the same way", () => {
    const scan = scanOf([mkProject({ dir: "jax-os" })]);
    const needsInputCapsule: Capsule = { status: "needs_input", eventId: 1, minutes: null, declaredAt: new Date(NOW - 1_000).toISOString(), mergeAsk: null, question: null };
    const hub = mkHub({ "jax-os": mkHubProject({ panes: [mkPane({ pane: "%1", capsule: needsInputCapsule })] }) });
    const { model } = buildMission(
      scan, hub, { ok: true, data: { "jax-os": { ok: true, prs: [], truncated: false } } }, NOW,
    );
    expect(model.inbox).toContainEqual({ kind: "attention", dir: "jax-os", name: "jax-os", status: "needs_input", transport: "tmux", pane: "%1", tmuxIncarnation: "100:1000" });
  });

  it("a pane with a pending question does NOT also get a duplicate attention row — the question row already covers it", () => {
    const scan = scanOf([mkProject({ dir: "jax-os" })]);
    const blockedCapsule: Capsule = { status: "blocked", eventId: 1, minutes: null, declaredAt: new Date(NOW - 1_000).toISOString(), mergeAsk: null, question: null };
    const pendingQuestion: PendingQuestion = {
      eventId: 9, pane: "%1", tmuxIncarnation: "100:1000",
      payload: { questions: [{ question: "q?", options: [{ label: "a" }] }] },
    };
    const hub = mkHub({ "jax-os": mkHubProject({ panes: [mkPane({ pane: "%1", capsule: blockedCapsule, pendingQuestion })] }) });
    const { model } = buildMission(
      scan, hub, { ok: true, data: { "jax-os": { ok: true, prs: [], truncated: false } } }, NOW,
    );
    expect(model.inbox.filter((r) => r.kind === "attention" || r.kind === "question")).toHaveLength(1);
    expect(model.inbox[0].kind).toBe("question");
  });
});

describe("buildMission — a project missing from an otherwise-resolved prs map (cold review round 3, Finding 4)", () => {
  it("a new project between polls reports pr:{status:'unknown'}, not 'none' — and readiness cannot claim confidence", () => {
    // mission/projects polls every 10s, mission/prs every 60s, and the two routes call
    // getProjects() independently — a project that appears between polls is absent from the prs
    // envelope's map even though the envelope itself resolved {ok:true}. This can no longer mean
    // "this project has no GitHub remote" (cold review round 4, Finding 4): the collector now
    // resolves that case to an explicit map entry instead of omitting it (test below), so a
    // missing entry is always read as unknown, and it keeps prs pending so a merge-ready PR
    // nobody has fetched yet can never be silently reported as "nothing waits".
    const scan = scanOf([mkProject({ dir: "brand-new" })]);
    const hub = mkHub({ "brand-new": mkHubProject() });
    const { readiness, model } = buildMission(scan, hub, { ok: true, data: {} }, NOW);
    expect(model.cards[0].pr).toEqual({ status: "unknown" });
    expect(readiness).toEqual({ kind: "loading", pending: ["prs"] });
  });

  it("once the map resolves for that project, pr status reflects it and readiness can go ready", () => {
    const scan = scanOf([mkProject({ dir: "jax-os" })]);
    const hub = mkHub({ "jax-os": mkHubProject() });
    const { readiness, model } = buildMission(
      scan, hub, { ok: true, data: { "jax-os": { ok: true, prs: [], truncated: false } } }, NOW,
    );
    expect(model.cards[0].pr).toEqual({ status: "ok", count: 0, ci: "none", truncated: false, newest: null });
    expect(readiness).toEqual({ kind: "ready" });
  });

  it("a no-GitHub-remote project's explicit collector entry reaches readiness:'ready' with no PR line, never stuck loading forever (cold review round 4, Finding 4)", () => {
    const scan = scanOf([mkProject({ dir: "no-remote-project" })]);
    const hub = mkHub({ "no-remote-project": mkHubProject() });
    const { readiness, model } = buildMission(
      scan, hub, { ok: true, data: { "no-remote-project": { ok: true, prs: [], truncated: false } } }, NOW,
    );
    expect(model.cards[0].pr).toEqual({ status: "ok", count: 0, ci: "none", truncated: false, newest: null }); // no PR line (spec §5.1)
    expect(readiness).toEqual({ kind: "ready" }); // resolved, not "loading" forever
  });
});

describe("buildMission — hubReady on the card (branch review, Finding 1)", () => {
  it("hubReady is false while hub is pending or failed, true once hub resolves ok — even with the same 'unknown' headline", () => {
    const scan = scanOf([mkProject({ dir: "jax-os" })]);

    const { model: pending } = buildMission(scan, undefined, undefined, NOW);
    expect(pending.cards[0].hubReady).toBe(false);

    const { model: failed } = buildMission(scan, { ok: false, error: "boom" }, undefined, NOW);
    expect(failed.cards[0].hubReady).toBe(false);

    const { model: ready } = buildMission(scan, mkHub({ "jax-os": mkHubProject() }), undefined, NOW);
    expect(ready.cards[0].hubReady).toBe(true);
  });
});

describe("buildMission — session grouping keeps pane incarnations apart (branch review, Finding 2)", () => {
  it("two rows for the same bare pane id, different tmuxIncarnation, both session:null, land in two separate session buckets", () => {
    const scan = scanOf([mkProject({ dir: "jax-os" })]);
    const hub = mkHub({
      "jax-os": mkHubProject({
        // mkPane's own `session` fallback (`overrides.session ?? "s1"`) can't express an explicit
        // null override — building the pane, then overwriting `session`, sidesteps that fixture
        // default rather than construct HubPane literals by hand.
        panes: [
          { ...mkPane({ pane: "%1", tmuxIncarnation: "100:1000" }), session: null },
          { ...mkPane({ pane: "%1", tmuxIncarnation: "200:2000" }), session: null },
        ],
      }),
    });
    const { model } = buildMission(scan, hub, undefined, NOW);
    const tmuxSessions = model.cards[0].sessions.filter((s) => s.transport === "tmux");
    expect(tmuxSessions).toHaveLength(2); // NOT collapsed into one "%1" bucket
    for (const s of tmuxSessions) {
      if (s.transport !== "tmux") continue;
      expect(s.session).toBe("%1"); // display label stays the bare pane id either way
      expect(s.panes).toHaveLength(1);
    }
    const incarnations = tmuxSessions.flatMap((s) => (s.transport === "tmux" ? s.panes.map((p) => p.tmuxIncarnation) : [])).sort();
    expect(incarnations).toEqual(["100:1000", "200:2000"]);
  });
});

describe("buildMission — sortEpochMs compares real instants, not raw strings (branch review, Finding 6)", () => {
  it("a UTC 'Z' timestamp and an offset timestamp sort by actual instant, not lexicographically", () => {
    // 2026-08-27T12:00:00.000Z is 12:00 UTC; 2026-08-27T10:00:00-03:00 is 13:00 UTC — the first is
    // OLDER even though "10" sorts before "12" as a string. Same headline/rank (both idle) forces
    // the comparison down to sortEpochMs.
    const scan = scanOf([mkProject({ dir: "later" }), mkProject({ dir: "earlier" })]);
    const hub = mkHub({
      later: mkHubProject({ newestEventTs: "2026-08-27T10:00:00-03:00" }), // 13:00 UTC
      earlier: mkHubProject({ newestEventTs: "2026-08-27T12:00:00.000Z" }), // 12:00 UTC
    });
    const { model } = buildMission(scan, hub, undefined, NOW);
    expect(model.cards.map((c) => c.dir)).toEqual(["earlier", "later"]); // oldest (earliest UTC instant) first
  });

  it("an unparseable timestamp sorts last (+Infinity), never mistaken for the oldest", () => {
    const scan = scanOf([mkProject({ dir: "garbage" }), mkProject({ dir: "valid" })]);
    const hub = mkHub({
      garbage: mkHubProject({ newestEventTs: "not-a-date" }),
      valid: mkHubProject({ newestEventTs: "2026-08-27T12:00:00.000Z" }),
    });
    const { model } = buildMission(scan, hub, undefined, NOW);
    expect(model.cards.map((c) => c.dir)).toEqual(["valid", "garbage"]);
  });
});

describe("stripPresentation — pure title/warning decision (branch review, Finding 8)", () => {
  it("ready with an empty inbox is the empty title with no warning", () => {
    const { title, warning } = stripPresentation({ kind: "ready" }, { ...EMPTY_MODEL, inboxTotalCount: 0 });
    expect(title).toEqual({ kind: "empty" });
    expect(warning).toEqual({ kind: "none" });
  });

  it("ready with a non-empty inbox carries the count, no warning", () => {
    const { title, warning } = stripPresentation({ kind: "ready" }, { ...EMPTY_MODEL, inboxTotalCount: 3 });
    expect(title).toEqual({ kind: "count", count: 3 });
    expect(warning).toEqual({ kind: "none" });
  });

  it("failed always shows the source warning, with the neutral unknown title, regardless of card count", () => {
    const readiness: MissionReadiness = { kind: "failed", failed: ["hub"], pending: [] };
    const { title, warning } = stripPresentation(readiness, EMPTY_MODEL);
    expect(title).toEqual({ kind: "unknown" });
    expect(warning).toEqual({ kind: "source", sources: ["hub"] });
  });

  it("degraded with zero cards shows the source warning (Finding 3 — nothing else can carry it)", () => {
    const readiness: MissionReadiness = { kind: "degraded", degraded: ["prs"] };
    const { warning } = stripPresentation(readiness, { ...EMPTY_MODEL, cards: [] });
    expect(warning).toEqual({ kind: "source", sources: ["prs"] });
  });

  it("degraded with at least one card stays inline-only — no page-level warning here", () => {
    const readiness: MissionReadiness = { kind: "degraded", degraded: ["prs"] };
    const { warning } = stripPresentation(readiness, { ...EMPTY_MODEL, cards: [{ dir: "x" } as never] });
    expect(warning).toEqual({ kind: "none" });
  });

  it("loading with an empty inbox shows the skeleton", () => {
    const readiness: MissionReadiness = { kind: "loading", pending: ["prs"] };
    const { warning } = stripPresentation(readiness, { ...EMPTY_MODEL, inbox: [] });
    expect(warning).toEqual({ kind: "skeleton" });
  });

  it("loading with a non-empty inbox shows no skeleton — known rows already render", () => {
    const readiness: MissionReadiness = { kind: "loading", pending: ["prs"] };
    const { warning } = stripPresentation(readiness, { ...EMPTY_MODEL, inbox: [{ kind: "gate" } as never] });
    expect(warning).toEqual({ kind: "none" });
  });
});

describe("buildMission — activeRun presentation (MOA-464)", () => {
  const ACTIVE: HubActiveRun = {
    runId: "diff1", startedAt: "2026-09-09T10:02:00.000Z", kind: "diff",
    runtime: "claude", target: "fix/example", count: 1, lastStep: null,
  };

  function cardOf(activeRun: HubActiveRun | null | undefined, hub?: Envelope<HubEnvelopeData> | undefined) {
    const scan = scanOf([mkProject({
      dir: "p1", stage: "build", branch: "feat/old-build", builder: "codex",
      now: "Completed build 88015f165499", updated: "2026-09-09T09:00:00.000Z",
    })]);
    const envelope = hub !== undefined ? hub : mkHub({
      p1: mkHubProject({ activeRun, newestEventTs: "2026-09-09T11:00:00.000Z" }),
    });
    return buildMission(scan, envelope, { ok: true, data: { p1: { ok: true, prs: [], truncated: false } } }, NOW).model.cards[0];
  }

  it.each([
    ["build", "build", "building", "branch"],
    ["spec", "review", "reviewSpec", "document"],
    ["plan", "review", "reviewPlan", "document"],
    ["diff", "review", "reviewDiff", "branch"],
  ] as const)("maps kind %s to stage %s / %s / %s", (kind, stage, descriptionKey, targetKind) => {
    const card = cardOf({ ...ACTIVE, kind });
    expect(card.activeRun).toMatchObject({ kind, stage, descriptionKey, targetKind, runtime: "claude", target: "fix/example" });
    expect(card.stage).toBe("build");
    expect(card.branch).toBe("feat/old-build");
    expect(card.builder).toBe("codex");
    expect(card.now).toBe("Completed build 88015f165499");
  });

  it("unknown metadata renders generic and does not borrow status-file fields", () => {
    const card = cardOf({ ...ACTIVE, kind: null, runtime: null, target: null });
    expect(card.activeRun).toMatchObject({
      stage: null, descriptionKey: "generic", targetKind: null, runtime: null, target: null,
    });
    expect(card.branch).toBe("feat/old-build");
    expect(card.builder).toBe("codex");
  });

  it("newestEventTs does not change startedAt", () => {
    const card = cardOf(ACTIVE);
    expect(card.lastEventAgo).toBe("2026-09-09T11:00:00.000Z");
    expect(card.activeRun?.startedAt).toBe("2026-09-09T10:02:00.000Z");
  });

  it("absent, null, undefined hub, and failed hub produce null activeRun", () => {
    expect(cardOf(undefined).activeRun).toBeNull();
    expect(cardOf(null).activeRun).toBeNull();
    const scan = scanOf([mkProject({ dir: "p1" })]);
    expect(buildMission(scan, undefined, { ok: true, data: { p1: { ok: true, prs: [], truncated: false } } }, NOW).model.cards[0].activeRun).toBeNull();
    expect(cardOf(ACTIVE, { ok: false, error: "boom" }).activeRun).toBeNull();
  });

  it("activeRuns maps every in-flight run through the same kind->stage projection, role passed through", () => {
    const card = cardOf(ACTIVE, mkHub({
      p1: mkHubProject({
        activeRuns: [
          { runId: "diff1", role: "reviewer", kind: "diff", runtime: "claude", target: "fix/example", startedAt: "2026-09-09T10:02:00.000Z" },
          { runId: "build1", role: "builder", kind: "build", runtime: "codex", target: "feat/old", startedAt: "2026-09-09T10:00:00.000Z" },
        ],
      }),
    }));
    expect(card.activeRuns).toEqual([
      { runId: "diff1", role: "reviewer", kind: "diff", runtime: "claude", target: "fix/example", startedAt: "2026-09-09T10:02:00.000Z", stage: "review", descriptionKey: "reviewDiff", targetKind: "branch" },
      { runId: "build1", role: "builder", kind: "build", runtime: "codex", target: "feat/old", startedAt: "2026-09-09T10:00:00.000Z", stage: "build", descriptionKey: "building", targetKind: "branch" },
    ]);
  });

  it("hubReady false or missing hp yields an empty activeRuns array", () => {
    expect(cardOf(undefined).activeRuns).toEqual([]);
    const scan = scanOf([mkProject({ dir: "p1" })]);
    expect(buildMission(scan, undefined, { ok: true, data: { p1: { ok: true, prs: [], truncated: false } } }, NOW).model.cards[0].activeRuns).toEqual([]);
  });

  it("locale files expose every activity description key", () => {
    const keys = ["building", "reviewSpec", "reviewPlan", "reviewDiff", "generic", "started", "lastRecord", "otherRuns"] as const;
    for (const key of keys) {
      expect(pt.mission.activity[key]).toEqual({
        building: "Implementando",
        reviewSpec: "Revisando spec",
        reviewPlan: "Revisando plano",
        reviewDiff: "Revisando diff",
        generic: "Execução em andamento",
        started: "Iniciado",
        lastRecord: "Último registro",
        otherRuns: "{count, plural, one {+# outra execução} other {+# outras execuções}}",
      }[key]);
      expect(en.mission.activity[key]).toEqual({
        building: "Building",
        reviewSpec: "Reviewing spec",
        reviewPlan: "Reviewing plan",
        reviewDiff: "Reviewing diff",
        generic: "Run in progress",
        started: "Started",
        lastRecord: "Last recorded status",
        otherRuns: "{count, plural, one {+# other run} other {+# other runs}}",
      }[key]);
    }
  });
});

describe("buildMission — lastRun classification (spec §12.4 rendering table, MOA-474)", () => {
  const LAST_RUN_BASE: HubLastRun = {
    runId: "r1", role: "builder", kind: "build", runtime: "codex", target: "feat/x",
    finishedAt: "2026-09-09T10:02:00.000Z", outcome: null, contractStatus: "ok",
    stage: null, diagnostic: null, headSha: null, findings: null,
    runtimeModel: null, profileName: null, reportRel: null, targetRel: null,
    verifyCommand: null, buildCommand: null,
  };

  function cardWithLastRun(lastRun: HubLastRun | null) {
    const scan = scanOf([mkProject({ dir: "p1" })]);
    const hub = mkHub({ p1: mkHubProject({ lastRun }) });
    return buildMission(scan, hub, undefined, NOW).model.cards[0];
  }

  it.each([
    ["success", "buildOk", "success"],
    ["failure", "buildFailed", "danger"],
    ["blocked", "buildBlocked", "danger"],
  ] as const)("builder ok outcome %s maps to %s / %s", (outcome, labelKey, tone) => {
    const card = cardWithLastRun({ ...LAST_RUN_BASE, outcome });
    expect(card.lastRun).toMatchObject({ labelKey, tone });
  });

  it.each([
    ["approve", "reviewOk", "success"],
    ["approve-with-changes", "reviewChanges", "success"],
    ["reject", "reviewRejected", "danger"],
  ] as const)("reviewer ok outcome %s maps to %s / %s", (outcome, labelKey, tone) => {
    const card = cardWithLastRun({ ...LAST_RUN_BASE, role: "reviewer", outcome });
    expect(card.lastRun).toMatchObject({ labelKey, tone });
  });

  it("reviewer with no verdict (missing/invalid report) maps to reviewNoVerdict/danger", () => {
    const card = cardWithLastRun({ ...LAST_RUN_BASE, role: "reviewer", contractStatus: "invalid", stage: "report", diagnostic: "report invalid" });
    expect(card.lastRun).toMatchObject({ labelKey: "reviewNoVerdict", tone: "danger" });
  });

  it("stage worker always maps to *Interrupted/warning, checked before outcome (builder and reviewer)", () => {
    const builder = cardWithLastRun({ ...LAST_RUN_BASE, outcome: "failure", contractStatus: "interrupted", stage: "worker", diagnostic: "worker interrupted by SIGHUP" });
    expect(builder.lastRun).toMatchObject({ labelKey: "buildInterrupted", tone: "warning" });
    const reviewer = cardWithLastRun({ ...LAST_RUN_BASE, role: "reviewer", contractStatus: "interrupted", stage: "worker", diagnostic: "worker interrupted — no cause emitted" });
    expect(reviewer.lastRun).toMatchObject({ labelKey: "reviewInterrupted", tone: "warning" });
  });

  it("interrupted row 1a (stage runtime) reuses buildFailed/reviewNoVerdict at danger tone", () => {
    const builder = cardWithLastRun({ ...LAST_RUN_BASE, outcome: "failure", contractStatus: "interrupted", stage: "runtime", diagnostic: "APIError 403" });
    expect(builder.lastRun).toMatchObject({ labelKey: "buildFailed", tone: "danger" });
    const reviewer = cardWithLastRun({ ...LAST_RUN_BASE, role: "reviewer", contractStatus: "interrupted", stage: "runtime", diagnostic: "APIError 403" });
    expect(reviewer.lastRun).toMatchObject({ labelKey: "reviewNoVerdict", tone: "danger" });
  });

  it("cancelled maps to *Cancelled/warning for both roles", () => {
    const builder = cardWithLastRun({ ...LAST_RUN_BASE, contractStatus: "cancelled" });
    expect(builder.lastRun).toMatchObject({ labelKey: "buildCancelled", tone: "warning" });
    const reviewer = cardWithLastRun({ ...LAST_RUN_BASE, role: "reviewer", contractStatus: "cancelled" });
    expect(reviewer.lastRun).toMatchObject({ labelKey: "reviewCancelled", tone: "warning" });
  });

  it("a legacy row with no stage/diagnostic/outcome at all maps to the *NoResult/NoVerdict warning label", () => {
    const builder = cardWithLastRun({ ...LAST_RUN_BASE, contractStatus: "missing" });
    expect(builder.lastRun).toMatchObject({ labelKey: "buildNoResult", tone: "warning" });
    const reviewer = cardWithLastRun({ ...LAST_RUN_BASE, role: "reviewer", contractStatus: "missing" });
    expect(reviewer.lastRun).toMatchObject({ labelKey: "reviewNoVerdict", tone: "warning" });
  });

  it("null lastRun stays null on the card, and is null when hub is not ready", () => {
    expect(cardWithLastRun(null).lastRun).toBeNull();
    const scan = scanOf([mkProject({ dir: "p1" })]);
    expect(buildMission(scan, undefined, undefined, NOW).model.cards[0].lastRun).toBeNull();
  });

  it("run-finished timeline rows carry outcome/contractStatus/stage/diagnostic, other rows carry null", () => {
    const scan = scanOf([mkProject({ dir: "p1" })]);
    const hub = mkHub({
      p1: mkHubProject({
        timeline: [
          { id: 2, ts: "2026-09-09T10:02:00.000Z", type: "run-finished", pane: null, role: "builder", emitter: "wrapper", payload: { result: "failure", contract_status: "ok", stage: "build", diagnostic: "pnpm build exit 1" } },
          { id: 1, ts: "2026-09-09T10:00:00.000Z", type: "run-started", pane: null, role: "builder", emitter: "wrapper", payload: { kind: "build" } },
        ],
      }),
    });
    const card = buildMission(scan, hub, undefined, NOW).model.cards[0];
    expect(card.timeline[0]).toMatchObject({ type: "run-finished", outcome: "failure", contractStatus: "ok", stage: "build", diagnostic: "pnpm build exit 1" });
    expect(card.timeline[1]).toMatchObject({ type: "run-started", outcome: null, contractStatus: null, stage: null, diagnostic: null });
  });

  it("locale files expose every lastRun key (spec §12.4)", () => {
    const stageKeys = ["runtime", "verify", "build", "worker", "report"] as const;
    for (const key of stageKeys) {
      expect(typeof pt.mission.lastRun.stage[key]).toBe("string");
      expect(typeof en.mission.lastRun.stage[key]).toBe("string");
    }
    const labelKeys: LastRunLabelKey[] = [
      "buildOk", "buildFailed", "buildBlocked", "buildInterrupted", "buildCancelled", "buildNoResult",
      "reviewOk", "reviewChanges", "reviewRejected", "reviewInterrupted", "reviewCancelled", "reviewNoVerdict",
    ];
    for (const key of labelKeys) {
      expect(typeof pt.mission.lastRun[key]).toBe("string");
      expect(typeof en.mission.lastRun[key]).toBe("string");
    }
    expect(typeof pt.mission.lastRun.noCause).toBe("string");
    expect(typeof en.mission.lastRun.noCause).toBe("string");
    expect(pt.mission.lastRun.reportProblem).toContain("{status}");
    expect(en.mission.lastRun.reportProblem).toContain("{status}");
  });

  it("findings passes through from HubLastRun to MissionLastRun unchanged", () => {
    const card = cardWithLastRun({ ...LAST_RUN_BASE, role: "reviewer", outcome: "approve-with-changes", findings: { high: 0, medium: 3, low: 1 } });
    expect(card.lastRun?.findings).toEqual({ high: 0, medium: 3, low: 1 });
  });

  it("findings stays null through the pass-through when the hub row has none", () => {
    const card = cardWithLastRun({ ...LAST_RUN_BASE, findings: null });
    expect(card.lastRun?.findings).toBeNull();
  });
});

describe("buildMission — native Codex sessions (MOA-469 §4)", () => {
  const mkCodex = (overrides: Partial<import("../server/db/workflows").HubCodexSession> = {}) => ({
    threadId: overrides.threadId ?? "0191f0aa-1234-7000-8000-000000000001",
    status: overrides.status ?? "idle",
    activeFlags: overrides.activeFlags ?? [],
    sourceOk: overrides.sourceOk ?? true,
    working: overrides.working ?? false,
    attention: overrides.attention ?? false,
    pendingAttention: overrides.pendingAttention ?? false,
    runInFlight: overrides.runInFlight ?? false,
    capsule: overrides.capsule ?? null,
    lastEventTs: overrides.lastEventTs ?? null,
    subagentCount: overrides.subagentCount ?? 0,
  });

  it("a native Codex session renders as a discriminated row with no fabricated pane", () => {
    const scan = scanOf([mkProject({ dir: "p1" })]);
    const hub = mkHub({ p1: mkHubProject({ codexSessions: [mkCodex({ status: "active", working: true })] }) });
    const { model } = buildMission(scan, hub, { ok: true, data: { p1: { ok: true, prs: [], truncated: false } } }, NOW);
    expect(model.cards[0].headline).toBe("working");
    const codex = model.cards[0].sessions.find((s) => s.transport === "codex")!;
    expect(codex).toMatchObject({ transport: "codex", state: "working" });
    expect(JSON.stringify(codex)).not.toContain('"pane"');
  });

  it("attention flags make a native session needs-you with a codex inbox row", () => {
    const scan = scanOf([mkProject({ dir: "p1" })]);
    const hub = mkHub({ p1: mkHubProject({ codexSessions: [mkCodex({ status: "active", activeFlags: ["waitingOnUserInput"], attention: true })] }) });
    const { model } = buildMission(scan, hub, { ok: true, data: { p1: { ok: true, prs: [], truncated: false } } }, NOW);
    expect(model.cards[0].headline).toBe("needs-you");
    expect(model.inbox).toContainEqual({ kind: "attention", dir: "p1", name: "p1", status: "needs_input", transport: "codex", threadId: mkCodex().threadId });
  });

  it("a Codex source failure warning never hides a genuinely working Claude pane", () => {
    const scan = scanOf([mkProject({ dir: "p1" })]);
    const hub = mkHub({ p1: mkHubProject({ panes: [mkPane({ pane: "%1", working: true })] }) }, false, "failed");
    const { model } = buildMission(scan, hub, { ok: true, data: { p1: { ok: true, prs: [], truncated: false } } }, NOW);
    expect(model.cards[0].headline).toBe("working");
    expect(model.cards[0].codexSource).toBe("failed");
    expect(model.codexSource).toBe("failed");
    expect(model.cards[0].sessions).toHaveLength(1); // only the Claude pane row
  });

  it("surfaces a failed tmux source independently of a healthy Codex source (C4)", () => {
    const scan = scanOf([mkProject({ dir: "p1" })]);
    const hub = mkHub({ p1: mkHubProject({ codexSessions: [mkCodex({ status: "active", working: true })] }) }, false, "ok", "failed");
    const { model } = buildMission(scan, hub, undefined, NOW);
    expect(model.tmuxSource).toBe("failed");
    expect(model.codexSource).toBe("ok");
    expect(model.cards[0].headline).toBe("working");
  });

  it("a still-loading hub reports no source failure (no warning flash on page load)", () => {
    const scan = scanOf([mkProject({ dir: "p1" })]);
    const { model } = buildMission(scan, undefined, undefined, NOW);
    expect(model.tmuxSource).toBe("ok");
    expect(model.codexSource).toBe("ok");
    expect(model.cards[0].codexSource).toBe("ok");
    const failed = buildMission(scan, { ok: false, error: "down" }, undefined, NOW).model;
    expect(failed.tmuxSource).toBe("failed");
    expect(failed.codexSource).toBe("failed");
  });

  it("a native unknown state never downgrades a working Claude pane", () => {
    const scan = scanOf([mkProject({ dir: "p1" })]);
    const hub = mkHub({ p1: mkHubProject({
      panes: [mkPane({ pane: "%1", working: true })],
      codexSessions: [mkCodex({ sourceOk: false, status: "unknown" })],
    }) });
    const { model } = buildMission(scan, hub, undefined, NOW);
    expect(model.cards[0].headline).toBe("working");
    const codex = model.cards[0].sessions.find((s) => s.transport === "codex")!;
    expect(codex.state).toBe("unknown");
  });

  it("a waiting native capsule past its deadline reads stuck", () => {
    const scan = scanOf([mkProject({ dir: "p1" })]);
    const hub = mkHub({ p1: mkHubProject({ codexSessions: [mkCodex({ capsule: { status: "waiting", eventId: 1, minutes: 1, declaredAt: new Date(NOW - 60 * 60_000).toISOString(), mergeAsk: null, question: null } })] }) });
    const { model } = buildMission(scan, hub, undefined, NOW);
    expect(model.cards[0].headline).toBe("stuck");
  });

  it("a native Codex session with a pending capsule carries its event id, same as a pane's capsuleEventId (diff review F2)", () => {
    const scan = scanOf([mkProject({ dir: "p1" })]);
    const hub = mkHub({ p1: mkHubProject({ codexSessions: [mkCodex({ status: "active", pendingAttention: true, capsule: { status: "needs_input", eventId: 42, minutes: null, declaredAt: new Date(NOW).toISOString(), mergeAsk: null, question: null } })] }) });
    const { model } = buildMission(scan, hub, undefined, NOW);
    const codex = model.cards[0].sessions.find((s) => s.transport === "codex")!;
    expect(codex).toMatchObject({ state: "needs-you", capsuleEventId: 42 });
  });

  it("surfaces codexUntracked as card metadata", () => {
    const scan = scanOf([mkProject({ dir: "p1" })]);
    const hub = mkHub({ p1: mkHubProject({ codexUntracked: true }) });
    const { model } = buildMission(scan, hub, undefined, NOW);
    expect(model.cards[0].codexUntracked).toBe(true);
  });
});

describe("buildMission — MOA-469 live corrections: dead panes and unavailable sessions carry no current attention", () => {
  const mkCodex = (overrides: Partial<import("../server/db/workflows").HubCodexSession> = {}) => ({
    threadId: overrides.threadId ?? "0191f0aa-7777-7000-8000-000000000077",
    status: overrides.status ?? "idle",
    activeFlags: overrides.activeFlags ?? [],
    sourceOk: overrides.sourceOk ?? true,
    working: overrides.working ?? false,
    attention: overrides.attention ?? false,
    pendingAttention: overrides.pendingAttention ?? false,
    runInFlight: overrides.runInFlight ?? false,
    capsule: overrides.capsule ?? null,
    lastEventTs: overrides.lastEventTs ?? null,
    subagentCount: overrides.subagentCount ?? 0,
  });
  const needsInputCapsule: Capsule = { status: "needs_input", eventId: 1, minutes: null, declaredAt: new Date(NOW - 1_000).toISOString(), mergeAsk: null, question: null };
  const RESOLVED: Envelope<Record<string, PrResult>> = { ok: true, data: { p1: { ok: true, prs: [], truncated: false } } };

  it("a dead Claude pane carrying needs_input next to an active native Codex session: headline working, no actionable inbox row from the dead pane", () => {
    const scan = scanOf([mkProject({ dir: "p1" })]);
    const hub = mkHub({ p1: mkHubProject({
      panes: [mkPane({ pane: "%1", live: false, capsule: needsInputCapsule })],
      codexSessions: [mkCodex({ status: "active", working: true })],
    }) });
    const { model } = buildMission(scan, hub, RESOLVED, NOW);
    expect(model.cards[0].headline).toBe("working"); // dead pane contributes nothing; native working wins
    expect(model.cards[0].sessions).toHaveLength(2); // session display rows kept for both
    expect(model.inbox.filter((r) => r.kind === "attention" || r.kind === "question")).toHaveLength(0);
    expect(model.inboxTotalCount).toBe(0);
  });

  it("a dead pane's pending-question metadata yields no question row, no card pendingQuestions, and an idle headline", () => {
    const scan = scanOf([mkProject({ dir: "p1" })]);
    const pendingQuestion: PendingQuestion = {
      eventId: 9, pane: "%1", tmuxIncarnation: "100:1000",
      payload: { questions: [{ question: "q?", options: [{ label: "a" }] }] },
    };
    const hub = mkHub({ p1: mkHubProject({ panes: [mkPane({ pane: "%1", live: false, pendingQuestion })] }) });
    const { model } = buildMission(scan, hub, RESOLVED, NOW);
    expect(model.cards[0].headline).toBe("idle"); // stale question is not current attention
    expect(model.cards[0].pendingQuestions).toHaveLength(0);
    expect(model.inbox.filter((r) => r.kind === "question")).toHaveLength(0);
    expect(model.inboxTotalCount).toBe(0);
    const tmuxSession = model.cards[0].sessions.find((s) => s.transport === "tmux");
    expect(tmuxSession).toBeDefined(); // the pane still renders as a session display row
  });

  it("a dead needs_input pane plus a LIVE blocked pane: headline needs-you from the live pane, and only the live pane contributes an inbox row", () => {
    const scan = scanOf([mkProject({ dir: "p1" })]);
    const blockedCapsule: Capsule = { status: "blocked", eventId: 1, minutes: null, declaredAt: new Date(NOW - 1_000).toISOString(), mergeAsk: null, question: null };
    const hub = mkHub({ p1: mkHubProject({
      panes: [
        mkPane({ pane: "%1", live: false, capsule: needsInputCapsule }),
        mkPane({ pane: "%2", live: true, capsule: blockedCapsule }),
      ],
    }) });
    const { model } = buildMission(scan, hub, RESOLVED, NOW);
    expect(model.cards[0].headline).toBe("needs-you");
    expect(model.inbox).toContainEqual({ kind: "attention", dir: "p1", name: "p1", status: "blocked", transport: "tmux", pane: "%2", tmuxIncarnation: "100:1000" });
    expect(model.inbox.filter((r) => r.kind === "attention")).toHaveLength(1);
    expect(model.cards[0].pendingQuestions).toHaveLength(0);
  });

  it.each(["notLoaded", "systemError", "unknown"] as const)("an unavailable native %s session is unknown, never needs-you, even with stale pending attention and a needs_input capsule", (status) => {
    const scan = scanOf([mkProject({ dir: "p1" })]);
    const hub = mkHub({ p1: mkHubProject({ codexSessions: [mkCodex({ status, pendingAttention: true, capsule: needsInputCapsule })] }) });
    const { model } = buildMission(scan, hub, RESOLVED, NOW);
    const card = model.cards[0];
    expect(card.headline).toBe("idle"); // unavailable contributes nothing to the headline
    const codex = card.sessions.find((s) => s.transport === "codex")!;
    expect(codex.state).toBe("unknown");
    expect(model.inbox.filter((r) => r.kind === "attention" || r.kind === "question")).toHaveLength(0);
  });

  it("a genuinely live native session's pending attention still reads needs-you", () => {
    const scan = scanOf([mkProject({ dir: "p1" })]);
    const hub = mkHub({ p1: mkHubProject({ codexSessions: [mkCodex({ status: "idle", pendingAttention: true })] }) });
    const { model } = buildMission(scan, hub, RESOLVED, NOW);
    expect(model.cards[0].headline).toBe("needs-you");
    expect(model.inbox).toContainEqual({ kind: "attention", dir: "p1", name: "p1", status: "needs_input", transport: "codex", threadId: mkCodex().threadId });
  });
});

describe("bucketOf and summarizeBuckets — the four-pill fold (spec §7)", () => {
  function cardWith(headline: MissionState): MissionCard {
    return {
      dir: headline, name: headline, stage: "build", branch: "main", builder: "claude-code",
      gate: null, legacyGateField: false, now: "", residuals: [], updated: "",
      headline, hubReady: true, lastEventAgo: null, freshnessCount: null, freshnessWarning: false,
      sessions: [], pendingQuestions: [], timeline: [], pr: { status: "unknown" },
      codexSource: "ok", codexUntracked: false,
      heartbeat: [], lastAction: null, prefs: { pinned: false, hiddenAt: null }, visibility: "visible",
    };
  }

  it("bucketOf folds waiting into working and excludes unknown", () => {
    expect(bucketOf("waiting")).toBe("working");
    expect(bucketOf("unknown")).toBeNull();
    expect(bucketOf("needs-you")).toBe("needs-you");
    expect(bucketOf("stuck")).toBe("stuck");
    expect(bucketOf("working")).toBe("working");
    expect(bucketOf("idle")).toBe("idle");
  });

  it("counts a mixed board, folding waiting into working and excluding unknown", () => {
    const cards = [cardWith("needs-you"), cardWith("stuck"), cardWith("working"), cardWith("waiting"), cardWith("idle"), cardWith("unknown")];
    const { counts, worst } = summarizeBuckets({ kind: "ready" }, cards);
    expect(counts).toEqual({ "needs-you": 1, stuck: 1, working: 2, idle: 1 });
    expect(worst).toBe("needs-you");
  });

  it("a hidden card never reaches the pills, whatever its headline", () => {
    const hidden = { ...cardWith("needs-you"), visibility: "hidden" as const };
    const { counts, worst } = summarizeBuckets({ kind: "ready" }, [hidden, cardWith("working")]);
    expect(counts).toEqual({ "needs-you": 0, stuck: 0, working: 1, idle: 0 });
    expect(worst).toBe("working");
  });

  it("an all-idle board reports idle as the worst bucket", () => {
    const cards = [cardWith("idle"), cardWith("idle")];
    const { counts, worst } = summarizeBuckets({ kind: "ready" }, cards);
    expect(counts).toEqual({ "needs-you": 0, stuck: 0, working: 0, idle: 2 });
    expect(worst).toBe("idle");
  });

  it("a tie between two non-idle buckets resolves by the fixed rank order, not array order", () => {
    const cards = [cardWith("working"), cardWith("stuck")]; // working pushed first, stuck ranks higher
    const { worst } = summarizeBuckets({ kind: "ready" }, cards);
    expect(worst).toBe("stuck");
  });

  it("an empty card list reports idle with all-zero counts", () => {
    const { counts, worst } = summarizeBuckets({ kind: "ready" }, []);
    expect(counts).toEqual({ "needs-you": 0, stuck: 0, working: 0, idle: 0 });
    expect(worst).toBe("idle");
  });

  it("readiness.kind loading or failed reports the not-ready signal instead of a confident zero", () => {
    expect(summarizeBuckets({ kind: "loading", pending: ["hub"] }, [])).toEqual({ counts: null, worst: null });
    expect(summarizeBuckets({ kind: "failed", failed: ["hub"], pending: [] }, [cardWith("idle")])).toEqual({ counts: null, worst: null });
  });

  it("degraded never gates this feature — prs failure never affects card headlines", () => {
    const { counts, worst } = summarizeBuckets({ kind: "degraded", degraded: ["prs"] }, [cardWith("working")]);
    expect(counts).toEqual({ "needs-you": 0, stuck: 0, working: 1, idle: 0 });
    expect(worst).toBe("working");
  });
});

describe("PIPELINE_STAGES and stageDotIndex (spec §8, the stage-track row)", () => {
  it("lists the five real ProjectStage values in order", () => {
    expect(PIPELINE_STAGES).toEqual(["spec", "build", "review", "test", "ship"]);
  });

  it("maps every stage to its own dot index", () => {
    expect(PIPELINE_STAGES.map((s) => stageDotIndex(s))).toEqual([0, 1, 2, 3, 4]);
  });

  it("degrades to -1 (no dot highlighted) for an unrecognized value, never a crash", () => {
    expect(stageDotIndex("not-a-stage" as never)).toBe(-1);
  });
});

describe("buildMission — retained worktrees (spec §7, round 3 F2)", () => {
  it("retainedWorktrees is joined from the worktrees envelope by project.dir, independent of hub membership; null unresolved", () => {
    // F2 (round 2, MEDIUM): `worktrees` is a TRAILING optional 5th arg (after `nowMs`), so
    // every existing 4-argument caller above keeps binding `nowMs` to position 4 unchanged.
    const projects = { ok: true as const, data: { projects: [mkProject({ dir: "demo" })], skipped: 0, reposRoot: "" } };
    expect(buildMission(projects, undefined, undefined, NOW).model.cards[0].retainedWorktrees).toBeNull();
    const hub = undefined; // hub NOT ready / project absent from hubMap — must not matter
    const worktrees = { ok: true as const, data: { demo: { ok: true as const, count: 3, oldestAgeDays: 20 } } };
    const mission = buildMission(projects, hub, undefined, NOW, worktrees);
    expect(mission.model.cards[0].retainedWorktrees).toBe(3);
  });
});

const MERGE_Q = (labels: [string, string], question = "May I merge `feat/x` into `main`?"): MissionPendingQuestion => ({
  pane: "%1", eventId: 11, questions: [{ question, multiSelect: false, options: [{ label: labels[0] }, { label: labels[1] }] }],
});

describe("composeIndexedReply — the indexed grammar claimAnswer accepts (round-2 F1)", () => {
  it("one line per question, single or multi select, round-trips through parseIndexedReply", () => {
    const pq: MissionPendingQuestion = {
      pane: "%1", eventId: 5,
      questions: [
        { question: "Env?", multiSelect: false, options: [{ label: "dev" }, { label: "stg" }] },
        { question: "Checks?", multiSelect: true, options: [{ label: "t" }, { label: "b" }, { label: "l" }] },
      ],
    };
    const reply = composeIndexedReply(pq, [[2], [1, 3]]);
    expect(reply).toBe("1: 2\n2: 1,3");
    const shapes = pq.questions.map((q) => ({ multiSelect: q.multiSelect, option_count: q.options.length }));
    expect(parseIndexedReply(reply, shapes)).toEqual({ ok: true, answers: [
      { question_number: 1, kind: "options", values: [2] }, { question_number: 2, kind: "options", values: [1, 3] },
    ] });
    expect(composeIndexedReply(MERGE_Q(["Sim", "Não"]), [[1]])).toBe("1: 1");
    expect(parseIndexedReply("1: 1", [{ multiSelect: false, option_count: 2 }]).ok).toBe(true);
  });
});

const BASE: MissionCard = {
  dir: "p1", name: "p1", stage: "build", branch: "feat/x", builder: "codex", gate: null, legacyGateField: false, now: "", residuals: [],
  updated: "2026-09-18T00:00:00-03:00", headline: "needs-you", hubReady: true, lastEventAgo: null, freshnessCount: null, freshnessWarning: false,
  sessions: [], pendingQuestions: [], timeline: [], pr: { status: "unknown" }, activeRun: null, lastRun: null, codexSource: "ok", codexUntracked: false,
  heartbeat: [], lastAction: null, prefs: { pinned: false, hiddenAt: null }, visibility: "visible",
};
const pane = (p: string, over: Partial<MissionPane> = {}) => ({
  pane: p, tmuxIncarnation: "1:1", session: "jax-p1-lead", role: "lead" as const, state: "needs-you" as const, lastEventTs: null, pendingQuestion: null, capsuleEventId: null, subagentCount: 0,
  capsuleStatus: null as "needs_input" | "blocked" | null, capsuleMinutes: null as number | null, capsuleDeclaredAt: null as string | null,
  capsuleMergeAsk: null as number | null, capsuleQuestion: null as string | null,
  live: true, tmuxSession: "jax-p1-lead" as string | null, ...over,
});
const codexSession = (threadId: string, over: Partial<Extract<MissionSession, { transport: "codex" }>> = {}) => ({
  transport: "codex" as const, threadId, state: "needs-you" as const, lastEventTs: null, capsuleEventId: null, subagentCount: 0,
  capsuleStatus: null as "needs_input" | "blocked" | null, capsuleMinutes: null as number | null, capsuleDeclaredAt: null as string | null, ...over,
});

describe("selectPendingAction — deterministic single-event policy (round-3 F4)", () => {
  it("the first pane in pane order with a pending question wins; a second pane's question gets nothing", () => {
    const q1 = { ...MERGE_Q(["Sim", "Não"]), pane: "%1", eventId: 11 };
    const q2 = { ...MERGE_Q(["Sim", "Não"]), pane: "%2", eventId: 22 };
    const card: MissionCard = { ...BASE, pendingQuestions: [q1, q2], sessions: [{ transport: "tmux", session: "jax-p1-lead", panes: [pane("%1", { pendingQuestion: q1 }), pane("%2", { pendingQuestion: q2 })] }] };
    expect(selectPendingAction(card)).toEqual({ kind: "question", pendingQuestion: q1 });
  });
  it("with no structured question, the first needs-you pane carrying a capsuleEventId is a freeform action", () => {
    const card: MissionCard = { ...BASE, sessions: [{ transport: "tmux", session: "s", panes: [pane("%1", { state: "working" as const, capsuleEventId: 7 }), pane("%2", { capsuleEventId: 9 }), pane("%3", { capsuleEventId: 10 })] }] };
    expect(selectPendingAction(card)).toEqual({ kind: "freeform", pane: "%2", eventId: 9, capsuleStatus: null, transport: "tmux", role: "lead", mergeAsk: null, question: null, answerable: true });
  });
  it("nothing pending → null; a bare awaiting-approval gate is not a pending action", () => {
    expect(selectPendingAction(BASE)).toBeNull();
    expect(selectPendingAction({ ...BASE, gate: "awaiting-approval" })).toBeNull();
  });
  it("with no pane pending, a needs-you native Codex session's capsule is a freeform action (diff review F2)", () => {
    const card: MissionCard = { ...BASE, sessions: [codexSession("thread-1", { capsuleEventId: 42 })] };
    expect(selectPendingAction(card)).toEqual({ kind: "freeform", pane: "thread-1", eventId: 42, capsuleStatus: null, transport: "codex", role: null, mergeAsk: null, question: null, answerable: true });
  });
  it("a pane's capsuleEventId still wins over a native Codex session's when both are pending", () => {
    const card: MissionCard = {
      ...BASE,
      sessions: [
        { transport: "tmux", session: "s", panes: [pane("%1", { capsuleEventId: 7 })] },
        codexSession("thread-1", { capsuleEventId: 42 }),
      ],
    };
    expect(selectPendingAction(card)).toEqual({ kind: "freeform", pane: "%1", eventId: 7, capsuleStatus: null, transport: "tmux", role: "lead", mergeAsk: null, question: null, answerable: true });
  });
  it("an explicit eventId picks that pendingQuestion instead of always [0] (F2)", () => {
    const q1 = { ...MERGE_Q(["Sim", "Não"]), pane: "%1", eventId: 9 };
    const q2 = { ...MERGE_Q(["Sim", "Não"]), pane: "%2", eventId: 10 };
    const card: MissionCard = { ...BASE, pendingQuestions: [q1, q2] };
    expect(selectPendingAction(card, 10)).toEqual({ kind: "question", pendingQuestion: q2 });
    expect(selectPendingAction(card)).toEqual({ kind: "question", pendingQuestion: q1 });
  });
});

describe("isMergeAskEligible (spec §4)", () => {
  it("mergeAsk at/above the threshold is eligible regardless of question text", () => {
    expect(isMergeAskEligible(0.5)).toBe(true);
    expect(isMergeAskEligible(0.82)).toBe(true);
    expect(isMergeAskEligible(0.49)).toBe(false);
    expect(isMergeAskEligible(null)).toBe(false);
  });
  it("boundary values 0 and 1 (cold review round 1 F3)", () => {
    expect(isMergeAskEligible(0)).toBe(false);
    expect(isMergeAskEligible(1)).toBe(true);
  });
});

describe("selectPendingAction carries mergeAsk/question/role/transport on the freeform arm (spec §3 point 7)", () => {
  it("a tmux pane's freeform action carries its own role, mergeAsk and question", () => {
    const card: MissionCard = { ...BASE, sessions: [{ transport: "tmux", session: "s", panes: [
      pane("%1", { role: "adhoc" as const, capsuleEventId: 9, capsuleStatus: "needs_input", capsuleMergeAsk: 0.7, capsuleQuestion: "posso mergear?" }),
    ] }] };
    expect(selectPendingAction(card)).toEqual({
      kind: "freeform", pane: "%1", eventId: 9, capsuleStatus: "needs_input",
      transport: "tmux", role: "adhoc", mergeAsk: 0.7, question: "posso mergear?", answerable: true,
    });
  });
  it("a codex session's freeform action carries role: null and its own capsule mergeAsk/question (D2)", () => {
    const card: MissionCard = { ...BASE, sessions: [codexSession("thread-1", { capsuleEventId: 42, capsuleStatus: "needs_input" })] };
    expect(selectPendingAction(card)).toEqual({
      kind: "freeform", pane: "thread-1", eventId: 42, capsuleStatus: "needs_input",
      transport: "codex", role: null, mergeAsk: null, question: null, answerable: true,
    });
  });
  it("the FIRST needs-you tmux pane wins even when a later pane is the merge-ask-eligible one (Review Focus 5)", () => {
    const card: MissionCard = { ...BASE, sessions: [{ transport: "tmux", session: "s", panes: [
      pane("%1", { capsuleEventId: 7, capsuleStatus: "needs_input", capsuleMergeAsk: null, capsuleQuestion: null }),
      pane("%2", { capsuleEventId: 9, capsuleStatus: "needs_input", capsuleMergeAsk: 0.9, capsuleQuestion: "posso mergear?" }),
    ] }] };
    expect(selectPendingAction(card)).toEqual({
      kind: "freeform", pane: "%1", eventId: 7, capsuleStatus: "needs_input",
      transport: "tmux", role: "lead", mergeAsk: null, question: null, answerable: true,
    });
  });

  it("a native Codex session's freeform action carries its own mergeAsk/question/answerable, not a hardcoded null (D2)", () => {
    const card: MissionCard = { ...BASE, sessions: [codexSession("thread-1", { capsuleEventId: 42, capsuleStatus: "needs_input", capsuleMergeAsk: 0.9, capsuleQuestion: "posso mergear feat/x em main?" })] };
    expect(selectPendingAction(card)).toEqual({
      kind: "freeform", pane: "thread-1", eventId: 42, capsuleStatus: "needs_input",
      transport: "codex", role: null, mergeAsk: 0.9, question: "posso mergear feat/x em main?", answerable: true,
    });
  });

  it("a codex session with answerable: false carries it through unchanged (D4)", () => {
    const card: MissionCard = { ...BASE, sessions: [codexSession("thread-1", { capsuleEventId: 42, capsuleStatus: "blocked", capsuleAnswerable: false })] };
    expect(selectPendingAction(card)?.kind === "freeform" && (selectPendingAction(card) as { answerable: boolean }).answerable).toBe(false);
  });

  it("two native-Codex needs-you sessions: the EARLIER plain one wins over a LATER merge-ask-eligible one — no scanning ahead for 'the best' session (Review Focus 5)", () => {
    const card: MissionCard = { ...BASE, sessions: [
      codexSession("thread-1", { capsuleEventId: 7, capsuleStatus: "needs_input", capsuleMergeAsk: null, capsuleQuestion: null }),
      codexSession("thread-2", { capsuleEventId: 9, capsuleStatus: "needs_input", capsuleMergeAsk: 0.9, capsuleQuestion: "posso mergear feat/x em main?" }),
    ] };
    expect(selectPendingAction(card)).toEqual({
      kind: "freeform", pane: "thread-1", eventId: 7, capsuleStatus: "needs_input",
      transport: "codex", role: null, mergeAsk: null, question: null, answerable: true,
    });
  });
});

describe("postWorkflowAction (Task 9 — the one fetch wrapper the card's buttons use)", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("returns the parsed envelope, data included, on ok:true", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ json: async () => ({ ok: true, data: { run_id: "abc123def456" } }) }));
    const result = await postWorkflowAction("/api/workflow/dispatch", { project: "p1" });
    expect(result).toEqual({ ok: true, data: { run_id: "abc123def456" } });
  });

  it("passes an ok:false envelope through unchanged, including its code", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ json: async () => ({ ok: false, code: "mutation-rejected", error: "already-finished" }) }));
    const result = await postWorkflowAction("/api/workflow/cancel", { run_id: "x" });
    expect(result).toEqual({ ok: false, code: "mutation-rejected", error: "already-finished" });
  });

  it("never throws: a rejected fetch resolves to a network-error envelope", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("boom")));
    expect(await postWorkflowAction("/api/workflow/cancel", {})).toEqual({ ok: false, error: "network error" });
  });

  it("never throws: a non-JSON response resolves to an invalid-response envelope", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ json: async () => { throw new Error("not json"); } }));
    expect(await postWorkflowAction("/api/workflow/cancel", {})).toEqual({ ok: false, error: "invalid response" });
  });

  it("never throws: a shape with no ok field resolves to an unexpected-response envelope", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ json: async () => ({ weird: true }) }));
    expect(await postWorkflowAction("/api/workflow/cancel", {})).toEqual({ ok: false, error: "unexpected response" });
  });

  it("sends a JSON POST with the exact body given", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ json: async () => ({ ok: true, data: {} }) });
    vi.stubGlobal("fetch", fetchMock);
    await postWorkflowAction("/api/workflow/answer", { project: "p1", branch: "feat/x" });
    expect(fetchMock).toHaveBeenCalledWith("/api/workflow/answer", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ project: "p1", branch: "feat/x" }),
    });
  });
});

describe("answerEventId (merge-button-hide fix — keys the optimistic answered state)", () => {
  it("extracts the numeric event_id from an answer POST body", () => {
    expect(answerEventId({ event_id: 9, reply: "pode" })).toBe(9);
  });

  it("returns null for a body with no event_id (cancel/dispatch/prefs POSTs)", () => {
    expect(answerEventId({ run_id: "x" })).toBeNull();
    expect(answerEventId({ project: "p1", pinned: true })).toBeNull();
  });

  it("returns null for a non-numeric event_id, null, or a non-object body", () => {
    expect(answerEventId({ event_id: "9" })).toBeNull();
    expect(answerEventId(null)).toBeNull();
    expect(answerEventId(undefined)).toBeNull();
    expect(answerEventId("not an object")).toBeNull();
    expect(answerEventId([1, 2])).toBeNull();
  });
});

describe("Phase 3 — subagentCount folds into working (spec Decision 8)", () => {
  it("paneState is working when subagentCount > 0 even with no other working signal", () => {
    const base = { pane: "%1", tmuxIncarnation: "1:1", session: null, role: null, live: true, working: false,
      capsule: null, pendingQuestion: null, lastEventTs: "2026-09-19T00:00:00.000Z", subagentCount: 0, runtime: "claude" as const };
    expect(paneState({ ...base, subagentCount: 2 }, Date.now())).toBe("working");
    expect(paneState(base, Date.now())).toBe("idle");
  });
});

describe("Phase 3 — visibility derivation (spec §10, Decisions 14/15)", () => {
  const projectBase = { dir: "p1", name: "P1", stage: "build" as const, gate: null, legacyGateField: false,
    builder: "", branch: "", updated: "", now: "", residuals: [], statusMtime: "2026-09-01T00:00:00.000Z", archived: false };
  const HUB_BASE: HubProject = { panes: [], codexSessions: [], codexUntracked: false, pendingQuestion: null, freshnessCount: 0,
    newestEventTs: null, timeline: [], heartbeat: new Array(60).fill(0), lastAction: null, activeRun: undefined, activeRuns: [], lastRun: null };
  const NOW5 = Date.parse("2026-09-19T12:00:00.000Z");
  const hub = (byProject: Record<string, HubProject>, prefs: Record<string, import("../server/db/workflows").ProjectPrefs> = {}, archiveAfterDays = 14) => ({
    ok: true as const,
    data: { byProject, historicalTruncated: false, codexSource: "ok" as const, tmuxSource: "ok" as const, prefs, settings: { archiveAfterDays } },
  });
  const scan = (projects: Project[]) => ({ ok: true as const, data: { projects, skipped: 0, reposRoot: "" } });

  it("archived-but-fresh (a later AGENT event) is visible; archived-and-stale is excluded", () => {
    const fresh = buildMission(scan([{ ...projectBase, archived: true }]), hub({ p1: { ...HUB_BASE, freshnessCount: 1, newestEventTs: "2026-09-19T11:00:00.000Z" } }), undefined, NOW5);
    expect(fresh.model.cards[0].visibility).toBe("visible");
    const stale = buildMission(scan([{ ...projectBase, archived: true }]), hub({ p1: { ...HUB_BASE } }), undefined, NOW5);
    expect(stale.model.cards[0].visibility).toBe("archived");
  });

  it("pinning a hidden project does NOT clear hidden_at (F6); pinned overrides both hidden and archived", () => {
    const result = buildMission(scan([{ ...projectBase, archived: true }]), hub({}, { p1: { pinned: true, hiddenAt: "2026-09-19T10:00:00.000Z" } }), undefined, NOW5);
    expect(result.model.cards[0].visibility).toBe("visible");
    expect(result.model.cards[0].prefs).toEqual({ pinned: true, hiddenAt: "2026-09-19T10:00:00.000Z" });
  });

  it("a question-answered event (emitter jaxos) does not advance lastActivity and does not un-hide", () => {
    // A real post-hiddenAt event, unfiltered in hp.timeline (workflows.ts:728 — getProjectTimeline
    // filters nothing), but hp.newestEventTs stays null because the upstream SQL that computes it
    // (workflows.ts:1387-1421) excludes emitter='jaxos', and question-answered is jaxos-only
    // (EVENT_MATRIX). This proves buildModel reads newestEventTs as already-filtered and never
    // re-derives lastActivity by scanning hp.timeline itself.
    const jaxosEvent = { id: 1, ts: "2026-09-19T10:00:00.000Z", type: "question-answered", pane: null, role: "lead", emitter: "jaxos", payload: {} };
    const result = buildMission(scan([projectBase]), hub({ p1: { ...HUB_BASE, timeline: [jaxosEvent] } }, { p1: { pinned: false, hiddenAt: "2026-09-19T09:00:00.000Z" } }), undefined, NOW5);
    expect(result.model.cards[0].visibility).toBe("hidden");
  });

  it("a hidden project feeds no inbox row (gate, question, attention, PR) — hiding a card hides its strip rows too", () => {
    // Fresh statusMtime keeps the control case out of the auto-hide ladder; hiddenAt is later than it.
    const hiddenPrefs = { p1: { pinned: false, hiddenAt: "2026-09-19T11:45:00.000Z" } };
    const gated = { ...projectBase, gate: "awaiting-approval" as const, statusMtime: "2026-09-19T11:30:00.000Z" };
    const pane = { pane: "%1", tmuxIncarnation: "i1", capsule: { status: "blocked" as const, ts: "2026-09-01T00:00:00.000Z" }, state: "idle" as const };
    const hp = { ...HUB_BASE, panes: [pane as unknown as HubProject["panes"][number]], pendingQuestion: { pane: "%1", eventId: 7, questions: [{ question: "Q?" }] } as unknown as HubProject["pendingQuestion"] };
    const prs = { ok: true as const, data: { p1: { ok: true as const, prs: [{ number: 1, title: "pr", isDraft: false, url: "u", ciState: "green" as const, mergeReady: true, updatedAt: "2026-09-01T00:00:00.000Z" }], truncated: false } } };
    const hidden = buildMission(scan([gated]), hub({ p1: hp }, hiddenPrefs), prs, NOW5);
    expect(hidden.model.cards[0].visibility).toBe("hidden");
    expect(hidden.model.inbox).toEqual([]);
    expect(hidden.model.inboxTotalCount).toBe(0);
    const shown = buildMission(scan([gated]), hub({ p1: hp }), prs, NOW5);
    expect(shown.model.inbox.map((r) => r.kind)).toContain("gate");
    expect(shown.model.inbox.map((r) => r.kind)).toContain("pr");
  });

  it("an awaiting-approval gate follows the tech lead: working pane → working card and no gate row; stopped pane → needs-you and a gate row", () => {
    const gated = { ...projectBase, gate: "awaiting-approval" as const, statusMtime: "2026-09-19T11:30:00.000Z" };
    const livePane = { pane: "%1", tmuxIncarnation: "i1", session: "s", role: "lead", live: true, working: true, capsule: null, pendingQuestion: null, subagentCount: 0, lastEventTs: "2026-09-19T11:59:00.000Z" };
    const working = buildMission(scan([gated]), hub({ p1: { ...HUB_BASE, panes: [livePane as unknown as HubProject["panes"][number]] } }), undefined, NOW5);
    expect(working.model.cards[0].headline).toBe("working");
    expect(working.model.inbox.map((r) => r.kind)).not.toContain("gate");
    const stoppedPane = { ...livePane, working: false, capsule: { status: "needs_input", minutes: null, declaredAt: "2026-09-19T11:59:00.000Z", eventId: 9 } };
    const stopped = buildMission(scan([gated]), hub({ p1: { ...HUB_BASE, panes: [stoppedPane as unknown as HubProject["panes"][number]] } }), undefined, NOW5);
    expect(stopped.model.cards[0].headline).toBe("needs-you");
    expect(stopped.model.inbox.map((r) => r.kind)).toContain("gate");
  });

  it("an awaiting-approval gate does not claim Rafa while the tech lead waits on its own run (2026-09-24, arc)", () => {
    const gated = { ...projectBase, gate: "awaiting-approval" as const, statusMtime: "2026-09-19T11:30:00.000Z" };
    const waitingPane = { pane: "%1", tmuxIncarnation: "i1", session: "s", role: "lead", live: true, working: false, capsule: { status: "waiting", minutes: 120, declaredAt: "2026-09-19T11:59:00.000Z", eventId: 9 }, pendingQuestion: null, subagentCount: 0, lastEventTs: "2026-09-19T11:59:00.000Z" };
    const m = buildMission(scan([gated]), hub({ p1: { ...HUB_BASE, panes: [waitingPane as unknown as HubProject["panes"][number]] } }), undefined, NOW5);
    expect(m.model.cards[0].headline).toBe("waiting");
    expect(m.model.inbox.map((r) => r.kind)).not.toContain("gate");
  });

  it("a project absent from BOTH hubMap and prefs still derives visible via the client-side default", () => {
    // Plan fix: the plan's shared `projectBase.statusMtime` (2026-09-01) is 18 days older than
    // NOW5, so the auto-hide ladder legitimately yields "autoHidden" for it — the fixture, not
    // the ladder, was wrong for this test's own intent (spec §14: absent from both maps must
    // still get the default prefs and stay visible). A fresh statusMtime isolates the default
    // join from the separate no-measurable-activity case the F13 test already covers.
    const freshProject = { ...projectBase, statusMtime: "2026-09-19T11:30:00.000Z" };
    const result = buildMission(scan([freshProject]), hub({}), undefined, NOW5);
    expect(result.model.cards[0].visibility).toBe("visible");
    expect(result.model.cards[0].prefs).toEqual({ pinned: false, hiddenAt: null });
  });

  it("autoHidden when now - lastActivity >= archiveAfterDays; never with no measurable lastActivity (F13)", () => {
    const oldProject = { ...projectBase, updated: "2026-09-01T00:00:00-03:00", statusMtime: "2026-09-01T00:00:00.000Z" };
    const autoHidden = buildMission(scan([oldProject]), hub({ p1: { ...HUB_BASE } }), undefined, NOW5);
    expect(autoHidden.model.cards[0].visibility).toBe("autoHidden");
    const neverMeasured = { ...projectBase, updated: "", statusMtime: "" };
    const noSignal = buildMission(scan([neverMeasured]), hub({}), undefined, NOW5);
    expect(noSignal.model.cards[0].visibility).toBe("visible");
  });

  it("MissionModel carries archiveAfterDays, hiddenCount and autoHiddenCount", () => {
    const result = buildMission(scan([projectBase]), hub({}, { p1: { pinned: false, hiddenAt: "2026-09-19T00:00:00.000Z" } }, 7), undefined, NOW5);
    expect(result.model.archiveAfterDays).toBe(7);
    expect(result.model.hiddenCount).toBe(1);
    expect(result.model.autoHiddenCount).toBe(0);
  });
});

describe("Phase 3 — buildTicker (spec §9, Decisions 12/13)", () => {
  // Fixtures are all dated 2026-09-19; pin `nowMs` so the 7-day window (below) doesn't age
  // them out as the real clock moves past that date (they went stale on 2026-09-26).
  const FIXTURE_NOW = Date.parse("2026-09-19T12:00:00.000Z");

  it("merges timeline events, commits and open PRs, sorts newest first, excludes run-started", () => {
    const timeline: MissionTimelineEntry[] = [
      { ts: "2026-09-19T10:00:00.000Z", type: "run-finished", pane: null, capsuleStatus: null, outcome: "success", contractStatus: "ok", stage: null, role: null, diagnostic: null },
      { ts: "2026-09-19T09:00:00.000Z", type: "run-started", pane: null, capsuleStatus: null, outcome: null, contractStatus: null, stage: null, role: null, diagnostic: null },
    ];
    const cards = [{ dir: "p1", name: "P1", timeline }];
    const commits = [{ repo: "p1", hash: "abc", author: "rafa", message: "fix: x", epoch: Date.parse("2026-09-19T11:00:00.000Z") / 1000 }];
    const prs = { p1: { ok: true as const, truncated: false, prs: [{ number: 1, title: "T", isDraft: false, url: "https://x", ciState: "green" as const, mergeReady: true, updatedAt: "2026-09-19T08:00:00.000Z" }] } };
    const items = buildTicker(cards, commits, prs, FIXTURE_NOW);
    expect(items.map((i) => i.source)).toEqual(["commit", "event", "pr"]);
    expect(items[0]).toMatchObject({ source: "commit", ts: "2026-09-19T11:00:00.000Z" });
    expect(items[2]).toMatchObject({ source: "pr", number: 1, ci: "green" });
  });

  it("caps at 30 items even with more available", () => {
    const cards = [{ dir: "p1", name: "P1", timeline: Array.from({ length: 40 }, (_, i) => ({
      ts: new Date(Date.parse("2026-09-19T00:00:00.000Z") + i * 1000).toISOString(), type: "turn-stopped",
      pane: null, capsuleStatus: "done", outcome: null, contractStatus: null, stage: null, role: null, diagnostic: null,
    })) }];
    expect(buildTicker(cards, [], {}, FIXTURE_NOW)).toHaveLength(30);
  });

  // Rafa: "o activity tá muito grande... precisa ter um cut dos últimos 7 dias só" — the window
  // is applied BEFORE sort/cap, against an injected `nowMs` (default Date.now()).
  describe("7-day cutoff", () => {
    const NOW_ISO2 = "2026-09-19T12:00:00.000Z";
    const nowMs = Date.parse(NOW_ISO2);
    const entry = (ts: string): MissionTimelineEntry => ({
      ts, type: "run-finished", pane: null, capsuleStatus: null, outcome: "success", contractStatus: "ok", stage: null, role: null, diagnostic: null,
    });

    it("drops an item older than 7 days and keeps one within the window", () => {
      const eightDaysAgo = new Date(nowMs - 8 * 24 * 60 * 60 * 1000).toISOString();
      const sixDaysAgo = new Date(nowMs - 6 * 24 * 60 * 60 * 1000).toISOString();
      const cards = [{ dir: "p1", name: "P1", timeline: [entry(eightDaysAgo), entry(sixDaysAgo)] }];
      const items = buildTicker(cards, [], {}, nowMs);
      expect(items).toHaveLength(1);
      expect(items[0]).toMatchObject({ ts: sixDaysAgo });
    });

    it("drops an item with an invalid ts", () => {
      const cards = [{ dir: "p1", name: "P1", timeline: [entry("not-a-date")] }];
      expect(buildTicker(cards, [], {}, nowMs)).toHaveLength(0);
    });
  });
});

describe("pageSlice", () => {
  const items = Array.from({ length: 25 }, (_, i) => i);

  it("splits into pages of the given size", () => {
    expect(pageSlice(items, 1, 10)).toEqual({ items: items.slice(0, 10), page: 1, pages: 3 });
    expect(pageSlice(items, 3, 10)).toEqual({ items: items.slice(20, 25), page: 3, pages: 3 });
  });

  it("clamps an out-of-range page into [1, pages]", () => {
    expect(pageSlice(items, 99, 10)).toEqual({ items: items.slice(20, 25), page: 3, pages: 3 });
    expect(pageSlice(items, 0, 10)).toEqual({ items: items.slice(0, 10), page: 1, pages: 3 });
    expect(pageSlice(items, -5, 10)).toEqual({ items: items.slice(0, 10), page: 1, pages: 3 });
  });

  it("an empty list is one page (page 1 of 1), no items", () => {
    expect(pageSlice([], 1, 10)).toEqual({ items: [], page: 1, pages: 1 });
  });
});

describe("hasQuestionText (spec §6, Decision 9a, round-2 F3)", () => {
  const q = (question: string): MissionPendingQuestion => ({
    pane: "%1", eventId: 1, questions: [{ question, multiSelect: false, options: [{ label: "a" }, { label: "b" }] }],
  });
  it("true for non-empty trimmed text; false for missing, empty-array, and empty/whitespace text", () => {
    expect(hasQuestionText({ pendingQuestions: [q("Merge now?")] })).toBe(true);
    expect(hasQuestionText({ pendingQuestions: [] })).toBe(false);
    expect(hasQuestionText({ pendingQuestions: [{ pane: "%1", eventId: 1, questions: [] }] })).toBe(false);
    expect(hasQuestionText({ pendingQuestions: [q("")] })).toBe(false);
    expect(hasQuestionText({ pendingQuestions: [q("   ")] })).toBe(false);
  });
});

describe("capsuleStatus on MissionPane/codex MissionSession (spec §6, Decision 11a, round-2 F1)", () => {
  it("a live pane's blocked/needs_input capsule surfaces as capsuleStatus; waiting/null does not", () => {
    const card = (capsule: { status: string } | null) => {
      const hp = mkHubProject({ panes: [mkPane({ pane: "%1", live: true, capsule: capsule ? { status: capsule.status, minutes: null, declaredAt: NOW_ISO, eventId: 9, mergeAsk: null, question: null } : null })] });
      return buildMission(scanOf([mkProject({ dir: "p1" })]), mkHub({ p1: hp }), undefined, NOW).model.cards[0];
    };
    expect(card({ status: "blocked" }).sessions[0]).toMatchObject({ transport: "tmux", panes: [{ capsuleStatus: "blocked" }] });
    expect(card({ status: "needs_input" }).sessions[0]).toMatchObject({ panes: [{ capsuleStatus: "needs_input" }] });
    expect(card({ status: "waiting" }).sessions[0]).toMatchObject({ panes: [{ capsuleStatus: null }] });
    expect(card(null).sessions[0]).toMatchObject({ panes: [{ capsuleStatus: null }] });
  });
  it("a codex session's capsule is narrowed the same way", () => {
    const hp = mkHubProject({ codexSessions: [{
      threadId: "t1", status: "unknown", activeFlags: [], sourceOk: true, working: false, attention: false,
      pendingAttention: false, runInFlight: false, capsule: { status: "blocked", minutes: null, declaredAt: NOW_ISO, eventId: 3, mergeAsk: null, question: null },
      lastEventTs: null, subagentCount: 0,
    }] });
    const card = buildMission(scanOf([mkProject({ dir: "p1" })]), mkHub({ p1: hp }), undefined, NOW).model.cards[0];
    expect(card.sessions[0]).toMatchObject({ transport: "codex", capsuleStatus: "blocked" });
  });
});

describe("PendingAction.freeform.capsuleStatus (spec §6, Decision 11a)", () => {
  it("selectPendingAction's freeform result carries the same pane's capsuleStatus", () => {
    const card: MissionCard = { ...BASE, sessions: [{ transport: "tmux", session: "s", panes: [pane("%1", { state: "working" as const, capsuleEventId: 7, capsuleStatus: null }), pane("%2", { capsuleEventId: 9, capsuleStatus: "needs_input" })] }] };
    expect(selectPendingAction(card)).toEqual({ kind: "freeform", pane: "%2", eventId: 9, capsuleStatus: "needs_input", transport: "tmux", role: "lead", mergeAsk: null, question: null, answerable: true });
  });
  it("a codex session's freeform result carries its own capsuleStatus", () => {
    const card: MissionCard = { ...BASE, sessions: [codexSession("thread-1", { capsuleEventId: 42, capsuleStatus: "blocked" })] };
    expect(selectPendingAction(card)).toEqual({ kind: "freeform", pane: "thread-1", eventId: 42, capsuleStatus: "blocked", transport: "codex", role: null, mergeAsk: null, question: null, answerable: true });
  });
});

describe("MissionTimelineEntry.role (spec §9, Decision 21, round-2 F4)", () => {
  it("copies HubTimelineEvent.role straight through buildModel's timeline map", () => {
    const hp = mkHubProject({ timeline: [
      { id: 1, ts: "2026-09-19T10:00:00.000Z", type: "run-finished", pane: null, role: "builder", emitter: "codex-stop", payload: { result: "success" } },
      { id: 2, ts: "2026-09-19T09:00:00.000Z", type: "run-finished", pane: null, role: null, emitter: "codex-stop", payload: {} },
    ] });
    const card = buildMission(scanOf([mkProject({ dir: "p1" })]), mkHub({ p1: hp }), undefined, NOW).model.cards[0];
    expect(card.timeline.map((e) => e.role)).toEqual(["builder", null]);
  });
});

describe("tickerSentence (spec §9, Decisions 19/21, round-2 F4)", () => {
  const e = (over: Partial<MissionTimelineEntry> = {}): MissionTimelineEntry => ({
    ts: "2026-09-19T10:00:00.000Z", type: "run-finished", pane: null, capsuleStatus: null,
    outcome: null, contractStatus: null, stage: null, role: null, diagnostic: null, ...over,
  });
  it("non-run-finished types map one-to-one", () => {
    expect(tickerSentence(e({ type: "question" }))).toEqual({ key: "sentence.question" });
    expect(tickerSentence(e({ type: "question-resolved" }))).toEqual({ key: "sentence.questionResolved" });
    expect(tickerSentence(e({ type: "attention-needed" }))).toEqual({ key: "sentence.attentionNeeded" });
    expect(tickerSentence(e({ type: "merge-approved" }))).toEqual({ key: "sentence.mergeApproved" });
  });
  it("turn-stopped branches on capsuleStatus", () => {
    expect(tickerSentence(e({ type: "turn-stopped", capsuleStatus: "needs_input" }))).toEqual({ key: "sentence.turnStoppedNeedsInput" });
    expect(tickerSentence(e({ type: "turn-stopped", capsuleStatus: "blocked" }))).toEqual({ key: "sentence.turnStoppedBlocked" });
    expect(tickerSentence(e({ type: "turn-stopped", capsuleStatus: "done" }))).toEqual({ key: "sentence.turnStopped" });
    expect(tickerSentence(e({ type: "turn-stopped" }))).toEqual({ key: "sentence.turnStopped" });
  });
  it("run-finished: stage:worker beats everything (F3 precedence, row 1)", () => {
    expect(tickerSentence(e({ stage: "worker", outcome: "success", role: "builder" }))).toEqual({ key: "sentence.runFinishedInterrupted" });
    expect(tickerSentence(e({ stage: "worker", contractStatus: "cancelled" }))).toEqual({ key: "sentence.runFinishedInterrupted" });
  });
  it("run-finished: contractStatus:cancelled beats a non-null outcome (F3 precedence, row 2)", () => {
    expect(tickerSentence(e({ contractStatus: "cancelled", outcome: "success", role: "builder" }))).toEqual({ key: "sentence.runFinishedCancelled" });
  });
  it("run-finished: role gates the outcome-value rows (round-2 F4 — the exact collision F4 found)", () => {
    expect(tickerSentence(e({ role: "reviewer", outcome: "approve" }))).toEqual({ key: "sentence.reviewApprove" });
    expect(tickerSentence(e({ role: "reviewer", outcome: "approve-with-changes" }))).toEqual({ key: "sentence.reviewApproveChanges" });
    expect(tickerSentence(e({ role: "reviewer", outcome: "reject" }))).toEqual({ key: "sentence.reviewReject" });
    expect(tickerSentence(e({ role: "builder", outcome: "success" }))).toEqual({ key: "sentence.buildOk" });
    expect(tickerSentence(e({ role: "builder", outcome: "blocked" }))).toEqual({ key: "sentence.buildBlocked" });
    // The exact bug F4 found: a builder outcome equal to the literal "reject" must never
    // collide with the reviewer-only row now that role gates first.
    expect(tickerSentence(e({ role: "builder", outcome: "reject" }))).toEqual({ key: "sentence.buildFailed" });
    // And the inverse: a reviewer outcome the table doesn't name falls to the catch-all,
    // never to a builder row.
    expect(tickerSentence(e({ role: "reviewer", outcome: "success" }))).toEqual({ key: "sentence.runFinishedNoResult" });
  });
  it("run-finished: no outcome at all, or role neither reviewer nor builder, falls to the catch-all", () => {
    expect(tickerSentence(e({ role: "builder", outcome: null }))).toEqual({ key: "sentence.runFinishedNoResult" });
    expect(tickerSentence(e({ role: "lead", outcome: "success" }))).toEqual({ key: "sentence.runFinishedNoResult" });
    expect(tickerSentence(e({ role: null, outcome: null }))).toEqual({ key: "sentence.runFinishedNoResult" });
  });
  it("every key resolves in both locales", () => {
    const keys = ["question", "questionResolved", "attentionNeeded", "turnStoppedNeedsInput", "turnStoppedBlocked", "turnStopped",
      "runFinishedInterrupted", "runFinishedCancelled", "reviewApprove", "reviewApproveChanges", "reviewReject",
      "buildOk", "buildBlocked", "buildFailed", "runFinishedNoResult", "mergeApproved"] as const;
    for (const key of keys) {
      expect(typeof pt.mission.ticker.sentence[key]).toBe("string");
      expect(typeof en.mission.ticker.sentence[key]).toBe("string");
    }
  });
});

describe("formatClock (spec §9, Decision 20)", () => {
  // Round-2 F1: formatClock renders in the HOST's local timezone (no `timeZone` option —
  // the cockpit is single-user, host TZ America/Sao_Paulo, AGENTS.md) — that is the decision,
  // not UTC. Hardcoding an assumed UTC hour ("21:") broke under the real host offset
  // (18:41:12). Fixed by computing the expected string with the SAME Intl.DateTimeFormat call
  // the implementation uses, so the assertion holds under whatever TZ the test actually runs in.
  it("formats HH:MM:SS, hour12 false, in the host's local timezone, in both locales", () => {
    const ms = Date.parse("2026-09-19T21:41:12.000Z");
    const fmt = (locale: string) => new Intl.DateTimeFormat(locale, { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(ms);
    expect(formatClock(ms, "en-US")).toMatch(/^\d{2}:\d{2}:\d{2}$/);
    expect(formatClock(ms, "en-US")).toBe(fmt("en-US"));
    expect(formatClock(ms, "pt-BR")).toBe(fmt("pt-BR"));
  });
});

describe("MissionPane.live / .tmuxSession (Sessions mosaic spec §5 Decision 1 / round-2 F1 Decision 4a)", () => {
  it("live is copied straight from HubPane.live; tmuxSession is the RAW session (no pane-id fallback), independent of the display session field", () => {
    const hub = mkHub({ p1: mkHubProject({ panes: [
      mkPane({ pane: "%1", live: true }), // default session "s1"
      { ...mkPane({ pane: "%2", live: false }), session: null },
    ] }) });
    const { model } = buildMission(scanOf([mkProject({ dir: "p1" })]), hub, undefined, NOW);
    const panes = model.cards[0].sessions.flatMap((s) => (s.transport === "tmux" ? s.panes : []));
    const p1 = panes.find((p) => p.pane === "%1")!;
    const p2 = panes.find((p) => p.pane === "%2")!;
    expect(p1).toMatchObject({ live: true, tmuxSession: "s1", session: "s1" });
    // p2: HubPane.session is null -> display `session` falls back to the pane id, but the RAW
    // tmuxSession stays null (round-2 F1 — no fallback; every action that reaches real tmux must
    // read .tmuxSession, never .session, or it targets a pane id as if it were a session name).
    expect(p2).toMatchObject({ live: false, tmuxSession: null, session: "%2" });
  });
});

describe("MissionPane.runtime — session-runtime-label fix", () => {
  it("is copied straight from HubPane.runtime, independent of the project's own card.builder", () => {
    const hub = mkHub({ p1: mkHubProject({ panes: [mkPane({ pane: "%1", runtime: "claude" })] }) });
    // project.builder ("codex") must never leak onto the pane's own runtime field.
    const { model } = buildMission(scanOf([mkProject({ dir: "p1", builder: "codex" })]), hub, undefined, NOW);
    const panes = model.cards[0].sessions.flatMap((s) => (s.transport === "tmux" ? s.panes : []));
    expect(panes[0]).toMatchObject({ runtime: "claude" });
    expect(model.cards[0].builder).toBe("codex");
  });
});

describe("mosaicPaneState (Sessions mosaic spec §6, state table rows 5-8)", () => {
  const base: Parameters<typeof mosaicPaneState>[0] = { state: "idle", capsuleStatus: null, pendingQuestion: null };
  it("row 5: capsuleStatus blocked -> blocked-permission, regardless of state", () => {
    expect(mosaicPaneState({ ...base, capsuleStatus: "blocked", state: "working" })).toBe("blocked-permission");
  });
  it("row 6: capsuleStatus needs_input -> waiting-for-input", () => {
    expect(mosaicPaneState({ ...base, capsuleStatus: "needs_input" })).toBe("waiting-for-input");
  });
  it("row 6: a pendingQuestion with no capsuleStatus also -> waiting-for-input", () => {
    const pq = { pane: "%1", eventId: 1, questions: [{ question: "q?", multiSelect: false, options: [] }] };
    expect(mosaicPaneState({ ...base, pendingQuestion: pq })).toBe("waiting-for-input");
  });
  it("row 7: state working with no attention -> working", () => {
    expect(mosaicPaneState({ ...base, state: "working" })).toBe("working");
  });
  it("row 8: idle/waiting/stuck/unknown all collapse to idle-stale", () => {
    for (const state of ["idle", "waiting", "stuck", "unknown"] as const) {
      expect(mosaicPaneState({ ...base, state })).toBe("idle-stale");
    }
  });
});

describe("mosaicPaneAction (Sessions mosaic spec §8 Decision 17, round-1 F2/F3, rows 17a-17d)", () => {
  const simpleQ = { pane: "%1", eventId: 1, questions: [{ question: "Ship it?", multiSelect: false, options: [{ label: "yes" }, { label: "no" }] }] };
  it("17a: exactly one non-multiSelect question with real text -> numbered options", () => {
    expect(mosaicPaneAction({ pendingQuestion: simpleQ, capsuleStatus: null, capsuleEventId: null })).toEqual({ kind: "options", pendingQuestion: simpleQ });
  });
  it("17b: a multi-select question -> respond", () => {
    const multi = { ...simpleQ, questions: [{ ...simpleQ.questions[0], multiSelect: true }] };
    expect(mosaicPaneAction({ pendingQuestion: multi, capsuleStatus: null, capsuleEventId: null })).toEqual({ kind: "respond" });
  });
  it("17b: more than one question -> respond", () => {
    const twoQ = { ...simpleQ, questions: [simpleQ.questions[0], simpleQ.questions[0]] };
    expect(mosaicPaneAction({ pendingQuestion: twoQ, capsuleStatus: null, capsuleEventId: null })).toEqual({ kind: "respond" });
  });
  it("17b: needs_input capsule with no structured question -> respond", () => {
    expect(mosaicPaneAction({ pendingQuestion: null, capsuleStatus: "needs_input", capsuleEventId: 7 })).toEqual({ kind: "respond" });
  });
  it("17c: blocked with no structured question -> open-pane, carrying the capsule event id it targets", () => {
    expect(mosaicPaneAction({ pendingQuestion: null, capsuleStatus: "blocked", capsuleEventId: 42 })).toEqual({ kind: "open-pane", capsuleEventId: 42 });
  });
  it("17c: blocked with capsuleEventId null still returns open-pane, carrying null", () => {
    expect(mosaicPaneAction({ pendingQuestion: null, capsuleStatus: "blocked", capsuleEventId: null })).toEqual({ kind: "open-pane", capsuleEventId: null });
  });
  it("17d: nothing pending -> none", () => {
    expect(mosaicPaneAction({ pendingQuestion: null, capsuleStatus: null, capsuleEventId: null })).toEqual({ kind: "none" });
  });
  it("an empty-text single question falls to respond, not options (17a requires non-empty text)", () => {
    const emptyQ = { ...simpleQ, questions: [{ ...simpleQ.questions[0], question: "   " }] };
    expect(mosaicPaneAction({ pendingQuestion: emptyQ, capsuleStatus: null, capsuleEventId: null })).toEqual({ kind: "respond" });
  });
  it("takes only pane-shaped fields — no MissionCard, no gate/mergeNow leak possible by construction", () => {
    // mosaicPaneAction's parameter type is
    // Pick<MissionPane, "pendingQuestion" | "capsuleStatus" | "capsuleEventId">; there is no way
    // to pass a MissionCard's gate/mergeNow into it, so a card in "awaiting-approval" can never
    // surface its mergeNow button on an unrelated pane's cell (unlike the card-scoped
    // pendingUiAction/selectPendingAction, spec §8 Decision 17 intro).
    expect(mosaicPaneAction({ pendingQuestion: null, capsuleStatus: null, capsuleEventId: null })).toEqual({ kind: "none" });
  });
});

describe("buildMission — capsuleMinutes/capsuleDeclaredAt projected onto MissionPane and the codex arm (spec item 5, round 3)", () => {
  it("a waiting pane's capsule minutes/declaredAt reach the client card verbatim", () => {
    const capsule = { status: "waiting", minutes: 30, declaredAt: "2026-09-20T10:00:00.000Z", eventId: 1, mergeAsk: null, question: null };
    const scan = scanOf([mkProject({ dir: "p1" })]);
    const hub = mkHub({ p1: mkHubProject({ panes: [mkPane({ pane: "%1", capsule })] }) });
    const mission = buildMission(scan, hub, undefined, Date.parse("2026-09-20T10:05:00.000Z"));
    const pane = mission.model.cards[0].sessions[0];
    if (pane.transport !== "tmux") throw new Error("expected a tmux session");
    expect(pane.panes[0].capsuleMinutes).toBe(30);
    expect(pane.panes[0].capsuleDeclaredAt).toBe("2026-09-20T10:00:00.000Z");
  });
  it("a pane with no capsule at all projects both fields as null", () => {
    const scan = scanOf([mkProject({ dir: "p1" })]);
    const hub = mkHub({ p1: mkHubProject({ panes: [mkPane({ pane: "%1", capsule: null })] }) });
    const mission = buildMission(scan, hub, undefined, Date.now());
    const pane = mission.model.cards[0].sessions[0];
    if (pane.transport !== "tmux") throw new Error("expected a tmux session");
    expect(pane.panes[0].capsuleMinutes).toBeNull();
    expect(pane.panes[0].capsuleDeclaredAt).toBeNull();
  });
});

describe("buildMission — MissionCardPr.newest (spec item 12, round 3)", () => {
  const prResult = (prs: import("../server/collectors/github").Pr[]) => ({ p1: { ok: true as const, prs, truncated: false } });
  const pr = (n: number, updatedAt: string, ciState: import("../server/collectors/github").CiState = "green") =>
    ({ number: n, title: `PR ${n}`, url: `https://x/${n}`, isDraft: false, ciState, mergeReady: true, updatedAt });
  it("picks the PR with the latest valid updatedAt as newest", () => {
    const scan = scanOf([mkProject({ dir: "p1" })]);
    const hub = mkHub({ p1: mkHubProject() });
    const prs: Envelope<Record<string, PrResult>> = { ok: true, data: prResult([pr(1, "2026-09-19T00:00:00.000Z"), pr(2, "2026-09-20T00:00:00.000Z")]) };
    const mission = buildMission(scan, hub, prs, Date.now());
    const card = mission.model.cards[0].pr;
    if (card.status !== "ok") throw new Error("expected ok");
    expect(card.newest).toEqual({ number: 2, ci: "green" });
  });
  it("ties on updatedAt break toward the higher PR number", () => {
    const scan = scanOf([mkProject({ dir: "p1" })]);
    const hub = mkHub({ p1: mkHubProject() });
    const prs: Envelope<Record<string, PrResult>> = { ok: true, data: prResult([pr(5, "2026-09-20T00:00:00.000Z"), pr(3, "2026-09-20T00:00:00.000Z")]) };
    const mission = buildMission(scan, hub, prs, Date.now());
    const card = mission.model.cards[0].pr;
    if (card.status !== "ok") throw new Error("expected ok");
    expect(card.newest?.number).toBe(5);
  });
  it("every PR missing/invalid updatedAt falls back to the highest number, never null", () => {
    const scan = scanOf([mkProject({ dir: "p1" })]);
    const hub = mkHub({ p1: mkHubProject() });
    const prs: Envelope<Record<string, PrResult>> = { ok: true, data: prResult([pr(1, "not-a-date"), pr(9, "")]) };
    const mission = buildMission(scan, hub, prs, Date.now());
    const card = mission.model.cards[0].pr;
    if (card.status !== "ok") throw new Error("expected ok");
    expect(card.newest?.number).toBe(9);
  });
  it("an empty PR list gives newest: null", () => {
    const scan = scanOf([mkProject({ dir: "p1" })]);
    const hub = mkHub({ p1: mkHubProject() });
    const prs: Envelope<Record<string, PrResult>> = { ok: true, data: prResult([]) };
    const mission = buildMission(scan, hub, prs, Date.now());
    const card = mission.model.cards[0].pr;
    if (card.status !== "ok") throw new Error("expected ok");
    expect(card.newest).toBeNull();
  });
});

describe("newestPr (direct, spec item 12 — Global Constraints: every pure helper is exported and directly tested)", () => {
  // Add `newestPr` to this file's own `./mission` import line at the top (Step 1's anchor list
  // above only covers the anchors this task edits in mission.ts itself — this is a one-name
  // addition to the existing `import { ... } from "./mission"` line already at the top of this file).
  const pr = (n: number, updatedAt: string, ciState: import("../server/collectors/github").CiState = "green") =>
    ({ number: n, title: `PR ${n}`, url: `https://x/${n}`, isDraft: false, ciState, mergeReady: true, updatedAt });
  it("picks the PR with the latest valid updatedAt", () => {
    expect(newestPr([pr(1, "2026-09-19T00:00:00.000Z"), pr(2, "2026-09-20T00:00:00.000Z")])).toEqual({ number: 2, ci: "green" });
  });
  it("a tie on updatedAt breaks toward the higher number", () => {
    expect(newestPr([pr(5, "2026-09-20T00:00:00.000Z"), pr(3, "2026-09-20T00:00:00.000Z")])?.number).toBe(5);
  });
  it("every PR with an invalid/missing updatedAt falls back to the highest number", () => {
    expect(newestPr([pr(1, "not-a-date"), pr(9, "")])?.number).toBe(9);
  });
  it("an empty list returns null", () => {
    expect(newestPr([])).toBeNull();
  });
});

describe("statusPillSuffix (spec item 3, round 3)", () => {
  const NOW3 = Date.parse("2026-09-20T12:00:00.000Z");
  it("activeRun.startedAt wins over any session timestamp", () => {
    const card: MissionCard = { ...BASE, activeRun: { runId: "r", startedAt: "2026-09-20T11:42:00.000Z", kind: "build", runtime: "claude", target: "x", count: 1, lastStep: null, stage: "build", descriptionKey: "building", targetKind: "branch" } };
    expect(statusPillSuffix(card, NOW3)).toEqual({ minutes: 18, subagentCount: 0 });
  });
  it("no activeRun falls back to the newest session lastEventTs across panes/codex sessions", () => {
    const card: MissionCard = { ...BASE, activeRun: null, sessions: [
      { transport: "tmux", session: "s", panes: [pane("%1", { lastEventTs: "2026-09-20T11:52:00.000Z" })] },
      codexSession("t1", { lastEventTs: "2026-09-20T11:45:00.000Z" }),
    ] };
    expect(statusPillSuffix(card, NOW3).minutes).toBe(8);
  });
  it("neither present gives minutes: null", () => {
    expect(statusPillSuffix(BASE, NOW3).minutes).toBeNull();
  });
  it("subagentCount sums across every pane and codex session on the card", () => {
    const card: MissionCard = { ...BASE, sessions: [
      { transport: "tmux", session: "s", panes: [pane("%1", { subagentCount: 2 }), pane("%2", { subagentCount: 1 })] },
      codexSession("t1", { subagentCount: 3 }),
    ] };
    expect(statusPillSuffix(card, NOW3).subagentCount).toBe(6);
  });
  it("both minutes and subagentCount can be set together", () => {
    const card: MissionCard = {
      ...BASE, activeRun: { runId: "r", startedAt: "2026-09-20T11:30:00.000Z", kind: "build", runtime: "claude", target: "x", count: 1, lastStep: null, stage: "build", descriptionKey: "building", targetKind: "branch" },
      sessions: [{ transport: "tmux", session: "s", panes: [pane("%1", { subagentCount: 2 })] }],
    };
    expect(statusPillSuffix(card, NOW3)).toEqual({ minutes: 30, subagentCount: 2 });
  });
  // F3: a non-finite first pane timestamp used to latch newestTs and block every later valid
  // one from ever winning (NaN comparisons are always false).
  it("skips a non-finite pane timestamp and keeps the max finite one, whatever order they arrive in", () => {
    const card: MissionCard = { ...BASE, activeRun: null, sessions: [
      { transport: "tmux", session: "s", panes: [
        pane("%1", { lastEventTs: "not-a-date" }),
        pane("%2", { lastEventTs: "2026-09-20T11:52:00.000Z" }),
      ] },
    ] };
    expect(statusPillSuffix(card, NOW3).minutes).toBe(8);
  });
  // T2: elapsed minutes only render for the headlines where "elapsed" is meaningful.
  it("returns minutes: null for idle/waiting/unknown headlines even with a session timestamp, but keeps working/stuck/needs-you", () => {
    const withTs: MissionCard = { ...BASE, activeRun: null, sessions: [
      { transport: "tmux", session: "s", panes: [pane("%1", { lastEventTs: "2026-09-20T11:52:00.000Z" })] },
    ] };
    for (const headline of ["idle", "waiting", "unknown"] as const) {
      expect(statusPillSuffix({ ...withTs, headline }, NOW3).minutes).toBeNull();
    }
    for (const headline of ["working", "stuck", "needs-you"] as const) {
      expect(statusPillSuffix({ ...withTs, headline }, NOW3).minutes).toBe(8);
    }
  });
});

describe("worstPaneState (F2, round 3)", () => {
  it("picks the worst state in needs-you > stuck > working > waiting > idle > unknown order", () => {
    expect(worstPaneState([pane("%1", { state: "idle" }), pane("%2", { state: "needs-you" }), pane("%3", { state: "working" })])).toBe("needs-you");
    expect(worstPaneState([pane("%1", { state: "waiting" }), pane("%2", { state: "stuck" })])).toBe("stuck");
    expect(worstPaneState([pane("%1", { state: "unknown" }), pane("%2", { state: "idle" })])).toBe("idle");
    expect(worstPaneState([pane("%1", { state: "working" })])).toBe("working");
  });
});

describe("elapsedMinutesSince (spec item 5, round 3)", () => {
  const NOW3 = Date.parse("2026-09-20T12:00:00.000Z");
  it("clamps a future timestamp to 0, returns null on an invalid string", () => {
    expect(elapsedMinutesSince("2026-09-20T12:05:00.000Z", NOW3)).toBe(0);
    expect(elapsedMinutesSince("not-a-date", NOW3)).toBeNull();
  });
  it("computes whole minutes elapsed for a past timestamp", () => {
    expect(elapsedMinutesSince("2026-09-20T11:45:30.000Z", NOW3)).toBe(14);
  });
});

describe("stuckLine (spec item 5, round 3)", () => {
  const NOW3 = Date.parse("2026-09-20T12:00:00.000Z");
  it("rung 1: a valid declared-minutes capsule on the first stuck pane wins", () => {
    const card: MissionCard = { ...BASE, headline: "stuck", sessions: [{ transport: "tmux", session: "s", panes: [pane("%1", { state: "stuck" as MissionState, capsuleMinutes: 20, capsuleDeclaredAt: "2026-09-20T11:30:00.000Z" })] }] };
    expect(stuckLine(card, NOW3)).toEqual({ key: "stuckLine", declared: 20, elapsed: 30 });
  });
  it("rung 2: an invalid declaredAt falls through to the newest session lastEventTs", () => {
    const card: MissionCard = { ...BASE, headline: "stuck", sessions: [{ transport: "tmux", session: "s", panes: [pane("%1", { state: "stuck" as MissionState, capsuleMinutes: 20, capsuleDeclaredAt: "not-a-date", lastEventTs: "2026-09-20T11:50:00.000Z" })] }] };
    expect(stuckLine(card, NOW3)).toEqual({ key: "stuckLineFallback", elapsed: 10 });
  });
  it("rung 2: a future declaredAt (valid but after nowMs) also falls through, never clamped to 0 in place", () => {
    const card: MissionCard = { ...BASE, headline: "stuck", sessions: [{ transport: "tmux", session: "s", panes: [pane("%1", { state: "stuck" as MissionState, capsuleMinutes: 20, capsuleDeclaredAt: "2026-09-20T12:10:00.000Z", lastEventTs: "2026-09-20T11:50:00.000Z" })] }] };
    expect(stuckLine(card, NOW3)).toEqual({ key: "stuckLineFallback", elapsed: 10 });
  });
  it("rung 3: neither declared nor session timestamp falls to card.updated", () => {
    const card: MissionCard = { ...BASE, headline: "stuck", updated: "2026-09-20T11:00:00.000Z", sessions: [] };
    expect(stuckLine(card, NOW3)).toEqual({ key: "stuckLineFallback", elapsed: 60 });
  });
  it("rung 4: none of the three present returns null", () => {
    const card: MissionCard = { ...BASE, headline: "stuck", updated: "not-a-date", sessions: [] };
    expect(stuckLine(card, NOW3)).toBeNull();
  });
  // F7 (cold review 92a14d784fb3): a non-stuck pane must never contribute its own capsule data
  // to rung 1 — only the FIRST pane/session whose state is "stuck" counts, even if an earlier,
  // non-stuck pane happens to carry capsule fields.
  it("a non-stuck pane's own capsule data is ignored — only the first stuck pane's own capsule counts", () => {
    const card: MissionCard = { ...BASE, headline: "stuck", sessions: [
      { transport: "tmux", session: "s1", panes: [pane("%1", { state: "working" as MissionState, capsuleMinutes: 99, capsuleDeclaredAt: "2026-09-20T10:00:00.000Z" })] },
      { transport: "tmux", session: "s2", panes: [pane("%2", { state: "stuck" as MissionState, capsuleMinutes: 20, capsuleDeclaredAt: "2026-09-20T11:30:00.000Z" })] },
    ] };
    expect(stuckLine(card, NOW3)).toEqual({ key: "stuckLine", declared: 20, elapsed: 30 });
  });
});

describe("gateWaitingLine (spec item 5, round 3)", () => {
  const NOW3 = Date.parse("2026-09-20T12:00:00.000Z");
  // Tech-lead correction after the first worktree render: the gate is set by status.md, so its
  // age is card.updated — the same timestamp the inbox gate row shows. Session activity is the
  // tech lead WORKING on the approval, not the wait itself (it rendered "há 0" while the strip
  // said "há 13 minutos").
  it("uses card.updated, ignoring newer session activity", () => {
    const card: MissionCard = { ...BASE, headline: "needs-you", gate: "awaiting-approval", updated: "2026-09-20T11:40:00.000Z", sessions: [{ transport: "tmux", session: "s", panes: [pane("%1", { lastEventTs: "2026-09-20T11:59:00.000Z" })] }] };
    expect(gateWaitingLine(card, NOW3)).toEqual({ key: "gateWaitingLine", elapsed: 20 });
  });
  it("unparseable card.updated returns null", () => {
    const card: MissionCard = { ...BASE, headline: "needs-you", gate: "awaiting-approval", updated: "not-a-date", sessions: [] };
    expect(gateWaitingLine(card, NOW3)).toBeNull();
  });
});

describe("heartbeatPath (spec item 10, round 3)", () => {
  it("returns a non-empty M…L… path with one point per bucket", () => {
    const minutes = Array.from({ length: 60 }, (_, i) => (i === 30 ? 5 : 0));
    const path = heartbeatPath(minutes);
    expect(path.startsWith("M")).toBe(true);
    expect((path.match(/L/g) ?? []).length).toBe(59);
  });
});

describe("worstFinding (spec item 11, round 3)", () => {
  it("high wins when non-zero", () => {
    expect(worstFinding({ high: 1, medium: 2, low: 3 })).toEqual({ severity: "high", count: 1 });
  });
  it("medium wins when high is 0", () => {
    expect(worstFinding({ high: 0, medium: 2, low: 3 })).toEqual({ severity: "medium", count: 2 });
  });
  it("low wins when both high and medium are 0", () => {
    expect(worstFinding({ high: 0, medium: 0, low: 3 })).toEqual({ severity: "low", count: 3 });
  });
  it("null when all three are 0, and when findings itself is null", () => {
    expect(worstFinding({ high: 0, medium: 0, low: 0 })).toBeNull();
    expect(worstFinding(null)).toBeNull();
  });
});

describe("dispatchBodyFor (spec item 8, round 3)", () => {
  const fields: DispatchFields = { command: "review", kind: "diff", target: "abc123def456", focus: "", plan: "", phase: "", branch: "", whitelist: "", verify: "", build: "", profile: "default" };
  it("reviewSpec/reviewPlan/reviewDiff map to command:review with the matching kind", () => {
    expect(dispatchBodyFor("reviewSpec", { ...fields, target: ".local/docs/specs/x.md" }, "p1")).toEqual({ project: "p1", command: "review", kind: "spec", target: ".local/docs/specs/x.md" });
    expect(dispatchBodyFor("reviewPlan", { ...fields, target: ".local/docs/plans/x.md" }, "p1")).toEqual({ project: "p1", command: "review", kind: "plan", target: ".local/docs/plans/x.md" });
    expect(dispatchBodyFor("reviewDiff", fields, "p1")).toEqual({ project: "p1", command: "review", kind: "diff", target: "abc123def456" });
  });
  it("focus is included only when non-empty", () => {
    expect(dispatchBodyFor("reviewDiff", { ...fields, focus: "auth" }, "p1")).toEqual({ project: "p1", command: "review", kind: "diff", target: "abc123def456", focus: "auth" });
  });
  it("build maps to command:build with every build field, profile passed through, kind absent", () => {
    const buildFields: DispatchFields = { ...fields, plan: ".local/docs/plans/x.md", phase: "x-phase", branch: "feat/x", whitelist: "src/a.ts", verify: "pnpm test", build: "pnpm build", profile: "fallback" };
    expect(dispatchBodyFor("build", buildFields, "p1")).toEqual({ project: "p1", command: "build", plan: ".local/docs/plans/x.md", phase: "x-phase", branch: "feat/x", whitelist: "src/a.ts", verify: "pnpm test", build: "pnpm build", profile: "fallback" });
  });
});

describe("messages — mission.expanded.sessionLineNoRole / mission.pit.hidden (round-3 review T3/F5)", () => {
  // T3: this template is used ONLY for the codex arm, where the caller already builds `name`
  // as `${t("codexSession")} ${prefix}` (already carries the "sessão"/"session" word) — the
  // template must not embed that word a second time (ProjectCard.tsx double-prefix bug).
  it("sessionLineNoRole has no leading session/sessão word, in either locale", () => {
    expect(en.mission.expanded.sessionLineNoRole.startsWith("{name}")).toBe(true);
    expect(pt.mission.expanded.sessionLineNoRole.startsWith("{name}")).toBe(true);
    expect(en.mission.expanded.sessionLineNoRole.toLowerCase()).not.toContain("session {name}");
    expect(pt.mission.expanded.sessionLineNoRole.toLowerCase()).not.toContain("sessão {name}");
  });
  // F5: the plural "other" variant needs the same " · " separator the "one" variant already has.
  it("pit.hidden's plural variants both carry the ' · ' separator", () => {
    expect(en.mission.pit.hidden).toContain("one { · # hidden}");
    expect(en.mission.pit.hidden).toContain("other { · # hidden}");
    expect(pt.mission.pit.hidden).toContain("one { · # oculto}");
    expect(pt.mission.pit.hidden).toContain("other { · # ocultos}");
  });
});

describe("liveSessions", () => {
  const pane = (pane: string, live: boolean) => ({ pane, live }) as never;
  it("keeps only live tmux panes and loaded codex threads, drops sessions left empty", () => {
    const out = liveSessions([
      { transport: "tmux", session: "jax-os", panes: [pane("%1", true), pane("%15", false)] },
      { transport: "tmux", session: "jax-os-lead", panes: [pane("%160", false)] },
      { transport: "codex", threadId: "t1", state: "idle" } as never,
      { transport: "codex", threadId: "t2", state: "unknown" } as never,
    ]);
    expect(out.map((s) => (s.transport === "tmux" ? `${s.session}:${s.panes.map((p) => p.pane).join(",")}` : "codex"))).toEqual(["jax-os:%1", "codex"]);
  });
});
