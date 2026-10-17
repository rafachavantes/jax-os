// src/components/sessions/Mosaic.test.tsx
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string, values?: Record<string, string>) => (values ? `${key}:${JSON.stringify(values)}` : key),
  useLocale: () => "en-US",
}));
vi.mock("@/components/RelativeTime", () => ({ RelativeTime: () => null }));

import type { MissionCard, MissionPane, MissionReadiness } from "@/lib/mission";
import { Mosaic, MosaicAnswerOptions, MosaicQuestionText, mobileCellOrder } from "./Mosaic";
import { mosaicGroups, selectSpotlightPane } from "@/lib/mosaic";

const cardBase: MissionCard = {
  dir: "p1", name: "Alpha", stage: "build", branch: "main", builder: "claude-code",
  gate: null, legacyGateField: false, now: "", residuals: [], updated: "2026-09-19T10:00:00.000Z",
  headline: "working", hubReady: true, lastEventAgo: null, freshnessCount: null, freshnessWarning: false,
  sessions: [], pendingQuestions: [], timeline: [], pr: { status: "unknown" },
  codexSource: "ok", codexUntracked: false,
  heartbeat: new Array(60).fill(0), lastAction: null, prefs: { pinned: false, hiddenAt: null }, visibility: "visible",
};

function pane(overrides: Partial<MissionPane>): MissionPane {
  return {
    pane: "%1", tmuxIncarnation: "1:1", session: "s1", role: "lead", state: "working",
    lastEventTs: null, pendingQuestion: null, subagentCount: 0, capsuleEventId: null, capsuleStatus: null,
    capsuleMinutes: null, capsuleDeclaredAt: null, capsuleMergeAsk: null, capsuleQuestion: null,
    live: true, tmuxSession: "s1",
    ...overrides,
  };
}

function cardWithPanes(overrides: Partial<MissionCard>, panes: MissionPane[]): MissionCard {
  return { ...cardBase, ...overrides, sessions: [{ transport: "tmux", session: panes[0]?.session ?? "s1", panes }] };
}

const readyEmpty: MissionReadiness = { kind: "ready" };

// Module scope (round-3 F2): shared by both the "pane-scoped action rendering" describe block
// and the standalone "MosaicAnswerOptions" describe block below — declaring it inside either
// block's callback leaves it out of the other's lexical scope and fails to compile.
const single = { pane: "%1", eventId: 9, questions: [{ question: "Env?", multiSelect: false, options: [{ label: "dev" }, { label: "stg" }] }] };

describe("Mosaic — readiness gating (spec §8 D18)", () => {
  it("loading renders 3 skeleton cells, no group headers, no empty text", () => {
    const html = renderToStaticMarkup(createElement(Mosaic, {
      readiness: { kind: "loading", pending: ["projects"] }, cards: [cardWithPanes({}, [pane({})])],
      panesTail: null, busy: false, onAction: () => {}, onOpenDetail: () => {}, onOpenMobile: () => {},
    }));
    expect(html).not.toContain("Alpha");
    expect(html).not.toContain("empty");
  });
  it("failed renders SourceWarning, never the empty text", () => {
    const html = renderToStaticMarkup(createElement(Mosaic, {
      readiness: { kind: "failed", failed: ["hub"], pending: [] }, cards: [],
      panesTail: null, busy: false, onAction: () => {}, onOpenDetail: () => {}, onOpenMobile: () => {},
    }));
    expect(html).toContain("unavailable");
    expect(html).not.toContain("mosaic.empty");
  });
  it("ready with zero live panes anywhere renders the empty string", () => {
    const html = renderToStaticMarkup(createElement(Mosaic, {
      readiness: readyEmpty, cards: [cardBase], panesTail: null, busy: false,
      onAction: () => {}, onOpenDetail: () => {}, onOpenMobile: () => {},
    }));
    // Plan correction: the mocked useTranslations returns the bare key, so the empty state
    // renders "empty" (namespace-stripped), never the literal "mosaic.empty" the plan asserted.
    expect(html).toContain("empty");
  });
});

describe("Mosaic — grouping and stale-pane exclusion (spec §5 rows 1-3, round-1 F1)", () => {
  it("groups cells by project; a card with zero live panes gets no header/cell", () => {
    const cards = [
      cardWithPanes({ dir: "p1", name: "Alpha" }, [pane({ pane: "%1" })]),
      cardWithPanes({ dir: "p2", name: "Beta" }, [pane({ pane: "%2", live: false, pendingQuestion: { pane: "%2", eventId: 1, questions: [{ question: "Q?", multiSelect: false, options: [{ label: "A" }] }] } })]),
    ];
    const html = renderToStaticMarkup(createElement(Mosaic, {
      readiness: readyEmpty, cards, panesTail: null, busy: false,
      onAction: () => {}, onOpenDetail: () => {}, onOpenMobile: () => {},
    }));
    expect(html).toContain("Alpha");
    expect(html).not.toContain("Beta");
  });
});

describe("Mosaic — spotlight span/dim (spec §8 D14, round-2 F3)", () => {
  it("exactly one spotlight cell dims every other cell but keeps its own bar color", () => {
    const needsYou = pane({ pane: "%1", state: "needs-you", capsuleStatus: "blocked" });
    const idle = pane({ pane: "%2", state: "idle", tmuxSession: "s2" });
    const cards = [cardWithPanes({ dir: "p1", name: "Alpha" }, [needsYou, idle])];
    const html = renderToStaticMarkup(createElement(Mosaic, {
      readiness: readyEmpty, cards, panesTail: null, busy: false,
      onAction: () => {}, onOpenDetail: () => {}, onOpenMobile: () => {},
    }));
    expect(html).toContain('data-spotlight="true"');
    expect(html).toContain("opacity-[.55]");
  });
});

describe("Mosaic — tmuxSession === null (spec §8 D13a, round-2 F1)", () => {
  it("renders lines/state bar but a muted unknownSession caption, no detail click, no Take Control path", () => {
    const p = pane({ pane: "%1", tmuxSession: null });
    const cards = [cardWithPanes({ dir: "p1", name: "Alpha" }, [p])];
    let opened: string | null = null;
    const html = renderToStaticMarkup(createElement(Mosaic, {
      readiness: readyEmpty, cards, panesTail: null, busy: false,
      onAction: () => {}, onOpenDetail: (s) => { opened = s; }, onOpenMobile: () => {},
    }));
    expect(html).toContain("unknownSession");
    expect(html).not.toContain("openPane");
    expect(html).not.toContain('role="button"');
    expect(opened).toBeNull();
  });
});

describe("Mosaic — pane-scoped action rendering (spec §8 rows 17a-17d)", () => {
  const multi = { pane: "%1", eventId: 9, questions: [{ question: "Q1?", multiSelect: true, options: [{ label: "a" }] }, { question: "Q2?", multiSelect: false, options: [{ label: "b" }] }] };
  function renderPane(p: MissionPane) {
    return renderToStaticMarkup(createElement(Mosaic, {
      readiness: readyEmpty, cards: [cardWithPanes({}, [p])], panesTail: null, busy: false,
      onAction: () => {}, onOpenDetail: () => {}, onOpenMobile: () => {},
    }));
  }
  it("17a: a simple single-select renders numbered options, no Responder button", () => {
    const html = renderPane(pane({ pane: "%1", state: "needs-you", pendingQuestion: single }));
    expect(html).toContain("dev");
    expect(html).not.toContain("actions.respond");
  });
  it("17b: a multi-question pendingQuestion renders the Responder button, no numbered options", () => {
    const html = renderPane(pane({ pane: "%1", state: "needs-you", pendingQuestion: multi }));
    expect(html).toContain("actions.respond");
  });
  it("17c: a bare blocked capsule renders status text and no button", () => {
    const html = renderPane(pane({ pane: "%1", state: "needs-you", capsuleStatus: "blocked" }));
    expect(html).toContain("inbox.blocked");
    expect(html).not.toContain("actions.respond");
  });
  it("17d: nothing pending renders no action controls", () => {
    const html = renderPane(pane({ pane: "%1", state: "idle" }));
    expect(html).not.toContain("actions.respond");
    expect(html).not.toContain("inbox.blocked");
  });
});

describe("Mosaic — lines preview (spec §7 row 11, round-1 F5)", () => {
  it("a pane with lines: null renders the muted noOutput line, never an empty/broken cell", () => {
    const p = pane({ pane: "%1" });
    const html = renderToStaticMarkup(createElement(Mosaic, {
      readiness: readyEmpty, cards: [cardWithPanes({}, [p])], panesTail: [{ pane: "%1", lines: null }], busy: false,
      onAction: () => {}, onOpenDetail: () => {}, onOpenMobile: () => {},
    }));
    expect(html).toContain("noOutput");
  });
  it("a pane with real lines renders them, sliced to 4 (desktop grid)", () => {
    const p = pane({ pane: "%1" });
    const html = renderToStaticMarkup(createElement(Mosaic, {
      readiness: readyEmpty, cards: [cardWithPanes({}, [p])],
      panesTail: [{ pane: "%1", lines: ["a", "b", "c", "d", "e", "f"] }], busy: false,
      onAction: () => {}, onOpenDetail: () => {}, onOpenMobile: () => {},
    }));
    // Plan correction: the plan asserted `html).not.toContain("e")`, but class names alone
    // ("border", "flex") contain single letters, so the whole-document assertion can never pass.
    // Extracting the <pre> body tests the same claim (slice(0, 4) drops the 5th line) without
    // matching markup.
    const preview = html.match(/<pre[^>]*>([\s\S]*?)<\/pre>/)?.[1] ?? "";
    expect(preview).toContain("a");
    expect(preview).toContain("d");
    expect(preview).not.toContain("e");
  });
});

describe("mobileCellOrder (spec §9 D19, direct test)", () => {
  it("moves the spotlight entry to the front, keeps the rest in traversal order", () => {
    const p1 = pane({ pane: "%1" });
    const p2 = pane({ pane: "%2", state: "needs-you", capsuleStatus: "blocked" });
    const groups = mosaicGroups([
      cardWithPanes({ dir: "p1", name: "Alpha" }, [p1, p2]),
    ]);
    const spotlight = selectSpotlightPane(groups);
    const order = mobileCellOrder(groups, spotlight);
    expect(order.map((e) => e.pane.pane)).toEqual(["%2", "%1"]);
  });
  it("no spotlight leaves traversal order untouched", () => {
    const p1 = pane({ pane: "%1" });
    const groups = mosaicGroups([cardWithPanes({ dir: "p1", name: "Alpha" }, [p1])]);
    expect(mobileCellOrder(groups, null).map((e) => e.pane.pane)).toEqual(["%1"]);
  });
});

describe("MosaicAnswerOptions (exported, reused by Task 2)", () => {
  it("renders one button per option, default (grid-cell) size has no 44px minimum", () => {
    const html = renderToStaticMarkup(createElement(MosaicAnswerOptions, { pendingQuestion: single, busy: false, onAction: () => {} }));
    expect(html).toContain("dev");
    expect(html).toContain("stg");
    expect(html).not.toContain("min-h-11");
  });
  it("size='mobile' renders the 44px minimum target (spec §10 D26, round-3 F5)", () => {
    const html = renderToStaticMarkup(createElement(MosaicAnswerOptions, { pendingQuestion: single, busy: false, onAction: () => {}, size: "mobile" }));
    expect(html).toContain("min-h-11");
    expect(html).toContain("min-w-11");
  });
  it("labels each option with its 1-based index (round-4 F1)", () => {
    const html = renderToStaticMarkup(createElement(MosaicAnswerOptions, { pendingQuestion: single, busy: false, onAction: () => {} }));
    expect(html).toContain("1 · dev");
    expect(html).toContain("2 · stg");
  });
});

describe("Mosaic — pending question text (round-4 F2)", () => {
  function renderSingleQuestionCell(question: string) {
    const withQuestion = { ...single, questions: [{ ...single.questions[0], question }] };
    const p = pane({ pane: "%1", state: "needs-you", pendingQuestion: withQuestion });
    return renderToStaticMarkup(createElement(Mosaic, {
      readiness: readyEmpty, cards: [cardWithPanes({}, [p])], panesTail: null, busy: false,
      onAction: () => {}, onOpenDetail: () => {}, onOpenMobile: () => {},
    }));
  }
  it("renders the trimmed question text above the numbered options", () => {
    const html = renderSingleQuestionCell("  Env?  ");
    expect(html).toContain("Env?");
    expect(html).toContain("1 · dev");
  });
  it("MosaicQuestionText renders nothing for a blank question (direct render)", () => {
    const blank = { ...single, questions: [{ ...single.questions[0], question: "   " }] };
    const html = renderToStaticMarkup(createElement(MosaicQuestionText, { pendingQuestion: blank }));
    expect(html).toBe("");
  });
});

describe("Mosaic — mobile 44px cell target (round-4 F3)", () => {
  it("cell container carries the mobile min-h-11, reset back to 0 at the desktop (lg) breakpoint", () => {
    const html = renderToStaticMarkup(createElement(Mosaic, {
      readiness: readyEmpty, cards: [cardWithPanes({}, [pane({})])], panesTail: null, busy: false,
      onAction: () => {}, onOpenDetail: () => {}, onOpenMobile: () => {},
    }));
    expect(html).toContain("min-h-11 lg:min-h-0");
  });
});
