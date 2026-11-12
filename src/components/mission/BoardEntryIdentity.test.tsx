import { Children, isValidElement, type ReactElement, type ReactNode } from "react";
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

import type { BoardEntry, MissionCard } from "@/lib/mission";
import { ProjectCard } from "./ProjectCard";
import { renderBoardEntry } from "./boardEntry";

const base = (over: Partial<MissionCard>): MissionCard => ({
  dir: "p", name: "p", stage: "build", branch: "main", builder: "claude-code", gate: null,
  legacyGateField: false, now: "", residuals: [], updated: "", headline: "idle", hubReady: true,
  lastEventAgo: null, freshnessCount: null, freshnessWarning: false, sessions: [], pendingQuestions: [],
  timeline: [], pr: { status: "unknown" }, codexSource: "ok", codexUntracked: false, heartbeat: [],
  lastAction: null, prefs: { pinned: false, hiddenAt: null }, visibility: "visible", ...over,
});
const seat = (id: string): MissionCard => base({ key: `p#session:${id}`, scope: { kind: "session", sessionId: id, transport: "tmux", runtime: "claude", name: null, tmuxSessions: [], panes: [] } as unknown as MissionCard["scope"] });
const A = seat("A");
const B = seat("B");
const plain: BoardEntry = { kind: "plain", card: A };
const group: BoardEntry = {
  kind: "group", project: base({ group: { cards: [A, B], footer: { retainedWorktrees: 1, unownedLastRun: null } } }), cards: [A, B],
};
const opts = { expandedDir: null, onToggleExpand: () => {}, counts: null };

type El = ReactElement<{ children?: ReactNode }>;
const kids = (e: El): ReactNode[] => Children.toArray(e.props.children).length === 0 ? [] : (Array.isArray(e.props.children) ? e.props.children : [e.props.children]);
// Path of [type, key] pairs from the outer element down to the ProjectCard keyed `target`.
function pathTo(e: El, target: string, acc: [unknown, unknown][] = []): [unknown, unknown][] | null {
  const here: [unknown, unknown][] = [...acc, [e.type, e.key]];
  if (e.type === ProjectCard && e.key === target) return here;
  for (const c of kids(e).flatMap((k) => (Array.isArray(k) ? k : [k]))) {
    if (isValidElement(c)) { const r = pathTo(c as El, target, here); if (r) return r; }
  }
  return null;
}

describe("renderBoardEntry keeps a session card's React identity across plain and group (review F3)", () => {
  const p = renderBoardEntry(plain, opts) as El;
  const g = renderBoardEntry(group, opts) as El;

  it("outer wrappers share key (project dir) and type", () => {
    expect(p.key).toBe("p");
    expect(g.key).toBe("p");
    expect(p.type).toBe(g.type);
  });

  it("the type/key path down to A's ProjectCard is identical", () => {
    const pp = pathTo(p, "p#session:A");
    const gp = pathTo(g, "p#session:A");
    expect(pp).not.toBeNull();
    expect(gp).toEqual(pp);
  });

  it("header and footer slots are null in plain mode, set in group mode; container at the same index", () => {
    const [ph, pc, pf] = kids(p);
    const [gh, gc, gf] = kids(g);
    expect(ph).toBeNull();
    expect(pf).toBeNull();
    expect(gh).not.toBeNull();
    expect(gf).not.toBeNull();
    expect(kids(p)).toHaveLength(3);
    expect(kids(g)).toHaveLength(3);
    expect((pc as El).type).toBe((gc as El).type);
  });

  it("plain markup keeps the wrapper attributes and basis classes", () => {
    const html = renderToStaticMarkup(p);
    expect(html).toContain('data-card-dir="p"');
    expect(html).toContain('data-card-key="p#session:A"');
    expect(html).toContain("basis-full sm:basis-[45%] lg:basis-[30%]");
  });
});

describe("group grid: two cards per row, an odd last card takes the whole row", () => {
  const C = seat("C");
  const layout = (cards: MissionCard[]) => {
    const grid = kids(renderBoardEntry({ ...group, cards } as BoardEntry, opts) as El)[1] as ReactElement<{ className: string; children?: ReactNode }>;
    const spans = (kids(grid as El) as ReactElement<{ rootAttrs: { className: string } }>[]).map((c) => c.props.rootAttrs.className);
    return { grid: grid.props.className, spans };
  };

  it("uses two columns", () => {
    expect(layout([A, B]).grid).toContain("sm:grid-cols-2");
  });

  it("two cards share one row", () => {
    expect(layout([A, B]).spans).toEqual(["", ""]);
  });

  it("the third card stretches across its row", () => {
    expect(layout([A, B, C]).spans).toEqual(["", "", "sm:col-span-full"]);
  });
});
