import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

// Plan review round 1 (72cd1de344ad) F3: the sanctioned shape used across this codebase's
// static-render tests (e.g. `ProjectCard.test.tsx`, `ProjectsColumn.test.tsx`, `AfkToggle.test.tsx`)
// — `useTranslations` ignores its namespace argument and returns the key verbatim, so an
// assertion below reads the bare key ("empty"/"title"/"ci.green"), never the real copy.
// `useLocale` is also required: the real `RelativeTime` child component reads it, and every
// sibling static-render test mocks the same pair (`ProjectCard.test.tsx` et al).
// Round-2 F2: a call with no `values` still returns the bare key (existing assertions
// unaffected); a call with values appends them for the value-bearing sentence test.
vi.mock("next-intl", () => ({ useTranslations: () => (key: string, values?: Record<string, string>) => (values ? `${key}:${JSON.stringify(values)}` : key), useLocale: () => "en-US" })); // replaces this file's existing next-intl mock
const sentenceOverride = vi.hoisted(() => ({ current: null as null | ((e: unknown) => { key: string; values?: Record<string, string> }) }));
vi.mock("@/lib/mission", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/mission")>();
  return { ...actual, tickerSentence: (e: unknown) => sentenceOverride.current?.(e) ?? actual.tickerSentence(e as Parameters<typeof actual.tickerSentence>[0]) };
});

import type { TickerItem } from "@/lib/mission";
import { EventTicker } from "./EventTicker";

describe("EventTicker", () => {
  it("renders one line per item", () => {
    const items: TickerItem[] = [
      { source: "commit", dir: "p1", name: "P1", ts: "2026-09-19T10:00:00.000Z", author: "rafa", message: "fix: x", hash: "abc1234" },
      { source: "pr", dir: "p1", name: "P1", ts: "2026-09-19T09:00:00.000Z", number: 1, url: "https://x", ci: "green", title: "T" },
    ];
    const html = renderToStaticMarkup(createElement(EventTicker, { items }));
    expect(html).toContain("fix: x");
    expect(html).toContain("#1");
    expect(html).toContain("ci.green");
  });

  it("renders an empty hint with no items", () => {
    expect(renderToStaticMarkup(createElement(EventTicker, { items: [] }))).toContain("empty");
  });

  // Decision 19 supersedes the raw-field render: an "event" item now renders its ticker
  // sentence (a role-less outcome falls to row 9's catch-all) plus the diagnostic suffix.
  it("renders an event item's ticker sentence and diagnostic, not the raw TimelineEntryText fields", () => {
    const items: TickerItem[] = [
      {
        source: "event", dir: "p1", name: "P1", ts: "2026-09-19T10:00:00.000Z",
        type: "run-finished", pane: null, capsuleStatus: null, role: null, outcome: "failure",
        contractStatus: "ok", stage: "build", diagnostic: "pnpm build exit 1",
      },
    ];
    const html = renderToStaticMarkup(createElement(EventTicker, { items }));
    expect(html).toContain("sentence.runFinishedNoResult");
    expect(html).toContain("pnpm build exit 1");
    expect(html).not.toContain("run-finished");
  });
});

describe("EventTicker — sentences and a fixed clock (spec §9, Decisions 19/20)", () => {
  const runFinished = { source: "event" as const, dir: "p1", name: "P1", ts: "2026-09-19T21:41:12.000Z", type: "run-finished" as const, pane: null, capsuleStatus: null, contractStatus: "ok" as const, stage: null, role: "builder" as const, diagnostic: null };
  it("an event item renders its ticker sentence key and an HH:MM:SS clock, not TimelineEntryText's raw fields", () => {
    const html = renderToStaticMarkup(createElement(EventTicker, { items: [{ ...runFinished, outcome: "success" }] }));
    expect(html).toContain("sentence.buildOk"); // t() echoes the key verbatim, scoped to "mission.ticker" — "sentence.buildOk", not "ticker.sentence.buildOk"
    // Plan deviation: formatClock is host-local by design (Part 1 round-2 F1 documents the host
    // TZ America/Sao_Paulo), so the clock is computed with the SAME Intl call instead of the
    // plan's hardcoded UTC "21:41:12", which reads 18:41:12 here.
    const clock = new Intl.DateTimeFormat("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(Date.parse(runFinished.ts));
    expect(html).toMatch(/\d{2}:\d{2}:\d{2}/);
    expect(html).toContain(clock);
    expect(html).not.toContain("run-finished"); // TimelineEntryText's raw type string
  });
  it("round-2 F2: a value-bearing sentence passes .values through to t(), not just .key", () => {
    sentenceOverride.current = () => ({ key: "sentence.buildFailed", values: { findings: "3" } });
    const html = renderToStaticMarkup(createElement(EventTicker, { items: [{ ...runFinished, outcome: "reject" }] }));
    sentenceOverride.current = null;
    // React escapes `"` in text nodes to `&quot;` — the JSON stays intact, only HTML-encoded.
    expect(html).toContain("sentence.buildFailed:{&quot;findings&quot;:&quot;3&quot;}");
  });
});

describe("EventTicker — paging (Rafa, 2026-09-19)", () => {
  const runFinished = { source: "event" as const, dir: "p1", name: "P1", type: "run-finished" as const, pane: null, capsuleStatus: null, contractStatus: "ok" as const, stage: null, role: "builder" as const, outcome: "success", diagnostic: null };
  const items = (n: number): TickerItem[] =>
    Array.from({ length: n }, (_, i) => ({ ...runFinished, ts: new Date(Date.parse("2026-09-19T00:00:00.000Z") + i * 1000).toISOString() }));

  it("hides paging controls when everything fits on one page", () => {
    const html = renderToStaticMarkup(createElement(EventTicker, { items: items(5) }));
    expect(html).not.toContain("pageInfo");
  });

  it("shows 10 items on the first page plus page info when there are more", () => {
    const html = renderToStaticMarkup(createElement(EventTicker, { items: items(25) }));
    expect(html).toContain("pageInfo:{&quot;page&quot;:1,&quot;pages&quot;:3}");
  });
});

describe("EventTicker — title/subtitle + one-line rows (spec item 7)", () => {
  it("renders the subtitle alongside the title", () => {
    const html = renderToStaticMarkup(createElement(EventTicker, { items: [] }));
    expect(html).toContain("subtitle");
  });
  it("a row renders the clock, name, and sentence in one flex row, no separate clock line below", () => {
    const items: TickerItem[] = [
      { source: "commit", dir: "p1", name: "P1", ts: "2026-09-19T10:00:00.000Z", author: "rafa", message: "fix: x", hash: "abc1234" },
    ];
    const html = renderToStaticMarkup(createElement(EventTicker, { items }));
    const rowStart = html.indexOf('data-testid="ticker-row"');
    const rowOpen = html.lastIndexOf("<div", rowStart);
    const rowClose = html.indexOf("</div>", rowStart);
    const row = html.slice(rowOpen, rowClose + "</div>".length);
    // Three spans: the clock wrapper, TickerClock's own inner span, and the sentence span.
    // The plan's ≤2 count assumed no inner span; assert the actual intent instead — the clock
    // leads inline inside the row, and no separate `text-[11px] text-muted` clock line sits
    // stacked beneath the sentence.
    expect((row.match(/<span/g) ?? []).length).toBeLessThanOrEqual(3);
    const clockIdx = row.indexOf('class="flex-none text-[11px] text-muted"');
    expect(clockIdx).toBeGreaterThan(-1);
    expect(clockIdx).toBeLessThan(row.indexOf("fix: x"));
    expect(row).not.toContain('class="text-[11px] text-muted"');
  });
});
