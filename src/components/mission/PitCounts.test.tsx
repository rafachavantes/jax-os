import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("next-intl", () => ({ useTranslations: () => (key: string, values?: Record<string, unknown>) => (values ? `${key}:${JSON.stringify(values)}` : key) }));

import type { MissionCounts } from "@/lib/mission";
import { PitCounts } from "./PitCounts";

const COUNTS: MissionCounts = { "needs-you": 3, stuck: 1, working: 2, idle: 4 };

describe("PitCounts — count-first order + hidden dot (spec item 6)", () => {
  it("renders count before the label", () => {
    const html = renderToStaticMarkup(createElement(PitCounts, { counts: COUNTS, active: null, onToggle: () => {}, hiddenCount: 0, autoHiddenCount: 0 }));
    // renderToStaticMarkup has no hydration comment separators, so the count and label share one
    // text run: "...</span>3 states.needs-you</button>" — assert that order directly.
    expect(html).toMatch(/3 states\.needs-you/);
    expect(html).not.toContain("states.needs-you · 3");
  });
  it("the idle pill appends the hidden suffix when hiddenCount+autoHiddenCount > 0", () => {
    const html = renderToStaticMarkup(createElement(PitCounts, { counts: COUNTS, active: null, onToggle: () => {}, hiddenCount: 2, autoHiddenCount: 1 }));
    // renderToStaticMarkup HTML-escapes the quotes the mock's JSON.stringify emits.
    expect(html).toContain("pit.hidden:{&quot;count&quot;:3}");
  });
  it("no hidden suffix when both counts are 0", () => {
    const html = renderToStaticMarkup(createElement(PitCounts, { counts: COUNTS, active: null, onToggle: () => {}, hiddenCount: 0, autoHiddenCount: 0 }));
    expect(html).not.toContain("pit.hidden");
  });
});

describe("PitCounts — pit-hint line (spec item 15)", () => {
  it("renders the hint with the live working count interpolated", () => {
    const html = renderToStaticMarkup(createElement(PitCounts, { counts: COUNTS, active: null, onToggle: () => {}, hiddenCount: 0, autoHiddenCount: 0 }));
    expect(html).toContain("pitHint:{&quot;workingCount&quot;:2}");
  });
});
