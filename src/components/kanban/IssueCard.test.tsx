import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { KanbanIssue } from "@/server/collectors/linear";
import { IssueCard } from "./IssueCard";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en-US",
}));

const issue: KanbanIssue = {
  id: "i1",
  identifier: "MOA-1",
  title: "Build the cockpit",
  priority: 2,
  url: "https://x",
  stateId: "s1",
  createdAt: "2026-01-01T00:00:00Z",
  updatedAt: "2026-01-01T00:00:00Z",
  labels: [],
};

describe("IssueCard", () => {
  it("opens via a real, keyboard-operable button without nested interactives", () => {
    const html = renderToStaticMarkup(createElement(IssueCard, { issue, onOpen: () => {} }));
    expect(html).toContain("<button");
    expect(html).toContain("Build the cockpit");
    expect(html).toContain('type="button"');
    expect(html).not.toContain("<a"); // no nested link
    expect(html).not.toContain("<select");
  });

  it("keeps pointer drag and drops the click handler when drag is off", () => {
    const html = renderToStaticMarkup(createElement(IssueCard, { issue, onDragStart: () => {}, onOpen: () => {} }));
    expect(html).toContain('draggable="true"');
    const staticOnly = renderToStaticMarkup(createElement(IssueCard, { issue }));
    expect(staticOnly).toContain('disabled=""');
  });

  it("renders no button when no open handler is provided", () => {
    const html = renderToStaticMarkup(createElement(IssueCard, { issue }));
    expect(html).toContain("disabled");
  });
});