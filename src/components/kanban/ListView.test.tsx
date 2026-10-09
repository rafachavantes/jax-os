import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { KanbanIssue, KanbanState } from "@/server/collectors/linear";
import { ListView } from "./ListView";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en-US",
}));

const state: KanbanState = { id: "s1", name: "In Progress", color: "#0f0", position: 1, type: "started" };
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

describe("ListView", () => {
  it("renders each row as a real button", () => {
    const html = renderToStaticMarkup(createElement(ListView, { states: [state], issues: [issue], onOpenIssue: () => {} }));
    expect(html).toContain("In Progress");
    expect(html).toContain("Build the cockpit");
    expect(html.match(/<button/g)?.length).toBe(1);
    expect(html).toContain('type="button"');
    expect(html).not.toContain("<a");
  });
});