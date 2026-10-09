import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";

const harness = vi.hoisted(() => ({
  members: { data: { ok: true, data: [] } as unknown, isError: false },
  labels: { data: { ok: true, data: [] } as unknown, isError: false },
}));

vi.mock("next-intl", () => ({ useTranslations: () => (key: string) => key }));
vi.mock("@tanstack/react-query", () => ({
  useQuery: ({ queryKey }: { queryKey: string[] }) => {
    if (queryKey.includes("members")) return harness.members;
    if (queryKey.includes("labels")) return harness.labels;
    return { data: { ok: true, data: [] }, isError: false };
  },
}));

import { CreateIssueOverlay } from "./CreateIssueOverlay";

const states = [{ id: "s1", name: "Todo", color: "#e2e2e2", position: 1, type: "unstarted" }];
const issues = [
  { id: "i1", identifier: "MOA-1", title: "t", priority: 0, url: "", stateId: "s1",
    createdAt: "2026-01-01T00:00:00Z", updatedAt: "2026-01-01T00:00:00Z",
    project: "Jax OS", projectId: "p1", labels: [] },
];

function render(defaultProjectName: string | null) {
  return renderToStaticMarkup(
    createElement(CreateIssueOverlay, { teamId: "t1", states, issues, defaultProjectName, onClose: () => {}, onCreated: () => {} }),
  );
}

beforeEach(() => {
  harness.members = { data: { ok: true, data: [] }, isError: false };
  harness.labels = { data: { ok: true, data: [] }, isError: false };
});

describe("CreateIssueOverlay", () => {
  it("renders required title, optional description, and a disabled submit without a title", () => {
    const html = render(null);
    expect(html).toContain("<dialog");
    expect(html).toContain('aria-label="createTitlePlaceholder"');
    expect(html).toContain('aria-label="createDescriptionPlaceholder"');
    expect(html).toContain(">Jax OS<");
    expect(html).toContain('disabled=""');
  });
  it("preselects the project matching defaultProjectName", () => {
    expect(render("Jax OS")).toContain('value="p1"');
  });
});

describe("CreateIssueOverlay option source failures (error contract)", () => {
  it("renders no warning when both sources succeed", () => {
    const html = render(null);
    expect(html).not.toContain("createMembersUnavailable");
    expect(html).not.toContain("createLabelsUnavailable");
  });

  it("warns when the members envelope is {ok:false}", () => {
    harness.members = { data: { ok: false, error: "down" }, isError: false };
    const html = render(null);
    expect(html).toContain("createMembersUnavailable");
    expect(html).not.toContain("createLabelsUnavailable");
  });

  it("warns when the labels fetch rejects", () => {
    harness.labels = { data: undefined, isError: true };
    const html = render(null);
    expect(html).toContain("createLabelsUnavailable");
    expect(html).not.toContain("createMembersUnavailable");
  });
});
