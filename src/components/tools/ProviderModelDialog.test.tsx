import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { EditorRevision, ModelChoice } from "@/lib/agent-settings";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
}));
vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({ invalidateQueries: vi.fn() }),
}));

import { ProviderModelDialog, resolveEffortSource } from "./ProviderModelDialog";

const REV: EditorRevision = {
  settings: "a".repeat(32),
  sources: { "opencode.json": "absent", "opencode.jsonc": "absent" },
};

function model(over: Partial<ModelChoice> = {}): ModelChoice {
  return {
    id: "m/1",
    label: "M One",
    origin: "override",
    efforts: ["high", "low"],
    no_effort: false,
    compatible: true,
    reason: null,
    context: 128000,
    effort_template: "reasoning",
    ...over,
  };
}

function render(model?: ModelChoice | null) {
  return renderToStaticMarkup(createElement(ProviderModelDialog, {
    connection: "acme",
    model,
    expected: REV,
    onClose: () => {},
  }));
}

describe("ProviderModelDialog", () => {
  it("creates a model with an editable id and no output/raw controls", () => {
    const html = render(null);
    expect(html).toContain("addModel");
    expect(html).not.toContain('readOnly=""');
    expect(html).not.toContain("outputTokens");
    expect(html).not.toContain("reasoningSupported");
    expect(html).toContain("dialogHeader");
    expect(html).toContain("dialogFooter");
    expect(html).toContain("dialogClose");
  });

  it("groups optional metadata without dropping existing values", () => {
    const fresh = render(null);
    expect(fresh).toContain("modelMetadataSummary");
    expect(fresh).toMatch(/<details[^>]*\bopen=""/);
    const edited = render(model());
    expect(edited).toMatch(/<details(?![^>]*\bopen\b)[^>]*>/);
    expect(edited).toContain('value="128000"');
    expect(edited).toContain('value="high, low"');
  });

  it("edits an existing model with a readonly id and prefilled context/efforts", () => {
    const html = render(model());
    expect(html).toContain("editModel");
    expect(html).toContain('readOnly=""');
    expect(html).toContain('value="128000"');
    expect(html).toContain('value="high, low"');
    expect(html).not.toContain("effortUnsupported");
    expect(html).toContain("outputInternal");
  });

  it("explains an unsupported effort representation while preserving variants", () => {
    const html = render(model({ effort_template: null }));
    expect(html).toContain("effortUnsupported");
    expect(html).toContain("</details>");
    expect(html.indexOf("effortUnsupported")).toBeGreaterThan(html.indexOf("</details>"));
    expect(html.indexOf("outputInternal")).toBeGreaterThan(html.indexOf("</details>"));
    const efforts = html.match(/<input[^>]*value="high, low"[^>]*>/);
    expect(efforts?.[0]).toContain("disabled");
  });

  it("keeps create-mode effort editing off until a configured model matches", () => {
    const configured = model({ id: "known", efforts: ["high"], effort_template: "reasoning" });
    expect(resolveEffortSource(false, null, [configured], "known")).toBe(configured);
    expect(resolveEffortSource(false, null, [configured], "unknown")).toBeNull();
    expect(resolveEffortSource(false, null, undefined, "known")).toBeNull();
    expect(resolveEffortSource(false, null, [configured], "")).toBeNull();
    expect(resolveEffortSource(true, configured, undefined, "known")).toBe(configured);
    const html = render(null);
    expect(html).not.toContain("effortUnsupported");
  });
});
