import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en-US",
}));

import { initialOrganizeState, enterOrganize } from "@/lib/cockpitPrefsQuery";
import type { CockpitSnapshot } from "@/lib/cockpitLayout";
import { ProjectLayoutControls, ProjectOrganizeHandle, PROJECT_DRAG_MIME, readDraggedProject } from "./ProjectLayoutControls";

const snapshot: CockpitSnapshot = {
  revision: 4, prScope: "mine",
  layout: [{ project: "Alpha", width: "half" }, { project: "Beta", width: "third" }],
};

const baseProps = {
  unavailable: false,
  organize: initialOrganizeState,
  onEnter: () => {}, onSave: () => {}, onCancel: () => {}, onReset: () => {}, onReload: () => {},
  prScope: "mine" as const, scopeBusy: false, scopeError: null, onScope: () => {},
};

describe("Organize controls (spec AC3, plan Task 3)", () => {
  it("offers Enter and Reset in view mode, with the visible PR scope", () => {
    const html = renderToStaticMarkup(createElement(ProjectLayoutControls, baseProps));
    expect(html).toContain("controls.organize");
    expect(html).toContain("controls.reset");
    expect(html).not.toContain("controls.save");
    expect(html).toContain('aria-pressed="true"');
    expect(html).toContain("controls.mine");
  });

  it("in Organize mode offers Save, Cancel and the width/move alternatives, and surfaces a conflict with Reload", () => {
    const organize = enterOrganize(snapshot, [{ dir: "Alpha" }, { dir: "Beta" }]);
    const html = renderToStaticMarkup(createElement(ProjectLayoutControls, {
      ...baseProps, organize: { ...organize, status: "conflict" }, prScope: "all",
    }));
    expect(html).toContain("controls.save");
    expect(html).toContain("controls.cancel");
    expect(html).toContain("controls.reload");
    expect(html).toContain('aria-pressed="true"'); // All selected
  });

  it("a save failure is visible and an unavailable source disables organize/reset/scope", () => {
    const failed = renderToStaticMarkup(createElement(ProjectLayoutControls, {
      ...baseProps, organize: { ...enterOrganize(snapshot, [{ dir: "Alpha" }]), status: "error", error: "disk" },
    }));
    expect(failed).toContain("controls.saveError");
    const unavailable = renderToStaticMarkup(createElement(ProjectLayoutControls, { ...baseProps, unavailable: true }));
    expect(unavailable).toContain("controls.unavailable");
    expect(unavailable.match(/disabled/g)?.length ?? 0).toBeGreaterThanOrEqual(4);
  });

  it("a scope persistence error is surfaced while the prior choice stays selected", () => {
    const html = renderToStaticMarkup(createElement(ProjectLayoutControls, {
      ...baseProps, prScope: "all", scopeError: "failed",
    }));
    expect(html).toContain("controls.scopeError");
    expect(html).toContain("controls.all");
  });
});

describe("per-project handle (dedicated drag handle + keyboard/button alternative)", () => {
  const gripTag = (html: string) => html.match(/<span[^>]*data-project-dir="Alpha"[^>]*>/)?.[0] ?? "";

  it("renders a presentational pointer grip, a width select (third/half/full) and disabled boundary buttons", () => {
    const html = renderToStaticMarkup(createElement(ProjectOrganizeHandle, {
      dir: "Alpha", name: "Alpha", width: "half", index: 0, count: 2,
      onMove: () => {}, onWidth: () => {},
    }));
    const grip = gripTag(html);
    expect(grip).toContain(`data-project-dir="Alpha"`);
    expect(grip).toContain('draggable="true"');
    // F5: pointer-only affordance — hidden from the accessibility tree; the labeled width select
    // and move buttons below are the real keyboard controls.
    expect(grip).toContain('aria-hidden="true"');
    expect(grip).not.toContain("role=");
    expect(grip).not.toContain("tabindex");
    expect(grip).not.toContain("aria-label");
    expect(grip).not.toContain("aria-disabled");
    // The namespace-less next-intl mock renders the bare keys.
    expect(html).toContain("width");
    expect(html).toContain(">third<");
    expect(html).toContain(">half<");
    expect(html).toContain(">full<");
    expect(html).toContain("moveEarlier");
    expect(html).toContain("moveLater");
    // first entry: Move earlier is disabled; last entry: Move later is disabled.
    expect(/disabled[^>]*>[^<]*↑|↑<\/button>/.test(html)).toBe(true);
    expect(html).toContain("disabled");
  });

  it("drop payloads are checked against the current draft", () => {
    const draft = [{ project: "Alpha", width: "third" as const }];
    const known = { getData: (type: string) => (type === PROJECT_DRAG_MIME ? "Alpha" : "") };
    const unknown = { getData: (type: string) => (type === PROJECT_DRAG_MIME ? "Other" : "") };
    const unsafe = { getData: (type: string) => (type === PROJECT_DRAG_MIME ? "../escape" : "") };
    expect(readDraggedProject(known, draft)).toBe("Alpha");
    expect(readDraggedProject(unknown, draft)).toBeNull();
    expect(readDraggedProject(unsafe, draft)).toBeNull();
  });
});
describe("pending save disables the organize controls (plan Task 3)", () => {
  const handle = (disabled: boolean) => renderToStaticMarkup(createElement(ProjectOrganizeHandle, {
    dir: "Alpha", name: "Alpha", width: "half", index: 0, count: 2,
    onMove: () => {}, onWidth: () => {}, disabled,
  }));

  it("disables the pointer drag and the real keyboard controls while saving", () => {
    const html = handle(true);
    const grip = html.match(/<span[^>]*data-project-dir="Alpha"[^>]*>/)?.[0] ?? "";
    expect(grip).toContain('draggable="false"');
    expect(grip).toContain('aria-hidden="true"');
    expect(grip).not.toContain("aria-disabled");
    expect((html.match(/disabled=""/g) ?? []).length).toBeGreaterThanOrEqual(3);
  });

  it("keeps the controls enabled when no write is pending", () => {
    const html = handle(false);
    const grip = html.match(/<span[^>]*data-project-dir="Alpha"[^>]*>/)?.[0] ?? "";
    expect(grip).toContain('draggable="true"');
    expect(grip).not.toContain("aria-disabled");
  });

  it("reads only the project MIME and validates against bounded normalized draft identities", () => {
    const draft = [{ project: "projetos/ação web", width: "third" as const }];
    const mime = { getData: (t: string) => (t === PROJECT_DRAG_MIME ? "projetos/ação web" : "") };
    const textOnly = { getData: (t: string) => (t === "text/plain" ? "projetos/ação web" : "") };
    const malformed = { getData: (t: string) => (t === PROJECT_DRAG_MIME ? "../escape" : "") };
    const unrelated = { getData: (t: string) => (t === PROJECT_DRAG_MIME ? "outra coisa" : "") };
    expect(readDraggedProject(mime, draft)).toBe("projetos/ação web");
    expect(readDraggedProject(textOnly, draft)).toBeNull();
    expect(readDraggedProject(malformed, draft)).toBeNull();
    expect(readDraggedProject(unrelated, draft)).toBeNull();
  });
});
