import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { Breadcrumb } from "./Breadcrumb";

describe("Breadcrumb rootClickable (spec §4.3)", () => {
  it("renders the root as a button by default", () => {
    const html = renderToStaticMarkup(createElement(Breadcrumb, { rootLabel: "Arquivos", segments: [], onRootClick: () => {}, onSegmentClick: () => {} }));
    expect(html).toContain("<button");
  });
  it("renders the root as a non-interactive span, styled like a current crumb, when false", () => {
    const html = renderToStaticMarkup(createElement(Breadcrumb, { rootLabel: "Arquivos", rootClickable: false, segments: [], onRootClick: () => {}, onSegmentClick: () => {} }));
    expect(html).not.toContain("<button");
    expect(html).toContain("text-ink font-semibold");
    expect(html).toContain(">Arquivos<");
  });
});
