import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { ModalDialog, requestClose } from "./ModalDialog";

describe("requestClose (dialog pending-close guard)", () => {
  it("blocks close while a write is pending and allows it once settled", () => {
    const close = vi.fn();
    requestClose(true, close);
    expect(close).not.toHaveBeenCalled();
    requestClose(false, close);
    expect(close).toHaveBeenCalledTimes(1);
  });
});

describe("ModalDialog", () => {
  it("renders a native dialog passing aria-label, className and children through", () => {
    const html = renderToStaticMarkup(
      createElement(ModalDialog, { onClose: () => {}, "aria-label": "x", className: "c" }, "body"),
    );
    expect(html).toContain("<dialog");
    expect(html).toContain('aria-label="x"');
    expect(html).toContain('class="c"');
    expect(html).toContain("body");
  });
});
