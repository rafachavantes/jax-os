import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en-US",
}));

import { RuleDiffModal, requestRuleClose } from "./RuleDiffModal";

describe("RuleDiffModal", () => {
  it("renders a native dialog with the diff and a pending apply control", () => {
    const html = renderToStaticMarkup(
      createElement(RuleDiffModal, {
        appLabel: "Claude Code",
        expected: "rules",
        disk: "disk",
        missing: false,
        status: null,
        onApply: async () => {},
        onClose: () => {},
      }),
    );
    expect(html).toContain("<dialog");
    expect(html).toContain("Claude Code");
    expect(html).toContain("diffDisk");
    expect(html).toContain("diffExpected");
    expect(html).toContain("apply");
  });

  it("marks a missing file explicitly", () => {
    const html = renderToStaticMarkup(
      createElement(RuleDiffModal, {
        appLabel: "Claude Code",
        expected: "rules",
        disk: "",
        missing: true,
        status: null,
        onApply: async () => {},
        onClose: () => {},
      }),
    );
    expect(html).toContain("diffMissing");
  });

  it("renders the supplied error and pending text inside an accessible live region", () => {
    for (const [tone, text] of [
      ["pending", "applying"],
      ["error", "writeUnconfirmed"],
    ] as const) {
      const html = renderToStaticMarkup(
        createElement(RuleDiffModal, {
          appLabel: "Claude Code",
          expected: "rules",
          disk: "disk",
          missing: false,
          status: { tone, text },
          onApply: async () => {},
          onClose: () => {},
        }),
      );
      expect(html).toContain('role="status"');
      expect(html).toContain('aria-live="polite"');
      expect(html).toContain(text);
    }
  });

  it("renders no live status region while no settlement exists", () => {
    const html = renderToStaticMarkup(
      createElement(RuleDiffModal, {
        appLabel: "Claude Code",
        expected: "rules",
        disk: "disk",
        missing: false,
        status: null,
        onApply: async () => {},
        onClose: () => {},
      }),
    );
    expect(html).not.toContain('role="status"');
  });

  it("requires an in-dialog confirmation before applying (no browser confirm)", () => {
    const html = renderToStaticMarkup(
      createElement(RuleDiffModal, {
        appLabel: "Claude Code",
        expected: "rules",
        disk: "disk",
        missing: false,
        status: null,
        onApply: async () => {},
        onClose: () => {},
      }),
    );
    expect(html).toContain("confirmApply");
    expect(html).toContain('type="checkbox"');
  });

  it("disables apply and explains when the preview snapshot went stale", () => {
    const html = renderToStaticMarkup(
      createElement(RuleDiffModal, {
        appLabel: "Claude Code",
        expected: "rules",
        disk: "disk",
        missing: false,
        status: null,
        stale: true,
        onApply: async () => {},
        onClose: () => {},
      }),
    );
    expect(html).toContain("previewStale");
  });
});

describe("requestRuleClose (dialog pending-close guard)", () => {
  it("blocks close while a write is pending and allows it once settled", () => {
    const close = vi.fn();
    requestRuleClose(true, close);
    expect(close).not.toHaveBeenCalled();
    requestRuleClose(false, close);
    expect(close).toHaveBeenCalledTimes(1);
  });
});