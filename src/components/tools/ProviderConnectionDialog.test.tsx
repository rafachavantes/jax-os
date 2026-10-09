import { readFileSync } from "node:fs";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { fileURLToPath } from "node:url";
import { describe, expect, it, vi } from "vitest";
import type { ConnectionChoice, EditorRevision } from "@/lib/agent-settings";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
}));
vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({ invalidateQueries: vi.fn() }),
}));

import { ProviderConnectionDialog } from "./ProviderConnectionDialog";

const REV: EditorRevision = {
  settings: "a".repeat(32),
  sources: { "opencode.json": "absent", "opencode.jsonc": "absent" },
};

function connection(over: Partial<ConnectionChoice> = {}): ConnectionChoice {
  return {
    id: "acme",
    label: "Acme",
    adapter: "xai",
    base_url: null,
    auth: "api-key",
    health: "registered",
    credential: { kind: "native" },
    editable: true,
    reason: null,
    used_by: [],
    models: [],
    ...over,
  };
}

function render(model?: ConnectionChoice | null) {
  return renderToStaticMarkup(createElement(ProviderConnectionDialog, {
    connection: model,
    expected: REV,
    onClose: () => {},
  }));
}

describe("ProviderConnectionDialog", () => {
  it("carries the pinned dialog class with no credential step (MOA-498 D4/12)", () => {
    const html = render(null);
    expect(html).toContain("providerDialog");
    expect(html).not.toContain("credentialOnboarding");
    expect(html).not.toContain("onboardingNew");
    expect(html).not.toContain("onboardingLink");
    expect(html).not.toContain("onboardingLater");
    expect(html).not.toContain("registrationConfirmed");
    expect(html).toContain("connectionIdHint");
    expect(html).toContain("dialogHeader");
    expect(html).toContain("dialogFooter");
    expect(html).toContain("dialogClose");
  });

  it("edits an existing connection with a readonly identity and no credential fields", () => {
    const html = render(connection());
    expect(html).toContain('readOnly=""');
    expect(html).not.toContain("credentialOnboarding");
    expect(html).not.toContain('type="password"');
    expect(html).not.toContain("secretId");
  });

  it("registration-then-credential flow is gone from the source (MOA-498 D4/12)", () => {
    const text = readFileSync(fileURLToPath(new URL("ProviderConnectionDialog.tsx", import.meta.url)), "utf8");
    expect(text).not.toContain("/api/opencode-providers/credentials");
    expect(text).not.toContain("ProviderCredentialDialog");
    expect(text).not.toContain("useCredentialForm");
    expect(text).not.toContain("credentialMode");
  });
});
