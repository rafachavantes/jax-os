import { readFileSync } from "node:fs";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { fileURLToPath } from "node:url";
import { describe, expect, it, vi } from "vitest";
import type { EditorRevision } from "@/lib/agent-settings";
import en from "../../../messages/en-US.json";
import pt from "../../../messages/pt-BR.json";

vi.mock("next-intl", () => ({ useTranslations: () => (key: string) => key }));
vi.mock("@tanstack/react-query", () => ({ useQueryClient: () => ({ invalidateQueries: vi.fn() }) }));

import { ProviderCredentialDialog } from "./ProviderCredentialDialog";

const REV: EditorRevision = {
  settings: "a".repeat(32),
  sources: { "opencode.json": "absent", "opencode.jsonc": "absent" },
};

function render(current: { kind: "native" } | { kind: "env"; env: string } | null) {
  return renderToStaticMarkup(createElement(ProviderCredentialDialog, {
    request: { providerId: "acme", current, adapter: "xai" },
    expected: REV,
    onClose: () => {},
  }));
}

function source(name: string): string {
  return readFileSync(fileURLToPath(new URL(name, import.meta.url)), "utf8");
}

describe("ProviderCredentialDialog (MOA-498 D4/12 — native/env only, no BWS flow)", () => {
  it("renders a native/env choice, no create/link/replace fields", () => {
    const html = render({ kind: "native" });
    expect(html).toContain("credentialKind");
    expect(html).not.toContain("secretId");
    expect(html).not.toContain("credentialKey");
    expect(html).not.toContain('type="password"');
  });

  it("shows the env name field, prefilled, when the current binding is env-kind", () => {
    const html = render({ kind: "env", env: "JAX_PROVIDER_ACME_API_KEY" });
    expect(html).toContain("JAX_PROVIDER_ACME_API_KEY");
    expect(html).toContain("credentialEnvName");
    for (const messages of [en.tools, pt.tools]) {
      expect(messages).toHaveProperty("envConfigured");
      expect(messages).toHaveProperty("envMissing");
      expect(messages).toHaveProperty("editCredential");
      expect(messages).toHaveProperty("invalidEnvName");
    }
  });
});

describe("the BWS credential route is never called (MOA-498 D4/12)", () => {
  it("no credential-facing tool component references the deleted route", () => {
    for (const name of ["ProviderCredentialDialog.tsx", "OpenCodeProvidersSection.tsx", "ProviderConnectionDialog.tsx"]) {
      expect(source(name)).not.toContain("/api/opencode-providers/credentials");
    }
  });

  it("the dialog saves a binding through bind-credential on the providers route and checks the env status", () => {
    const text = source("ProviderCredentialDialog.tsx");
    expect(text).toContain('action: "bind-credential"');
    expect(text).toContain('postJson("/api/opencode-providers"');
    expect(text).toContain("credential-env=");
  });
});
