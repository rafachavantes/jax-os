import { createElement } from "react";
import { NextIntlClientProvider } from "next-intl";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { BOOTSTRAP_DRAFT, type ConnectionChoice, type EditorRevision } from "@/lib/agent-settings";
import type { ToolsDraftApi } from "./ToolsDraftProvider";
import pt from "../../../messages/pt-BR.json";
import en from "../../../messages/en-US.json";

const harness = vi.hoisted(() => ({
  api: {} as unknown,
  query: { data: undefined as unknown, isError: false, dataUpdatedAt: 0, error: undefined as unknown },
  settings: { data: undefined as unknown },
}));

vi.mock("@tanstack/react-query", () => ({
  useQuery: (opts?: { queryKey?: unknown[] }) => (opts?.queryKey?.[0] === "settings" ? harness.settings : harness.query),
  useQueryClient: () => ({ invalidateQueries: vi.fn() }),
}));
vi.mock("./ToolsDraftProvider", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./ToolsDraftProvider")>();
  return { ...actual, useToolsDraft: () => harness.api as never };
});

import { AgentSettingsSection } from "./AgentSettingsSection";
import { classifyWrite } from "./ToolsDraftProvider";

const REV: EditorRevision = {
  settings: "a".repeat(32),
  sources: { "opencode.json": "absent", "opencode.jsonc": "absent" },
};

function connection(id: string, adapter: string, modelId: string, efforts: string[]): ConnectionChoice {
  return {
    id,
    label: id,
    adapter,
    base_url: null,
    auth: "api-key",
    health: "registered",
    credential: { kind: "native" },
    editable: true,
    reason: null,
    used_by: [],
    models: [
      { id: modelId, label: modelId, origin: "catalog", efforts, no_effort: false, compatible: true, reason: null, context: null, effort_template: null },
    ],
  };
}

function api(over: Partial<ToolsDraftApi> = {}): ToolsDraftApi {
  return {
    dirty: true,
    conflict: false,
    draft: JSON.parse(JSON.stringify(BOOTSTRAP_DRAFT)),
    baseline: JSON.parse(JSON.stringify(BOOTSTRAP_DRAFT)),
    revision: REV,
    connections: [
      connection("xai", "xai", "grok-4.6", ["high", "xhigh"]),
      connection("openrouter", "openrouter", "deepseek/deepseek-v4-flash-0731", ["high"]),
    ],
    settingsPresent: true,
    busy: false,
    resetEpoch: 0,
    setBusy: vi.fn(),
    setDraft: vi.fn(),
    discard: vi.fn(),
    applyServer: vi.fn(),
    markSaved: vi.fn(),
    credential: null,
    openCredential: vi.fn(),
    closeCredential: vi.fn(),
    ...over,
  };
}

function render(current: ToolsDraftApi, locale: "pt-BR" | "en-US" = "pt-BR", agents?: { claude: boolean; codex: boolean; opencode: boolean }) {
  harness.api = current;
  harness.settings = agents ? { data: { ok: true, data: { integrations: { agents } } } } : { data: undefined };
  harness.query = {
    data: { ok: true, data: { editor_revision: REV, settings: { revision: REV.settings }, draft: current.draft, connections: current.connections } },
    isError: false,
    dataUpdatedAt: 0,
    error: undefined,
  };
  return renderToStaticMarkup(createElement(NextIntlClientProvider, {
    locale,
    messages: locale === "pt-BR" ? pt : en,
    timeZone: "UTC",
    children: createElement(AgentSettingsSection, { active: true }),
  }));
}

function buttonDisabled(html: string, testid: string): boolean {
  const match = html.match(new RegExp(`data-testid="${testid}"[^>]*`));
  return !!match && / disabled(=""|>)/.test(match[0]);
}

function draftWith(over: { connection?: string; model?: string } = {}) {
  const d = JSON.parse(JSON.stringify(BOOTSTRAP_DRAFT)) as typeof BOOTSTRAP_DRAFT;
  if (over.connection) d.builders.default.connection = over.connection;
  if (over.model !== undefined) d.builders.default.model = over.model;
  return d;
}

describe("AgentSettingsSection direct save", () => {
  it("offers Save and Discard and no preview UI", () => {
    const html = render(api());
    expect(html).toContain('data-testid="agent-save"');
    expect(html).toContain('data-testid="agent-discard"');
    expect(html).not.toContain(pt.tools.previewStale);
    expect(html).not.toContain('data-testid="agent-preview"');
  });

  it("enables a dirty valid save without any preview", () => {
    const html = render(api({ dirty: true }));
    expect(buttonDisabled(html, "agent-save")).toBe(false);
  });

  it("keeps an unchanged save disabled", () => {
    const html = render(api({ dirty: false }));
    expect(buttonDisabled(html, "agent-save")).toBe(true);
  });

  it("blocks save while busy or invalid", () => {
    expect(buttonDisabled(render(api({ busy: true })), "agent-save")).toBe(true);
    expect(buttonDisabled(render(api({ connections: [] })), "agent-save")).toBe(true);
  });

  it("initial activation shows the inline acknowledgement and stays disabled until acknowledged", () => {
    const html = render(api({ settingsPresent: false, dirty: false }));
    expect(html).toContain(pt.tools.activationAcknowledge);
    expect(html).toContain('type="checkbox"');
    expect(buttonDisabled(html, "agent-save")).toBe(true);
  });

  it("does not show the acknowledgement for normal saves", () => {
    expect(render(api({ settingsPresent: true }))).not.toContain(pt.tools.activationAcknowledge);
  });

  it("offers reconcile only under conflict", () => {
    expect(render(api({ conflict: true }))).toContain('data-testid="agent-reconcile"');
    expect(render(api())).not.toContain('data-testid="agent-reconcile"');
  });
});

describe("Agent model inputs and reviewer mapping", () => {
  it("renders OpenRouter as a plain text field with no list or datalist", () => {
    const html = render(api({
      draft: draftWith({ connection: "openrouter", model: "deepseek/deepseek-v4-flash-0731" }),
    }));
    expect(html).toContain('type="text"');
    expect(html).not.toContain("models-default");
    expect(html).not.toContain("<datalist");
  });

  it("renders configured-model search for non-OpenRouter connections", () => {
    const html = render(api());
    expect(html).toContain('type="search"');
    expect(html).toContain('list="models-default"');
    expect(html).toContain('<datalist id="models-default"');
  });

  it("always offers the blank provider-default effort option", () => {
    const html = render(api());
    expect(html).toContain(`>${pt.tools.effortDefault}<`);
  });

  it("keeps an unavailable saved connection and model visible", () => {
    const html = render(api({ draft: draftWith({ connection: "ghost", model: "ghost-model" }) }));
    expect(html).toContain('value="ghost"');
    expect(html).toContain(pt.tools.unavailableShort);
    expect(html).toContain(pt.tools.modelUnavailable);
    expect(html).toContain(pt.tools.manageConnection);
  });

  it("maps a Claude lead to reviewers.codex and a Codex lead to reviewers.claude", () => {
    const current = api();
    const html = render(current);
    expect(html).toContain(`aria-label="Modelo do revisor Codex" value="${current.draft.reviewers.codex.model}"`);
    expect(html).toContain(`aria-label="Modelo do revisor Claude" value="${current.draft.reviewers.claude.model}"`);
  });

  it("renders both manage-connection actions as outlined buttons", () => {
    const buttons = render(api()).match(new RegExp(`<button\\b[^>]*>${pt.tools.manageConnection} →<\\/button>`, "g")) ?? [];
    expect(buttons).toHaveLength(2);
    for (const button of buttons) expect(button).toContain("border border-line");
  });

  it.each([
    ["pt-BR", "Revisor", "Execução", "atribuição protegida", "Modelo do revisor"],
    ["en-US", "Reviewer", "Runtime", "protected assignment", "Reviewer model"],
  ] as const)("presents localized protected reviewer roles in %s", (locale, role, runtime, protectedText, model) => {
    const html = render(api(), locale);
    for (const [lead, reviewer] of [["Claude", "Codex"], ["Codex", "Claude"]]) {
      expect(html).toMatch(new RegExp(`Lead ${lead} <span[^>]*>→</span> <strong>${role} ${reviewer}</strong>`));
      expect(html).toContain(`${runtime}: ${reviewer} · ${protectedText}</small>`);
      expect(html).toContain(`${model} ${reviewer}<input`);
    }
  });

  it("renders the Builder and Reviewers section headings with the runtime badge", () => {
    const html = render(api());
    expect(html).toContain(pt.tools.builderHeading);
    expect(html).toContain(pt.tools.builderSubtitle);
    expect(html).toContain("opencode-builder");
    expect(html).toContain(pt.tools.reviewersHeading);
    expect(html).toContain(pt.tools.reviewersFixed);
    expect(html).toContain(pt.tools.builderFlowNote);
  });
});

describe("Agents tab follows the agent switches (MOA-504 D10)", () => {
  const A = (claude: boolean, codex: boolean, opencode: boolean) => ({ claude, codex, opencode });

  it("OpenCode off: the builder section is replaced by one muted line", () => {
    const html = render(api(), "pt-BR", A(true, true, false));
    expect(html).toContain(pt.tools.builderOff);
    expect(html).not.toContain("opencode-builder");
    expect(html).not.toContain(pt.tools.builderFlowNote);
    expect(html).toContain(pt.tools.reviewersHeading);
  });

  it("OpenCode on keeps the builder cards and has no builderOff line", () => {
    const html = render(api(), "pt-BR", A(true, true, true));
    expect(html).toContain("opencode-builder");
    expect(html).not.toContain(pt.tools.builderOff);
  });

  it("OpenCode off: an invalid hidden builder does not block a valid reviewer save", () => {
    const bad = api({ connections: [] });
    expect(buttonDisabled(render(bad, "pt-BR", A(true, true, false)), "agent-save")).toBe(false);
  });

  it("OpenCode on: an invalid builder still blocks save", () => {
    const bad = api({ connections: [] });
    expect(buttonDisabled(render(bad, "pt-BR", A(true, true, true)), "agent-save")).toBe(true);
  });

  it("Codex off: only the Claude lead row, reviewed by Claude itself, tagged fallback", () => {
    const html = render(api(), "pt-BR", A(true, false, true));
    expect(html).toContain('aria-label="Modelo do revisor Claude"');
    expect(html).not.toContain('aria-label="Modelo do revisor Codex"');
    expect(html.match(/data-testid="reviewer-fallback"/g)).toHaveLength(1);
    expect(html).toMatch(/Lead Claude <span[^>]*>→<\/span> <strong>Revisor Claude<\/strong>/);
  });

  it("both reviewer agents on: two rows, cross-assigned, no fallback tag", () => {
    const html = render(api(), "pt-BR", A(true, true, true));
    expect(html).toContain('aria-label="Modelo do revisor Claude"');
    expect(html).toContain('aria-label="Modelo do revisor Codex"');
    expect(html).not.toContain('data-testid="reviewer-fallback"');
  });

  it("neither Claude nor Codex on: a single note and no reviewer rows", () => {
    const html = render(api(), "pt-BR", A(false, false, true));
    expect(html).toContain(pt.tools.reviewersNone);
    expect(html).not.toContain("Modelo do revisor");
  });
});

describe("classifyWrite outcome classification", () => {
  it("treats a confirmed save with a revision as ok and carries the revision", () => {
    const out = classifyWrite({ ok: true, data: { editor_revision: REV } });
    expect(out.kind).toBe("ok");
    expect(out.revision).toEqual(REV);
  });

  it("treats an ok-shaped response without a revision as unconfirmed-eligible", () => {
    const out = classifyWrite({ ok: true, data: {} });
    expect(out.kind).toBe("ok");
    expect(out.revision).toBeUndefined();
  });

  it("classifies pending, error and refused outcomes", () => {
    expect(classifyWrite({ ok: false, effect: "activation-pending", error: "x" }).kind).toBe("activation-pending");
    expect(classifyWrite({ ok: false, effect: "unconfirmed", error: "x" }).kind).toBe("unconfirmed");
    expect(classifyWrite({ ok: false, error: "invalid payload" }).kind).toBe("refused");
    expect(classifyWrite(null).kind).toBe("unconfirmed");
  });
});
