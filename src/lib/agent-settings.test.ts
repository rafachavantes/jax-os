import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { beforeEach, describe, expect, it } from "vitest";
import sample from "../../workflow/fixtures/agent-settings-v1.json";
import corpus from "../../workflow/fixtures/agent-settings-validation.json";
import modelDefaults from "../../workflow/fixtures/model-defaults-v1.json";
import {
  BOOTSTRAP_DRAFT,
  CLAUDE_REVIEWER_EFFORTS,
  CODEX_REVIEWER_EFFORTS,
  CONNECTION_ADAPTERS,
  compactRouting,
  configuredModels,
  applyPendingOperations,
  decodeAgentSettings,
  draftIsValid,
  isSettingsDraftDirty,
  nextToolsTab,
  resolveToolsTab,
  visibleToolsTabs,
  type ToolsTab,
  parseEditorRevision,
  QUANTIZATIONS,
  activationSaveEnabled,
  previewFingerprint,
  referencedProfiles,
  saveIsStale,
  selectableConnections,
  settingsWriteKind,
  shouldConfirmLeaveTools,
  TOOLS_TABS,
  type ActivationGate,
  type ConnectionChoice,
  type EditorRevision,
  type ModelChoice,
  type SettingsDraft,
  validateModelInput,
} from "./agent-settings";

describe("TOOLS_TABS", () => {
  it("TOOLS_TABS puts general first, keeps the other four in their existing order (MOA-496 decision 2)", () => {
    expect(TOOLS_TABS).toEqual(["general", "agents", "providers", "rules", "inventory"]);
  });
});

const fixtures = join(fileURLToPath(new URL(".", import.meta.url)), "../../workflow/fixtures");

type CorpusCase = {
  name: string;
  expect: string | null;
  kind: "file" | "utf8" | "hex" | "fill" | "repeat";
  path?: string;
  text?: string;
  hex?: string;
  byte?: number;
  count?: number;
  suffix_hex?: string;
  suffix_count?: number;
};

function corpusBytes(c: CorpusCase): Uint8Array {
  switch (c.kind) {
    case "file":
      return new Uint8Array(readFileSync(join(fixtures, c.path!)));
    case "utf8":
      return new TextEncoder().encode(c.text!);
    case "hex":
      return Uint8Array.from(Buffer.from(c.hex!, "hex"));
    case "fill":
      return new Uint8Array(c.count!).fill(c.byte!);
    case "repeat": {
      const unit = Buffer.from(c.hex!, "hex");
      const suffix = Buffer.from(c.suffix_hex ?? "", "hex");
      const out = Buffer.alloc(unit.length * (c.count ?? 0) + suffix.length * (c.suffix_count ?? 0));
      for (let i = 0; i < (c.count ?? 0); i++) unit.copy(out, i * unit.length);
      for (let i = 0; i < (c.suffix_count ?? 0); i++) {
        suffix.copy(out, unit.length * (c.count ?? 0) + i * suffix.length);
      }
      return new Uint8Array(out);
    }
  }
}

function decodeCode(bytes: Uint8Array): string | null {
  try {
    decodeAgentSettings(bytes);
    return null;
  } catch (err) {
    if (err instanceof Error) return err.message;
    throw err;
  }
}

describe("decodeAgentSettings corpus", () => {
  it("reject duplicate authority keys before JSON parsing erases them", () => {
    const raw = JSON.stringify(sample).replace(
      '"schema_version":1',
      '"schema_version":1,"schema_version":1',
    );
    expect(() => decodeAgentSettings(new TextEncoder().encode(raw))).toThrow(
      "agent-settings-malformed",
    );
  });

  it.each((corpus as { cases: CorpusCase[] }).cases)("$name", (c) => {
    expect(decodeCode(corpusBytes(c))).toBe(c.expect);
  });
});

describe("executor-keyed bootstrap, not caller-keyed runtime tables", () => {
  it("maps Claude-lead Codex reviewer onto reviewers.codex", () => {
    expect(BOOTSTRAP_DRAFT.reviewers.codex).toEqual({ model: "gpt-5.6-luna", effort: "xhigh" });
    expect(BOOTSTRAP_DRAFT.reviewers.claude).toEqual({ model: "sonnet", effort: "xhigh" });
    expect((CODEX_REVIEWER_EFFORTS as readonly string[]).includes(BOOTSTRAP_DRAFT.reviewers.codex.effort)).toBe(true);
    expect((CLAUDE_REVIEWER_EFFORTS as readonly string[]).includes(BOOTSTRAP_DRAFT.reviewers.claude.effort)).toBe(true);
  });

  it("splits bootstrap CLI ids on the first slash only", () => {
    expect(BOOTSTRAP_DRAFT.builders.default).toMatchObject({
      connection: "xai",
      model: "grok-4.6",
      effort: "xhigh",
      credential: { kind: "native" },
      routing: null,
    });
    expect(BOOTSTRAP_DRAFT.builders.fallback).toMatchObject({
      connection: "openrouter",
      model: "deepseek/deepseek-v4-flash-0731",
      effort: "high",
      credential: { kind: "native" },
      routing: { sort: "price", allow_fallbacks: true },
    });
  });
});

describe("BOOTSTRAP_DRAFT model/effort leaves match the shared fixture (spec: One source of model defaults)", () => {
  it("reviewers", () => {
    expect(BOOTSTRAP_DRAFT.reviewers.claude).toMatchObject(modelDefaults.reviewers.claude);
    expect(BOOTSTRAP_DRAFT.reviewers.codex).toMatchObject(modelDefaults.reviewers.codex);
  });
  it("builders (model/effort only — connection/credential/routing stay TS-only)", () => {
    expect(BOOTSTRAP_DRAFT.builders.default).toMatchObject(modelDefaults.builders.default);
    expect(BOOTSTRAP_DRAFT.builders.fallback).toMatchObject(modelDefaults.builders.fallback);
  });
});

describe("QUANTIZATIONS", () => {
  it("lists the twelve OpenRouter quantization levels in wire order", () => {
    expect(QUANTIZATIONS).toEqual([
      "int4", "int8", "fp4", "mxfp4", "nvfp4", "fp6", "fp8", "mxfp8", "fp16", "bf16", "fp32", "unknown",
    ]);
  });
});

describe("reference and draft derivations", () => {
  const settings = decodeAgentSettings(new Uint8Array(readFileSync(join(fixtures, "agent-settings-v1.json"))));

  it("blocks removal of a connection or model still referenced by either profile", () => {
    expect(referencedProfiles(settings, "fixture")).toEqual(["default", "fallback"]);
    expect(referencedProfiles(settings, "fixture", "wire-real-model")).toEqual(["default", "fallback"]);
    expect(referencedProfiles(settings, "other")).toEqual([]);
  });

  it("treats identical drafts as clean and a field change as dirty", () => {
    const draft = {
      reviewers: settings.reviewers,
      builders: settings.builders,
    };
    expect(isSettingsDraftDirty(draft, draft)).toBe(false);
    expect(
      isSettingsDraftDirty(draft, {
        ...draft,
        builders: {
          ...draft.builders,
          default: { ...draft.builders.default, model: "other" },
        },
      }),
    ).toBe(true);
  });

  it("drops empty optional routing controls but keeps a zero price ceiling", () => {
    expect(compactRouting(null)).toBeNull();
    expect(
      compactRouting({
        sort: "price",
        allow_fallbacks: true,
        preferred_min_throughput: null,
        preferred_max_latency: null,
        max_price: { prompt: 0 },
        only: [],
        ignore: [],
        quantizations: [],
      }),
    ).toEqual({ sort: "price", allow_fallbacks: true, max_price: { prompt: 0 } });
  });

  it("keeps a non-empty quantizations list", () => {
    expect(
      compactRouting({ sort: "price", allow_fallbacks: true, quantizations: ["fp8", "fp4"] }),
    ).toEqual({ sort: "price", allow_fallbacks: true, quantizations: ["fp8", "fp4"] });
  });
});

const REV: EditorRevision = {
  settings: "a".repeat(32),
  sources: { "opencode.json": "b".repeat(64), "opencode.jsonc": "c".repeat(64) },
};

function model(partial: Partial<ModelChoice> & { id: string }): ModelChoice {
  return {
    label: partial.id,
    origin: "catalog",
    efforts: ["high"],
    no_effort: false,
    compatible: true,
    reason: null,
    context: null,
    effort_template: null,
    ...partial,
  };
}

function conn(partial: Partial<ConnectionChoice> & { id: string }): ConnectionChoice {
  return {
    label: partial.id,
    adapter: "xai",
    base_url: null,
    auth: "api-key",
    health: "registered",
    credential: { kind: "native" },
    editable: true,
    reason: null,
    used_by: [],
    models: [],
    ...partial,
  };
}

describe("tools UI derivations", () => {
  it("cycles tools tabs with Arrow/Home/End without remounting identity", () => {
    expect(nextToolsTab("agents", "ArrowRight")).toBe("providers");
    expect(nextToolsTab("inventory", "ArrowRight")).toBe("general");
    expect(nextToolsTab("rules", "ArrowLeft")).toBe("providers");
    expect(nextToolsTab("providers", "Home")).toBe("general");
    expect(nextToolsTab("agents", "End")).toBe("inventory");
  });

  it("hides the providers tab while OpenCode is off and keyboard nav skips it (MOA-504 D10)", () => {
    const on = visibleToolsTabs({ opencode: true });
    const off = visibleToolsTabs({ opencode: false });
    expect(on).toEqual(TOOLS_TABS);
    expect(off).toEqual(["general", "agents", "rules", "inventory"]);
    expect(nextToolsTab("agents", "ArrowRight", off)).toBe("rules");
    expect(nextToolsTab("rules", "ArrowLeft", off)).toBe("agents");
    expect(nextToolsTab("general", "ArrowLeft", off)).toBe("inventory");
    expect(nextToolsTab("agents", "End", off)).toBe("inventory");
    expect(nextToolsTab("agents", "ArrowRight", on)).toBe("providers"); // default list unchanged
  });

  it("a vanished tab resets the stored selection, so re-enabling OpenCode stays on General", () => {
    const on = visibleToolsTabs({ opencode: true });
    const off = visibleToolsTabs({ opencode: false });
    let selected: ToolsTab = "providers";
    selected = resolveToolsTab(selected, off);
    expect(selected).toBe("general");
    expect(resolveToolsTab(selected, on)).toBe("general");
    expect(resolveToolsTab("providers", on)).toBe("providers");
  });

  it("save is stale when any expected revision drifted from the server", () => {
    expect(saveIsStale(REV, REV)).toBe(false);
    expect(saveIsStale(REV, { ...REV, settings: "d".repeat(32) })).toBe(true);
    expect(
      saveIsStale(REV, {
        ...REV,
        sources: { ...REV.sources, "opencode.json": "e".repeat(64) },
      }),
    ).toBe(true);
  });

  it("builder connection choices are configured-and-credentialed only", () => {
    const rows = [
      conn({ id: "ok" }),
      conn({ id: "oauth", health: "oauth-managed", credential: { kind: "native" } }),
      conn({ id: "openrouter", adapter: "openrouter" }),
      conn({ id: "env-only", adapter: "openrouter", health: "missing", credential: null }),
      conn({ id: "catalog-only", adapter: "xai", health: "missing", credential: null }),
      conn({ id: "sync", health: "sync-pending" }),
      conn({ id: "invalid", health: "invalid" }),
    ];
    expect(selectableConnections(rows).map((c) => c.id)).toEqual(["ok", "oauth", "openrouter"]);
    expect(CONNECTION_ADAPTERS).toEqual(["xai", "openrouter", "openai-compatible"]);
  });

  it("model selector uses compatible configured models, not aliases", () => {
    const c = conn({
      id: "or",
      models: [model({ id: "grok" }), model({ id: "bad", compatible: false, origin: "override" })],
    });
    expect(configuredModels(c).map((m) => m.id)).toEqual(["grok"]);
    expect(configuredModels(undefined)).toEqual([]);
  });

  it("warns on leaving tools with a dirty draft, not on tab changes", () => {
    expect(shouldConfirmLeaveTools("/settings", "/files", true)).toBe(true);
    expect(shouldConfirmLeaveTools("/settings", "/settings", true)).toBe(false);
    expect(shouldConfirmLeaveTools("/settings", "/kanban", false)).toBe(false);
    expect(shouldConfirmLeaveTools("/files", "/kanban", true)).toBe(false);
  });

  it("does not flatten applied-unrecorded or activation-pending into a generic failed save", () => {
    expect(settingsWriteKind({ ok: true, data: {} })).toBe("ok");
    expect(settingsWriteKind({ ok: false, effect: "applied", audit: "pending", error: "x" })).toBe(
      "applied-unrecorded",
    );
    expect(settingsWriteKind({ ok: false, effect: "activation-pending", error: "x" })).toBe(
      "activation-pending",
    );
    expect(settingsWriteKind({ ok: false, effect: "unconfirmed", error: "x" })).toBe("unconfirmed");
    expect(settingsWriteKind({ ok: false, error: "invalid payload" })).toBe("refused");
  });
});

describe("parseEditorRevision", () => {
  const valid = {
    settings: "a".repeat(32),
    sources: { "opencode.json": "b".repeat(64), "opencode.jsonc": "absent" },
  };

  it("accepts the exact revision shape with absent sentinels", () => {
    expect(parseEditorRevision(valid)).toEqual(valid);
    expect(parseEditorRevision({ settings: "absent", sources: { "opencode.json": "absent", "opencode.jsonc": "absent" } }))
      .toEqual({ settings: "absent", sources: { "opencode.json": "absent", "opencode.jsonc": "absent" } });
  });

  it("rejects malformed objects, extra keys and short enums", () => {
    expect(parseEditorRevision(null)).toBeNull();
    expect(parseEditorRevision([])).toBeNull();
    expect(parseEditorRevision({ ...valid, extra: 1 })).toBeNull();
    expect(parseEditorRevision({ ...valid, sources: { ...valid.sources, extra: "absent" } })).toBeNull();
    expect(parseEditorRevision({ ...valid, settings: "a".repeat(31) })).toBeNull();
    expect(parseEditorRevision({ ...valid, sources: { "opencode.json": "b".repeat(63), "opencode.jsonc": "absent" } }))
      .toBeNull();
    expect(parseEditorRevision({ settings: "absent", sources: { "opencode.json": "absent" } })).toBeNull();
  });
});

describe("model form contract", () => {
  it("accepts catalog and plain explicit models", () => {
    expect(validateModelInput({ id: "grok" })).toBe(true);
    expect(validateModelInput({ id: "grok", label: "Grok", limit: { context: 1000, output: 100 } })).toBe(true);
  });

  it("accepts none/reasoning/reasoning_effort with balanced capabilities", () => {
    expect(validateModelInput({
      id: "x", limit: { context: 1, output: 1 }, reasoning: false, tool_call: true, effort_template: "none",
    })).toBe(true);
    expect(validateModelInput({
      id: "x", limit: { context: 1, output: 1 }, reasoning: true, tool_call: true,
      effort_template: "reasoning", efforts: ["high"],
    })).toBe(true);
    expect(validateModelInput({
      id: "x", limit: { context: 1, output: 1 }, reasoning: true, tool_call: true,
      effort_template: "reasoning_effort", efforts: ["high"],
    })).toBe(true);
  });

  it("accepts a positive integer context and rejects it alongside limit", () => {
    expect(validateModelInput({ id: "x", context: 1000 })).toBe(true);
    expect(validateModelInput({ id: "x", context: 1000, label: "X" })).toBe(true);
    expect(validateModelInput({ id: "x", context: 0 })).toBe(false);
    expect(validateModelInput({ id: "x", context: -1 })).toBe(false);
    expect(validateModelInput({ id: "x", context: 1.5 })).toBe(false);
    expect(validateModelInput({ id: "x", context: 1000, limit: { context: 1, output: 1 } })).toBe(false);
  });

  it("rejects unknowns, bad limits, duplicates, and unbalanced capabilities", () => {
    const bad = [
      { id: "x", extra: 1 },
      { id: "x", limit: { context: 0, output: 1 } },
      { id: "x", limit: { context: 1, output: 1 }, reasoning: false, tool_call: true, effort_template: "reasoning", efforts: ["high"] },
      { id: "x", limit: { context: 1, output: 1 }, reasoning: true, tool_call: false, effort_template: "none" },
      { id: "x", limit: { context: 1, output: 1 }, reasoning: true, tool_call: true, effort_template: "reasoning", efforts: ["high", "high"] },
      { id: "x", limit: { context: 1, output: 1 }, reasoning: true, tool_call: true, effort_template: "reasoning" },
      { id: "jaxflow-builder-default" },
      { id: "x", efforts: ["high"], reasoning: true, tool_call: true, effort_template: "none", limit: { context: 1, output: 1 } },
    ];
    for (const m of bad) expect(validateModelInput(m)).toBe(false);
  });
});

describe("current preview gate", () => {
  let draft: SettingsDraft;
  let rev: EditorRevision;
  beforeEach(() => {
    draft = JSON.parse(JSON.stringify(BOOTSTRAP_DRAFT));
    rev = { settings: "a".repeat(32), sources: { "opencode.json": "absent", "opencode.jsonc": "absent" } };
  });

  it("fingerprint binds the full draft and complete editor/source revision", () => {
    const fp = previewFingerprint(draft, rev);
    expect(fp).toBe(previewFingerprint(JSON.parse(JSON.stringify(draft)), rev));
    draft.builders.default.model = "grok-4.6b";
    expect(previewFingerprint(draft, rev)).not.toBe(fp);
    expect(previewFingerprint(draft, null)).toBeNull();
    expect(previewFingerprint(draft, { ...rev, sources: { "opencode.json": "b".repeat(64), "opencode.jsonc": "absent" } })).not.toBe(fp);
  });

});

describe("activation save gate", () => {
  function gate(over: Partial<ActivationGate> = {}): ActivationGate {
    return {
      firstActivation: false,
      dirty: false,
      revision: { settings: "a".repeat(32), sources: { "opencode.json": "absent", "opencode.jsonc": "absent" } },
      conflict: false,
      valid: true,
      busy: false,
      acknowledged: false,
      ...over,
    };
  }

  it("normal dirty valid save is allowed without any preview", () => {
    expect(activationSaveEnabled(gate({ dirty: true }))).toBe(true);
  });

  it("unchanged normal save stays disabled", () => {
    expect(activationSaveEnabled(gate())).toBe(false);
  });

  it("initial activation needs explicit acknowledgement even when unchanged", () => {
    expect(activationSaveEnabled(gate({ firstActivation: true }))).toBe(false);
    expect(activationSaveEnabled(gate({ firstActivation: true, acknowledged: true }))).toBe(true);
  });

  it("invalid selection, missing revision, conflict or busy always block", () => {
    expect(activationSaveEnabled(gate({ dirty: true, valid: false }))).toBe(false);
    expect(activationSaveEnabled(gate({ dirty: true, revision: null }))).toBe(false);
    expect(activationSaveEnabled(gate({ dirty: true, conflict: true }))).toBe(false);
    expect(activationSaveEnabled(gate({ dirty: true, busy: true }))).toBe(false);
    expect(activationSaveEnabled(gate({ firstActivation: true, acknowledged: true, valid: false }))).toBe(false);
    expect(activationSaveEnabled(gate({ firstActivation: true, acknowledged: true, busy: true }))).toBe(false);
  });
});

describe("draft validity", () => {
  const validDraft = (): SettingsDraft => JSON.parse(JSON.stringify(BOOTSTRAP_DRAFT));
  const rows = (): ConnectionChoice[] => [
    conn({
      id: "xai",
      models: [model({ id: "grok-4.6", efforts: ["high", "xhigh"] })],
    }),
    conn({
      id: "openrouter",
      adapter: "openrouter",
      models: [model({ id: "deepseek/deepseek-v4-flash-0731", efforts: ["high"] })],
    }),
  ];

  it("accepts both credentialed selections with compatible configured models", () => {
    expect(draftIsValid(validDraft(), rows())).toBe(true);
  });

  it("rejects a builder whose connection is not credentialed", () => {
    const connections = rows();
    connections[0] = conn({ id: "xai", health: "missing", credential: null, models: [model({ id: "grok-4.6", efforts: ["high"] })] });
    expect(draftIsValid(validDraft(), connections)).toBe(false);
  });

  it("rejects a builder whose model is not compatible/configured", () => {
    const connections = rows();
    connections[0] = conn({ id: "xai", models: [model({ id: "grok-4.6", compatible: false, efforts: ["high"] })] });
    expect(draftIsValid(validDraft(), connections)).toBe(false);
  });

  it("rejects an effort that the configured model does not offer", () => {
    const d = validDraft();
    d.builders.default.effort = "unknown-effort";
    expect(draftIsValid(d, rows())).toBe(false);
  });

  it("allows a null effort even when the model offers efforts", () => {
    const d = validDraft();
    d.builders.default.effort = null;
    expect(draftIsValid(d, rows())).toBe(true);
  });

  it("marks a pending connection unselectable and invalid for both profiles", () => {
    const connections = applyPendingOperations(rows(), [{ provider_id: "xai" }]);
    expect(connections.find((c) => c.id === "xai")?.pending).toBe(true);
    expect(selectableConnections(connections).map((c) => c.id)).toEqual(["openrouter"]);
    expect(draftIsValid(validDraft(), connections)).toBe(false);
  });

  it("leaves connections untouched when nothing is pending", () => {
    const input = rows();
    expect(applyPendingOperations(input, [])).toBe(input);
    expect(applyPendingOperations(input, [{ provider_id: "ghost" }])[0].pending).toBeUndefined();
  });

  it("rejects empty reviewer fields and unsupported reviewer efforts", () => {
    const empty = validDraft();
    empty.reviewers.claude.model = "";
    expect(draftIsValid(empty, rows())).toBe(false);
    const badEffort = validDraft();
    badEffort.reviewers.codex.effort = "max";
    expect(draftIsValid(badEffort, rows())).toBe(false);
  });
});
