import { describe, expect, it, vi } from "vitest";
import type { HelperRunner } from "./agent-settings";
import {
  applyProviders,
  eligibleSecretReadAllowed,
  parseProviderConnection,
  previewProviders,
  sanitizeConnections,
  snapshotProviders,
} from "./opencode-providers";

const REV = {
  settings: "absent",
  sources: { "opencode.json": "absent", "opencode.jsonc": "absent" },
};
const OP = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee";

const LEAKY = {
  editor_revision: REV,
  connections: [
    {
      id: "fixture",
      label: "Fixture",
      adapter: "xai",
      base_url: "https://api.x.ai/v1",
      auth: "api-key",
      health: "registered",
      credential: { kind: "native" },
      editable: true,
      reason: null,
      used_by: ["default", "fallback"],
      models: [
        {
          id: "wire-real-model",
          label: "Wire",
          origin: "override",
          efforts: ["high"],
          no_effort: false,
          compatible: true,
          reason: null,
          context: 128000,
          effort_template: "reasoning",
        },
        {
          id: "jaxflow-builder-default",
          label: "alias",
          origin: "override",
          efforts: [],
          no_effort: true,
          compatible: true,
          reason: null,
          context: null,
          effort_template: null,
        },
      ],
      apiKey: "s3cret",
      options: { apiKey: "s3cret", Authorization: "Bearer s3cret" },
    },
  ],
  catalog: [{ id: "catalog-only", apiKey: "s3cret" }],
};

describe("sanitizeConnections", () => {
  it("drops catalog-only rows, jaxflow aliases, and any raw auth object", () => {
    const connections = sanitizeConnections(LEAKY);
    expect(connections.map((row) => row.id)).toEqual(["fixture"]);
    expect(connections[0]?.models.map((row) => row.id)).toEqual(["wire-real-model"]);
    expect(connections[0]).not.toHaveProperty("apiKey");
    expect(connections[0]).not.toHaveProperty("options");
    expect(JSON.stringify(connections)).not.toContain("s3cret");
    expect(JSON.stringify(connections)).not.toContain("catalog-only");
  });

  it("keeps the sanitized context/effort_template DTO and drops unlisted native keys", () => {
    const connections = sanitizeConnections({
      connections: [
        {
          id: "fixture",
          models: [
            {
              id: "m",
              label: "M",
              origin: "override",
              efforts: ["high"],
              no_effort: false,
              compatible: true,
              reason: null,
              context: 128000,
              effort_template: "reasoning",
              limit: { output: 163840 },
              options: { secret: true },
            },
          ],
          apiKey: "s3cret",
        },
      ],
    });
    expect(connections[0]?.models[0]).toEqual({
      id: "m",
      label: "M",
      origin: "override",
      efforts: ["high"],
      no_effort: false,
      compatible: true,
      reason: null,
      context: 128000,
      effort_template: "reasoning",
    });
    expect(JSON.stringify(connections)).not.toContain("163840");
    expect(JSON.stringify(connections)).not.toContain("secret");
  });
});

describe("provider helper wrappers", () => {
  it("snapshot omits native_metadata and never paths", async () => {
    const run = vi.fn<HelperRunner>(async () => LEAKY);
    const result = await snapshotProviders({ providers: { fixture: {} } }, run);
    expect(run).toHaveBeenCalledWith("snapshot", {});
    expect(run.mock.calls[0]?.[1]).not.toHaveProperty("native_metadata");
    expect(run.mock.calls[0]?.[1]).not.toHaveProperty("paths");
    expect(result.connections[0]?.id).toBe("fixture");
    expect(result.connections[0]?.adapter).toBe("xai");
    expect(JSON.stringify(result)).not.toContain("s3cret");
  });

  it("preview and apply pass provider intents without paths", async () => {
    const intent = { kind: "save-connection", connection: { id: "fixture", adapter: "xai" } };
    const run = vi.fn<HelperRunner>(async (op) => (
      op === "apply" ? { effect: "published", editor_revision: REV, settings: null } : { affected: ["opencode.json"], aliases: { before: [], after: [] } }
    ));
    await previewProviders(intent, REV, {}, run);
    await applyProviders(intent, REV, OP, {}, run);
    expect(run.mock.calls[0]?.[0]).toBe("preview");
    expect(run.mock.calls[1]?.[0]).toBe("apply");
    for (const [, body] of run.mock.calls) {
      expect(body).not.toHaveProperty("paths");
      expect(body.intent).toEqual(intent);
    }
  });
});

describe("parseProviderConnection", () => {
  it("accepts supported adapters with an optional label and https base url", () => {
    expect(parseProviderConnection({ id: "acme", adapter: "xai" })).toEqual({ id: "acme", adapter: "xai" });
    expect(parseProviderConnection({
      id: "acme", adapter: "openai-compatible", label: "Acme", base_url: "https://api.example.com/v1",
    })).toEqual({ id: "acme", adapter: "openai-compatible", label: "Acme", base_url: "https://api.example.com/v1" });
  });

  it("refuses unknown adapters, bad ids, extra keys and unsafe urls", () => {
    for (const value of [
      null,
      { id: "acme", adapter: "oauth" },
      { id: "-bad", adapter: "xai" },
      { id: "acme", adapter: "xai", extra: 1 },
      { id: "acme", adapter: "openai-compatible", base_url: "http://127.0.0.1/v1" },
      { id: "acme", adapter: "openai-compatible", base_url: "https://api.example.com/v1?q=1" },
    ]) {
      expect(parseProviderConnection(value)).toBeNull();
    }
  });
});

describe("eligibleSecretReadAllowed", () => {
  it("respects an existing connection's real auth and safety controls", () => {
    expect(eligibleSecretReadAllowed({ editable: true, auth: "api-key" }, "acme", null)).toBe(true);
    expect(eligibleSecretReadAllowed({ editable: false, auth: "api-key" }, "acme", null)).toBe(false);
    expect(eligibleSecretReadAllowed({ editable: true, auth: "oauth" }, "acme", null)).toBe(false);
    expect(eligibleSecretReadAllowed({ editable: true, auth: "unknown" }, "acme", null)).toBe(false);
  });

  it("allows only a matching new draft when no connection exists", () => {
    const draft = { id: "fresh", adapter: "xai" };
    expect(eligibleSecretReadAllowed(undefined, "fresh", draft)).toBe(true);
    expect(eligibleSecretReadAllowed(undefined, "other", draft)).toBe(false);
    expect(eligibleSecretReadAllowed(undefined, "fresh", null)).toBe(false);
  });
});
