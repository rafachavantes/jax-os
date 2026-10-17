import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const fsState = vi.hoisted(() => ({ files: {} as Record<string, string> }));
vi.mock("node:fs", () => ({
  readFileSync: (path: string) => {
    if (!(path in fsState.files)) throw new Error("ENOENT");
    return fsState.files[path];
  },
}));
vi.mock("node:os", () => ({ homedir: () => "/home/test" }));

import {
  getSubscriptions, nativeCredentialsConfigured, parseClaudeCreds, parseClaudeUsage, parseCodexCreds, parseCodexUsage, parseOpenCodeGoUsage,
  type ProviderUsage, type SubscriptionsFetchers,
} from "./subscriptions";

const usage = (provider: ProviderUsage["provider"]): ProviderUsage =>
  ({ provider, plan: null, fiveHour: null, weekly: null, models: [] });

const fakeFetchers = (): SubscriptionsFetchers => ({
  claude: vi.fn().mockResolvedValue(usage("anthropic")),
  codex: vi.fn().mockResolvedValue(usage("codex")),
  opencodeGo: vi.fn().mockResolvedValue(usage("opencode")),
});

describe("getSubscriptions — per-provider gate (spec decision 14)", () => {
  it("a disabled slot returns {ok:false,error:disabled}", async () => {
    const result = await getSubscriptions({ claude: false, codex: true, opencodeGo: true }, fakeFetchers());
    expect(result.anthropic).toEqual({ ok: false, error: "disabled" });
  });
  it("turning off exactly one provider leaves the other two slots' pass/fail behavior untouched", async () => {
    const fetchers = fakeFetchers();
    const allOn = await getSubscriptions({ claude: true, codex: true, opencodeGo: true }, fetchers);
    const codexOff = await getSubscriptions({ claude: true, codex: false, opencodeGo: true }, fetchers);
    expect(codexOff.anthropic).toEqual(allOn.anthropic);
    expect(codexOff.opencode).toEqual(allOn.opencode);
    expect(codexOff.codex).toEqual({ ok: false, error: "disabled" });
  });
});

// F4 (cold review): the injectable fetch seam's whole point — every provider proven, in one
// matrix, to never invoke its OWN fetcher when disabled.
describe("getSubscriptions — per-provider fetch seam matrix (F4, cold review)", () => {
  it.each([
    { name: "claude", enabled: { claude: false, codex: true, opencodeGo: true }, slot: "anthropic" as const, fetcherKey: "claude" as const },
    { name: "codex", enabled: { claude: true, codex: false, opencodeGo: true }, slot: "codex" as const, fetcherKey: "codex" as const },
    { name: "opencodeGo", enabled: { claude: true, codex: true, opencodeGo: false }, slot: "opencode" as const, fetcherKey: "opencodeGo" as const },
  ])("disabled $name returns {ok:false,error:disabled} without invoking its own fetcher", async ({ enabled, slot, fetcherKey }) => {
    const fetchers = fakeFetchers();
    const result = await getSubscriptions(enabled, fetchers);
    expect(result[slot]).toEqual({ ok: false, error: "disabled" });
    expect(fetchers[fetcherKey]).not.toHaveBeenCalled();
  });
});

describe("parseClaudeUsage", () => {
  it("maps camelCase windows and reads utilization as percent", () => {
    const u = parseClaudeUsage(
      {
        fiveHour: { utilization: 42, resetsAt: "2026-07-08T15:00:00Z" },
        sevenDay: { utilization: 87, resetsAt: "2026-07-12T00:00:00Z" },
        sevenDayOpus: { utilization: 10, resetsAt: "2026-07-12T00:00:00Z" },
      },
      "max",
    );
    expect(u.fiveHour).toEqual({ usedPercent: 42, resetsAt: "2026-07-08T15:00:00Z" });
    expect(u.weekly?.usedPercent).toBe(87);
    expect(u.models).toEqual([{ name: "Opus", weekly: { usedPercent: 10, resetsAt: "2026-07-12T00:00:00Z" } }]);
    expect(u.plan).toBe("max");
  });

  it("treats utilization 1 as 1%, not a full window", () => {
    const u = parseClaudeUsage({ fiveHour: { utilization: 1, resetsAt: "x" } }, null);
    expect(u.fiveHour?.usedPercent).toBe(1);
  });

  it("accepts snake_case aliases and missing windows", () => {
    const u = parseClaudeUsage({ five_hour: { utilization: 50, resets_at: "x" } }, null);
    expect(u.fiveHour).toEqual({ usedPercent: 50, resetsAt: "x" });
    expect(u.weekly).toBeNull();
    expect(u.models).toEqual([]);
  });
});

describe("parseClaudeCreds", () => {
  const NOW = 1_783_339_200_000;
  it("extracts token and plan", () => {
    const c = parseClaudeCreds(
      { claudeAiOauth: { accessToken: "tok", expiresAt: NOW + 3_600_000, subscriptionType: "max" } },
      NOW,
    );
    expect(c).toEqual({ token: "tok", plan: "max" });
  });
  it("throws expired when under 5 minutes left", () => {
    expect(() =>
      parseClaudeCreds({ claudeAiOauth: { accessToken: "tok", expiresAt: NOW + 60_000 } }, NOW),
    ).toThrow("expired");
  });
  it("throws missing-credentials on malformed input", () => {
    expect(() => parseClaudeCreds({}, NOW)).toThrow("missing-credentials");
  });
});

describe("parseCodexUsage", () => {
  it("maps primary→5h and secondary→weekly with epoch reset", () => {
    const u = parseCodexUsage({
      plan_type: "plus",
      rate_limit: {
        primary_window: { used_percent: 12, limit_window_seconds: 18000, reset_at: 1783350000 },
        secondary_window: { usage_percent: 55.4, limit_window_seconds: 604800, reset_at: 1783600000 },
      },
    });
    expect(u.fiveHour).toEqual({ usedPercent: 12, resetsAt: new Date(1783350000 * 1000).toISOString() });
    expect(u.weekly?.usedPercent).toBe(55);
    expect(u.plan).toBe("plus");
  });

  it("classifies a lone window by its length", () => {
    const u = parseCodexUsage({
      rate_limit: { primary_window: { used_percent: 9, limit_window_seconds: 604800 } },
    });
    expect(u.fiveHour).toBeNull();
    expect(u.weekly?.usedPercent).toBe(9);
  });
});

describe("parseCodexCreds", () => {
  it("extracts access token + account id", () => {
    expect(parseCodexCreds({ tokens: { access_token: "t", account_id: "a" } })).toEqual({ token: "t", accountId: "a" });
  });
  it("throws missing-credentials without tokens", () => {
    expect(() => parseCodexCreds({ OPENAI_API_KEY: "k" })).toThrow("missing-credentials");
  });
});

describe("nativeCredentialsConfigured (diff review e52d9e6dc555 F3)", () => {
  const CLAUDE_CREDS = "/home/test/.claude/.credentials.json";
  const CODEX_CREDS = "/home/test/.codex/auth.json";

  beforeEach(() => {
    fsState.files = {};
    delete process.env.OPENCODE_GO_API_KEY;
  });
  afterEach(() => {
    delete process.env.OPENCODE_GO_API_KEY;
  });

  it("claude true only when ~/.claude/.credentials.json has a usable oauth token", () => {
    expect(nativeCredentialsConfigured().claude).toBe(false); // absent
    fsState.files[CLAUDE_CREDS] = JSON.stringify({ claudeAiOauth: { accessToken: "tok", expiresAt: Date.now() + 3_600_000 } });
    expect(nativeCredentialsConfigured().claude).toBe(true);
    fsState.files[CLAUDE_CREDS] = JSON.stringify({ claudeAiOauth: { accessToken: "tok", expiresAt: Date.now() + 60_000 } });
    expect(nativeCredentialsConfigured().claude).toBe(false); // same expired rule parseClaudeCreds throws on
  });

  it("codex true only when ~/.codex/auth.json parses via parseCodexCreds", () => {
    expect(nativeCredentialsConfigured().codex).toBe(false);
    fsState.files[CODEX_CREDS] = JSON.stringify({ tokens: { access_token: "t" } });
    expect(nativeCredentialsConfigured().codex).toBe(true);
    fsState.files[CODEX_CREDS] = JSON.stringify({ OPENAI_API_KEY: "k" });
    expect(nativeCredentialsConfigured().codex).toBe(false);
  });

  it("opencodeGo true only when OPENCODE_GO_API_KEY is set", () => {
    expect(nativeCredentialsConfigured().opencodeGo).toBe(false);
    process.env.OPENCODE_GO_API_KEY = "x";
    expect(nativeCredentialsConfigured().opencodeGo).toBe(true);
  });

  it("never returns the token value itself — booleans only", () => {
    fsState.files[CLAUDE_CREDS] = JSON.stringify({ claudeAiOauth: { accessToken: "SECRET-TOKEN", expiresAt: Date.now() + 3_600_000 } });
    process.env.OPENCODE_GO_API_KEY = "SECRET-ENV";
    const result = nativeCredentialsConfigured();
    expect(Object.values(result).every((v) => typeof v === "boolean")).toBe(true);
    expect(JSON.stringify(result)).not.toContain("SECRET");
  });
});

describe("parseOpenCodeGoUsage", () => {
  it("maps rolling/weekly/monthly windows from the proven response shape", () => {
    const u = parseOpenCodeGoUsage({
      usage: {
        rolling: { status: "ok", percent: 12, resetsAt: "2026-09-19T20:00:00Z" },
        weekly: { status: "ok", percent: 48.6, resetsAt: "2026-09-22T00:00:00Z" },
        monthly: { status: "warning", percent: 91, resetsAt: "2026-10-01T00:00:00Z" },
      },
    });
    expect(u.fiveHour).toEqual({ usedPercent: 12, resetsAt: "2026-09-19T20:00:00Z" });
    expect(u.weekly).toEqual({ usedPercent: 49, resetsAt: "2026-09-22T00:00:00Z" });
    expect(u.monthly).toEqual({ usedPercent: 91, resetsAt: "2026-10-01T00:00:00Z" });
    expect(u.plan).toBeNull();
    expect(u.models).toEqual([]);
  });

  it("tolerates a missing usage block", () => {
    const u = parseOpenCodeGoUsage({});
    expect(u.fiveHour).toBeNull();
    expect(u.weekly).toBeNull();
    expect(u.monthly).toBeNull();
  });
});
