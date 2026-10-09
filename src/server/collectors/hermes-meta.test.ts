import { afterEach, describe, expect, it, vi } from "vitest";
import { getHermesMeta, parsePromptSize, parseSkillsUsage, resetHermesPromptCache } from "./hermes-meta";

describe("parsePromptSize", () => {
  const payload = JSON.stringify({
    platform: "cli",
    model: "gpt-5.5",
    system_prompt: { chars: 28967, bytes: 29529 },
    skills_index: { chars: 8561, bytes: 8589 },
    memory: { chars: 2134, bytes: 2351 },
    user_profile: { chars: 1495, bytes: 1718 },
    tools: { count: 44, json_bytes: 61873 },
  });

  it("parses components in fixed order and totals bytes", () => {
    const b = parsePromptSize(payload);
    expect(b.components.map((c) => c.key)).toEqual([
      "systemPrompt", "skillsIndex", "memory", "userProfile", "toolSchemas",
    ]);
    expect(b.totalBytes).toBe(29529 + 8589 + 2351 + 1718 + 61873);
    expect(b.toolCount).toBe(44);
    expect(b.model).toBe("gpt-5.5");
  });

  it("skips CLI noise before the JSON and defaults missing fields to 0", () => {
    const b = parsePromptSize("  Bitwarden: applied 35 secrets\n" + JSON.stringify({ tools: {} }));
    expect(b.totalBytes).toBe(0);
    expect(b.platform).toBeNull();
  });

  it("throws when no JSON at all", () => {
    expect(() => parsePromptSize("boom")).toThrow();
  });

  it("skips brace-bearing banner fragments and parses the real payload", () => {
    const b = parsePromptSize(
      'Banner {"leaked":true} noise\n' +
        JSON.stringify({ tools: { count: 44, json_bytes: 61873 }, system_prompt: { bytes: 100 } }),
    );
    expect(b.toolCount).toBe(44);
    expect(b.components.find((c) => c.key === "systemPrompt")!.bytes).toBe(100);
  });
});

describe("parseSkillsUsage", () => {
  const NOW = Date.parse("2026-07-08T12:00:00Z");
  const D = 86_400_000;
  const skills = JSON.stringify({
    fresh: { state: "active", use_count: 9, last_used_at: new Date(NOW - 2 * D).toISOString() },
    stale: { state: "active", use_count: 5, last_used_at: new Date(NOW - 45 * D).toISOString() },
    unused: { state: "active", use_count: 0, last_used_at: null },
    archived: { state: "archived", use_count: 0, last_used_at: null },
  });

  it("flags active+never-used and active+stale, never archived; sorts by useCount desc", () => {
    const out = parseSkillsUsage(skills, NOW);
    expect(out.map((s) => s.name)).toEqual(["fresh", "stale", "unused", "archived"]);
    expect(out.find((s) => s.name === "fresh")!.flagged).toBe(false);
    expect(out.find((s) => s.name === "stale")!.flagged).toBe(true);
    expect(out.find((s) => s.name === "unused")!.flagged).toBe(true);
    expect(out.find((s) => s.name === "archived")!.flagged).toBe(false);
  });

  it("treats an unparseable last_used_at as stale (fails safe)", () => {
    const out = parseSkillsUsage(
      JSON.stringify({ corrupt: { state: "active", use_count: 5, last_used_at: "not-a-date" } }),
      NOW,
    );
    expect(out[0].flagged).toBe(true);
  });
});

const HERMES_OPTS = { encoding: "utf8" as const, timeout: 30_000, maxBuffer: 2 * 1024 * 1024 };
const SENTINEL = "SECRET_TOKEN=super-secret-cli-stderr";

describe("getHermesMeta prompt execution", () => {
  afterEach(() => {
    resetHermesPromptCache();
  });

  it("maps executor timeout to timeout and never leaks sentinel text", async () => {
    const exec = vi.fn(async () => {
      throw Object.assign(new Error(`Command failed: hermes ${SENTINEL}`), {
        code: "ETIMEDOUT",
        killed: true,
        stderr: SENTINEL,
        stdout: SENTINEL,
      });
    });
    const res = await getHermesMeta(exec);
    expect(res.prompt).toEqual({ ok: false, error: "timeout" });
    expect(JSON.stringify(res.prompt)).not.toContain(SENTINEL);
    expect(exec).toHaveBeenCalledWith(
      expect.stringMatching(/hermes$/),
      ["prompt-size", "--json"],
      HERMES_OPTS,
    );
  });

  it("maps unavailable/nonzero to unavailable without raw exception text", async () => {
    const exec = vi.fn(async () => {
      throw Object.assign(new Error(`spawn hermes ENOENT ${SENTINEL}`), { code: "ENOENT", stderr: SENTINEL });
    });
    const res = await getHermesMeta(exec);
    expect(res.prompt).toEqual({ ok: false, error: "unavailable" });
    expect(JSON.stringify(res.prompt)).not.toContain(SENTINEL);
  });

  it("maps overflow and malformed output to invalid-output", async () => {
    const overflow = vi.fn(async () => {
      throw Object.assign(new Error(`maxBuffer ${SENTINEL}`), {
        code: "ERR_CHILD_PROCESS_STDIO_MAXBUFFER",
        stdout: SENTINEL,
      });
    });
    expect((await getHermesMeta(overflow)).prompt).toEqual({ ok: false, error: "invalid-output" });
    resetHermesPromptCache();
    const malformed = vi.fn(async () => ({ stdout: "not json at all" }));
    expect((await getHermesMeta(malformed)).prompt).toEqual({ ok: false, error: "invalid-output" });
    expect(JSON.stringify((await getHermesMeta(malformed)).prompt)).not.toContain(SENTINEL);
  });

  it("caches a classified failure for ten minutes without a retry", async () => {
    const exec = vi.fn(async () => {
      throw Object.assign(new Error(SENTINEL), { code: 1 });
    });
    await getHermesMeta(exec);
    await getHermesMeta(exec);
    expect(exec).toHaveBeenCalledOnce();
  });
});
