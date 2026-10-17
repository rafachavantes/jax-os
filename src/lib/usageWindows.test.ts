import { describe, expect, it } from "vitest";
import { formatResetLine, highestWindow } from "./usageWindows";
import type { ProviderUsage } from "@/server/collectors/subscriptions";

const t = (key: string) => key;
// a values-capturing variant, needed wherever a test must see the actual interpolation.
const tv = (key: string, values?: Record<string, string | number>) => (values ? `${key}:${JSON.stringify(values)}` : key);

describe("highestWindow — the ring shows whichever window is closest to its cap", () => {
  it("picks weekly over fiveHour when weekly is higher", () => {
    const usage: ProviderUsage = {
      provider: "anthropic", plan: "max",
      fiveHour: { usedPercent: 40, resetsAt: null },
      weekly: { usedPercent: 85, resetsAt: "2026-09-20T00:00:00.000Z" },
      models: [],
    };
    expect(highestWindow(usage, t)?.percent).toBe(85);
  });

  it("picks a per-model weekly window over the top-level weekly one when it's higher", () => {
    const usage: ProviderUsage = {
      provider: "anthropic", plan: "max",
      fiveHour: { usedPercent: 10, resetsAt: null },
      weekly: { usedPercent: 20, resetsAt: null },
      models: [{ name: "Opus", weekly: { usedPercent: 92, resetsAt: null } }],
    };
    expect(highestWindow(usage, t)?.percent).toBe(92);
  });

  it("returns null when the provider has no windows at all", () => {
    const usage: ProviderUsage = { provider: "opencode", plan: null, fiveHour: null, weekly: null, models: [] };
    expect(highestWindow(usage, t)).toBeNull();
  });

  it("picks monthly over fiveHour/weekly when it is the highest (OpenCode GO)", () => {
    const usage: ProviderUsage = {
      provider: "opencode", plan: null,
      fiveHour: { usedPercent: 20, resetsAt: null },
      weekly: { usedPercent: 50, resetsAt: null },
      monthly: { usedPercent: 91, resetsAt: "2026-10-01T00:00:00.000Z" },
      models: [],
    };
    expect(highestWindow(usage, t)?.percent).toBe(91);
  });

  it("the tooltip lists every window", () => {
    const usage: ProviderUsage = {
      provider: "codex", plan: null,
      fiveHour: { usedPercent: 10, resetsAt: null },
      weekly: { usedPercent: 20, resetsAt: "2026-09-20T00:00:00.000Z" },
      models: [],
    };
    const w = highestWindow(usage, t);
    expect(w?.tooltip).toContain("10%");
    expect(w?.tooltip).toContain("20%");
  });
});

// Round-2 F1 precedent (mission.test.ts formatClock): the cockpit is single-user, host TZ
// America/Sao_Paulo (AGENTS.md), never UTC — formatResetLine renders in the HOST's local
// timezone (no `timeZone` option). Hardcoding an assumed offset breaks under the real host
// offset, so the expected weekday/time strings below are computed with the SAME Intl call /
// Date accessors the implementation uses, making the assertion hold under whatever TZ the
// test actually runs in.
function expectedTime(ms: number): string {
  const d = new Date(ms);
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}
function expectedWeekday(ms: number, locale: string): string {
  return new Intl.DateTimeFormat(locale, { weekday: "long" }).format(ms);
}

describe("formatResetLine — exact reset time, never vague text", () => {
  const NOW = Date.parse("2026-09-19T12:00:00.000Z");

  it("null resetsAt or an unparseable string omits the line", () => {
    expect(formatResetLine(null, NOW, "en-US", t)).toBeNull();
    expect(formatResetLine("not-a-date", NOW, "en-US", t)).toBeNull();
  });

  it("a resetsAt already in the past (or exactly now) is stale — omit the line, never 'in 0 min'", () => {
    expect(formatResetLine(new Date(NOW - 60_000).toISOString(), NOW, "en-US", t)).toBeNull();
    expect(formatResetLine(new Date(NOW).toISOString(), NOW, "en-US", t)).toBeNull();
  });

  it("under 1h: exact minutes, rounded up", () => {
    expect(formatResetLine(new Date(NOW + 15 * 60_000).toISOString(), NOW, "en-US", tv)).toBe('rings.resetInMinutes:{"minutes":15}');
    // 40m30s out rounds up to 41, never an optimistic 40
    expect(formatResetLine(new Date(NOW + 40 * 60_000 + 30_000).toISOString(), NOW, "en-US", tv)).toBe('rings.resetInMinutes:{"minutes":41}');
  });

  it("under 24h: exact hours + minutes", () => {
    expect(formatResetLine(new Date(NOW + 4 * 3_600_000 + 37 * 60_000).toISOString(), NOW, "en-US", tv)).toBe(
      'rings.resetInHoursMinutes:{"hours":4,"minutes":37}',
    );
  });

  it("24h boundary: strictly under 24h always uses hours+minutes; at/over 24h switches to the weekday+time form", () => {
    expect(formatResetLine(new Date(NOW + 23 * 3_600_000 + 59 * 60_000).toISOString(), NOW, "en-US", tv)).toBe(
      'rings.resetInHoursMinutes:{"hours":23,"minutes":59}',
    );
    const atMs = NOW + 24 * 3_600_000;
    expect(formatResetLine(new Date(atMs).toISOString(), NOW, "en-US", tv)).toBe(
      `rings.resetOnAt:{"weekday":"${expectedWeekday(atMs, "en-US")}","time":"${expectedTime(atMs)}"}`,
    );
  });

  it("24h or more: weekday + local 24h time, locale-aware weekday, in both locales", () => {
    const atMs = NOW + 2 * 86_400_000 + 11 * 3_600_000;
    const target = new Date(atMs).toISOString();
    expect(formatResetLine(target, NOW, "en-US", tv)).toBe(
      `rings.resetOnAt:{"weekday":"${expectedWeekday(atMs, "en-US")}","time":"${expectedTime(atMs)}"}`,
    );
    expect(formatResetLine(target, NOW, "pt-BR", tv)).toBe(
      `rings.resetOnAt:{"weekday":"${expectedWeekday(atMs, "pt-BR")}","time":"${expectedTime(atMs)}"}`,
    );
  });

  it("7 days or more: date instead of an ambiguous weekday (monthly windows)", () => {
    const atMs = NOW + 20 * 86_400_000;
    const date = new Intl.DateTimeFormat("pt-BR", { day: "2-digit", month: "2-digit" }).format(atMs);
    expect(formatResetLine(new Date(atMs).toISOString(), NOW, "pt-BR", tv)).toBe(
      `rings.resetOnAt:{"weekday":"${date}","time":"${expectedTime(atMs)}"}`,
    );
  });
});
