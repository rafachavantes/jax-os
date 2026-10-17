import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

const harness = vi.hoisted(() => ({ mission: new Map<string, { data: unknown }>(), subscriptions: { data: undefined as unknown } }));

vi.mock("next-intl", () => ({ useTranslations: () => (key: string) => key, useLocale: () => "en-US" }));
const nav = vi.hoisted(() => ({ pathname: "/" }));
vi.mock("next/navigation", () => ({ usePathname: () => nav.pathname }));
vi.mock("next/link", () => ({
  default: ({ href, children }: { href: string; children: React.ReactNode }) => createElement("a", { href }, children),
}));
vi.mock("./MobileNav", () => ({ MobileNav: () => null }));
vi.mock("./LanguageSelector", () => ({ LanguageSelector: () => null }));
vi.mock("./ThemeToggle", () => ({ ThemeToggle: () => null }));
vi.mock("./mission/AfkToggle", () => ({ AfkToggle: () => createElement("div", { "data-testid": "afk-toggle" }) }));
vi.mock("@/lib/useMission", () => ({
  useMission: (endpoint: string) => harness.mission.get(endpoint) ?? { data: undefined },
}));
vi.mock("@/lib/useTokensApi", () => ({ useTokensApi: () => harness.subscriptions }));

import { hubStatusKey, liveSourceCount, meterShown, pillReadinessOf, Topbar } from "./Topbar";

describe("pillReadinessOf (round-1 F1)", () => {
  it("is ready once both resolve (regardless of prs, never passed in), loading while either is unresolved, failed when either errors", () => {
    expect(pillReadinessOf({ ok: true }, { ok: true })).toEqual({ kind: "ready" });
    expect(pillReadinessOf(undefined, { ok: true })).toEqual({ kind: "loading", pending: ["projects"] });
    expect(pillReadinessOf({ ok: false }, { ok: true })).toEqual({ kind: "failed", failed: ["projects"], pending: [] });
  });
});

const usage = (provider: string) => ({ provider, plan: null, fiveHour: { usedPercent: 12, resetsAt: null }, weekly: null, monthly: null, models: [] });
const RING = /<svg width="40" height="40"/g;
function seedMission() {
  harness.mission.set("projects", { data: { ok: true, data: { projects: [], skipped: 0 } } });
  harness.mission.set("hub", { data: { ok: true, data: { byProject: {}, historicalTruncated: false, codexSource: "ok", tmuxSource: "ok" } } });
}
function renderWith(payload: unknown) {
  seedMission();
  harness.subscriptions.data = payload === undefined ? undefined : { ok: true, data: payload };
  try {
    return renderToStaticMarkup(createElement(Topbar, { initialTheme: "dark", sidebar: "expanded", onToggleSidebar: () => {} }));
  } finally {
    harness.subscriptions.data = undefined;
  }
}

describe("Topbar rendering", () => {
  it("renders the worst state, not a skeleton, alongside the ring slots of enabled agents", () => {
    harness.mission.set("projects", { data: { ok: true, data: { projects: [], skipped: 0 } } });
    harness.mission.set("hub", { data: { ok: true, data: { byProject: {}, historicalTruncated: false, codexSource: "ok", tmuxSource: "ok" } } });
    const html = renderWith({ anthropic: { ok: true, data: usage("anthropic") }, codex: { ok: true, data: usage("codex") }, opencode: { ok: true, data: usage("opencode") } });
    expect(html).not.toContain("animate-pulse");
    expect(html).toContain("states.idle");
    expect((html.match(/<svg width="40" height="40"/g) ?? []).length).toBe(3);
    // MOA-486 follow-up: each ring's slot is now a tappable button, aria-labelled with the provider name.
    expect(html).toContain('aria-label="Anthropic"');
    expect(html).toContain('aria-label="OpenAI Codex"');
    expect(html).toContain('aria-label="OpenCode GO"');
    expect((html.match(/<button/g) ?? []).length).toBeGreaterThanOrEqual(3);
    // MOA-486 follow-up: a short provider name renders under every ring, on every viewport.
    expect(html).toContain(">Claude<");
    expect(html).toContain(">Codex<");
    expect(html).toContain(">Go<");
  });
});

describe("liveSourceCount (spec §7, Decision 14)", () => {
  it("counts exactly the 4 named signals; truncated codex still counts as live", () => {
    expect(liveSourceCount(true, true, "ok", "ok")).toBe(4);
    expect(liveSourceCount(true, true, "ok", "truncated")).toBe(4);
    expect(liveSourceCount(true, true, "failed", "ok")).toBe(3);
    expect(liveSourceCount(true, true, "ok", "failed")).toBe(3);
    expect(liveSourceCount(false, undefined, "failed", "failed")).toBe(0);
  });
});

describe("Topbar — composed subtitle (spec §7, Decision 14)", () => {
  it("renders the composed key, not the static tagline", () => {
    // t() echoes only the key (this file's mock) — asserts the right key is reached, not the interpolated count; liveSourceCount/hubStatusKey's own logic has its dedicated unit test.
    harness.mission.set("projects", { data: { ok: true, data: { projects: [], skipped: 0 } } });
    harness.mission.set("hub", { data: { ok: true, data: { byProject: {}, historicalTruncated: false, codexSource: "ok", tmuxSource: "ok" } } });
    const html = renderToStaticMarkup(createElement(Topbar, { initialTheme: "dark", sidebar: "expanded", onToggleSidebar: () => {} }));
    expect(html).toContain("topbar.liveSources");
    expect(html).toContain("topbar.hubOk");
    expect(html).not.toContain("subtitle.mission");
  });
});

describe("Topbar — locale + theme controls move to the MobileNav drawer below md", () => {
  it("wraps LanguageSelector/ThemeToggle in a group hidden below md, shown at md+ (CSS-only split, no behaviour change at desktop)", () => {
    harness.mission.set("projects", { data: { ok: true, data: { projects: [], skipped: 0 } } });
    harness.mission.set("hub", { data: { ok: true, data: { byProject: {}, historicalTruncated: false, codexSource: "ok", tmuxSource: "ok" } } });
    const html = renderToStaticMarkup(createElement(Topbar, { initialTheme: "dark", sidebar: "expanded", onToggleSidebar: () => {} }));
    expect(html).toContain('class="hidden items-center gap-4 md:flex"');
  });
});

describe("Topbar — AFK toggle is always in the header", () => {
  it("renders the AFK toggle on the mission route", () => {
    harness.mission.set("projects", { data: { ok: true, data: { projects: [], skipped: 0 } } });
    harness.mission.set("hub", { data: { ok: true, data: { byProject: {}, historicalTruncated: false, codexSource: "ok", tmuxSource: "ok" } } });
    const html = renderToStaticMarkup(createElement(Topbar, { initialTheme: "dark", sidebar: "expanded", onToggleSidebar: () => {} }));
    expect(html).toContain('data-testid="afk-toggle"');
  });

  it("renders the AFK toggle on every other route too", () => {
    nav.pathname = "/files";
    try {
      const html = renderToStaticMarkup(createElement(Topbar, { initialTheme: "dark", sidebar: "expanded", onToggleSidebar: () => {} }));
      expect(html).toContain('data-testid="afk-toggle"');
    } finally {
      nav.pathname = "/";
    }
  });
});

describe("hubStatusKey (spec §7, Decision 14, round-1 F5 — tri-state + zero-source)", () => {
  it("loading/ok/degraded by hub state; liveCount 0 overrides all three to noSources", () => {
    expect(hubStatusKey(undefined, 2)).toBe("topbar.hubLoading");
    expect(hubStatusKey({ ok: true }, 4)).toBe("topbar.hubOk");
    expect(hubStatusKey({ ok: false }, 3)).toBe("topbar.hubDegraded");
    expect(hubStatusKey(undefined, 0)).toBe("topbar.noSources");
    expect(hubStatusKey({ ok: true }, 0)).toBe("topbar.noSources");
    expect(hubStatusKey({ ok: false }, 0)).toBe("topbar.noSources");
  });
});

describe("Topbar — pit pill count-first (spec item 6)", () => {
  it("the worst-state pill renders the count before the state label", () => {
    harness.mission.set("projects", { data: { ok: true, data: { projects: [], skipped: 0 } } });
    harness.mission.set("hub", { data: { ok: true, data: { byProject: {}, historicalTruncated: false, codexSource: "ok", tmuxSource: "ok" } } });
    const html = renderToStaticMarkup(createElement(Topbar, { initialTheme: "dark", sidebar: "expanded", onToggleSidebar: () => {} }));
    expect(html).toMatch(/>\s*0 states\.idle/);
  });
});

describe("Topbar — third ring is OpenCode GO", () => {
  it("renders an active ring (not the unavailable placeholder) for opencode data", () => {
    harness.mission.set("projects", { data: { ok: true, data: { projects: [], skipped: 0 } } });
    harness.mission.set("hub", { data: { ok: true, data: { byProject: {}, historicalTruncated: false, codexSource: "ok", tmuxSource: "ok" } } });
    harness.subscriptions.data = {
      ok: true,
      data: {
        anthropic: { ok: false, error: "missing-credentials" },
        codex: { ok: false, error: "missing-credentials" },
        opencode: { ok: true, data: { provider: "opencode", plan: null, fiveHour: { usedPercent: 12, resetsAt: null }, weekly: null, monthly: null, models: [] } },
      },
    };
    const html = renderToStaticMarkup(createElement(Topbar, { initialTheme: "dark", sidebar: "expanded", onToggleSidebar: () => {} }));
    harness.subscriptions.data = undefined;
    expect(html).toContain("12%");
  });
});

describe("Topbar meters follow the agents (MOA-504 D6)", () => {
  it("renders no ring slot until the payload exists", () => {
    const html = renderWith(undefined);
    expect(html.match(RING) ?? []).toHaveLength(0);
    expect(html).not.toContain('aria-label="Anthropic"');
  });

  it("omits slots whose result is disabled or missing-credentials, keeps the others", () => {
    const html = renderWith({
      anthropic: { ok: false, error: "disabled" },
      codex: { ok: false, error: "missing-credentials" },
      opencode: { ok: true, data: usage("opencode") },
    });
    expect(html.match(RING) ?? []).toHaveLength(1);
    expect(html).toContain('aria-label="OpenCode GO"');
    expect(html).not.toContain('aria-label="Anthropic"');
    expect(html).not.toContain('aria-label="OpenAI Codex"');
  });

  it("keeps the unavailable placeholder for expired / rate-limited (a credential exists, the meter is down)", () => {
    const html = renderWith({
      anthropic: { ok: false, error: "expired" },
      codex: { ok: false, error: "rate-limited" },
      opencode: { ok: false, error: "disabled" },
    });
    expect(html.match(RING) ?? []).toHaveLength(2);
    expect(html).toContain('aria-label="Anthropic"');
    expect(html).toContain('aria-label="OpenAI Codex"');
  });

  it("meterShown: only disabled / missing-credentials / no result hide a slot", () => {
    expect(meterShown(undefined)).toBe(false);
    expect(meterShown({ ok: false, error: "disabled" })).toBe(false);
    expect(meterShown({ ok: false, error: "missing-credentials" })).toBe(false);
    expect(meterShown({ ok: false, error: "expired" })).toBe(true);
    expect(meterShown({ ok: true, data: usage("codex") as never })).toBe(true);
  });
});
