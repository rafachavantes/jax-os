import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ProviderUsageModal, UsageRing, UsageRingUnavailable } from "./UsageRing";
import type { ProviderUsage, SubscriptionsPayload } from "@/server/collectors/subscriptions";

const t = (key: string) => key;

// Header cleanup: the ring is the compact indicator on its own now — no adjacent text
// label / reset line at lg+ (that detail moved one tap away, into ProviderUsageModal).
describe("UsageRing — the ring IS the compact indicator, no adjacent text label", () => {
  it("renders the percent inside the button that wraps the svg, and nothing else beside it", () => {
    const html = renderToStaticMarkup(
      createElement(UsageRing, {
        percent: 82,
        tooltip: "Claude · weekly: 82% (resets Sat)",
        providerName: "Anthropic",
        shortName: "Claude",
        onOpenDetails: () => {},
      }),
    );
    // the ring is a real <button>, tap target, accessible name = provider name, tooltip kept for desktop hover
    expect(html).toContain("<button");
    expect(html).toContain('aria-label="Anthropic"');
    expect(html).toContain('title="Claude · weekly: 82% (resets Sat)"');
    // the percent overlay AND the short name both live inside the same button, nothing else
    const buttonMatch = html.match(/<button[^>]*>([\s\S]*?)<\/button>/);
    expect(buttonMatch).not.toBeNull();
    expect(buttonMatch![1]).toContain("<svg");
    expect(buttonMatch![1]).toContain(">82%<");
    expect(buttonMatch![1]).toContain(">Claude<");
    // no leftover text-label markup beside the ring
    expect(html).not.toContain("hidden flex-col leading-tight lg:flex");
  });
});

// round-1 F4: the ring's slot never disappears, even with no data/window.
describe("UsageRingUnavailable", () => {
  it("renders the short name and the given caption as the tooltip, no adjacent text label", () => {
    const html = renderToStaticMarkup(
      createElement(UsageRingUnavailable, { label: "Anthropic", shortName: "Claude", caption: "rings.unavailable", onOpenDetails: () => {} }),
    );
    expect(html).toContain('title="rings.unavailable"');
    expect(html).toContain("<button");
    expect(html).toContain('aria-label="Anthropic"');
    // the short name renders inside the button, beneath the placeholder ring
    const buttonMatch = html.match(/<button[^>]*>([\s\S]*?)<\/button>/);
    expect(buttonMatch).not.toBeNull();
    expect(buttonMatch![1]).toContain(">Claude<");
    expect(html).not.toContain("hidden flex-col leading-tight lg:flex");
  });
});

// MOA-486 follow-up: one dialog for all three providers — a single tap on any ring
// shows the full picture (every window of every provider, plus an unavailable row).
describe("ProviderUsageModal", () => {
  const providers = [
    { key: "anthropic", name: "Anthropic" },
    { key: "codex", name: "OpenAI Codex" },
    { key: "opencode", name: "OpenCode GO" },
  ] as const;

  it("lists every window of every provider, and an unavailable row for an ok:false provider", () => {
    const anthropicUsage: ProviderUsage = {
      provider: "anthropic",
      plan: "max",
      fiveHour: { usedPercent: 40, resetsAt: null },
      weekly: { usedPercent: 85, resetsAt: null },
      models: [{ name: "Opus", weekly: { usedPercent: 92, resetsAt: null } }],
    };
    const opencodeUsage: ProviderUsage = {
      provider: "opencode",
      plan: null,
      fiveHour: { usedPercent: 10, resetsAt: null },
      weekly: { usedPercent: 20, resetsAt: null },
      monthly: { usedPercent: 60, resetsAt: null },
      models: [],
    };
    const payload: SubscriptionsPayload = {
      anthropic: { ok: true, data: anthropicUsage },
      codex: { ok: false, error: "missing-credentials" },
      opencode: { ok: true, data: opencodeUsage },
    };
    const html = renderToStaticMarkup(
      createElement(ProviderUsageModal, {
        providers: providers as unknown as readonly { key: keyof SubscriptionsPayload; name: string }[],
        payload,
        locale: "en-US",
        t,
        tMission: t,
        onClose: () => {},
      }),
    );
    expect(html).toContain("<dialog");
    // every provider heading
    expect(html).toContain("Anthropic");
    expect(html).toContain("OpenAI Codex");
    expect(html).toContain("OpenCode GO");
    // Anthropic: fiveHour, weekly, per-model weekly
    expect(html).toContain("40%");
    expect(html).toContain("85%");
    expect(html).toContain("92%");
    // OpenCode GO: fiveHour, weekly, monthly
    expect(html).toContain("10%");
    expect(html).toContain("20%");
    expect(html).toContain("60%");
    // Codex is ok:false — unavailable row, no stray percent for it
    expect(html).toContain("unavailable");
    // a close button/backdrop-close affordance exists
    expect(html).toContain("rings.close");
  });

  it("shows the exact reset time (tv-values passthrough), not a vague key with no values", () => {
    const tv = (key: string, values?: Record<string, string | number>) => (values ? `${key}:${JSON.stringify(values)}` : key);
    const usage: ProviderUsage = {
      provider: "anthropic",
      plan: "max",
      fiveHour: { usedPercent: 40, resetsAt: new Date(Date.now() + 15 * 60_000).toISOString() },
      weekly: null,
      models: [],
    };
    const payload: SubscriptionsPayload = {
      anthropic: { ok: true, data: usage },
      codex: { ok: false, error: "missing-credentials" },
      opencode: { ok: false, error: "missing-credentials" },
    };
    const html = renderToStaticMarkup(
      createElement(ProviderUsageModal, {
        providers: providers as unknown as readonly { key: keyof SubscriptionsPayload; name: string }[],
        payload,
        locale: "en-US",
        t,
        tMission: tv,
        onClose: () => {},
      }),
    );
    expect(html).toContain("rings.resetInMinutes");
    expect(html).toContain("&quot;minutes&quot;:15"); // React escapes the JSON's quotes in a text node
  });
});
