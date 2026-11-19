import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en-US",
}));
vi.mock("next/link", () => ({
  default: ({ href, children, ...rest }: { href: string; children: React.ReactNode } & Record<string, unknown>) =>
    createElement("a", { href, ...rest }, children),
}));

import {
  TerminalPanel, accessFailureStatus, accessLifecycle, initialAccessLifecycle,
} from "./TerminalPanel";

describe("M3b frame lifecycle (deferred races, pause/resume, expiry, healthy polls)", () => {
  it("deferred A -> B -> A frames can never replace the current selection's frame", () => {
    let state = accessLifecycle(initialAccessLifecycle, { type: "open" }); // A
    const aGeneration = state.generation;
    state = accessLifecycle(state, { type: "open" }); // B while A validates
    state = accessLifecycle(state, { type: "open" }); // back to A
    expect(state.generation).not.toBe(aGeneration);
    // A late RO validation for the ORIGINAL A must not attach: only pendingAttach gates an attach,
    // and it belongs to the CURRENT generation.
    const late = accessLifecycle({ ...state, pendingAttach: false }, { type: "validation", roUrl: "http://ro/old" });
    expect(late.frame).toBeNull();
    const current = accessLifecycle(state, { type: "validation", roUrl: "http://ro/current" });
    expect(current.frame).toMatchObject({ url: "http://ro/current", key: state.generation, mode: "ro" });
  });

  it("a healthy metadata refresh keeps the same iframe key/src", () => {
    const live = accessLifecycle(accessLifecycle(initialAccessLifecycle, { type: "open" }), { type: "validation", roUrl: "http://ro/a" });
    const polled = accessLifecycle(live, { type: "validation", roUrl: "http://ro/other" });
    expect(polled).toBe(live);
    expect(polled.frame).toEqual(live.frame);
  });

  it("hidden removes the RW frame and resets to RO; visible requires a fresh RO validation", () => {
    let state = accessLifecycle(initialAccessLifecycle, { type: "open" });
    state = accessLifecycle(state, { type: "validation", roUrl: "http://ro/a" });
    state = accessLifecycle(state, { type: "takeStart" });
    const rwToken = state.generation;
    state = accessLifecycle(state, { type: "takeOk", rwUrl: "http://rw/a" });
    expect(state.frame?.mode).toBe("rw");

    const hidden = accessLifecycle(state, { type: "hide" });
    expect(hidden.frame).toBeNull();
    expect(hidden.status).toBe("paused");
    expect(hidden.generation).toBeGreaterThan(rwToken);

    const shown = accessLifecycle(hidden, { type: "show" });
    const stale = accessLifecycle({ ...shown, pendingAttach: false }, { type: "validation", roUrl: "http://ro/old" });
    expect(stale.frame).toBeNull(); // cached/old results never restore the frame
    expect(accessLifecycle(shown, { type: "validation", roUrl: "http://ro/fresh" }).frame?.mode).toBe("ro");
  });

  it("expiry/loss removes the frame, invalidates the generation and never auto-remounts on a later poll", () => {
    let state = accessLifecycle(initialAccessLifecycle, { type: "open" });
    state = accessLifecycle(state, { type: "validation", roUrl: "http://ro/a" });
    const generation = state.generation;
    const expired = accessLifecycle(state, { type: "failure", status: "expired" });
    expect(expired.frame).toBeNull();
    expect(expired.generation).toBeGreaterThan(generation);
    expect(accessLifecycle(expired, { type: "validation", roUrl: "http://ro/b" })).toBe(expired);
  });

  it("release drops RW synchronously and requires a fresh RO attach", () => {
    let state = accessLifecycle(initialAccessLifecycle, { type: "open" });
    state = accessLifecycle(state, { type: "takeStart" });
    state = accessLifecycle(state, { type: "takeOk", rwUrl: "http://rw/a" });
    const released = accessLifecycle(state, { type: "releaseStart" });
    expect(released.frame).toBeNull();
    expect(released.pendingAttach).toBe(true);
    expect(accessLifecycle(released, { type: "validation", roUrl: "http://ro/a" }).frame?.mode).toBe("ro");
  });

  it("a failed Take reports the existing unknown state, never indefinite pending", () => {
    let state = accessLifecycle(initialAccessLifecycle, { type: "open" });
    state = accessLifecycle(state, { type: "validation", roUrl: "http://ro/a" });
    state = accessLifecycle(state, { type: "takeStart" });
    const failed = accessLifecycle(state, { type: "takeFailed" });
    expect(failed.frame).toBeNull();
    expect(failed.pendingAttach).toBe(false);
    expect(failed.status).toBe("unknown");
    // An ordinary metadata success cannot attach; only an explicit reconnect starts RO again.
    expect(accessLifecycle(failed, { type: "validation", roUrl: "http://ro/later" })).toBe(failed);
    const reconnected = accessLifecycle(failed, { type: "reconnect" });
    expect(reconnected.pendingAttach).toBe(true);
    expect(accessLifecycle(reconnected, { type: "validation", roUrl: "http://ro/fresh" }).frame?.mode).toBe("ro");
  });
});

describe("accessFailureStatus (bounded target codes)", () => {
  it("distinguishes unknown source from a lost/unavailable/unconfigured target", () => {
    expect(accessFailureStatus("source-failure")).toBe("unknown");
    expect(accessFailureStatus("lost-target")).toBe("lost-target");
    expect(accessFailureStatus("unavailable")).toBe("unavailable");
    expect(accessFailureStatus("unconfigured")).toBe("unconfigured");
    expect(accessFailureStatus("invalid-selector")).toBe("unknown");
  });
});

function buttonTag(html: string, label: string): string {
  const labelAt = html.indexOf(`>${label}`);
  const start = html.lastIndexOf("<button", labelAt === -1 ? html.indexOf(label) : labelAt);
  const end = html.indexOf(">", start);
  return start === -1 || end === -1 ? "" : html.slice(start, end);
}

function iframeOf(html: string): string {
  return html.match(/<iframe[\s\S]*?<\/iframe>/)?.[0] ?? "";
}

describe("TerminalPanel prepared-frame rendering (M3b)", () => {
  const base = {
    session: "project agent", frameUrl: null, frameKey: 0, frameMode: "ro" as const,
    configLoading: false, configured: true, mode: "ro" as const, busy: false, error: null,
    validatedAt: null,
    onTakeControl: () => {}, onReleaseControl: () => {}, onReconnect: () => {}, largerHref: "/tmux?kind=tmux&session=x",
  };

  it("renders ONLY the prepared frame URL and opens the larger view as a rel=noopener Jax link", () => {
    const html = renderToStaticMarkup(createElement(TerminalPanel, {
      ...base, frameUrl: "http://127.0.0.1:10000/ro/?arg=%242%3A%403.%257&disableReconnect=true", status: "live",
    }));
    expect(html).toContain("http://127.0.0.1:10000/ro/?arg=");
    expect(html).toContain('href="/tmux?kind=tmux&amp;session=x"');
    expect(html).toContain('target="_blank"');
    expect(html).toContain('rel="noopener"');
  });

  it("paused/lost/unknown/expired remove the frame and show the state, never a config-derived URL", () => {
    for (const [status, key] of [["paused", "paused"], ["lost-target", "targetLost"], ["unknown", "unknown"], ["expired", "expired"]] as const) {
      const html = renderToStaticMarkup(createElement(TerminalPanel, { ...base, status }));
      expect(html).not.toContain("<iframe");
      expect(html).toContain(key);
    }
  });

  it("a taken RW frame renders exactly its provided rwUrl; the mode chip follows the frame", () => {
    const html = renderToStaticMarkup(createElement(TerminalPanel, {
      ...base, frameUrl: "http://127.0.0.1:10001/rw/?arg=%242%3A%403.%257", frameMode: "rw", mode: "rw", status: "live",
    }));
    expect(html).toContain("http://127.0.0.1:10001/rw/?arg=");
    expect(html).toContain("writable");
    expect(html).toContain("releaseControl");
  });

  it("announces a pending control action and keeps errors visible", () => {
    const html = renderToStaticMarkup(createElement(TerminalPanel, {
      ...base, frameUrl: "http://ro/a", status: "live", busy: true, error: "recording failed",
    }));
    expect(html).toContain("takingControl");
    expect(html).toContain("recording failed");
    expect(html).toContain('aria-busy="true"');
    expect(html).toContain('role="status"');
  });

  it("shows the metadata validation time and both shared-control limits near the controls", () => {
    const html = renderToStaticMarkup(createElement(TerminalPanel, {
      ...base, frameUrl: "http://ro/a", status: "live", validatedAt: 1_700_000_000_000,
    }));
    expect(html).toContain("lastValidated");
    expect(html).toContain("attachMovesSelection");
    expect(html).toContain("closedPaneInputRisk");
    expect(html).toContain("nativeEnterException");
    expect(html).toContain("modeChangeScrollback");
    const absent = renderToStaticMarkup(createElement(TerminalPanel, { ...base, frameUrl: "http://ro/a", status: "live" }));
    expect(absent).not.toContain("lastValidated");
  });

  it("not configured renders the config warning instead of a frame", () => {
    const html = renderToStaticMarkup(createElement(TerminalPanel, { ...base, configured: false, status: "unconfigured" }));
    expect(html).toContain("notConfigured");
    expect(html).not.toContain("<iframe");
  });

  it("a prepared live frame survives the separate config poll (false/true/loading) with the same node and URL", () => {
    const live = { ...base, frameUrl: "http://127.0.0.1:10000/ro/?arg=%242%3A%403.%257", status: "live" as const };
    const ready = renderToStaticMarkup(createElement(TerminalPanel, live));
    const frame = iframeOf(ready);
    expect(frame).toContain("http://127.0.0.1:10000/ro/?arg=");
    expect(iframeOf(renderToStaticMarkup(createElement(TerminalPanel, { ...live, configured: false })))).toBe(frame);
    expect(iframeOf(renderToStaticMarkup(createElement(TerminalPanel, { ...live, configLoading: true })))).toBe(frame);
    expect(iframeOf(renderToStaticMarkup(createElement(TerminalPanel, { ...live, configured: true })))).toBe(frame);
  });

  it("a live RW frame keeps its iframe and an enabled Release even when the config poll fails", () => {
    const html = renderToStaticMarkup(createElement(TerminalPanel, {
      ...base, configured: false, frameUrl: "http://127.0.0.1:10001/rw/?arg=%242%3A%403.%257",
      frameMode: "rw", mode: "rw", status: "live",
    }));
    expect(iframeOf(html)).toContain("http://127.0.0.1:10001/rw/?arg=");
    expect(html).toContain("releaseControl");
    expect(buttonTag(html, "releaseControl")).not.toContain("disabled=\"\"");
  });

  it("Take remains gated on the configured transport for a live RO frame", () => {
    const live = { ...base, frameUrl: "http://127.0.0.1:10000/ro/?arg=x", status: "live" as const };
    const enabled = renderToStaticMarkup(createElement(TerminalPanel, { ...live, configured: true }));
    const gated = renderToStaticMarkup(createElement(TerminalPanel, { ...live, configured: false }));
    expect(buttonTag(enabled, "takeControl")).not.toContain("disabled=\"\"");
    expect(buttonTag(gated, "takeControl")).toContain("disabled=\"\"");
    expect(iframeOf(gated)).toBe(iframeOf(enabled));
  });

  it("a failed Take shows the existing control error and an available Reconnect", () => {
    const html = renderToStaticMarkup(createElement(TerminalPanel, {
      ...base, configured: true, status: "unknown", error: "controlFailed",
    }));
    expect(html).not.toContain("<iframe");
    expect(html).toContain("unknown");
    expect(html).toContain("controlFailed");
    expect(buttonTag(html, "reconnect")).not.toContain("disabled=\"\"");
  });
});

