import { createElement, type ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { NextIntlClientProvider } from "next-intl";
import { beforeEach, describe, expect, it, vi } from "vitest";
import en from "../../../messages/en-US.json";
import pt from "../../../messages/pt-BR.json";

const harness = vi.hoisted(() => ({
  results: new Map<string, { data: unknown; isError: boolean }>(),
  calls: [] as { queryKey: unknown[] }[],
}));

vi.mock("@tanstack/react-query", () => ({
  useQuery: (opts: { queryKey: unknown[] }) => {
    harness.calls.push(opts);
    return harness.results.get(opts.queryKey.join(":")) ?? { data: undefined, isError: false };
  },
}));

import { EnvironmentDiagnostics } from "./EnvironmentDiagnostics";

const catalogs = { "en-US": en, "pt-BR": pt } as const;
type Locale = keyof typeof catalogs;

function render(node: ReactElement, locale: Locale = "en-US") {
  const errors: unknown[] = [];
  const html = renderToStaticMarkup(createElement(NextIntlClientProvider, {
    locale, messages: catalogs[locale], timeZone: "UTC", onError: (error: unknown) => errors.push(error), children: node,
  }));
  return { html, errors };
}

const COMPONENTS = {
  observedAt: "2026-11-19T00:00:00.000Z",
  components: [
    { id: "database", state: "healthy", detail: "select-1", observedAt: "2026-11-19T00:00:00.000Z", lastGoodAt: null },
    { id: "ttyd", state: "disabled-default", detail: null, observedAt: "2026-11-19T00:00:00.000Z", lastGoodAt: null },
    { id: "github", state: "needs-configuration", detail: "cli-missing", observedAt: "2026-11-19T00:00:00.000Z", lastGoodAt: null },
    { id: "linear", state: "unknown", detail: "credential-present", observedAt: "2026-11-19T00:00:00.000Z", lastGoodAt: null },
  ],
};

const RESOURCES = {
  observedAt: "2026-11-19T00:00:00.000Z", lastGoodAt: "2026-11-19T00:00:00.000Z",
  state: "ok", stale: false, warning: null,
  tmp: { path: "/tmp", mountPath: "/tmp", fsType: "tmpfs", tmpfs: true, sharedRoot: false, usedGb: 1.25, totalGb: 4, availableGb: 2.75 },
};

const WORKLOADS = {
  observedAt: "2026-11-19T00:00:00.000Z", state: "ok", totalProcesses: 3, truncated: false,
  coverage: { ownershipKnown: 2, ownershipUnknown: 1, partial: false, reasons: { "no-pane": 1 } },
  groups: [
    {
      key: "pane:%1:codex", family: "codex", count: 2, running: 1, zombies: 1, rssMb: 120,
      maxAgeSeconds: 60, owner: { kind: "pane", pane: "%1" }, project: "jax-os", session: "s1",
      contextObservedAt: "2026-11-18T23:00:00.000Z", purpose: "tmux-pane",
      pids: [{ pid: 30, ppid: 20, state: "S+", ageSeconds: 60, rssMb: 120, parentCommand: null }], detailsTruncated: true,
    },
    {
      key: "unknown:1000:chrome:1", family: "chrome", count: 1, running: 1, zombies: 0, rssMb: null,
      maxAgeSeconds: 90, owner: { kind: "unknown", reason: "no-pane" }, project: null, session: null,
      contextObservedAt: null, purpose: null,
      pids: [{ pid: 40, ppid: 1, state: "Sl", ageSeconds: 90, rssMb: null, parentCommand: null }], detailsTruncated: false,
    },
  ],
};

const PACKAGES = {
  observedAt: "2026-11-19T00:00:00.000Z", lastGoodAt: "2026-11-19T00:00:00.000Z",
  state: "ok", truncated: false, warnings: [],
  counts: { total: 2, shown: 2, remainder: 0, selectedNow: 1, held: 1, deferred: 0, unknown: 0 },
  candidates: [
    { key: "openssl:amd64", name: "openssl", arch: "amd64", fromVersion: "1", toVersion: "2", security: true, held: false, selectedNow: true, conflict: false, reason: "selected-now" },
    { key: "curl:amd64", name: "curl", arch: "amd64", fromVersion: "1", toVersion: "2", security: false, held: true, selectedNow: false, conflict: false, reason: "held" },
  ],
};

function setAll() {
  harness.results.set("environment:components", { data: { ok: true, data: COMPONENTS }, isError: false });
  harness.results.set("environment:resources", { data: { ok: true, data: RESOURCES }, isError: false });
  harness.results.set("environment:workloads", { data: { ok: true, data: WORKLOADS }, isError: false });
  harness.results.set("environment:packages", { data: { ok: true, data: PACKAGES }, isError: false });
}

beforeEach(() => {
  harness.calls.length = 0;
  harness.results.clear();
});

describe("EnvironmentDiagnostics (M5 Task 4)", () => {
  it("polls the four sections independently with their own intervals", () => {
    setAll();
    render(createElement(EnvironmentDiagnostics));
    expect(harness.calls.map((c) => c.queryKey.join(":"))).toEqual([
      "environment:components", "environment:resources", "environment:workloads", "environment:packages",
    ]);
  });

  it("renders distinct state explanations and no enable prompt for disabled defaults", () => {
    setAll();
    const { html } = render(createElement(EnvironmentDiagnostics));
    expect(html).toContain("Database");
    expect(html).toContain("Healthy");
    expect(html).toContain("Own database answered a read query.");
    expect(html).toContain("Disabled (default)");
    expect(html).toContain("Needs configuration");
    expect(html).toContain("CLI executable was not found.");
    expect(html).toContain("Credential present; remote health not verified.");
    expect(html).toContain("Unknown");
    expect(html).toContain("No action needed.");
    expect(html).toContain("Optional and disabled by default.");
    expect(html).not.toMatch(/enable it|turn it on/i);
  });

  it("keeps sibling sections visible while one source fails", () => {
    setAll();
    harness.results.set("environment:resources", { data: undefined, isError: true });
    const { html } = render(createElement(EnvironmentDiagnostics));
    expect(html).toContain("Environment data unavailable");
    expect(html).toContain("Workloads");
    expect(html).toContain("openssl");
    expect(html).toContain("Database");
  });

  it("shows measured tmpfs capacity, backing explanation and decimal units", () => {
    setAll();
    const { html } = render(createElement(EnvironmentDiagnostics));
    expect(html).toContain('id="tmp"');
    expect(html).toContain("1.3 GB");
    expect(html).toContain("4.0 GB");
    expect(html).toContain("2.8 GB available");
    expect(html).toContain("Backed by RAM/swap");
    expect(html).not.toContain("Shmem");
  });

  it("a failed /tmp observation keeps a visible warning and the last good value", () => {
    setAll();
    harness.results.set("environment:resources", {
      data: { ok: true, data: { ...RESOURCES, state: "unavailable", stale: true, warning: "statfs-failed" } }, isError: false,
    });
    const { html } = render(createElement(EnvironmentDiagnostics));
    expect(html).toContain("Temporary capacity unavailable");
    expect(html).toContain("Showing the last good observation.");
    expect(html).toContain("4.0 GB");
  });

  it("explains workloads with proven/unknown origin, approximate RSS and zombie lifecycle", () => {
    setAll();
    const { html } = render(createElement(EnvironmentDiagnostics));
    expect(html).toContain("pane %1 (proven ancestry)");
    expect(html).toContain("origin unknown");
    expect(html).toContain("last known project jax-os");
    expect(html).toContain("1 zombie");
    expect(html).toContain("RSS is approximate");
    expect(html).toContain("Purpose unavailable");
    expect(html).toContain("not reclaimable memory");
  });

  it("shows package candidates, holds and simulation selection with honest counts and no repair command", () => {
    setAll();
    const { html } = render(createElement(EnvironmentDiagnostics));
    expect(html).toContain("2 candidates");
    expect(html).toContain("1 selected now");
    expect(html).toContain("1 held");
    expect(html).toContain("selected by simulation");
    expect(html).toContain("held");
    expect(html).toContain("security");
    expect(html).toContain("install updates manually");
    expect(html).not.toMatch(/apt(-get)? (upgrade|install)|kill -/i);
  });

  it("both actual catalogs resolve every environment key", () => {
    setAll();
    for (const locale of ["en-US", "pt-BR"] as const) {
      const { html, errors } = render(createElement(EnvironmentDiagnostics), locale);
      expect(errors).toEqual([]);
      expect(html).not.toMatch(/health\.environment|componentsList\.|nextStep\.|details\./);
    }
  });

  it("renders a long package identifier without forcing a nowrap page", () => {
    setAll();
    const longName = "very-long-package-name-with-architecture-and-lots-of-segments";
    harness.results.set("environment:packages", {
      data: { ok: true, data: { ...PACKAGES, candidates: [{ ...PACKAGES.candidates[0], key: `${longName}:amd64`, name: longName }] } },
      isError: false,
    });
    const { html } = render(createElement(EnvironmentDiagnostics));
    expect(html).toContain(longName);
    expect(html).toContain("[overflow-wrap:anywhere]");
  });
});
