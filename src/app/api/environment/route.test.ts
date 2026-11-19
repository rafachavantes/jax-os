import { describe, expect, it, vi } from "vitest";
import { handleEnvironmentGet, parseSection, type EnvironmentHandlerDeps } from "./handler";
import * as route from "./route";

const COMPONENTS = { observedAt: "2026-11-19T00:00:00.000Z", components: [] };
const RESOURCES = { observedAt: "2026-11-19T00:00:00.000Z", state: "ok" as const, stale: false, tmp: null, warning: null, lastGoodAt: null };
const WORKLOADS = {
  observedAt: "2026-11-19T00:00:00.000Z", state: "ok" as const, groups: [], totalProcesses: 0,
  truncated: false, coverage: { ownershipKnown: 0, ownershipUnknown: 0, partial: false, reasons: {} },
};
const PACKAGES = {
  observedAt: "2026-11-19T00:00:00.000Z", lastGoodAt: null, state: "unavailable" as const,
  candidates: [], counts: null, truncated: false, warnings: ["package-query-failed" as const],
};

type RuntimeOverrides = Partial<EnvironmentHandlerDeps["runtime"]>;

function deps(over: RuntimeOverrides = {}): EnvironmentHandlerDeps {
  return {
    runtime: {
      components: over.components ?? (async () => COMPONENTS),
      resources: over.resources ?? (async () => RESOURCES),
      workloads: over.workloads ?? (async () => WORKLOADS),
      packages: over.packages ?? (async () => PACKAGES),
    },
  };
}

const call = async (query: string, d: EnvironmentHandlerDeps = deps()) =>
  (await handleEnvironmentGet(new Request(`http://127.0.0.1:3100/api/environment${query}`), d)).json();

describe("GET /api/environment (M5 Task 4)", () => {
  it("defaults to components and returns the existing envelope", async () => {
    expect(await call("")).toEqual({ ok: true, data: COMPONENTS });
    expect(parseSection(null)).toBe("components");
    expect(parseSection("")).toBe("components");
  });

  it("serves each fixed section independently", async () => {
    for (const [section, expected] of [
      ["components", COMPONENTS], ["resources", RESOURCES], ["workloads", WORKLOADS], ["packages", PACKAGES],
    ] as const) {
      expect(await call(`?section=${section}`)).toEqual({ ok: true, data: expected });
    }
  });

  it("refuses any other section value and any section failure with the safe envelope", async () => {
    expect(await call("?section=../../etc/passwd")).toMatchObject({ ok: false });
    const failed = await call("?section=resources", deps({ resources: async () => { throw new Error("boom"); } }));
    expect(failed).toEqual({ ok: false, error: "boom" });
  });

  it("exposes GET only — no mutation surface", () => {
    expect(typeof route.GET).toBe("function");
    expect(route).not.toHaveProperty("POST");
    expect(route).not.toHaveProperty("PUT");
    expect(route).not.toHaveProperty("DELETE");
  });

  it("does not poll every section for one request", async () => {
    const components = vi.fn(async () => COMPONENTS);
    const resources = vi.fn(async () => RESOURCES);
    await call("?section=components", deps({ components, resources }));
    expect(components).toHaveBeenCalledTimes(1);
    expect(resources).not.toHaveBeenCalled();
  });
});
