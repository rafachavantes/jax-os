import { afterEach, describe, expect, it, vi } from "vitest";

const seed = vi.hoisted(() => vi.fn());
vi.mock("./server/db", () => ({ getDb: () => "DB" }));
vi.mock("./server/settings", () => ({ seedAgentsIfFirstRun: seed }));
vi.mock("./server/collectors/cli-presence", () => ({ isCliPresent: (name: string) => name === "claude" }));

import { register } from "./instrumentation";

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
  seed.mockReset();
});

describe("register() (MOA-504 D4)", () => {
  it("seeds on the node runtime, probing the CLIs through isCliPresent", async () => {
    vi.stubEnv("NEXT_RUNTIME", "nodejs");
    await register();
    expect(seed).toHaveBeenCalledTimes(1);
    expect(seed.mock.calls[0][0]).toBe("DB");
    expect(seed.mock.calls[0][1]("claude")).toBe(true);
    expect(seed.mock.calls[0][1]("codex")).toBe(false);
  });

  it.each([
    ["the edge runtime", "edge", undefined],
    ["the production build phase", "nodejs", "phase-production-build"],
  ])("does nothing on %s", async (_n, runtime, phase) => {
    vi.stubEnv("NEXT_RUNTIME", runtime);
    vi.stubEnv("NEXT_PHASE", phase ?? "");
    await register();
    expect(seed).not.toHaveBeenCalled();
  });

  it("never throws; a seed failure and an audit failure are logged distinctly", async () => {
    vi.stubEnv("NEXT_RUNTIME", "nodejs");
    const log = vi.spyOn(console, "error").mockImplementation(() => {});
    seed.mockImplementationOnce(() => { throw new Error("disk full"); });
    await expect(register()).resolves.toBeUndefined();
    const audit = Object.assign(new Error("settings-audit-failed"), { name: "SettingsAuditFailedError" });
    seed.mockImplementationOnce(() => { throw audit; });
    await expect(register()).resolves.toBeUndefined();
    expect(String(log.mock.calls[0][0])).toContain("defaults apply");
    expect(String(log.mock.calls[1][0])).toContain("audit");
  });
});
