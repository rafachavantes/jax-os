import { describe, expect, it, vi } from "vitest";
import { parseSettings, settingsKeyPresence, type GeneralSettings } from "../settings";
import type { PackageObservation, PackageRow } from "./alerts";
import {
  buildComponents, buildPackageSection, capacityFromStatfs, createEnvironmentRuntime, createSourceObserver,
  decodeMountField, loopbackBase, MAX_PATH_SEGMENTS, parseMountInfo, probeCliPresence, readBoundedText,
  selectMount, systemEnvironmentDeps, type ComponentFacts, type EnvironmentDeps,
} from "./environment";

const NOW = Date.parse("2026-11-19T00:00:00.000Z");
const DEFAULTS = (parseSettings(undefined) as { ok: true; data: GeneralSettings }).data;

function facts(over: Partial<ComponentFacts> = {}): ComponentFacts {
  return {
    nowMs: NOW,
    settings: { ok: true, data: DEFAULTS },
    presence: settingsKeyPresence({}),
    db: { state: "ok", schemaVersion: 15 },
    projects: "ok",
    tmux: "ok",
    ttyd: { roConfigured: false, rwConfigured: false, probeBase: null, reachable: null },
    clis: { claude: "missing", codex: "missing", opencode: "missing" },
    githubCli: "missing",
    linearCredential: false,
    metrics: { state: "ok", latest: new Date(NOW).toISOString() },
    envFile: "ok",
    hostProbes: { resources: true, workloads: true, packages: true },
    ...over,
  };
}

function byId(list: ReturnType<typeof buildComponents>, id: string) {
  return list.find((c) => c.id === id);
}

const ENABLED: GeneralSettings = {
  ...DEFAULTS,
  integrations: {
    ...DEFAULTS.integrations,
    ttyd: true, github: true, linear: true,
    agents: { claude: true, codex: true, opencode: true },
  },
};

describe("buildComponents — explainable state matrix (AC2)", () => {
  it("healthy local components stay healthy and never claim remote auth", () => {
    const list = buildComponents(facts({
      settings: { ok: true, data: ENABLED },
      presence: settingsKeyPresence({ integrations: {
        ttyd: true, github: true, linear: true,
        agents: { claude: true, codex: true, opencode: true },
      } }),
      ttyd: { roConfigured: true, rwConfigured: true, probeBase: "http://127.0.0.1:10000", reachable: true },
      clis: { claude: "present", codex: "present", opencode: "present" },
      githubCli: "present",
      linearCredential: true,
    }));
    expect(byId(list, "database")?.state).toBe("healthy");
    expect(byId(list, "projects")).toMatchObject({ state: "healthy", detail: "root-readable-unverified" });
    expect(byId(list, "tmux")?.state).toBe("healthy");
    expect(byId(list, "metrics")?.state).toBe("healthy");
    expect(byId(list, "host-probes")).toMatchObject({ state: "healthy", detail: "sections-observed" });
    expect(byId(list, "env-file")?.state).toBe("healthy");
    expect(byId(list, "ttyd")).toMatchObject({ state: "healthy", detail: "http-ok-both" });
    expect(byId(list, "claude")).toMatchObject({ state: "healthy", detail: "cli-present" });
    // token/config presence is never proof of remote health
    expect(byId(list, "github")).toMatchObject({ state: "unknown", detail: "configured-unverified" });
    expect(byId(list, "linear")).toMatchObject({ state: "unknown", detail: "credential-present" });
  });

  it("distinguishes a defaulted false from an explicit false, with no error state either way", () => {
    const defaulted = buildComponents(facts());
    expect(byId(defaulted, "ttyd")?.state).toBe("disabled-default");
    expect(byId(defaulted, "github")?.state).toBe("disabled-default");
    expect(byId(defaulted, "linear")?.state).toBe("disabled-default");
    expect(byId(defaulted, "claude")?.state).toBe("disabled-default");

    const explicit = buildComponents(facts({
      presence: settingsKeyPresence({ integrations: { ttyd: false, github: false, linear: false, agents: { claude: false } } }),
    }));
    expect(byId(explicit, "ttyd")?.state).toBe("disabled-explicit");
    expect(byId(explicit, "github")?.state).toBe("disabled-explicit");
    expect(byId(explicit, "linear")?.state).toBe("disabled-explicit");
    expect(byId(explicit, "claude")?.state).toBe("disabled-explicit");
  });

  it("enabled but unconfigured components need configuration; failures stay per source", () => {
    const list = buildComponents(facts({
      settings: { ok: true, data: ENABLED },
      presence: settingsKeyPresence({ integrations: { ttyd: true, github: true, linear: true, agents: { codex: true } } }),
      ttyd: { roConfigured: false, rwConfigured: false, probeBase: null, reachable: null },
      clis: { claude: "missing", codex: "missing", opencode: "missing" },
      githubCli: "missing",
      linearCredential: false,
      tmux: "failed",
      db: { state: "failed", error: "database-unavailable" },
      metrics: { state: "failed" },
      projects: "failed",
    }));
    expect(byId(list, "ttyd")).toMatchObject({ state: "needs-configuration", detail: "urls-missing" });
    expect(byId(list, "github")).toMatchObject({ state: "needs-configuration", detail: "cli-missing" });
    expect(byId(list, "linear")).toMatchObject({ state: "needs-configuration", detail: "credential-missing" });
    expect(byId(list, "codex")).toMatchObject({ state: "needs-configuration", detail: "cli-missing" });
    expect(byId(list, "tmux")).toMatchObject({ state: "unavailable", detail: "tmux-unavailable" });
    expect(byId(list, "database")).toMatchObject({ state: "failed", detail: "database-unavailable" });
    expect(byId(list, "metrics")).toMatchObject({ state: "unavailable", detail: "database-unavailable" });
    expect(byId(list, "projects")).toMatchObject({ state: "failed", detail: "root-unreadable" });
    // enabled by settings but its CLI is absent — a configuration need, never disabled.
    expect(byId(list, "claude")).toMatchObject({ state: "needs-configuration", detail: "cli-missing" });
  });

  it("a permission/budget-inconclusive CLI probe is unknown, never missing", () => {
    const list = buildComponents(facts({
      settings: { ok: true, data: ENABLED },
      presence: settingsKeyPresence({ integrations: { agents: { claude: true, codex: true, opencode: true }, github: true } }),
      clis: { claude: "unknown", codex: "unknown", opencode: "unknown" },
      githubCli: "unknown",
    }));
    for (const id of ["claude", "codex", "opencode", "github"]) {
      expect(byId(list, id)).toMatchObject({ state: "unknown", detail: "cli-unverified" });
    }
  });

  it("malformed settings are unknown, never intentional disablement", () => {
    const list = buildComponents(facts({
      settings: { ok: false, error: "settings-malformed" },
      presence: null,
    }));
    for (const id of ["ttyd", "github", "linear", "claude", "codex", "opencode", "projects"]) {
      expect(byId(list, id)?.state).toBe("unknown");
    }
    expect(byId(list, "database")?.state).toBe("healthy");
  });

  it("metrics freshness uses the existing 180s boundary and never a zero substitute", () => {
    const fresh = buildComponents(facts({ metrics: { state: "ok", latest: new Date(NOW - 180_000).toISOString() } }));
    expect(byId(fresh, "metrics")?.state).toBe("healthy");
    const stale = buildComponents(facts({ metrics: { state: "ok", latest: new Date(NOW - 181_000).toISOString() } }));
    expect(byId(stale, "metrics")).toMatchObject({ state: "unavailable", detail: "stale" });
    const none = buildComponents(facts({ metrics: { state: "ok", latest: null } }));
    expect(byId(none, "metrics")).toMatchObject({ state: "unavailable", detail: "no-sample" });
    const unknown = buildComponents(facts({ metrics: { state: "unknown" } }));
    expect(byId(unknown, "metrics")).toMatchObject({ state: "unknown", detail: "database-unverified" });
  });

  it("an insecure env file needs configuration while a missing one is an optional default", () => {
    expect(byId(buildComponents(facts({ envFile: "missing" })), "env-file")?.state).toBe("disabled-default");
    expect(byId(buildComponents(facts({ envFile: "bad-mode" })), "env-file")).toMatchObject({
      state: "needs-configuration", detail: "bad-mode",
    });
  });

  it("an unobserved host-probe matrix is unknown/not-checked, never a false all-clear", () => {
    const none = buildComponents(facts({ hostProbes: { resources: null, workloads: null, packages: null } }));
    expect(byId(none, "host-probes")).toMatchObject({ state: "unknown", detail: "not-checked" });
    const partial = buildComponents(facts({ hostProbes: { resources: true, workloads: false, packages: null } }));
    expect(byId(partial, "host-probes")).toMatchObject({ state: "unavailable", detail: "sections-partial" });
  });
});

describe("mount parsing and /tmp capacity (AC4)", () => {
  const MOUNTINFO = [
    "22 1 8:1 / / rw,relatime - ext4 /dev/sda1 rw",
    "23 22 0:30 / /tmp rw,nosuid - tmpfs tmpfs rw,size=1000000k",
    "24 22 8:1 /var/tmp /mnt/my\\040disk rw - ext4 /dev/sda1 rw",
    "25 22 0:31 / /tmp rw - tmpfs tmpfs rw,size=2",
  ].join("\n");

  it("decodes octal escapes and selects the longest prefix with the last overmount winning", () => {
    const mounts = parseMountInfo(MOUNTINFO);
    expect(decodeMountField("/mnt/my\\040disk")).toBe("/mnt/my disk");
    expect(mounts.find((m) => m.mountPoint === "/mnt/my disk")?.mountId).toBe("24");
    const tmp = selectMount(mounts, "/tmp");
    expect(tmp?.mountId).toBe("25"); // equal prefix, last line wins
    expect(selectMount(mounts, "/tmp/sub")?.mountId).toBe("25");
    expect(selectMount(mounts, "/var/tmp/x")?.mountPoint).toBe("/");
    expect(selectMount([], "/tmp")).toBeNull();
  });

  it("rejects invalid counters and a zero total instead of inventing capacity", () => {
    expect(capacityFromStatfs({ bsize: 4096, blocks: 100, bfree: 10, bavail: 5 })).toEqual({
      usedGb: (90 * 4096) / 1e9, totalGb: (100 * 4096) / 1e9, availableGb: (5 * 4096) / 1e9,
    });
    expect(capacityFromStatfs({ bsize: 0, blocks: 0, bfree: 0, bavail: 0 })).toBeNull();
    expect(capacityFromStatfs({ bsize: NaN, blocks: 1, bfree: 0, bavail: 0 })).toBeNull();
    expect(capacityFromStatfs({ bsize: 4096, blocks: 100, bfree: 90, bavail: 95 })).toBeNull();
    expect(capacityFromStatfs({ bsize: 4096, blocks: 100, bfree: 110, bavail: 5 })).toBeNull();
    expect(capacityFromStatfs({ bsize: 4096, blocks: 100, bfree: 10, bavail: -1 })).toBeNull();
  });

  it("validates only loopback ttyd bases", () => {
    expect(loopbackBase("http://127.0.0.1:10000/x")).toBe("http://127.0.0.1:10000");
    expect(loopbackBase("https://localhost:10000")).toBe("https://localhost:10000");
    expect(loopbackBase("http://example.com:10000")).toBeNull();
    expect(loopbackBase("file:///tmp/x")).toBeNull();
    expect(loopbackBase(undefined)).toBeNull();
  });

  function runtimeDeps(over: Partial<EnvironmentDeps> = {}): EnvironmentDeps {
    return {
      now: () => NOW,
      sleep: () => new Promise<void>(() => {}),
      readSettingsRaw: () => ({ ok: true, raw: undefined, file: "absent" }),
      ownDb: async () => ({ ok: true, schemaVersion: 15, latestMetricTs: null, paneContext: null }),
      checkProjects: async () => ({ ok: true }),
      listPanes: async () => ({ paneIds: new Set(), incarnation: null, paneCommands: new Map() }),
      ttydTransport: () => ({ ro: null, rw: null }),
      httpOk: async () => true,
      cli: async () => "missing",
      linearCredential: () => false,
      envFile: () => "ok",
      readMountInfo: async () => MOUNTINFO,
      statfs: async () => ({ bsize: 4096, blocks: 1_000_000, bfree: 250_000, bavail: 200_000 }),
      realpath: () => "/tmp",
      packages: async () => observation(),
      exec: async () => "",
      ps: async () => "",
      readProc: async () => { throw new Error("no proc"); },
      ...over,
    };
  }

  it("reports a distinct tmpfs mount with decimal GB and no shared-root claim", async () => {
    const runtime = createEnvironmentRuntime(runtimeDeps());
    const section = await runtime.resources();
    expect(section.state).toBe("ok");
    expect(section.tmp).toMatchObject({ path: "/tmp", mountPath: "/tmp", fsType: "tmpfs", tmpfs: true, sharedRoot: false });
    expect(section.tmp?.totalGb).toBeCloseTo((1_000_000 * 4096) / 1e9);
    expect(section.tmp?.usedGb).toBeCloseTo((750_000 * 4096) / 1e9);
  });

  it("explains a shared root by device major:minor and a failure keeps the last good observation stale", async () => {
    let now = NOW;
    let fail = false;
    const runtime = createEnvironmentRuntime(runtimeDeps({
      now: () => now,
      readMountInfo: async () => "22 1 8:1 / / rw - ext4 /dev/sda1 rw\n23 22 8:1 / /tmp rw - ext4 /dev/sda1 rw",
      statfs: async () => {
        if (fail) throw new Error("statfs failed");
        return { bsize: 4096, blocks: 100, bfree: 40, bavail: 40 };
      },
    }));
    const shared = await runtime.resources();
    expect(shared.tmp).toMatchObject({ sharedRoot: true, tmpfs: false });
    now += 6_000;
    fail = true;
    const failed = await runtime.resources();
    expect(failed.state).toBe("unavailable");
    expect(failed.stale).toBe(true);
    expect(failed.warning).toBe("statfs-failed");
    expect(failed.tmp?.totalGb).toBe(shared.tmp?.totalGb);
    expect(failed.lastGoodAt).toBe(shared.lastGoodAt);
  });

  it("an initial failure has no last-good/stale claim and a mount read failure is distinguished from statfs", async () => {
    const runtime = createEnvironmentRuntime(runtimeDeps({
      readMountInfo: async () => { throw new Error("unreadable"); },
    }));
    const section = await runtime.resources();
    expect(section).toMatchObject({ state: "unavailable", stale: false, tmp: null, warning: "mount-facts", lastGoodAt: null });

    const unparsable = createEnvironmentRuntime(runtimeDeps({ readMountInfo: async () => "garbage" }));
    expect((await unparsable.resources()).warning).toBe("mount-facts");

    const badCounters = createEnvironmentRuntime(runtimeDeps({
      statfs: async () => ({ bsize: 4096, blocks: 0, bfree: 0, bavail: 0 }),
    }));
    expect((await badCounters.resources()).warning).toBe("statfs-failed");
  });

  it("two concurrent cold resource reads call each source exactly once", async () => {
    const readMountInfo = vi.fn(async () => MOUNTINFO);
    const statfs = vi.fn(async () => ({ bsize: 4096, blocks: 100, bfree: 40, bavail: 40 }));
    const runtime = createEnvironmentRuntime(runtimeDeps({ readMountInfo, statfs }));
    await Promise.all([runtime.resources(), runtime.resources()]);
    expect(readMountInfo).toHaveBeenCalledTimes(1);
    expect(statfs).toHaveBeenCalledTimes(1);
  });

  it("a pending package promise cannot delay resources or components", async () => {
    const runtime = createEnvironmentRuntime(runtimeDeps({
      packages: () => new Promise(() => {}),
    }));
    const timeout = new Promise((_, reject) => setTimeout(() => reject(new Error("blocked")), 100));
    await expect(Promise.race([Promise.all([runtime.resources(), runtime.components()]).then(() => "ok"), timeout])).resolves.toBe("ok");
  });
});

describe("source observer — one original pending operation (M5 Task 1)", () => {
  it("a timeout retains the original load; a repeated poll cannot start a copy", async () => {
    let now = NOW;
    const sleeps: (() => void)[] = [];
    const sleep = (ms: number) => {
      expect(ms).toBeGreaterThan(0);
      return new Promise<void>((resolve) => sleeps.push(resolve));
    };
    const observe = createSourceObserver(() => now, sleep);
    let resolveLoad: ((value: string) => void) | undefined;
    let loads = 0;
    const load = () => {
      loads += 1;
      return new Promise<string>((resolve) => { resolveLoad = resolve; });
    };
    const first = observe("source", 60_000, 5_000, load);
    const second = observe("source", 60_000, 5_000, load);
    expect(loads).toBe(1);
    sleeps.shift()?.();
    sleeps.shift()?.();
    expect(await first).toMatchObject({ ok: false, lastGood: null, lastGoodAt: null });
    expect(await second).toMatchObject({ ok: false });

    now += 1_000;
    resolveLoad?.("done");
    await new Promise((resolve) => setTimeout(resolve, 0));
    const third = observe("source", 60_000, 5_000, load);
    expect(loads).toBe(1);
    expect(await third).toMatchObject({ ok: true, value: "done" });
  });

  it("keeps the original observation time, not the serving time", async () => {
    let now = NOW;
    const observe = createSourceObserver(() => now);
    const first = await observe("value", 60_000, 1_000, async () => "a");
    expect(first).toMatchObject({ ok: true, value: "a", observedAt: NOW });
    now += 10_000;
    const cached = await observe("value", 60_000, 1_000, async () => "b");
    expect(cached).toMatchObject({ ok: true, value: "a", observedAt: NOW });
  });
});

describe("fixed probe budgets and CLI presence (M5 Task 1)", () => {
  it("passes the approved locale and budgets to ps and apt/dpkg probes", async () => {
    const calls: { file: string; options: Record<string, unknown> }[] = [];
    const deps = systemEnvironmentDeps({
      env: { PATH: "/bin", LINEAR_API_KEY: "present", NODE_ENV: "test" },
      execFileImpl: async (file, _args, options) => {
        calls.push({ file, options });
        return { stdout: "" };
      },
    });
    await deps.ps();
    await deps.exec("dpkg", ["--print-architecture"], { timeout: 2_000, maxBuffer: 4 * 1024 });
    expect(calls[0]).toMatchObject({ file: "ps", options: { timeout: 5_000, maxBuffer: 2 * 1024 * 1024 } });
    expect(calls[0].options.env).toMatchObject({ LC_ALL: "C", LANG: "C" });
    expect(calls[1]).toMatchObject({ file: "dpkg", options: { timeout: 2_000, maxBuffer: 4 * 1024 } });
    expect(calls[1].options.env).toMatchObject({ LC_ALL: "C", LANG: "C" });
  });

  it("probes only the fixed CLI names without launching them", async () => {
    const fs = {
      stat: vi.fn(async (path: string) => ({ isFile: () => path.endsWith("/claude") })),
      access: vi.fn(async () => {}),
    };
    expect(await probeCliPresence("claude", { PATH: "/a:/b", NODE_ENV: "test" }, fs)).toBe("present");
    expect(await probeCliPresence("codex", { PATH: "/a:/b", NODE_ENV: "test" }, fs)).toBe("missing");
    expect(await probeCliPresence("bash" as never, { PATH: "/a", NODE_ENV: "test" }, fs)).toBe("unknown");
    expect(fs.access).toHaveBeenCalledTimes(1);
  });

  it("reports unknown on a permission error or an unbounded PATH", async () => {
    const denied = {
      stat: async () => { throw Object.assign(new Error("denied"), { code: "EACCES" }); },
      access: async () => {},
    };
    expect(await probeCliPresence("claude", { PATH: "/a", NODE_ENV: "test" }, denied)).toBe("unknown");
    const dirs = Array.from({ length: MAX_PATH_SEGMENTS + 1 }, (_, i) => `/p${i}`).join(":");
    expect(await probeCliPresence("claude", { PATH: dirs, NODE_ENV: "test" }, { stat: async () => ({ isFile: () => true }), access: async () => {} })).toBe("unknown");
  });

  it("keeps the OpenCode installer fallback and times out to unknown", async () => {
    const fallback = {
      stat: vi.fn(async (path: string) => ({ isFile: () => path === "/home/x/.opencode/bin/opencode" })),
      access: async () => {},
    };
    expect(await probeCliPresence("opencode", { PATH: "/bin", HOME: "/home/x", NODE_ENV: "test" }, fallback)).toBe("present");
    const hanging = { stat: () => new Promise<{ isFile(): boolean }>(() => {}), access: async () => {} };
    expect(await probeCliPresence("claude", { PATH: "/a", NODE_ENV: "test" }, hanging, 5)).toBe("unknown");
  });

  it("bounds a filesystem read by bytes", async () => {
    await expect(readBoundedText("/x", { timeoutMs: 50, maxBytes: 4 }, async () => "12345")).rejects.toThrow("read-too-large");
    await expect(readBoundedText("/x", { timeoutMs: 50, maxBytes: 8 }, async () => "12345")).resolves.toBe("12345");
  });
});

function childFacts() {
  const at = new Date(NOW).toISOString();
  return { ok: true, observedAt: at, lastGoodAt: at };
}

function observation(over: Partial<PackageObservation> = {}): PackageObservation {
  const at = new Date(NOW).toISOString();
  return {
    rows: [],
    holds: [],
    simulation: [],
    nativeArch: "amd64",
    state: "ok",
    observedAt: at,
    lastGoodAt: at,
    stale: false,
    error: null,
    warnings: [],
    children: { candidates: childFacts(), holds: childFacts(), simulation: childFacts(), architecture: childFacts() },
    ...over,
  };
}

function pkg(name: string, arch: string, over: Partial<PackageRow> = {}): PackageRow {
  return { name, arch, key: `${name}:${arch}`, fromVersion: "1", toVersion: "2", security: false, ...over };
}

describe("package candidates (AC7)", () => {
  it("selects a bare simulation fact for the Arch:all candidate using the bracket architecture", () => {
    const section = buildPackageSection(observation({
      rows: [pkg("bar", "all")],
      simulation: [{ name: "bar", arch: "all", version: "2" }],
    }), { now: () => NOW });
    const allCandidate = section.candidates.find((candidate) => candidate.key === "bar:all");
    expect(allCandidate).toMatchObject({ selectedNow: true, reason: "selected-now", conflict: false });
    expect(section.counts).toMatchObject({ total: 1, selectedNow: 1, held: 0, deferred: 0 });
  });

  it("treats a bare showhold name as held while a foreign architecture stays distinct", () => {
    const section = buildPackageSection(observation({
      rows: [pkg("foo", "amd64"), pkg("foo", "i386")],
      holds: [{ name: "foo", arch: null }],
    }), { now: () => NOW });
    expect(section.candidates.find((c) => c.key === "foo:amd64")).toMatchObject({ held: true, reason: "held" });
    expect(section.candidates.find((c) => c.key === "foo:i386")).toMatchObject({ held: false, reason: "deferred-unknown" });
  });

  it("keeps multiple eligible bare matches and an unresolved native architecture unknown", () => {
    const ambiguous = buildPackageSection(observation({
      rows: [pkg("bar", "amd64"), pkg("bar", "all")],
      holds: [{ name: "bar", arch: null }],
    }), { now: () => NOW });
    expect(ambiguous.candidates.every((c) => c.reason === "unknown")).toBe(true);
    expect(ambiguous.counts).toMatchObject({ selectedNow: null, held: null, deferred: null });

    const unresolved = buildPackageSection(observation({
      rows: [pkg("bar", "")],
      nativeArch: null,
      holds: [{ name: "bar", arch: null }],
    }), { now: () => NOW });
    expect(unresolved.candidates[0]).toMatchObject({ key: "bar:unknown", reason: "unknown" });
    expect(unresolved.counts).toMatchObject({ total: 1, selectedNow: null, held: null, deferred: null });
  });

  it("marks held-and-simulation-selected evidence as a conflict, not confidently held or ready", () => {
    const conflict = buildPackageSection(observation({
      rows: [pkg("bar", "amd64")],
      holds: [{ name: "bar", arch: "amd64" }],
      simulation: [{ name: "bar", arch: "amd64", version: "2" }],
    }), { now: () => NOW });
    expect(conflict.candidates[0]).toMatchObject({ held: true, selectedNow: false, conflict: true, reason: "unknown" });
    expect(conflict.state).toBe("partial");
  });

  it("does not claim a selected candidate when the simulation version differs", () => {
    const section = buildPackageSection(observation({
      rows: [pkg("bar", "amd64", { toVersion: "3" })],
      simulation: [{ name: "bar", arch: "amd64", version: "2" }],
    }), { now: () => NOW });
    expect(section.candidates[0]).toMatchObject({ selectedNow: false, reason: "deferred-unknown" });
    expect(section.counts).toMatchObject({ selectedNow: 0, deferred: 1 });
  });

  it("builds totals and counts before the 500-row display cap and keeps the remainder", () => {
    const rows = Array.from({ length: 502 }, (_, i) => pkg(`pkg${String(i).padStart(3, "0")}`, "amd64"));
    const section = buildPackageSection(observation({ rows }), { now: () => NOW });
    expect(section.candidates).toHaveLength(500);
    expect(section.truncated).toBe(true);
    expect(section.counts).toEqual({
      total: 502, shown: 500, remainder: 2, selectedNow: 0, held: 0, deferred: 502, unknown: 0,
    });
    expect(section.state).toBe("partial");
  });

  it("preserves known candidates under a partial child warning but keeps an unknown total honest", () => {
    const section = buildPackageSection(observation({
      rows: [pkg("bar", "amd64")],
      simulation: null,
      warnings: ["simulation-unavailable"],
      children: { ...observation().children, simulation: { ok: false, observedAt: new Date(NOW).toISOString(), lastGoodAt: null } },
    }), { now: () => NOW });
    expect(section.state).toBe("partial");
    expect(section.warnings).toContain("simulation-unavailable");
    expect(section.candidates).toHaveLength(1);
    expect(section.counts).toMatchObject({ total: 1, selectedNow: null, deferred: null });
  });

  it("a total package-query failure is unavailable, never an all-clear zero", () => {
    const failed = observation({
      rows: [], state: "failed", lastGoodAt: null, stale: false, error: "package-query-failed",
      warnings: ["package-query-failed"],
      children: { ...observation().children, candidates: { ok: false, observedAt: new Date(NOW).toISOString(), lastGoodAt: null } },
    });
    const section = buildPackageSection(failed, { now: () => NOW });
    expect(section).toMatchObject({ state: "unavailable", counts: null });
    expect(section.warnings).toContain("package-query-failed");
    expect(section.candidates).toEqual([]);
  });
});
