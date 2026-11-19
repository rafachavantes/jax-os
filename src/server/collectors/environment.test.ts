import { describe, expect, it, vi } from "vitest";
import { parseSettings, settingsKeyPresence, type GeneralSettings } from "../settings";
import { createPackageObservationReader, type PackageObservation, type PackageRow } from "./alerts";
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
    ttyd: { roConfigured: false, rwConfigured: false, ro: { base: null, reachable: null }, rw: { base: null, reachable: null } },
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
      ttyd: { roConfigured: true, rwConfigured: true, ro: { base: "http://127.0.0.1:10000", reachable: true }, rw: { base: "http://127.0.0.1:10000", reachable: true } },
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
    ttyd: {
      roConfigured: false, rwConfigured: false,
      ro: { base: null, reachable: null }, rw: { base: null, reachable: null },
    },
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
    const some = buildComponents(facts({ hostProbes: { resources: true, workloads: null, packages: null } }));
    expect(byId(some, "host-probes")).toMatchObject({ state: "unknown", detail: "sections-partial" });
  });
});

const MOUNTINFO = [
    "22 1 8:1 / / rw,relatime - ext4 /dev/sda1 rw",
    "23 22 0:30 / /tmp rw,nosuid - tmpfs tmpfs rw,size=1000000k",
    "24 22 8:1 /var/tmp /mnt/my\\040disk rw - ext4 /dev/sda1 rw",
    "25 22 0:31 / /tmp rw - tmpfs tmpfs rw,size=2",
  ].join("\n");

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
    ps: async () => "",
    readProc: async () => { throw new Error("no proc"); },
    ...over,
  };
}

describe("mount parsing and /tmp capacity (AC4)", () => {

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

  it("decodes a work-dir escape but never matches it against fixed /tmp", () => {
    const mounts = parseMountInfo("30 1 8:1 / / rw - ext4 /dev/sda1 rw\n31 30 8:1 / /tmp\\040work rw - ext4 /dev/sda1 rw");
    expect(selectMount(mounts, "/tmp")?.mountId).toBe("30"); // the root, never the escaped work-dir mount
    expect(selectMount(mounts, "/tmp work/sub")?.mountId).toBe("31");
    expect(selectMount(mounts, "/tmp work")?.mountId).toBe("31");
  });

  it("validates only loopback ttyd bases", () => {
    expect(loopbackBase("http://127.0.0.1:10000/x")).toBe("http://127.0.0.1:10000");
    expect(loopbackBase("https://localhost:10000")).toBe("https://localhost:10000");
    expect(loopbackBase("http://example.com:10000")).toBeNull();
    expect(loopbackBase("file:///tmp/x")).toBeNull();
    expect(loopbackBase(undefined)).toBeNull();
  });


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
      monotonic: () => now,
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

describe("host-probe component (M5 Task 4)", () => {
  it("reports only already-observed section availability and never starts a package query", async () => {
    const packages = vi.fn(async () => observation());
    const runtime = createEnvironmentRuntime(runtimeDeps({ packages }));
    const none = await runtime.components();
    expect(byId(none.components, "host-probes")).toMatchObject({ state: "unknown", detail: "not-checked" });
    expect(packages).not.toHaveBeenCalled();

    await runtime.resources();
    const partial = await runtime.components();
    expect(byId(partial.components, "host-probes")).toMatchObject({ state: "unknown", detail: "sections-partial" });

    await runtime.workloads();
    await runtime.packages();
    const all = await runtime.components();
    expect(byId(all.components, "host-probes")).toMatchObject({ state: "healthy", detail: "sections-observed" });
  });

  it("shows a partial/unavailable host probe without hiding healthy components", async () => {
    const runtime = createEnvironmentRuntime(runtimeDeps({
      readMountInfo: async () => { throw new Error("unreadable"); },
    }));
    await runtime.resources();
    const section = await runtime.components();
    expect(byId(section.components, "host-probes")).toMatchObject({ state: "unavailable", detail: "sections-partial" });
    expect(byId(section.components, "database")?.state).toBe("healthy");
  });
});

describe("ttyd transport facts (M5 Task 4)", () => {
  const ttydSettings = () => ({ ok: true as const, raw: { integrations: { ttyd: true } }, file: "present" as const });

  async function ttydComponent(over: Partial<EnvironmentDeps>) {
    const runtime = createEnvironmentRuntime(runtimeDeps({
      readSettingsRaw: ttydSettings,
      ...over,
    }));
    const section = await runtime.components();
    return section.components.find((component) => component.id === "ttyd");
  }

  it("requires both configured RO and RW sides", async () => {
    const httpOk = vi.fn(async () => true);
    expect(await ttydComponent({ ttydTransport: () => ({ ro: null, rw: null }), httpOk })).toMatchObject({
      state: "needs-configuration", detail: "urls-missing",
    });
    expect(await ttydComponent({ ttydTransport: () => ({ ro: "http://localhost:10000", rw: null }), httpOk })).toMatchObject({
      state: "needs-configuration", detail: "urls-missing",
    });
    expect(await ttydComponent({ ttydTransport: () => ({ ro: null, rw: "http://localhost:10000" }), httpOk })).toMatchObject({
      state: "needs-configuration", detail: "urls-missing",
    });
    expect(httpOk).not.toHaveBeenCalled();
  });

  it("keeps external configured transports unverified and never probes them", async () => {
    const httpOk = vi.fn(async () => true);
    expect(await ttydComponent({
      ttydTransport: () => ({ ro: "http://example.com:10000", rw: "https://example.org:10000" }),
      httpOk,
    })).toMatchObject({ state: "unknown", detail: "external-unverified-ro-rw" });
    expect(httpOk).not.toHaveBeenCalled();
  });

  it("probes each allowed loopback side and reports a mixed external/local pair unverified", async () => {
    const httpOk = vi.fn(async () => true);
    expect(await ttydComponent({
      ttydTransport: () => ({ ro: "http://example.com:10000", rw: "http://localhost:10000/y" }),
      httpOk,
    })).toMatchObject({ state: "unknown", detail: "external-unverified-ro" });
    expect(httpOk).toHaveBeenCalledTimes(1);
    expect(httpOk).toHaveBeenCalledWith("http://localhost:10000");
  });

  it("treats a malformed configured URL as missing configuration and starts no probe", async () => {
    const httpOk = vi.fn(async () => true);
    expect(await ttydComponent({
      ttydTransport: () => ({ ro: "not a url", rw: "http://localhost:10000" }),
      httpOk,
    })).toMatchObject({ state: "needs-configuration", detail: "urls-missing" });
    expect(httpOk).not.toHaveBeenCalled();
  });

  it("probes both validated loopback sides and one failure cannot be healthy", async () => {
    const httpOk = vi.fn(async (_url: string) => true);
    expect(await ttydComponent({
      ttydTransport: () => ({ ro: "http://127.0.0.1:10000/x", rw: "http://localhost:10001/y" }),
      httpOk,
    })).toMatchObject({ state: "healthy", detail: "http-ok-both" });
    expect(httpOk).toHaveBeenCalledTimes(2);
    expect(httpOk).toHaveBeenCalledWith("http://127.0.0.1:10000");
    expect(httpOk).toHaveBeenCalledWith("http://localhost:10001");

    httpOk.mockImplementation(async (url: string) => url !== "http://localhost:10001");
    expect(await ttydComponent({
      ttydTransport: () => ({ ro: "http://127.0.0.1:10000/x", rw: "http://localhost:10001/y" }),
      httpOk,
    })).toMatchObject({ state: "failed", detail: "http-unreachable" });
  });

  it("does not even read transport env when ttyd is disabled", async () => {
    const transport = vi.fn(() => ({ ro: "http://127.0.0.1:10000", rw: null }));
    const runtime = createEnvironmentRuntime(runtimeDeps({
      readSettingsRaw: () => ({ ok: true, raw: undefined, file: "absent" }),
      ttydTransport: transport,
    }));
    const section = await runtime.components();
    expect(section.components.find((c) => c.id === "ttyd")?.state).toBe("disabled-default");
    expect(transport).not.toHaveBeenCalled();
  });
});

describe("native pane alive evidence (final contracts Task 2)", () => {
  const PANE_PS = " 10 1 1000 S 1 1 tmux\n 100 10 1000 R 1 1 codex";

  function procStatText(pid: number, ppid: number, start: number, state = "S"): string {
    return `${pid} (fixture) ${[state, String(ppid), ...Array<string>(17).fill("0"), String(start)].join(" ")}`;
  }

  function panesWith(dead: [boolean | undefined, boolean | undefined]) {
    let count = 0;
    return async () => {
      const value = dead[Math.min(count, dead.length - 1)];
      count += 1;
      const paneDead = new Map<string, boolean>();
      if (value !== undefined) paneDead.set("%1", value);
      return {
        paneIds: new Set(["%1"]),
        paneCommands: new Map([["%1", "tmux"]]),
        panePids: new Map([["%1", 10]]),
        incarnation: "10:20",
        paneDead,
      };
    };
  }

  async function workloadOwner(dead: [boolean | undefined, boolean | undefined]) {
    const runtime = createEnvironmentRuntime(runtimeDeps({
      ps: async () => PANE_PS,
      readProc: async (pid) => procStatText(pid, pid === 100 ? 10 : 1, 5),
      listPanes: panesWith(dead),
    }));
    const result = await runtime.workloads();
    return result.groups.find((group) => group.family === "codex")?.owner;
  }

  it("keeps an initially dead pane unknown when the fresh snapshot claims alive", async () => {
    expect(await workloadOwner([true, false])).toEqual({ kind: "unknown", reason: "race" });
  });

  it("keeps a pane unknown when both snapshots omit the native dead fact", async () => {
    expect(await workloadOwner([undefined, undefined])).toEqual({ kind: "unknown", reason: "race" });
  });

  it("keeps an initially alive pane unknown when the fresh snapshot reports dead", async () => {
    expect(await workloadOwner([false, true])).toEqual({ kind: "unknown", reason: "race" });
  });

  it("proves the pane owner only when both observations report it alive", async () => {
    expect(await workloadOwner([false, false])).toEqual({ kind: "pane", pane: "%1" });
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

  it("subtracts startup time already spent from the original caller budget", async () => {
    let now = 0;
    const sleeps: number[] = [];
    const sleep = (ms: number) => { sleeps.push(ms); return new Promise<void>(() => {}); };
    const observe = createSourceObserver(() => now, sleep, () => now);
    let release!: (value: string) => void;
    const pending = observe("source", 60_000, 200, async () => {
      now += 150;
      return new Promise<string>((resolve) => { release = resolve; });
    });
    expect(sleeps).toEqual([50]);
    release("done");
    await expect(pending).resolves.toMatchObject({ ok: true, value: "done" });
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

  it("uses the monotonic clock, never a wall step, for the public allowance", async () => {
    let wall = 1_000;
    let mono = 0;
    const sleeps: number[] = [];
    const sleep = (ms: number) => { sleeps.push(ms); return Promise.resolve(); };
    const observe = createSourceObserver(() => wall, sleep, () => mono);
    expect(await observe("source", 5_000, 200, () => new Promise<string>(() => {}))).toMatchObject({ ok: false });
    expect(sleeps).toEqual([200]);
    wall = 900; // backward wall step neither extends nor shortens the original allowance
    mono = 150;
    expect(await observe("source", 5_000, 200, () => new Promise<string>(() => {}))).toMatchObject({ ok: false });
    expect(sleeps).toEqual([200, 50]);
  });

  it("expires the retained observation by monotonic time, never by a wall step", async () => {
    let wall = 1_000;
    let mono = 0;
    const observe = createSourceObserver(() => wall, () => new Promise<void>(() => {}), () => mono);
    const first = await observe("source", 200, 5_000, async () => "a");
    expect(first).toMatchObject({ ok: true, value: "a", observedAt: 1_000 });
    wall = 900;
    mono = 201;
    const second = await observe("source", 200, 5_000, async () => "b");
    expect(second).toMatchObject({ ok: true, value: "b", observedAt: 900 });
  });

  it("fails the attempt when monotonic time passes the budget before a timeout callback", async () => {
    let mono = 0;
    const observe = createSourceObserver(() => 1_000, () => new Promise<void>(() => {}), () => mono);
    const result = await observe("source", 5_000, 200, async () => {
      mono = 201;
      return "late";
    });
    expect(result).toMatchObject({ ok: false, observedAt: 1_000, cause: "deadline" });
  });

  it("reuses the original monotonic deadline for callers with a different budget", async () => {
    let mono = 0;
    const sleeps: number[] = [];
    const sleep = (ms: number) => { sleeps.push(ms); return Promise.resolve(); };
    const observe = createSourceObserver(() => 1_000, sleep, () => mono);
    await observe("source", 5_000, 200, () => new Promise<string>(() => {}));
    expect(sleeps).toEqual([200]);
    mono = 150;
    await observe("source", 5_000, 10_000, () => new Promise<string>(() => {}));
    expect(sleeps).toEqual([200, 50]);
  });

  it("keeps wall observation timestamps while elapsed decisions use monotonic time", async () => {
    let wall = 1_000;
    let mono = 0;
    const observe = createSourceObserver(() => wall, () => new Promise<void>(() => {}), () => mono);
    const first = await observe("value", 60_000, 1_000, async () => {
      mono = 500;
      wall = 9_000;
      return "a";
    });
    expect(first).toMatchObject({ ok: true, value: "a", observedAt: 1_000 });
    wall = 12_000;
    const cached = await observe("value", 60_000, 1_000, async () => "b");
    expect(cached).toMatchObject({ ok: true, value: "a", observedAt: 1_000 });
  });
});

describe("own-DB failure state and original operation (observation completion Task 2)", () => {
  it("projects a typed unsuccessful snapshot as failed with the original last-good time", async () => {
    let now = NOW;
    let mode: "ok" | "failed" = "ok";
    const ownDb = async () => mode === "ok"
      ? { ok: true as const, schemaVersion: 15, latestMetricTs: new Date(now).toISOString(), paneContext: null }
      : { ok: false as const, error: "database-unavailable" as const };
    const runtime = createEnvironmentRuntime(runtimeDeps({ now: () => now, monotonic: () => now, ownDb }));
    expect(byId((await runtime.components()).components, "database")?.state).toBe("healthy");
    const successAt = new Date(now).toISOString();
    now += 6 * 60_000;
    mode = "failed";
    const second = await runtime.components();
    expect(byId(second.components, "database")).toMatchObject({ state: "failed", lastGoodAt: successAt });
    expect(byId(second.components, "metrics")?.state).not.toBe("healthy");
  });

  it("a first typed failure has no healthy badge and no last-good claim", async () => {
    const runtime = createEnvironmentRuntime(runtimeDeps({
      ownDb: async () => ({ ok: false, error: "database-unavailable" }),
    }));
    const section = await runtime.components();
    expect(byId(section.components, "database")).toMatchObject({ state: "failed", lastGoodAt: null });
  });

  it("keeps one original pending read across repeated caller timeouts", async () => {
    let now = NOW;
    let mode: "ok" | "pending" = "ok";
    const ownDb = vi.fn(async () => mode === "ok"
      ? { ok: true as const, schemaVersion: 15, latestMetricTs: null, paneContext: null }
      : new Promise<never>(() => {}));
    const runtime = createEnvironmentRuntime(runtimeDeps({ now: () => now, monotonic: () => now, ownDb, sleep: async () => {} }));
    await runtime.components();
    const afterFirst = ownDb.mock.calls.length;
    now += 10 * 60_000;
    mode = "pending";
    const firstFailed = await runtime.components();
    const afterSecond = ownDb.mock.calls.length;
    const secondFailed = await runtime.components();
    expect(afterSecond - afterFirst).toBe(2); // one original load per own-DB source, never a duplicate
    expect(ownDb.mock.calls.length).toBe(afterSecond);
    expect(byId(firstFailed.components, "database")).toMatchObject({ state: "unknown", detail: "database-unverified" });
    expect(byId(secondFailed.components, "database")).toMatchObject({ state: "unknown", detail: "database-unverified" });
  });
});

describe("unsuccessful observation causes (F3)", () => {
  it("maps an explicit first tmux rejection to unavailable even without last-good evidence", async () => {
    const runtime = createEnvironmentRuntime(runtimeDeps({
      listPanes: async () => { throw new Error("tmux refused"); },
    }));
    expect(byId((await runtime.components()).components, "tmux")).toMatchObject({
      state: "unavailable", detail: "tmux-unavailable",
    });
    expect((await runtime.workloads()).warning).toBe("tmux-unavailable");
  });

  it("maps a still-pending tmux deadline to unknown and tmux-unverified", async () => {
    let mono = 0;
    let panesCalls = 0;
    const sleeps: (() => void)[] = [];
    const sleep = (ms: number) => new Promise<void>((resolve) => { sleeps.push(resolve); });
    const runtime = createEnvironmentRuntime(runtimeDeps({
      monotonic: () => mono,
      sleep,
      listPanes: () => {
        panesCalls += 1;
        if (panesCalls === 1) return new Promise(() => {});
        return Promise.resolve({ paneIds: new Set<string>(), incarnation: null, paneCommands: new Map<string, string>() });
      },
    }));
    const components = runtime.components();
    await Promise.resolve();
    expect(sleeps.length).toBeGreaterThan(0);
    mono = 5_000;
    sleeps.shift()?.();
    expect(byId((await components).components, "tmux")).toMatchObject({
      state: "unknown", detail: "tmux-unverified",
    });

    mono = 10_000;
    const workloads = runtime.workloads();
    await Promise.resolve();
    expect(sleeps.length).toBeGreaterThan(0);
    sleeps.shift()?.(); // the cached tmux source deadline expires; ps itself resolves
    expect((await workloads).warning).toBe("tmux-unverified");
  });

  it("keeps a prior last-good timestamp when a later own-DB load overruns its allowance", async () => {
    let mono = 0;
    let mode: "ok" | "slow" = "ok";
    const ownDb = async () => {
      if (mode === "slow") mono += 250; // settles only after the original 200ms allowance
      return { ok: true as const, schemaVersion: 15, latestMetricTs: null, paneContext: null };
    };
    const runtime = createEnvironmentRuntime(runtimeDeps({
      now: () => NOW, monotonic: () => mono, ownDb,
    }));
    const first = await runtime.components();
    expect(byId(first.components, "database")?.state).toBe("healthy");
    const firstObservedAt = byId(first.components, "database")?.observedAt;
    mode = "slow";
    mono += 6 * 60_000;
    const second = await runtime.components();
    expect(byId(second.components, "database")).toMatchObject({
      state: "unknown", detail: "database-unverified", lastGoodAt: firstObservedAt,
    });
    expect(byId(second.components, "metrics")).toMatchObject({
      state: "unknown", detail: "database-unverified",
    });
  });

  it("keeps an in-budget own-DB rejection as a failed database", async () => {
    const runtime = createEnvironmentRuntime(runtimeDeps({
      ownDb: async () => { throw new Error("database-unavailable"); },
    }));
    expect(byId((await runtime.components()).components, "database")).toMatchObject({
      state: "failed", detail: "database-unavailable", lastGoodAt: null,
    });
  });

  it("does not turn a retained last-good value into a healthy badge after a deadline", async () => {
    let mono = 0;
    let mode: "ok" | "pending" = "ok";
    const ownDb = () => mode === "ok"
      ? Promise.resolve({ ok: true as const, schemaVersion: 15, latestMetricTs: null, paneContext: null })
      : new Promise<never>(() => {});
    const runtime = createEnvironmentRuntime(runtimeDeps({
      now: () => NOW, monotonic: () => mono, ownDb, sleep: async () => {},
    }));
    await runtime.components();
    mode = "pending";
    mono += 6 * 60_000;
    const section = await runtime.components();
    expect(byId(section.components, "database")?.state).toBe("unknown");
    expect(byId(section.components, "metrics")?.state).not.toBe("healthy");
  });
});

describe("fixed probe budgets and CLI presence (M5 Task 1)", () => {
  it("passes the approved locale and budgets to the ps probe", async () => {
    const calls: { file: string; options: Record<string, unknown> }[] = [];
    const deps = systemEnvironmentDeps({
      env: { PATH: "/bin", LINEAR_API_KEY: "present", NODE_ENV: "test" },
      execFileImpl: async (file, _args, options) => {
        calls.push({ file, options });
        return { stdout: "" };
      },
    });
    await deps.ps();
    expect(calls[0]).toMatchObject({ file: "ps", options: { timeout: 5_000, maxBuffer: 2 * 1024 * 1024 } });
    expect(calls[0].options.env).toMatchObject({ LC_ALL: "C", LANG: "C" });
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

  it("keeps the OpenCode installer fallback and treats inconclusive filesystem errors as unknown", async () => {
    const fallback = {
      stat: vi.fn(async (path: string) => ({ isFile: () => path === "/home/x/.opencode/bin/opencode" })),
      access: async () => {},
    };
    expect(await probeCliPresence("opencode", { PATH: "/bin", HOME: "/home/x", NODE_ENV: "test" }, fallback)).toBe("present");
    const eio = {
      stat: async () => { throw Object.assign(new Error("io"), { code: "EIO" }); },
      access: async () => {},
    };
    expect(await probeCliPresence("claude", { PATH: "/a", NODE_ENV: "test" }, eio)).toBe("unknown");
    const absent = {
      stat: async () => { throw Object.assign(new Error("absent"), { code: "ENOENT" }); },
      access: async () => {},
    };
    expect(await probeCliPresence("claude", { PATH: "/a", NODE_ENV: "test" }, absent)).toBe("missing");
  });

  it("leaves a pending CLI read owned by the source observer instead of opening a second one", async () => {
    let cliCalls = 0;
    const runtime = createEnvironmentRuntime(runtimeDeps({
      readSettingsRaw: () => ({ ok: true, raw: { integrations: { agents: { claude: true } } }, file: "present" }),
      sleep: async () => {},
      cli: () => { cliCalls += 1; return new Promise<never>(() => {}); },
    }));
    const first = await runtime.components();
    expect(cliCalls).toBe(4);
    expect(byId(first.components, "claude")?.state).toBe("unknown");
    await runtime.components();
    expect(cliCalls).toBe(4);
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

  it("selects the raw Arch:all candidate through the real parser-to-section path", async () => {
    const exec = vi.fn(async (file: string) => {
      if (file === "apt") return "Listing...\nbar/stable 2 all [upgradable from: 1]";
      if (file === "apt-mark") return "";
      if (file === "apt-get") {
        return [
          "NOTE: This is only a simulation!",
          " apt-get needs root privileges for real execution.",
          " Keep also in mind that locking is deactivated,",
          " so don't depend on the relevance to the real current situation!",
          "Reading package lists...",
          "Building dependency tree...",
          "Reading state information...",
          "The following packages will be upgraded:",
          "  bar",
          "Inst bar [1] (2 stable [all])",
          "1 upgraded, 0 newly installed, 0 to remove and 0 not upgraded.",
        ].join("\n");
      }
      if (file === "dpkg") return "amd64";
      throw new Error(`unexpected ${file}`);
    });
    const observation = await createPackageObservationReader(exec, () => NOW).read();
    const section = buildPackageSection(observation, { now: () => NOW + 300_000 });
    expect(section.candidates[0]).toMatchObject({ key: "bar:all", selectedNow: true, reason: "selected-now" });
    expect(section.counts).toMatchObject({ total: 1, selectedNow: 1, held: 0, deferred: 0, unknown: 0 });
  });

  it("preserves the source observation time instead of renewing it on projection", async () => {
    const original = observation({ observedAt: new Date(NOW - 600_000).toISOString() });
    const section = buildPackageSection(original, { now: () => NOW });
    expect(section.observedAt).toBe(original.observedAt);
  });

  it("keeps missing simulation evidence unknown with nullable selection counts", () => {
    const section = buildPackageSection(observation({
      rows: [pkg("bar", "amd64")],
      simulation: null,
      warnings: ["simulation-unavailable"],
    }), { now: () => NOW });
    expect(section.candidates[0]).toMatchObject({ reason: "unknown", selectedNow: false });
    expect(section.counts).toMatchObject({ total: 1, selectedNow: null, held: null, deferred: null });
    expect(section.state).toBe("partial");
  });

  it("does not claim current ready work from retained solver facts after a failed child", () => {
    const staleChildren = {
      ...observation().children,
      holds: { ok: false, observedAt: new Date(NOW).toISOString(), lastGoodAt: new Date(NOW - 60_000).toISOString() },
      simulation: { ok: false, observedAt: new Date(NOW).toISOString(), lastGoodAt: new Date(NOW - 60_000).toISOString() },
    };
    const section = buildPackageSection(observation({
      rows: [pkg("bar", "amd64")],
      holds: [{ name: "bar", arch: "amd64" }],
      simulation: [{ name: "bar", arch: "amd64", version: "2" }],
      state: "partial", stale: true, warnings: ["simulation-unavailable"],
      children: staleChildren,
    }), { now: () => NOW });
    expect(section.candidates[0]).toMatchObject({ reason: "unknown", selectedNow: false });
    expect(section.counts?.selectedNow).toBeNull();
    expect(section.state).toBe("partial");
  });

  it("classifies a held-and-selected conflict only as unknown", () => {
    const section = buildPackageSection(observation({
      rows: [pkg("bar", "amd64")],
      holds: [{ name: "bar", arch: "amd64" }],
      simulation: [{ name: "bar", arch: "amd64", version: "2" }],
    }), { now: () => NOW });
    expect(section.candidates[0]).toMatchObject({ reason: "unknown", selectedNow: false, held: true });
    expect(section.counts).toMatchObject({ total: 1, selectedNow: 0, held: 0, deferred: 0, unknown: 1 });
    expect(section.state).toBe("partial");
  });

  it("keeps conflicting duplicate versions as one unknown candidate", () => {
    const section = buildPackageSection(observation({
      rows: [pkg("bar", "amd64"), pkg("bar", "amd64", { fromVersion: "0", toVersion: "3" })],
      simulation: [{ name: "bar", arch: "amd64", version: "2" }],
    }), { now: () => NOW });
    expect(section.candidates).toHaveLength(1);
    expect(section.candidates[0]).toMatchObject({ key: "bar:amd64", reason: "unknown", selectedNow: false, conflict: true });
    expect(section.counts).toMatchObject({ total: 1, unknown: 1, selectedNow: 0, held: 0, deferred: 0 });
    expect(section.state).toBe("partial");
  });

  it("keeps known rows from a partial listing as an explicitly incomplete count", () => {
    const section = buildPackageSection(observation({
      rows: [pkg("bar", "amd64")],
      warnings: ["candidates-unrecognized"],
      children: {
        ...observation().children,
        candidates: { ok: false, observedAt: new Date(NOW).toISOString(), lastGoodAt: null },
      },
    }), { now: () => NOW });
    expect(section.state).toBe("partial");
    expect(section.warnings).toContain("candidates-unrecognized");
    expect(section.counts).toMatchObject({ total: 1, shown: 1, remainder: 0 });
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
