import { describe, expect, it, vi } from "vitest";
import { parseSettings, settingsKeyPresence, type GeneralSettings } from "../settings";
import {
  buildComponents, buildPackageSection, capacityFromStatfs, createEnvironmentRuntime,
  decodeMountField, loopbackBase, parseMountInfo, selectMount, type ComponentFacts, type EnvironmentDeps,
} from "./environment";

const NOW = Date.parse("2026-11-19T00:00:00.000Z");
const DEFAULTS = (parseSettings(undefined) as { ok: true; data: GeneralSettings }).data;

function facts(over: Partial<ComponentFacts> = {}): ComponentFacts {
  return {
    nowMs: NOW,
    settings: { ok: true, data: DEFAULTS },
    presence: settingsKeyPresence({}),
    db: { ok: true, schemaVersion: 15, error: null },
    projects: { ok: true },
    tmux: { ok: true },
    ttyd: { configured: false, reachable: null },
    clis: { claude: false, codex: false, opencode: false },
    githubCli: false,
    linearCredential: false,
    metricsLatest: new Date(NOW).toISOString(),
    envFile: "ok",
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
      ttyd: { configured: true, reachable: true },
      clis: { claude: true, codex: true, opencode: true },
      githubCli: true,
      linearCredential: true,
    }));
    expect(byId(list, "database")?.state).toBe("healthy");
    expect(byId(list, "projects")?.state).toBe("healthy");
    expect(byId(list, "tmux")?.state).toBe("healthy");
    expect(byId(list, "metrics")?.state).toBe("healthy");
    expect(byId(list, "env-file")?.state).toBe("healthy");
    expect(byId(list, "ttyd")).toMatchObject({ state: "healthy", detail: "http-ok" });
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
      ttyd: { configured: false, reachable: null },
      clis: { claude: false, codex: false, opencode: false },
      githubCli: false,
      linearCredential: false,
      tmux: { ok: false },
      db: { ok: false, schemaVersion: null, error: "database-unavailable" },
      projects: { ok: false },
    }));
    expect(byId(list, "ttyd")).toMatchObject({ state: "needs-configuration", detail: "urls-missing" });
    expect(byId(list, "github")).toMatchObject({ state: "needs-configuration", detail: "cli-missing" });
    expect(byId(list, "linear")).toMatchObject({ state: "needs-configuration", detail: "credential-missing" });
    expect(byId(list, "codex")).toMatchObject({ state: "needs-configuration", detail: "cli-missing" });
    expect(byId(list, "tmux")?.state).toBe("unavailable");
    expect(byId(list, "database")?.state).toBe("failed");
    expect(byId(list, "projects")?.state).toBe("failed");
    // enabled by settings but its CLI is absent — a configuration need, never disabled.
    expect(byId(list, "claude")).toMatchObject({ state: "needs-configuration", detail: "cli-missing" });
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
    const fresh = buildComponents(facts({ metricsLatest: new Date(NOW - 180_000).toISOString() }));
    expect(byId(fresh, "metrics")?.state).toBe("healthy");
    const stale = buildComponents(facts({ metricsLatest: new Date(NOW - 181_000).toISOString() }));
    expect(byId(stale, "metrics")).toMatchObject({ state: "unavailable", detail: "stale" });
    const none = buildComponents(facts({ metricsLatest: null }));
    expect(byId(none, "metrics")).toMatchObject({ state: "unavailable", detail: "no-sample" });
  });

  it("an insecure env file needs configuration while a missing one is an optional default", () => {
    expect(byId(buildComponents(facts({ envFile: "missing" })), "env-file")?.state).toBe("disabled-default");
    expect(byId(buildComponents(facts({ envFile: "bad-mode" })), "env-file")).toMatchObject({
      state: "needs-configuration", detail: "bad-mode",
    });
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
  });

  it("rejects invalid counters and a zero total instead of inventing capacity", () => {
    expect(capacityFromStatfs({ bsize: 4096, blocks: 100, bfree: 10, bavail: 5 })).toEqual({
      usedGb: (90 * 4096) / 1e9, totalGb: (100 * 4096) / 1e9, availableGb: (5 * 4096) / 1e9,
    });
    expect(capacityFromStatfs({ bsize: 0, blocks: 0, bfree: 0, bavail: 0 })).toBeNull();
    expect(capacityFromStatfs({ bsize: NaN, blocks: 1, bfree: 0, bavail: 0 })).toBeNull();
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
      readSettingsRaw: () => ({ ok: true, raw: undefined, file: "absent" }),
      checkDatabase: () => ({ ok: true, schemaVersion: 15, error: null }),
      checkProjects: () => ({ ok: true }),
      listPanes: async () => ({ paneIds: new Set(), incarnation: null, paneCommands: new Map() }),
      ttydBase: () => null,
      httpOk: async () => true,
      cli: () => false,
      linearCredential: () => false,
      metricsLatest: () => null,
      envFile: () => "ok",
      readMountInfo: async () => MOUNTINFO,
      statfs: async () => ({ bsize: 4096, blocks: 1_000_000, bfree: 250_000, bavail: 200_000 }),
      realpath: () => "/tmp",
      packages: async () => ({ rows: [], state: "ok", observedAt: new Date(NOW).toISOString(), lastGoodAt: new Date(NOW).toISOString(), stale: false, error: null }),
      exec: async () => "",
      ps: async () => "",
      readProc: async () => { throw new Error("no proc"); },
      paneContext: () => null,
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
    expect(failed.tmp?.totalGb).toBe(shared.tmp?.totalGb);
  });
});

describe("package candidates (AC7)", () => {
  const observation = {
    rows: [
      { name: "openssl", arch: "amd64", key: "openssl:amd64", fromVersion: "1", toVersion: "2", security: true },
      { name: "bare", arch: "", key: "bare:", fromVersion: "1", toVersion: "2", security: false },
      { name: "held", arch: "amd64", key: "held:amd64", fromVersion: "1", toVersion: "2", security: false },
    ],
    state: "ok" as const,
    observedAt: new Date(NOW).toISOString(), lastGoodAt: new Date(NOW).toISOString(), stale: false, error: null,
  };
  const execFor = (map: Record<string, string | Error>) => vi.fn(async (file: string) => {
    const value = map[file];
    if (value instanceof Error) throw value;
    if (value === undefined) throw new Error(`unexpected ${file}`);
    return value;
  });

  it("separates holds, simulation-selected rows and unknown deferrals with consistent counts", async () => {
    const deps = { now: () => NOW, exec: execFor({
      dpkg: "amd64\n",
      "apt-mark": "held\nbare\n",
      "apt-get": "Inst bare [1] (noble 2 amd64 [amd64])\n",
    }) };
    const section = await buildPackageSection(observation, deps);
    expect(section.state).toBe("ok");
    expect(section.counts).toEqual({ total: 3, selectedNow: 0, held: 2, deferred: 1 });
    expect(section.candidates.map((c) => [c.key, c.reason])).toEqual([
      ["bare:amd64", "held"],
      ["held:amd64", "held"],
      ["openssl:amd64", "deferred-unknown"],
    ]);
  });

  it("a failed simulation keeps candidates but makes ready/deferred counts unknown", async () => {
    const deps = { now: () => NOW, exec: execFor({
      dpkg: "amd64\n", "apt-mark": "", "apt-get": new Error("simulated apt lock"),
    }) };
    const section = await buildPackageSection(observation, deps);
    expect(section.state).toBe("partial");
    expect(section.counts).toBeNull();
    expect(section.warning).toBe("simulation-unavailable");
    expect(section.candidates.every((c) => c.reason === "unknown")).toBe(true);
    expect(section.candidates).toHaveLength(3);
  });

  it("an unknown native architecture keeps bare keys explicitly ambiguous", async () => {
    const deps = { now: () => NOW, exec: execFor({ dpkg: new Error("no dpkg"), "apt-mark": "", "apt-get": "" }) };
    const section = await buildPackageSection(observation, deps);
    expect(section.candidates.find((c) => c.name === "bare")?.key).toBe("bare:unknown");
  });

  it("a total package-query failure is unavailable, never an all-clear zero", async () => {
    const deps = { now: () => NOW, exec: execFor({}) };
    const section = await buildPackageSection({
      ...observation, rows: [], state: "failed", error: "package-query-failed",
    }, deps);
    expect(section).toMatchObject({ state: "unavailable", counts: null, warning: "package-query-failed" });
    expect(section.candidates).toEqual([]);
  });
});
