import { describe, expect, it, vi } from "vitest";
import { openDb } from "../../../../server/db/index";
import { insertEvent, getHubData } from "../../../../server/db/workflows";
import { handleHubGet, type HubRouteDeps } from "./handler";

const NO_CODEX = async () => ({ ok: false as const, ts: "2026-09-10T00:00:00.000Z", error: "codex runtime unavailable" });
const THREAD = "0191f0aa-cccc-7000-8000-00000000000c";
const CODEX_SNAP = async () => ({
  ok: true as const, ts: "2026-09-10T00:00:00.000Z", truncated: false,
  threads: [{ threadId: THREAD, cwd: "/home/rafa/repos/p1", owner: "p1", source: "cli", status: "idle", activeFlags: [], canAcceptDirectInput: true }],
});
const CODEX_EVENT = {
  run_id: null, project: "p1", role: "lead" as const, type: "turn-stopped" as const,
  source: "behavioral" as const, emitter: "codex-stop" as const, harness_session: THREAD,
  payload: { message_tail: "x" }, pane: null, tmux_incarnation: null,
};

describe("GET /api/mission/hub (wiring)", () => {
  it("assembles the nested {byProject, historicalTruncated} envelope from injected deps", async () => {
    const db = openDb(":memory:");
    insertEvent(
      db,
      { run_id: null, project: "p1", role: "lead", type: "turn-started", source: "deterministic",
        emitter: "claude-userprompt", payload: {}, pane: "%1", tmux_incarnation: "100:1" },
      new Date("2026-08-27T10:00:00Z"),
    );
    const deps: HubRouteDeps = {
      getDb: () => db,
      getProjects: () =>
        ({
          projects: [{ dir: "p1", statusMtime: "2026-08-01T00:00:00.000Z" }],
          skipped: 0,
        }) as unknown as ReturnType<HubRouteDeps["getProjects"]>,
      listLivePanes: async () => ({ paneIds: new Set(["%1"]), incarnation: "100:1", paneCommands: new Map([["%1", "claude"]]) }),
      collectCodexSnapshot: NO_CODEX,
      isCodexCliPresent: () => true,
      getHubData,
    };
    const res = await handleHubGet(deps);
    const body = await res.json();
    expect(body.ok).toBe(true);
    expect(body.data.byProject.p1).toBeDefined();
    expect(body.data.byProject.p1.panes[0]).toMatchObject({ pane: "%1", working: true });
    expect(body.data.tmuxSource).toBe("ok");
    expect(typeof body.data.historicalTruncated).toBe("boolean");
    db.close();
  });

  it("a genuine listLivePanes failure is a source warning with native state intact, not a whole-hub failure (C4)", async () => {
    // handleHubGet reads Date.now() directly (not deps-injectable) to pass `now` to getHubData,
    // which admits a project with no live pane only via the 30-day historical window in
    // getHubMembership — pin the clock so CODEX_EVENT's fixed ts stays inside that window
    // regardless of when this test runs (it aged out on 2026-09-26, 30 days after the fixture).
    vi.useFakeTimers();
    try {
      vi.setSystemTime(new Date("2026-09-10T00:00:00.000Z"));
      const db = openDb(":memory:");
      insertEvent(db, CODEX_EVENT, new Date("2026-08-27T10:00:00Z"));
      const deps: HubRouteDeps = {
        getDb: () => db,
        getProjects: () => ({ projects: [], skipped: 0 }) as unknown as ReturnType<HubRouteDeps["getProjects"]>,
        listLivePanes: async () => { throw new Error("spawn tmux ENOENT"); },
        collectCodexSnapshot: CODEX_SNAP,
        isCodexCliPresent: () => true,
        getHubData,
      };
      const res = await handleHubGet(deps);
      const body = await res.json();
      expect(body.ok).toBe(true);
      expect(body.data.tmuxSource).toBe("failed");
      expect(body.data.codexSource).toBe("ok");
      expect(body.data.byProject.p1.codexSessions).toHaveLength(1);
      db.close();
    } finally {
      vi.useRealTimers();
    }
  });

  it("a legitimate no-server empty snapshot is healthy, not a warning (C4)", async () => {
    const db = openDb(":memory:");
    const deps: HubRouteDeps = {
      getDb: () => db,
      getProjects: () => ({ projects: [], skipped: 0 }) as unknown as ReturnType<HubRouteDeps["getProjects"]>,
      listLivePanes: async () => ({ paneIds: new Set(), incarnation: null, paneCommands: new Map() }),
      collectCodexSnapshot: NO_CODEX,
      isCodexCliPresent: () => true,
      getHubData,
    };
    const res = await handleHubGet(deps);
    const body = await res.json();
    expect(body.ok).toBe(true);
    expect(body.data.tmuxSource).toBe("ok");
    db.close();
  });

  it("a Codex collector failure is a source warning — tmux data survives and codexSource is failed", async () => {
    const db = openDb(":memory:");
    insertEvent(
      db,
      { run_id: null, project: "p1", role: "lead", type: "turn-started", source: "deterministic",
        emitter: "claude-userprompt", payload: {}, pane: "%1", tmux_incarnation: "100:1" },
      new Date("2026-08-27T10:00:00Z"),
    );
    const deps: HubRouteDeps = {
      getDb: () => db,
      getProjects: () =>
        ({ projects: [{ dir: "p1", statusMtime: "2026-08-01T00:00:00.000Z" }], skipped: 0 }) as unknown as ReturnType<HubRouteDeps["getProjects"]>,
      listLivePanes: async () => ({ paneIds: new Set(["%1"]), incarnation: "100:1", paneCommands: new Map([["%1", "claude"]]) }),
      collectCodexSnapshot: async () => { throw new Error("no socket"); },
      isCodexCliPresent: () => true,
      getHubData,
    };
    const res = await handleHubGet(deps);
    const body = await res.json();
    expect(body.ok).toBe(true);
    expect(body.data.codexSource).toBe("failed");
    expect(body.data.tmuxSource).toBe("ok");
    expect(body.data.byProject.p1.panes[0]).toMatchObject({ pane: "%1", working: true });
    expect(body.data.byProject.p1.codexSessions).toEqual([]);
    db.close();
  });

  it("both sources failing is still a usable hub with both warnings, never a crash (C4)", async () => {
    const db = openDb(":memory:");
    const deps: HubRouteDeps = {
      getDb: () => db,
      getProjects: () => ({ projects: [], skipped: 0 }) as unknown as ReturnType<HubRouteDeps["getProjects"]>,
      listLivePanes: async () => { throw new Error("spawn tmux ENOENT"); },
      collectCodexSnapshot: async () => { throw new Error("no socket"); },
      isCodexCliPresent: () => true,
      getHubData,
    };
    const res = await handleHubGet(deps);
    const body = await res.json();
    expect(body.ok).toBe(true);
    expect(body.data.tmuxSource).toBe("failed");
    expect(body.data.codexSource).toBe("failed");
    db.close();
  });
  it("readChildLogTail and relToRepos, when supplied via HubRouteDeps, reach getHubData untouched (spec §8/§11 wiring)", async () => {
    const db = openDb(":memory:");
    const readChildLogTail = () => null, relToRepos = () => null;
    let seenArgs: unknown[] = [];
    const spyGetHubData = ((...args: Parameters<typeof getHubData>) => (seenArgs = args, getHubData(...args))) as typeof getHubData;
    const deps: HubRouteDeps = {
      getDb: () => db, getProjects: () => ({ projects: [], skipped: 0 }) as unknown as ReturnType<HubRouteDeps["getProjects"]>,
      listLivePanes: async () => ({ paneIds: new Set(), incarnation: null, paneCommands: new Map() }),
      collectCodexSnapshot: NO_CODEX, isCodexCliPresent: () => true,
      getHubData: spyGetHubData, readChildLogTail, relToRepos,
    };
    const res = await handleHubGet(deps);
    expect((await res.json()).ok).toBe(true);
    expect(seenArgs[6]).toBe(readChildLogTail); expect(seenArgs[7]).toBe(relToRepos); // getHubData's 7th/8th positional params
    db.close();
  });

  // MOA-502 Decision 3: a Claude-only install must not warn about an absent codex CLI.
  it("codexAbsent skips collectCodexSnapshot entirely when the codex CLI is not on PATH", async () => {
    const db = openDb(":memory:");
    const collectCodexSnapshot = vi.fn(async () => {
      throw new Error("must not be called");
    });
    const deps: HubRouteDeps = {
      getDb: () => db,
      getProjects: () => ({ projects: [], skipped: 0 }) as unknown as ReturnType<HubRouteDeps["getProjects"]>,
      listLivePanes: async () => ({ paneIds: new Set(), incarnation: null, paneCommands: new Map() }),
      collectCodexSnapshot, isCodexCliPresent: () => false, getHubData,
    };
    const res = await handleHubGet(deps);
    const body = await res.json();
    expect(body.data.codexSource).toBe("absent");
    expect(collectCodexSnapshot).not.toHaveBeenCalled();
    db.close();
  });

  it("calls collectCodexSnapshot as today when the codex CLI is on PATH", async () => {
    const db = openDb(":memory:");
    const collectCodexSnapshot = vi.fn(async () => ({
      ok: true as const, ts: "2026-09-10T00:00:00.000Z", truncated: false, threads: [],
    }));
    const deps: HubRouteDeps = {
      getDb: () => db,
      getProjects: () => ({ projects: [], skipped: 0 }) as unknown as ReturnType<HubRouteDeps["getProjects"]>,
      listLivePanes: async () => ({ paneIds: new Set(), incarnation: null, paneCommands: new Map() }),
      collectCodexSnapshot, isCodexCliPresent: () => true, getHubData,
    };
    const res = await handleHubGet(deps);
    const body = await res.json();
    expect(collectCodexSnapshot).toHaveBeenCalledTimes(1);
    expect(body.data.codexSource).toBe("ok");
    db.close();
  });

  it("codexEnabled false skips collectCodexSnapshot and reports absent even when the CLI is on PATH (MOA-504 D11)", async () => {
    const db = openDb(":memory:");
    const collectCodexSnapshot = vi.fn(async () => { throw new Error("must not be called"); });
    const deps: HubRouteDeps = {
      getDb: () => db,
      getProjects: () => ({ projects: [], skipped: 0 }) as unknown as ReturnType<HubRouteDeps["getProjects"]>,
      listLivePanes: async () => ({ paneIds: new Set(), incarnation: null, paneCommands: new Map() }),
      collectCodexSnapshot, isCodexCliPresent: () => true, codexEnabled: () => false, getHubData,
    };
    const body = await (await handleHubGet(deps)).json();
    expect(body.data.codexSource).toBe("absent");
    expect(collectCodexSnapshot).not.toHaveBeenCalled();
    db.close();
  });

  it("codexEnabled true (or omitted) polls as today", async () => {
    for (const codexEnabled of [() => true, undefined]) {
      const db = openDb(":memory:");
      const collectCodexSnapshot = vi.fn(async () => ({
        ok: true as const, ts: "2026-09-10T00:00:00.000Z", truncated: false, threads: [],
      }));
      const deps: HubRouteDeps = {
        getDb: () => db,
        getProjects: () => ({ projects: [], skipped: 0 }) as unknown as ReturnType<HubRouteDeps["getProjects"]>,
        listLivePanes: async () => ({ paneIds: new Set(), incarnation: null, paneCommands: new Map() }),
        collectCodexSnapshot, isCodexCliPresent: () => true, codexEnabled, getHubData,
      };
      const body = await (await handleHubGet(deps)).json();
      expect(collectCodexSnapshot).toHaveBeenCalledTimes(1);
      expect(body.data.codexSource).toBe("ok");
      db.close();
    }
  });
});
