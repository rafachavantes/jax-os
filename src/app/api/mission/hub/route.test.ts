import { describe, expect, it, vi } from "vitest";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
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

  describe("artifact enrichment resolves the project's discovered location", () => {
    const baseDeps = (db: ReturnType<typeof openDb>, scan: unknown): HubRouteDeps => ({
      getDb: () => db,
      getProjects: () => scan as ReturnType<HubRouteDeps["getProjects"]>,
      listLivePanes: async () => ({ paneIds: new Set(), incarnation: null, paneCommands: new Map() }),
      collectCodexSnapshot: NO_CODEX, isCodexCliPresent: () => true, getHubData,
    });
    const runEvents = (db: ReturnType<typeof openDb>, project: string, target: string, reportPath: string) => {
      insertEvent(db, { run_id: `run-${project}`, project, role: "builder", type: "run-started", source: "deterministic",
        emitter: "wrapper", payload: { kind: "spec", target }, pane: null, tmux_incarnation: null }, new Date());
      insertEvent(db, { run_id: `run-${project}`, project, role: "builder", type: "run-finished", source: "deterministic",
        emitter: "wrapper", payload: { contract_status: "ok", result: "success", report_path: reportPath }, pane: null, tmux_incarnation: null }, new Date());
    };

    it("normalizes nested relative spec targets and keeps absolute reports contained", async () => {
      const root = mkdtempSync(join(tmpdir(), "jax-hub-scan-"));
      try {
        const projectRoot = join(root, "company", "nested");
        mkdirSync(join(projectRoot, ".local", "docs", "specs"), { recursive: true });
        mkdirSync(join(projectRoot, ".local", "docs", "reports"), { recursive: true });
        writeFileSync(join(projectRoot, ".local", "docs", "specs", "x-spec.md"), "spec\n");
        writeFileSync(join(projectRoot, ".local", "docs", "reports", "x.md"), "report\n");
        const db = openDb(":memory:");
        runEvents(db, "nested", ".local/docs/specs/x-spec.md", join(projectRoot, ".local", "docs", "reports", "x.md"));
        const scan = { projects: [{ dir: "nested", path: "company/nested", statusMtime: "2026-08-01T00:00:00.000Z" }], skipped: 0, reposRoot: root };
        const body = await (await handleHubGet(baseDeps(db, scan))).json();
        expect(body.data.byProject.nested.lastRun.targetRel).toBe(".local/docs/specs/x-spec.md");
        expect(body.data.byProject.nested.lastRun.reportRel).toBe(".local/docs/reports/x.md");
        db.close();
      } finally {
        rmSync(root, { recursive: true, force: true });
      }
    });

    it("keeps a direct child rooted at reposRoot/dir", async () => {
      const root = mkdtempSync(join(tmpdir(), "jax-hub-scan-"));
      try {
        const projectRoot = join(root, "direct");
        mkdirSync(join(projectRoot, ".local", "docs", "specs"), { recursive: true });
        writeFileSync(join(projectRoot, ".local", "docs", "specs", "x-spec.md"), "spec\n");
        const db = openDb(":memory:");
        runEvents(db, "direct", ".local/docs/specs/x-spec.md", "");
        const scan = { projects: [{ dir: "direct", statusMtime: "2026-08-01T00:00:00.000Z" }], skipped: 0, reposRoot: root };
        const body = await (await handleHubGet(baseDeps(db, scan))).json();
        expect(body.data.byProject.direct.lastRun.targetRel).toBe(".local/docs/specs/x-spec.md");
        expect(body.data.byProject.direct.lastRun.reportRel).toBeNull();
        db.close();
      } finally {
        rmSync(root, { recursive: true, force: true });
      }
    });

    it("returns null enrichment for unknown or ambiguous identities", async () => {
      const root = mkdtempSync(join(tmpdir(), "jax-hub-scan-"));
      try {
        const db = openDb(":memory:");
        runEvents(db, "ghost", ".local/docs/specs/x-spec.md", "");
        const unknown = await (await handleHubGet(baseDeps(db, { projects: [], skipped: 0, reposRoot: root }))).json();
        expect(unknown.data.byProject.ghost.lastRun.targetRel).toBeNull();
        db.close();

        const ambiguousDb = openDb(":memory:");
        runEvents(ambiguousDb, "dup", ".local/docs/specs/x-spec.md", "");
        const ambiguousScan = {
          projects: [{ dir: "dup", path: "one/dup" }, { dir: "dup", path: "two/dup" }],
          skipped: 0, reposRoot: root,
        };
        const ambiguous = await (await handleHubGet(baseDeps(ambiguousDb, ambiguousScan))).json();
        expect(ambiguous.data.byProject.dup.lastRun.targetRel).toBeNull();
        ambiguousDb.close();
      } finally {
        rmSync(root, { recursive: true, force: true });
      }
    });

    it("drops relative and absolute targets that escape the discovered project root", async () => {
      const root = mkdtempSync(join(tmpdir(), "jax-hub-scan-"));
      try {
        const projectRoot = join(root, "company", "evil");
        mkdirSync(projectRoot, { recursive: true });
        writeFileSync(join(root, "company", "outside.md"), "outside\n");
        const db = openDb(":memory:");
        runEvents(db, "evil", "../outside.md", join(root, "company", "outside.md"));
        const scan = { projects: [{ dir: "evil", path: "company/evil" }], skipped: 0, reposRoot: root };
        const body = await (await handleHubGet(baseDeps(db, scan))).json();
        expect(body.data.byProject.evil.lastRun.targetRel).toBeNull();
        expect(body.data.byProject.evil.lastRun.reportRel).toBeNull();
        db.close();
      } finally {
        rmSync(root, { recursive: true, force: true });
      }
    });
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
