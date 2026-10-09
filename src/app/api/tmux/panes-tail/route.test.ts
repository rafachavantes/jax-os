import { describe, expect, it } from "vitest";
import { handlePanesTailGet, type PanesTailRouteDeps } from "./handler";

const NOOP_RUN = async () => "";

describe("GET /api/tmux/panes-tail (Sessions mosaic spec §7 Decisions 9-12)", () => {
  it("caps at 40 panes even when more are live", async () => {
    const paneIds = new Set(Array.from({ length: 45 }, (_, i) => `%${i + 1}`));
    const deps: PanesTailRouteDeps = {
      listLivePanes: async () => ({ paneIds, incarnation: "1:1", paneCommands: new Map() }),
      capturePaneTail: async (pane) => [`line for ${pane}`],
      run: NOOP_RUN,
      deadlineMs: 2_500,
    };
    const res = await handlePanesTailGet(deps);
    const body = await res.json();
    expect(body.ok).toBe(true);
    expect(body.data).toHaveLength(40);
  });

  it("a rejected capture maps to lines:null for that pane only — every other pane's array is untouched (round-1 F5, round-2 F2)", async () => {
    const deps: PanesTailRouteDeps = {
      listLivePanes: async () => ({ paneIds: new Set(["%1", "%2"]), incarnation: "1:1", paneCommands: new Map() }),
      capturePaneTail: async (pane) => { if (pane === "%2") throw new Error("capture failed"); return ["ok"]; },
      run: NOOP_RUN,
      deadlineMs: 2_500,
    };
    const res = await handlePanesTailGet(deps);
    const body = await res.json();
    expect(body.ok).toBe(true);
    expect(body.data).toEqual(expect.arrayContaining([
      { pane: "%1", lines: ["ok"] },
      { pane: "%2", lines: null },
    ]));
  });

  it("total handler time stays bounded by the deadline even when a capture hangs forever", async () => {
    const deps: PanesTailRouteDeps = {
      listLivePanes: async () => ({ paneIds: new Set(["%1"]), incarnation: "1:1", paneCommands: new Map() }),
      capturePaneTail: () => new Promise(() => {}), // never resolves
      run: NOOP_RUN,
      deadlineMs: 50,
    };
    const start = Date.now();
    const res = await handlePanesTailGet(deps);
    const body = await res.json();
    expect(Date.now() - start).toBeLessThan(1_000);
    expect(body.ok).toBe(true);
    expect(body.data).toEqual([{ pane: "%1", lines: null }]);
  });

  it("one absolute batch deadline is shared by every capture — three simultaneous hangs all come back lines:null within it, a fourth fast capture keeps its lines (round-2 review F2)", async () => {
    const deps: PanesTailRouteDeps = {
      listLivePanes: async () => ({ paneIds: new Set(["%1", "%2", "%3", "%4"]), incarnation: "1:1", paneCommands: new Map() }),
      // %1-%3 hang forever; %4 resolves immediately. All four are raced against the SAME
      // deadlineAt computed once for the batch, not four independent per-capture timers.
      capturePaneTail: async (pane: string) => (pane === "%4" ? ["fast"] : new Promise<string[]>(() => {})),
      run: NOOP_RUN,
      deadlineMs: 50,
    };
    const start = Date.now();
    const res = await handlePanesTailGet(deps);
    const body = await res.json();
    expect(Date.now() - start).toBeLessThan(1_000);
    expect(body.ok).toBe(true);
    expect(body.data).toEqual(expect.arrayContaining([
      { pane: "%1", lines: null },
      { pane: "%2", lines: null },
      { pane: "%3", lines: null },
      { pane: "%4", lines: ["fast"] },
    ]));
  });

  it("empty listLivePanes() is a quiet empty, not a warning", async () => {
    const deps: PanesTailRouteDeps = {
      listLivePanes: async () => ({ paneIds: new Set(), incarnation: null, paneCommands: new Map() }),
      capturePaneTail: async () => ["unused"],
      run: NOOP_RUN,
      deadlineMs: 2_500,
    };
    const res = await handlePanesTailGet(deps);
    expect(await res.json()).toEqual({ ok: true, data: [] });
  });

  it("a listLivePanes failure is the ONLY case that yields ok:false — never a single bad capture", async () => {
    const deps: PanesTailRouteDeps = {
      listLivePanes: async () => { throw new Error("spawn tmux ENOENT"); },
      capturePaneTail: async () => ["unused"],
      run: NOOP_RUN,
      deadlineMs: 2_500,
    };
    const res = await handlePanesTailGet(deps);
    expect((await res.json()).ok).toBe(false);
  });
});
