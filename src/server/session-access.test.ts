import { describe, expect, it } from "vitest";
import type { HubPane } from "./db/workflows";
import { getCodexSessions } from "./db/workflows";
import type { SessionAccessRow, SessionAccessSnapshot } from "./collectors/session-access";
import type { CodexSnapshot } from "./collectors/workflow-codex";
import { resolveSessionAccess, takenSessionAccess, type SessionAccessConfig, type SessionAccessDeps } from "./session-access";
import { SessionAccessError, type SessionAccessSelector } from "../lib/sessionAccess";

const cardSelector: SessionAccessSelector = {
  kind: "card", project: "jax-os", harnessSession: "sess-a", pane: "%7", incarnation: "1234:1700000000",
};

function row(overrides: Partial<SessionAccessRow> = {}): SessionAccessRow {
  const base = {
    pane: "%7", sessionId: "$2", windowId: "@3", serverPid: "1234", serverStart: "1700000000",
    dead: false, windowActive: true, paneActive: true, sessionName: "project agent",
    target: "$2:@3.%7", incarnation: "1234:1700000000", eligible: true,
  };
  return { ...base, ...overrides };
}

function pane(overrides: Partial<HubPane> = {}): HubPane {
  return {
    pane: "%7", tmuxIncarnation: "1234:1700000000", session: "project agent", role: "lead", live: true,
    working: false, capsule: null, pendingQuestion: null, lastEventTs: "2026-11-19T10:00:00.000Z",
    subagentCount: 0, runtime: "claude", harnessSession: "sess-a", ...overrides,
  };
}

function deps(
  snapshot: SessionAccessSnapshot,
  panes: HubPane[],
  config: SessionAccessConfig = { enabled: true, roUrl: "http://127.0.0.1:10000/ro", rwUrl: "http://127.0.0.1:10001/rw" },
  codex: Partial<SessionAccessDeps> = {},
) {
  const calls = { collect: 0, projectPanes: 0, codex: 0 };
  const value: SessionAccessDeps & { calls: typeof calls } = {
    collect: async () => { calls.collect += 1; return snapshot; },
    projectPanes: () => { calls.projectPanes += 1; return panes; },
    config: () => config,
    now: () => 1_700_000_100_000,
    ...codex,
    calls,
  };
  return value;
}

function failing(error: Error): SessionAccessSnapshot {
  throw error;
}

describe("resolveSessionAccess — configuration and source", () => {
  it("refuses a disabled integration without any native read", async () => {
    const d = deps({ rows: [], serverIncarnation: null }, [], { enabled: false });
    await expect(resolveSessionAccess(cardSelector, undefined, d)).rejects.toMatchObject({ code: "unconfigured" });
    expect(d.calls.collect).toBe(0);
  });

  it.each([
    "http://host/ro?x=1",
    "https://host/ro#fragment",
    "ftp://host/ro",
    "not-a-url",
  ])("rejects the invalid RO base %s before any native collection", async (roUrl) => {
    const d = deps({ rows: [row()], serverIncarnation: "1234:1700000000" }, [pane()],
      { enabled: true, roUrl, rwUrl: "http://host/rw" });
    await expect(resolveSessionAccess(cardSelector, undefined, d)).rejects.toMatchObject({ code: "unconfigured" });
    expect(d.calls.collect).toBe(0);
  });

  it("maps a genuine collector failure to source-failure, but a quiet no-server snapshot is not one", async () => {
    const failingDeps = deps({} as SessionAccessSnapshot, []);
    failingDeps.collect = async () => failing(new Error("tmux broken"));
    await expect(resolveSessionAccess(cardSelector, undefined, failingDeps)).rejects.toMatchObject({ code: "source-failure" });

    const quiet = deps({ rows: [], serverIncarnation: null }, []);
    await expect(resolveSessionAccess({ kind: "tmux", session: "gone" }, undefined, quiet))
      .rejects.toMatchObject({ code: "unavailable" });
  });
});

describe("resolveSessionAccess — card binding", () => {
  const snapshot = { rows: [row()], serverIncarnation: "1234:1700000000" };

  it("resolves the exact project/pane/incarnation/harness seat with its runtime", async () => {
    const resolved = await resolveSessionAccess(cardSelector, undefined, deps(snapshot, [pane()]));
    expect(resolved.expected).toEqual({ target: "$2:@3.%7", incarnation: "1234:1700000000" });
    expect(resolved.sessionName).toBe("project agent");
    expect(resolved.runtime).toBe("claude");
    expect(resolved.observedAt).toBe("2023-11-14T22:15:00.000Z");
    const url = new URL(resolved.roUrl);
    expect(url.searchParams.getAll("arg")).toEqual(["$2:@3.%7"]);
    expect(url.searchParams.get("disableReconnect")).toBe("true");
  });

  it("refuses a same-project foreign pane, a different harness on a reused pane and an unproven seat", async () => {
    await expect(resolveSessionAccess(cardSelector, undefined, deps(snapshot, [pane({ pane: "%8" })])))
      .rejects.toMatchObject({ code: "unavailable" });

    await expect(resolveSessionAccess(cardSelector, undefined, deps(snapshot, [pane({ harnessSession: "other" })])))
      .rejects.toMatchObject({ code: "unavailable" });

    await expect(resolveSessionAccess({ ...cardSelector, harnessSession: null }, undefined, deps(snapshot, [pane()])))
      .rejects.toMatchObject({ code: "unavailable" });
  });

  it("resolves an explicit unidentified-pane seat as null, never a harness wildcard", async () => {
    const resolved = await resolveSessionAccess(
      { ...cardSelector, harnessSession: null },
      undefined,
      deps(snapshot, [pane({ harnessSession: null, runtime: "unknown", role: "adhoc" })]),
    );
    expect(resolved.runtime).toBe("unknown");
    expect(resolved.expected.target).toBe("$2:@3.%7");
  });

  it("does not treat an omitted harnessSession as the explicit null identity", async () => {
    const unrecorded = pane();
    delete (unrecorded as unknown as Record<string, unknown>).harnessSession;
    await expect(resolveSessionAccess(
      { ...cardSelector, harnessSession: null },
      undefined,
      deps(snapshot, [unrecorded], { enabled: true, roUrl: "http://127.0.0.1:10000/ro" }),
    )).rejects.toMatchObject({ code: "unavailable" });

    const resolved = await resolveSessionAccess(
      { ...cardSelector, harnessSession: null },
      undefined,
      deps(snapshot, [pane({ harnessSession: null })], { enabled: true, roUrl: "http://127.0.0.1:10000/ro" }),
    );
    expect(resolved.roUrl).toContain("http://127.0.0.1:10000/ro");
  });

  it("resolves the requested unlinked pane beside unrelated linked and over-cap rows", async () => {
    const rows = [
      row(),
      row({ pane: "%9", target: "$2:@3.%9" }),
      row({ pane: "%8", target: "", sessionName: "n".repeat(300), eligible: false }),
    ];
    const resolved = await resolveSessionAccess(cardSelector, undefined, deps({ rows, serverIncarnation: "1234:1700000000" }, [pane()]));
    expect(resolved.expected.target).toBe("$2:@3.%7");
  });

  it("refuses a requested linked or malformed pane", async () => {
    const rows = [row({ pane: "%9", target: "$2:@3.%9" }), row({ pane: "%8", target: "", eligible: false })];
    const linkedSnapshot = { rows, serverIncarnation: "1234:1700000000" };
    await expect(resolveSessionAccess(cardSelector, undefined, deps(linkedSnapshot, [pane({ pane: "%8" })])))
      .rejects.toMatchObject({ code: "unavailable" });
    await expect(resolveSessionAccess(cardSelector, undefined, deps({ rows: [row({ dead: true })], serverIncarnation: "1234:1700000000" }, [pane()])))
      .rejects.toMatchObject({ code: "lost-target" });
  });

  it("compares a pinned expected binding and reports a changed tuple as lost-target", async () => {
    const expected = { target: "$2:@3.%7", incarnation: "1234:1700000000" };
    await expect(resolveSessionAccess(cardSelector, expected, deps(snapshot, [pane()])))
      .resolves.toMatchObject({ expected });
    await expect(resolveSessionAccess(cardSelector, { ...expected, target: "$2:@4.%7" }, deps(snapshot, [pane()])))
      .rejects.toMatchObject({ code: "lost-target" });
    await expect(resolveSessionAccess(cardSelector, { ...expected, incarnation: "1234:1700000001" }, deps(snapshot, [pane()])))
      .rejects.toMatchObject({ code: "lost-target" });
  });
});

describe("resolveSessionAccess — pinned loss classification (F6)", () => {
  const snapshot = { rows: [row()], serverIncarnation: "1234:1700000000" };
  const expected = { target: "$2:@3.%7", incarnation: "1234:1700000000" };

  it("a missing live hub row is unavailable initially and lost-target when pinned", async () => {
    await expect(resolveSessionAccess(cardSelector, undefined, deps(snapshot, [])))
      .rejects.toMatchObject({ code: "unavailable" });
    await expect(resolveSessionAccess(cardSelector, expected, deps(snapshot, [])))
      .rejects.toMatchObject({ code: "lost-target" });
  });

  it("a harness ownership mismatch stays unavailable with or without a pinned binding", async () => {
    const foreign = [pane({ harnessSession: "other" })];
    await expect(resolveSessionAccess(cardSelector, undefined, deps(snapshot, foreign)))
      .rejects.toMatchObject({ code: "unavailable" });
    await expect(resolveSessionAccess(cardSelector, expected, deps(snapshot, foreign)))
      .rejects.toMatchObject({ code: "unavailable" });
  });

  it("a seat ownership mismatch stays unavailable with or without a pinned binding", async () => {
    // The matching harness row is found first, while the seat iteration sees a different owner
    // of the same pane/incarnation — a strict seat proof failure, never a guessed replacement.
    const reordered = [pane({ harnessSession: "sess-a", session: "zzz" }), pane({ harnessSession: "sess-other", session: "aaa" })];
    await expect(resolveSessionAccess(cardSelector, undefined, deps(snapshot, reordered)))
      .rejects.toMatchObject({ code: "unavailable" });
    await expect(resolveSessionAccess(cardSelector, expected, deps(snapshot, reordered)))
      .rejects.toMatchObject({ code: "unavailable" });
  });

  it("zero eligible native pane rows is unavailable initially and lost-target when pinned", async () => {
    const none = { rows: [], serverIncarnation: "1234:1700000000" };
    await expect(resolveSessionAccess(cardSelector, undefined, deps(none, [pane()])))
      .rejects.toMatchObject({ code: "unavailable" });
    await expect(resolveSessionAccess(cardSelector, expected, deps(none, [pane()])))
      .rejects.toMatchObject({ code: "lost-target" });
  });

  it("an eligible pane row that is dead or has another incarnation is lost-target either way", async () => {
    for (const rows of [[row({ dead: true })], [row({ incarnation: "999:1700009999" })]]) {
      const snap = { rows, serverIncarnation: "1234:1700000000" };
      await expect(resolveSessionAccess(cardSelector, undefined, deps(snap, [pane()])))
        .rejects.toMatchObject({ code: "lost-target" });
      await expect(resolveSessionAccess(cardSelector, expected, deps(snap, [pane()])))
        .rejects.toMatchObject({ code: "lost-target" });
    }
  });
});

describe("resolveSessionAccess — generic tmux selector", () => {
  it("requires exactly one active window/pane on initial access", async () => {
    const selector: SessionAccessSelector = { kind: "tmux", session: "project agent" };
    const twoActive = { rows: [row(), row({ pane: "%8", windowId: "@4", target: "$2:@4.%8" })], serverIncarnation: "1234:1700000000" };
    await expect(resolveSessionAccess(selector, undefined, deps(twoActive, []))).rejects.toMatchObject({ code: "unavailable" });

    const oneActive = { rows: [row(), row({ pane: "%8", windowId: "@4", windowActive: false, paneActive: false, target: "$2:@4.%8" })], serverIncarnation: "1234:1700000000" };
    const resolved = await resolveSessionAccess(selector, undefined, deps(oneActive, []));
    expect(resolved.expected.target).toBe("$2:@3.%7");
    expect(resolved.runtime).toBe("unknown");
  });

  it("keeps a rename bound to the same native IDs and updates the display name", async () => {
    const renamed = { rows: [row({ sessionName: "renamed agent" })], serverIncarnation: "1234:1700000000" };
    const resolved = await resolveSessionAccess(
      { kind: "tmux", session: "project agent" },
      { target: "$2:@3.%7", incarnation: "1234:1700000000" },
      deps(renamed, []),
    );
    expect(resolved.sessionName).toBe("renamed agent");
  });

  it("reports native movement and a changed incarnation as lost-target", async () => {
    const moved = { rows: [row({ windowId: "@4", target: "$2:@4.%7" })], serverIncarnation: "1234:1700000000" };
    await expect(resolveSessionAccess(
      { kind: "tmux", session: "project agent" },
      { target: "$2:@3.%7", incarnation: "1234:1700000000" },
      deps(moved, []),
    )).rejects.toMatchObject({ code: "lost-target" });

    const restarted = { rows: [row({ incarnation: "999:1700009999" })], serverIncarnation: "999:1700009999" };
    await expect(resolveSessionAccess(
      { kind: "tmux", session: "project agent" },
      { target: "$2:@3.%7", incarnation: "1234:1700000000" },
      deps(restarted, []),
    )).rejects.toMatchObject({ code: "lost-target" });
  });

  it("refuses ambiguous native identities under one name", async () => {
    const ambiguous = { rows: [row(), row({ pane: "%8", sessionId: "$5", windowActive: false, paneActive: false, target: "$5:@3.%8" })], serverIncarnation: "1234:1700000000" };
    await expect(resolveSessionAccess({ kind: "tmux", session: "project agent" }, undefined, deps(ambiguous, [])))
      .rejects.toMatchObject({ code: "unavailable" });
  });
});

describe("resolveSessionAccess — fresh native Codex card binding (M3b)", () => {
  const CODEX_ID = "01a1ea73-662c-7c31-9f94-cf1567bd851e";
  const OTHER_ID = "0191f0aa-bbbb-7000-8000-00000000000a";
  const CODEX_INC = "3421408:1793045754";
  const SHORT = `${CODEX_ID.slice(0, 29)}...`;
  const codexSelector: SessionAccessSelector = {
    kind: "card", project: "p1", harnessSession: CODEX_ID, pane: "%183", incarnation: CODEX_INC,
  };
  const expected = { target: "$2:@3.%183", incarnation: CODEX_INC };

  const codexRow = (over: Partial<SessionAccessRow> = {}): SessionAccessRow => row({
    pane: "%183", sessionName: "p1-lead", target: "$2:@3.%183", incarnation: CODEX_INC,
    command: "codex", title: `⠹ p1 | p1 | ${SHORT}`, ...over,
  });
  const snapshot = (...rows: SessionAccessRow[]): SessionAccessSnapshot => ({ rows, serverIncarnation: CODEX_INC });
  const codexSnapshot = (over: Partial<Extract<CodexSnapshot, { ok: true }>> = {}): CodexSnapshot => ({
    ok: true, ts: "2026-11-19T14:00:00.000Z", truncated: false, loadedThreadIds: [CODEX_ID],
    threads: [{ threadId: CODEX_ID, cwd: "/home/rafa/repos/p1", owner: "p1", source: "cli", status: "idle", activeFlags: [], canAcceptDirectInput: true }],
    ...over,
  });
  const membership = (ids: string[] = [CODEX_ID]): ReturnType<typeof getCodexSessions> => ({
    sessions: ids.map((threadId) => ({ threadId })), sourceOk: true, untracked: false,
  }) as unknown as ReturnType<typeof getCodexSessions>;
  const codexDeps = (over: Partial<SessionAccessDeps> = {}): Partial<SessionAccessDeps> => ({
    codexEnabled: () => true,
    codexSnapshot: async () => codexSnapshot(),
    codexSessions: () => membership(),
    ...over,
  });

  it("resolves the proven native Codex terminal with runtime codex and no Claude fallback", async () => {
    const d = deps(snapshot(codexRow()), [pane({ runtime: "claude", harnessSession: "sess-a" })], undefined, {});
    d.codexEnabled = () => true;
    d.codexSnapshot = async () => codexSnapshot();
    d.codexSessions = () => membership();
    const resolved = await resolveSessionAccess(codexSelector, undefined, d);
    expect(resolved.expected).toEqual(expected);
    expect(resolved.sessionName).toBe("p1-lead");
    expect(resolved.runtime).toBe("codex");
    expect(d.calls.projectPanes).toBe(0); // never consults a historical Claude seat
    const url = new URL(resolved.roUrl);
    expect(url.searchParams.getAll("arg")).toEqual(["$2:@3.%183"]);
  });

  it("refuses a stale Claude registration on a current Codex command", async () => {
    const selector: SessionAccessSelector = { ...codexSelector, harnessSession: "sess-a" };
    const d = deps(snapshot(codexRow()), [pane()], undefined, {});
    Object.assign(d, codexDeps());
    await expect(resolveSessionAccess(selector, undefined, d)).rejects.toMatchObject({ code: "unavailable" });
    expect(d.calls.projectPanes).toBe(0);
  });

  it("refuses a full-UUID mismatch, a project owner mismatch and missing event membership", async () => {
    await expect(resolveSessionAccess({ ...codexSelector, harnessSession: OTHER_ID }, undefined, deps(snapshot(codexRow()), [pane()], undefined, codexDeps())))
      .rejects.toMatchObject({ code: "unavailable" });

    const foreignOwner = deps(snapshot(codexRow()), [pane()], undefined, codexDeps({
      codexSnapshot: async () => codexSnapshot({
        threads: [{ threadId: CODEX_ID, cwd: "/home/rafa/repos/p2", owner: "p2", source: "cli", status: "idle", activeFlags: [], canAcceptDirectInput: true }],
      }),
    }));
    await expect(resolveSessionAccess(codexSelector, undefined, foreignOwner)).rejects.toMatchObject({ code: "unavailable" });

    const unmembered = deps(snapshot(codexRow()), [pane()], undefined, codexDeps({ codexSessions: () => membership([]) }));
    await expect(resolveSessionAccess(codexSelector, undefined, unmembered)).rejects.toMatchObject({ code: "unavailable" });
  });

  it("reports a switched title/conversation as unavailable initially and lost-target when pinned", async () => {
    const switched = snapshot(codexRow({ title: `⠹ p1 | p1 | ${OTHER_ID}` }));
    await expect(resolveSessionAccess(codexSelector, undefined, deps(switched, [pane()], undefined, { ...codexDeps() })))
      .rejects.toMatchObject({ code: "unavailable" });
    await expect(resolveSessionAccess(codexSelector, expected, deps(switched, [pane()], undefined, { ...codexDeps() })))
      .rejects.toMatchObject({ code: "lost-target" });
  });

  it("checks all native panes before binding a Codex card", async () => {
    const claimedTwice = snapshot(codexRow(), codexRow({ pane: "%184", target: "$2:@3.%184" }));
    const d = deps(claimedTwice, [pane()], undefined, codexDeps());
    await expect(resolveSessionAccess(codexSelector, undefined, d)).rejects.toMatchObject({ code: "unavailable" });
    await expect(resolveSessionAccess(codexSelector, expected, d)).rejects.toMatchObject({ code: "lost-target" });
    expect(d.calls.projectPanes).toBe(0);
  });

  it("treats dead/moved/incarnation-changed panes as lost-target when pinned", async () => {
    const dead = snapshot(codexRow({ dead: true }));
    await expect(resolveSessionAccess(codexSelector, expected, deps(dead, [pane()], undefined, codexDeps())))
      .rejects.toMatchObject({ code: "lost-target" });

    const moved = snapshot(codexRow({ pane: "%184", target: "$2:@3.%184" }));
    await expect(resolveSessionAccess(codexSelector, expected, deps(moved, [pane()], undefined, codexDeps())))
      .rejects.toMatchObject({ code: "lost-target" });

    const restarted = snapshot(codexRow({ incarnation: "3421408:1793045755" }));
    await expect(resolveSessionAccess(codexSelector, expected, deps(restarted, [pane()], undefined, codexDeps())))
      .rejects.toMatchObject({ code: "lost-target" });
  });

  it("maps a failed, truncated, proof-less or disabled Codex source to source-failure/unavailable", async () => {
    for (const codex of [
      { codexSnapshot: async () => ({ ok: false as const, ts: "x", error: "no socket" }) },
      { codexSnapshot: async () => codexSnapshot({ truncated: true }) },
      { codexSnapshot: async () => codexSnapshot({ loadedThreadIds: undefined }) },
      { codexSnapshot: async () => { throw new Error("timeout"); } },
    ]) {
      const pinned = deps(snapshot(codexRow()), [pane()], undefined, codexDeps(codex));
      await expect(resolveSessionAccess(codexSelector, expected, pinned)).rejects.toMatchObject({ code: "source-failure" });
    }

    const disabled = deps(snapshot(codexRow()), [pane()], undefined, codexDeps({ codexEnabled: () => false }));
    await expect(resolveSessionAccess(codexSelector, undefined, disabled)).rejects.toMatchObject({ code: "unavailable" });

    const absent = deps(snapshot(codexRow()), [pane()]);
    await expect(resolveSessionAccess(codexSelector, undefined, absent)).rejects.toMatchObject({ code: "unavailable" });
  });

  it("forwards a budget reduced by the elapsed native tmux read and refuses an exhausted one", async () => {
    let clock = 1_000;
    let seen: number | null = null;
    const withElapsed = (elapsed: number) => deps(snapshot(codexRow()), [pane()], undefined, {
      ...codexDeps(),
      now: () => clock,
      collect: async () => { clock += elapsed; return snapshot(codexRow()); },
      codexSnapshot: async (ms) => { seen = ms; return codexSnapshot(); },
    });

    await resolveSessionAccess(codexSelector, undefined, withElapsed(500));
    expect(seen).toBe(3_000); // capped by the 3s ceiling

    clock = 1_000; seen = null;
    await resolveSessionAccess(codexSelector, undefined, withElapsed(2_500));
    expect(seen).toBe(2_000); // 4_500 - 2_500 elapsed

    clock = 1_000; seen = null;
    const exhausted = withElapsed(5_000);
    await expect(resolveSessionAccess(codexSelector, undefined, exhausted)).rejects.toMatchObject({ code: "source-failure" });
    expect(seen).toBeNull(); // no RPC starts on a nonpositive budget
  });

  it("keeps ordinary Claude access and explicit native-session selection off the Codex RPC", async () => {
    const claude = deps(snapshot(row({ command: "claude" })), [pane()], undefined, {
      ...codexDeps(),
      codexSnapshot: async () => { throw new Error("must not read Codex"); },
    });
    const resolved = await resolveSessionAccess(cardSelector, undefined, claude);
    expect(resolved.runtime).toBe("claude");

    const generic = deps(snapshot(row({ command: "codex", title: `p1 | ${SHORT}` })), [], undefined, {
      ...codexDeps(),
      codexSnapshot: async () => { throw new Error("must not read Codex"); },
    });
    const named = await resolveSessionAccess({ kind: "tmux", session: "project agent" }, undefined, generic);
    expect(named.expected.target).toBe("$2:@3.%7");
  });
});

describe("takenSessionAccess", () => {
  it("is the only RW builder and re-validates the base", async () => {
    const resolved = await resolveSessionAccess(cardSelector, undefined, deps({ rows: [row()], serverIncarnation: "1234:1700000000" }, [pane()]));
    const taken = takenSessionAccess(resolved, { enabled: true, rwUrl: "http://127.0.0.1:10001/rw" });
    expect(new URL(taken.rwUrl).searchParams.get("arg")).toBe("$2:@3.%7");
    expect(() => takenSessionAccess(resolved, { enabled: true, rwUrl: "http://host/rw?x=1" }))
      .toThrow(SessionAccessError);
  });
});
