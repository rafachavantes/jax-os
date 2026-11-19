import { describe, expect, it } from "vitest";
import type { HubPane } from "./db/workflows";
import type { SessionAccessRow, SessionAccessSnapshot } from "./collectors/session-access";
import { resolveSessionAccess, takenSessionAccess, type SessionAccessDeps } from "./session-access";
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

function deps(snapshot: SessionAccessSnapshot, panes: HubPane[], config = { enabled: true, roUrl: "http://127.0.0.1:10000/ro", rwUrl: "http://127.0.0.1:10001/rw" }) {
  const calls = { collect: 0, projectPanes: 0 };
  const value: SessionAccessDeps & { calls: typeof calls } = {
    collect: async () => { calls.collect += 1; return snapshot; },
    projectPanes: () => { calls.projectPanes += 1; return panes; },
    config: () => config,
    now: () => 1_700_000_100_000,
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

  it("rejects a configured base with query/fragment/userinfo as unconfigured", async () => {
    const d = deps({ rows: [row()], serverIncarnation: "1234:1700000000" }, [pane()],
      { enabled: true, roUrl: "http://host/ro?x=1", rwUrl: "http://host/rw" });
    await expect(resolveSessionAccess(cardSelector, undefined, d)).rejects.toMatchObject({ code: "unconfigured" });
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

describe("takenSessionAccess", () => {
  it("is the only RW builder and re-validates the base", async () => {
    const resolved = await resolveSessionAccess(cardSelector, undefined, deps({ rows: [row()], serverIncarnation: "1234:1700000000" }, [pane()]));
    const taken = takenSessionAccess(resolved, { enabled: true, rwUrl: "http://127.0.0.1:10001/rw" });
    expect(new URL(taken.rwUrl).searchParams.get("arg")).toBe("$2:@3.%7");
    expect(() => takenSessionAccess(resolved, { enabled: true, rwUrl: "http://host/rw?x=1" }))
      .toThrow(SessionAccessError);
  });
});
