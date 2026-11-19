import { describe, expect, it } from "vitest";
import { codexTerminalBindings, type CodexTerminalObservation } from "./codex-tmux";
import type { CodexSnapshot, CodexThread } from "./workflow-codex";

// The real live observation: pane %183 in jax-os-lead, native title ending in the pinned short
// identifier. The full UUID is this root conversation's own harness session.
const ID = "01a1ea73-662c-7c31-9f94-cf1567bd851e";
const SHORT = `${ID.slice(0, 29)}...`;
// First 29 characters equal to ID's — the observed title cannot distinguish them.
const SIBLING = "01a1ea73-662c-7c31-9f94-cf1560000000";
const TITLE = `⠹ jax-os-new-features | jax-os | ${SHORT}`;

function thread(over: Partial<CodexThread> = {}): CodexThread {
  return {
    threadId: ID, cwd: "/home/rafa/repos/jax-os", owner: "jax-os", source: "cli", status: "active",
    activeFlags: [], canAcceptDirectInput: true, ...over,
  };
}

function snapshot(over: Partial<Extract<CodexSnapshot, { ok: true }>> = {}): CodexSnapshot {
  return { ok: true, ts: "2026-11-19T14:00:00Z", threads: [thread()], loadedThreadIds: [ID], truncated: false, ...over };
}

function row(over: Partial<CodexTerminalObservation> = {}): CodexTerminalObservation {
  return {
    pane: "%183", tmuxIncarnation: "3421408:1793045754", session: "jax-os-lead", command: "codex",
    dead: false, eligible: true, title: TITLE, ...over,
  };
}

const binding = { project: "jax-os", terminal: { pane: "%183", tmuxIncarnation: "3421408:1793045754", session: "jax-os-lead" } };

describe("codexTerminalBindings", () => {
  it("binds the observed native title's shortened identifier and project component", () => {
    expect(codexTerminalBindings(snapshot(), [row()]).get(ID)).toEqual(binding);
  });

  it("accepts the full native UUID component", () => {
    expect(codexTerminalBindings(snapshot(), [row({ title: `jax-os | ${ID}` })]).get(ID)).toEqual(binding);
  });

  it("uses the 24-grapheme native project render and rejects the untruncated long owner", () => {
    const owner = "abcdefghijklmnopqrstuvwxy"; // 25 graphemes
    const shortened = `${owner.slice(0, 21)}...`;
    expect(codexTerminalBindings(snapshot({ threads: [thread({ owner })] }), [row({ title: `${shortened} | ${SHORT}` })]).size).toBe(1);
    expect(codexTerminalBindings(snapshot({ threads: [thread({ owner })] }), [row({ title: `${owner} | ${SHORT}` })]).size).toBe(0);
    const exactly24 = "abcdefghijklmnopqrstuvwx";
    expect(codexTerminalBindings(snapshot({ threads: [thread({ owner: exactly24 })] }), [row({ title: `${exactly24} | ${SHORT}` })]).size).toBe(1);
  });

  it("only matches the last title component, never an earlier UUID or arbitrary prefix", () => {
    expect(codexTerminalBindings(snapshot(), [row({ title: `jax-os | ${SHORT} | jax-os` })]).size).toBe(0);
    expect(codexTerminalBindings(snapshot(), [row({ title: `jax-os | ${ID.slice(0, 20)}` })]).size).toBe(0);
    expect(codexTerminalBindings(snapshot(), [row({ title: `x | ${SHORT}` })]).size).toBe(0);
    expect(codexTerminalBindings(snapshot(), [row({ title: SHORT })]).size).toBe(0);
    expect(codexTerminalBindings(snapshot(), [row({ title: `jax-os | ${SHORT} extra` })]).size).toBe(0);
  });

  it("rejects control-bearing and oversized titles without leaking them", () => {
    expect(codexTerminalBindings(snapshot(), [row({ title: `${TITLE}\u0000` })]).size).toBe(0);
    expect(codexTerminalBindings(snapshot(), [row({ title: `jax-os | ${SHORT}\n` })]).size).toBe(0);
    const oversized = `${"x".repeat(1025)} | ${SHORT}`;
    expect(codexTerminalBindings(snapshot(), [row({ title: oversized })]).size).toBe(0);
    expect(codexTerminalBindings(snapshot(), [row({ title: "" })]).size).toBe(0);
  });

  it("keeps literal pipes inside earlier components as data", () => {
    expect(codexTerminalBindings(snapshot(), [row({ title: `thread | name | jax-os | ${SHORT}` })]).get(ID)).toEqual(binding);
  });

  it("requires a healthy, complete identity discovery", () => {
    expect(codexTerminalBindings({ ok: false, ts: "x", error: "codex runtime unavailable" }, [row()]).size).toBe(0);
    expect(codexTerminalBindings(snapshot({ truncated: true }), [row()]).size).toBe(0);
    expect(codexTerminalBindings({ ok: true, ts: "x", threads: [thread()], truncated: false }, [row()]).size).toBe(0);
    expect(codexTerminalBindings(snapshot({ loadedThreadIds: [ID, ID] }), [row()]).size).toBe(0);
    expect(codexTerminalBindings(null, [row()]).size).toBe(0);
  });

  it("refuses a shortened identifier shared with an excluded loaded thread", () => {
    expect(codexTerminalBindings(snapshot({ loadedThreadIds: [ID, SIBLING] }), [row()]).size).toBe(0);
  });

  it("requires exactly one admitted root with that identity", () => {
    expect(codexTerminalBindings(snapshot({ threads: [] }), [row()]).size).toBe(0);
    expect(codexTerminalBindings(snapshot({ threads: [thread(), thread()] }), [row()]).size).toBe(0);
    expect(codexTerminalBindings(snapshot({ threads: [thread({ owner: null })] }), [row()]).size).toBe(0);
    expect(codexTerminalBindings(snapshot({ threads: [thread({ source: "exec" })] }), [row()]).size).toBe(0);
    expect(codexTerminalBindings(snapshot({ threads: [thread({ status: "notLoaded" })] }), [row()]).size).toBe(0);
    expect(codexTerminalBindings(snapshot({ threads: [thread({ status: "unknown" })] }), [row()]).size).toBe(0);
  });

  it("requires one consistent server incarnation across the observations", () => {
    expect(codexTerminalBindings(snapshot(), [row(), row({ pane: "%184", tmuxIncarnation: "3421408:1793045755" })]).size).toBe(0);
    expect(codexTerminalBindings(snapshot(), [row({ tmuxIncarnation: "not-an-incarnation" })]).size).toBe(0);
  });

  it("refuses duplicate panes, dead panes, shell commands and ineligible rows", () => {
    expect(codexTerminalBindings(snapshot(), [row(), row()]).size).toBe(0);
    expect(codexTerminalBindings(snapshot(), [row({ dead: true })]).size).toBe(0);
    expect(codexTerminalBindings(snapshot(), [row({ command: "bash" })]).size).toBe(0);
    expect(codexTerminalBindings(snapshot(), [row({ command: "" })]).size).toBe(0);
    expect(codexTerminalBindings(snapshot(), [row({ eligible: false })]).size).toBe(0);
    expect(codexTerminalBindings(snapshot(), [row({ pane: "%bad" })]).size).toBe(0);
    expect(codexTerminalBindings(snapshot(), [row({ session: "" })]).size).toBe(0);
  });

  it("keeps every root bound to its own single pane", () => {
    const other = "0191f0aa-cccc-7000-8000-00000000000c";
    const snap = snapshot({
      threads: [thread(), thread({ threadId: other })],
      loadedThreadIds: [ID, other],
    });
    const bindings = codexTerminalBindings(snap, [
      row(),
      row({ pane: "%184", session: "jax-os-second", title: `jax-os | ${other}` }),
    ]);
    expect(bindings.get(ID)).toEqual(binding);
    expect(bindings.get(other)).toEqual({ project: "jax-os", terminal: { pane: "%184", tmuxIncarnation: "3421408:1793045754", session: "jax-os-second" } });
  });

  it("binds nothing when two panes claim the same identity", () => {
    const second = row({ pane: "%184", session: "jax-os-lead-2" });
    expect(codexTerminalBindings(snapshot(), [row(), second]).size).toBe(0);
  });
});
