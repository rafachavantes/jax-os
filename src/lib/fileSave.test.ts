import { describe, expect, it } from "vitest";
import { createEdit, patchEdit, type EditState, type EditPatch } from "./filesState";
import { saveEdit, type SaveResult } from "./fileSave";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
function harness() {
  let edits: Record<string, EditState> = { x: createEdit("A", "ha") };
  const pending = new Set<symbol>();
  const patch = (key: string, id: symbol, p: EditPatch) => {
    edits = patchEdit(edits, key, id, p);
  };
  return {
    get edits() { return edits; }, pending, patch,
    change(text: string, key = "x") {
      patch(key, edits[key].id, { content: text, justSaved: false });
    },
    close(key = "x") { const { [key]: removed, ...rest } = edits; edits = rest; },
    open(key = "x") { edits = { ...edits, [key]: createEdit("A", "ha") }; },
    start(key = "x") {
      const request = deferred<SaveResult>();
      const submitted = edits[key];
      let calls = 0;
      const write = () => { calls++; return request.promise; };
      const apply = (p: EditPatch) => patch(key, submitted.id, p);
      const done = saveEdit(submitted, pending, write, apply);
      return { request, submitted, write, apply, done, get calls() { return calls; } };
    },
  };
}

describe("save completion isolation", () => {
  it.each(["B", "C", "A"])("preserves current %s after saving B", async (current) => {
    const h = harness(); h.change("B");
    const r = h.start(); h.change(current);
    r.request.resolve({ ok: true, hash: "hb" }); await r.done;
    expect(h.edits.x).toMatchObject({ content: current, savedContent: "B", baseHash: "hb", saving: false });
    expect(h.edits.x.content !== h.edits.x.savedContent).toBe(current !== "B");
  });
  it("sends the acknowledged hash with the next draft save", async () => {
    const h = harness(); h.change("B"); const first = h.start(); h.change("C");
    first.request.resolve({ ok: true, hash: "hb" }); await first.done;
    const second = h.start();
    expect(second.submitted).toMatchObject({ content: "C", baseHash: "hb" });
    second.request.resolve({ ok: true, hash: "hc" }); await second.done;
    expect(h.edits.x).toMatchObject({ savedContent: "C", baseHash: "hc" });
  });
  it.each(["success", "conflict", "network"])("ignores closed/reopened old %s", async (outcome) => {
    const h = harness(); h.change("B"); const old = h.start();
    h.close(); h.open(); h.change("C"); const fresh = h.start();
    const current = h.edits.x;
    if (outcome === "success") old.request.resolve({ ok: true, hash: "hb" });
    else if (outcome === "conflict") old.request.resolve({ ok: false, error: "changed on disk" });
    else old.request.reject(new Error("offline"));
    await old.done;
    expect(h.edits.x).toBe(current);
    expect(h.pending.has(fresh.submitted.id)).toBe(true);
    fresh.request.resolve({ ok: true, hash: "hc" }); await fresh.done;
    expect(h.edits.x).toMatchObject({ content: "C", savedContent: "C", saving: false });
  });
  it.each([true, false])("does not recreate a closed draft, success=%s", async (success) => {
    const h = harness(); h.change("B"); const r = h.start(); h.close();
    r.request.resolve(success ? { ok: true, hash: "hb" } : { ok: false, error: "failed" });
    await r.done;
    expect(h.edits).toEqual({}); expect(h.pending.size).toBe(0);
  });
  it.each(["changed on disk", "network"])("preserves newer edits on %s", async (error) => {
    const h = harness(); h.change("B"); const r = h.start(); h.change("C");
    if (error === "network") r.request.reject(new Error("offline"));
    else r.request.resolve({ ok: false, error });
    await r.done;
    expect(h.edits.x).toMatchObject({
      content: "C", savedContent: "A", baseHash: "ha", saving: false,
      stale: error === "changed on disk",
      writeErr: false,
      unconfirmed: error === "network",
    });
  });
  it("blocks immediate duplicates and allows independent drafts", async () => {
    const h = harness(); h.change("B"); const x = h.start();
    await saveEdit(x.submitted, h.pending, x.write, x.apply);
    expect(x.calls).toBe(1);
    h.open("y"); h.change("Y", "y"); const y = h.start("y");
    expect(y.calls).toBe(1); expect(h.pending.size).toBe(2);
    const unchangedY = h.edits.y;
    x.request.resolve({ ok: true, hash: "hb" }); await x.done;
    expect(h.edits.y).toBe(unchangedY);
    y.request.resolve({ ok: true, hash: "hy" }); await y.done;
    expect(h.pending.size).toBe(0);
  });
  it("updates baseHash after applied audit failure and keeps newer text", async () => {
    const h = harness();
    h.change("B");
    const r = h.start();
    h.change("C");
    r.request.resolve({
      ok: false, code: "audit-finalization-failed", effect: "applied", audit: "pending",
      error: "action applied but audit recording failed; check current state before retrying",
      hash: "hb",
    });
    await r.done;
    expect(h.edits.x).toMatchObject({
      content: "C", savedContent: "B", baseHash: "hb", auditWarning: true, justSaved: false, saving: false,
    });
    h.change("D");
    expect(h.edits.x.auditWarning).toBe(true);
    expect(h.edits.x.baseHash).toBe("hb");
  });

  it("keeps baseline and text on unconfirmed write", async () => {
    const h = harness();
    h.change("B");
    const r = h.start();
    r.request.resolve({
      ok: false, code: "mutation-unconfirmed", effect: "unconfirmed", audit: "recorded",
      error: "action outcome unconfirmed; check current state before retrying",
    });
    await r.done;
    expect(h.edits.x).toMatchObject({
      content: "B", savedContent: "A", baseHash: "ha", unconfirmed: true, justSaved: false,
    });
  });

  it("marks audit-unavailable without applying", async () => {
    const h = harness();
    h.change("B");
    const r = h.start();
    r.request.resolve({
      ok: false, code: "audit-unavailable", effect: "not-applied", audit: "unavailable",
      error: "audit unavailable; action not performed",
    });
    await r.done;
    expect(h.edits.x).toMatchObject({
      content: "B", savedContent: "A", writeErr: true, auditUnavailable: true,
    });
  });

  it("treats an invalid response without effect as unconfirmed", async () => {
    const h = harness();
    h.change("B");
    const r = h.start();
    r.request.resolve({ ok: false, error: "no json" });
    await r.done;
    expect(h.edits.x).toMatchObject({
      content: "B", savedContent: "A", unconfirmed: true, writeErr: false, justSaved: false,
    });
  });

  it("retains an unresolved auditWarning across a later confirmed save", async () => {
    const h = harness();
    h.change("B");
    const applied = h.start();
    applied.request.resolve({
      ok: false, code: "audit-finalization-failed", effect: "applied", audit: "pending",
      error: "action applied but audit recording failed; check current state before retrying",
      hash: "hb",
    });
    await applied.done;
    expect(h.edits.x).toMatchObject({
      savedContent: "B", baseHash: "hb", auditWarning: true, justSaved: false,
    });
    h.change("C");
    const confirmed = h.start();
    confirmed.request.resolve({ ok: true, hash: "hc" });
    await confirmed.done;
    expect(h.edits.x).toMatchObject({
      savedContent: "C", baseHash: "hc", justSaved: true,
      auditWarning: true, unconfirmed: false, writeErr: false, auditUnavailable: false,
    });
  });

  it("clears stale auditUnavailable after a later confirmed save and keeps unresolved audit warnings", async () => {
    const h = harness();
    h.change("B");
    const denied = h.start();
    denied.request.resolve({
      ok: false, code: "audit-unavailable", effect: "not-applied", audit: "unavailable",
      error: "audit unavailable; action not performed",
    });
    await denied.done;
    expect(h.edits.x.auditUnavailable).toBe(true);
    const recovered = h.start();
    recovered.request.resolve({ ok: true, hash: "hb" });
    await recovered.done;
    expect(h.edits.x).toMatchObject({
      savedContent: "B", baseHash: "hb", justSaved: true,
      auditUnavailable: false, writeErr: false, unconfirmed: false, auditWarning: false,
    });
    h.change("C");
    const applied = h.start();
    applied.request.resolve({
      ok: false, code: "audit-finalization-failed", effect: "applied", audit: "pending",
      error: "action applied but audit recording failed", hash: "hc",
    });
    await applied.done;
    expect(h.edits.x.auditWarning).toBe(true);
    h.change("D");
    const lost = h.start();
    lost.request.reject(new Error("offline"));
    await lost.done;
    expect(h.edits.x).toMatchObject({ content: "D", auditWarning: true, unconfirmed: true });
  });

  it("does not start a save while reload is pending", async () => {
    const h = harness();
    h.change("B");
    h.patch("x", h.edits.x.id, { reloading: true });
    const r = h.start();
    expect(r.calls).toBe(0);
    expect(h.edits.x.saving).toBe(false);
  });

  it("clears saved acknowledgement on new editing without changing identity", async () => {
    const h = harness(); const id = h.edits.x.id; h.change("B"); const r = h.start();
    r.request.resolve({ ok: true, hash: "hb" }); await r.done;
    expect(h.edits.x.justSaved).toBe(true);
    h.change("C");
    expect(h.edits.x.justSaved).toBe(false); expect(h.edits.x.id).toBe(id);
  });
});
