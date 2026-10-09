import { describe, expect, it, vi } from "vitest";
import { createEdit, patchEdit, type EditPatch, type EditState } from "./filesState";
import { reloadEdit, reloadReadResult } from "./fileReload";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function harness() {
  let edits: Record<string, EditState> = { x: createEdit("A", "ha") };
  const pending = new Set<symbol>();
  const patchById = (id: symbol, p: EditPatch) => {
    const key = Object.keys(edits).find((k) => edits[k].id === id);
    if (!key) return;
    edits = patchEdit(edits, key, id, p);
  };
  return {
    get edits() { return edits; },
    pending,
    change(text: string) {
      patchById(edits.x.id, { content: text, justSaved: false });
    },
    close() { const { x: _removed, ...rest } = edits; edits = rest; },
    open() { edits = { ...edits, x: createEdit("A", "ha") }; },
    start() {
      const request = deferred<{ content: string; hash: string } | null>();
      const submitted = edits.x;
      let calls = 0;
      const read = () => { calls++; return request.promise; };
      const done = reloadEdit({
        getCurrent: () => edits.x,
        read,
        patch: patchById,
        pending,
      });
      return { request, submitted, done, get calls() { return calls; } };
    },
  };
}

describe("reloadReadResult", () => {
  it("ignores cached success when refetch failed", () => {
    const cached = { ok: true as const, data: { content: "stale", hash: "hs", binary: false } };
    expect(reloadReadResult({ isError: true, status: "error", data: cached })).toBeNull();
    expect(reloadReadResult({ isError: false, status: "success", data: cached })).toEqual({
      content: "stale", hash: "hs",
    });
  });
});

describe("reloadEdit", () => {
  it("does not read when the caller cancels confirmation", async () => {
    const h = harness();
    const read = vi.fn();
    const confirmed = false;
    if (confirmed) {
      await reloadEdit({ getCurrent: () => h.edits.x, read, patch: () => {}, pending: h.pending });
    }
    expect(read).not.toHaveBeenCalled();
    expect(h.edits.x.content).toBe("A");
  });

  it("applies a successful reload to the same revision", async () => {
    const h = harness();
    h.change("B");
    const r = h.start();
    r.request.resolve({ content: "disk", hash: "hd" });
    await r.done;
    expect(h.edits.x).toMatchObject({
      content: "disk", savedContent: "disk", baseHash: "hd", stale: false, justSaved: false,
      reloading: false, reloadKept: false,
    });
    expect(h.pending.size).toBe(0);
  });

  it("ignores a late read after edit, undo, or save-settled revision change", async () => {
    const h = harness();
    h.change("B");
    const r = h.start();
    h.change("C");
    h.change("B");
    r.request.resolve({ content: "disk", hash: "hd" });
    await r.done;
    expect(h.edits.x.content).toBe("B");
    expect(h.edits.x.savedContent).toBe("A");
    expect(h.edits.x.reloadKept).toBe(true);
    expect(h.edits.x.reloading).toBe(false);
  });

  it("does not recreate a closed or reopened document", async () => {
    const h = harness();
    h.change("B");
    const r = h.start();
    h.close();
    h.open();
    h.change("C");
    const current = h.edits.x;
    r.request.resolve({ content: "disk", hash: "hd" });
    await r.done;
    expect(h.edits.x).toBe(current);
    expect(h.edits.x.content).toBe("C");
  });

  it("blocks duplicate reload and save-overlapping dispatch synchronously", async () => {
    const h = harness();
    h.change("B");
    const first = h.start();
    expect(first.calls).toBe(1);
    const second = h.start();
    expect(second.calls).toBe(0);
    first.request.resolve({ content: "disk", hash: "hd" });
    await first.done;
    await second.done;
  });

  it("preserves content and hash on read failure and clears the same pending flag", async () => {
    const h = harness();
    h.change("B");
    const r = h.start();
    r.request.reject(new Error("offline"));
    await r.done;
    expect(h.edits.x).toMatchObject({
      content: "B", savedContent: "A", baseHash: "ha", writeErr: true, reloading: false,
    });
    expect(h.pending.size).toBe(0);
  });

  it("keeps current content when the document is path-locked", async () => {
    const h = harness();
    h.change("B");
    h.edits.x.pathLocked = true;
    const r = h.start();
    r.request.resolve({ content: "disk", hash: "hd" });
    await r.done;
    expect(h.edits.x.content).toBe("B");
    expect(h.edits.x.reloadKept).toBe(true);
  });
});
