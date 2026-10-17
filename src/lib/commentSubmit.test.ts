import { describe, expect, it, vi } from "vitest";
import { createCommentSender, draftMatches, reconcileComment, type CommentDraft } from "./commentSubmit";
import type { IssueComment } from "@/server/collectors/linear";

const draft = (over: Partial<CommentDraft> = {}): CommentDraft => ({
  issueId: "i1",
  body: "hello",
  revision: 3,
  ...over,
});

function deferred<T>() {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => { resolve = r; });
  return { promise, resolve };
}

const ok = { ok: true } as const;
const failed = { ok: false, code: "mutation-rejected", effect: "not-applied", audit: "recorded", error: "nope" } as const;
const appliedUnrecorded = { ok: false, code: "audit-finalization-failed", effect: "applied", audit: "pending", error: "recording failed" } as const;
const unconfirmed = { ok: false, code: "mutation-unconfirmed", effect: "unconfirmed", audit: "recorded", error: "timed out" } as const;

describe("draftMatches", () => {
  it("requires the same issue, body and revision", () => {
    expect(draftMatches(draft(), draft())).toBe(true);
    expect(draftMatches(draft(), draft({ body: "newer" }))).toBe(false);
    expect(draftMatches(draft(), draft({ revision: 4 }))).toBe(false);
    expect(draftMatches(draft(), draft({ issueId: "i2" }))).toBe(false);
  });
});

describe("createCommentSender", () => {
  it("sends once and reports whether the current draft is still the sent draft", async () => {
    const post = vi.fn().mockResolvedValue(ok);
    const sender = createCommentSender(post);
    const result = await sender.send({ draft: draft(), getCurrent: () => draft() });
    expect(result).toEqual({ outcome: ok, current: true });
    expect(post).toHaveBeenCalledTimes(1);
    expect(post).toHaveBeenCalledWith("i1", "hello");
    expect(sender.isPending()).toBe(false);
  });

  it("suppresses a double submit — duplicate clicks resolve to null, one request", async () => {
    const gate = deferred<typeof ok>();
    const post = vi.fn(() => gate.promise);
    const sender = createCommentSender(post);
    const first = sender.send({ draft: draft(), getCurrent: () => draft() });
    const second = await sender.send({ draft: draft(), getCurrent: () => draft() });
    expect(second).toBeNull();
    expect(post).toHaveBeenCalledTimes(1);
    expect(sender.isPending()).toBe(true);
    gate.resolve(ok);
    await expect(first).resolves.toEqual({ outcome: ok, current: true });
    expect(sender.isPending()).toBe(false);
    await expect(sender.send({ draft: draft(), getCurrent: () => draft() })).resolves.toEqual({ outcome: ok, current: true });
    expect(post).toHaveBeenCalledTimes(2);
  });

  it("keeps a newer draft typed during the send (current=false so the UI never clears it)", async () => {
    const gate = deferred<typeof ok>();
    const sender = createCommentSender(() => gate.promise);
    const send = sender.send({
      draft: draft(),
      getCurrent: () => draft({ body: "newer text typed meanwhile", revision: 4 }),
    });
    gate.resolve(ok);
    const result = await send;
    expect(result?.outcome).toEqual(ok);
    expect(result?.current).toBe(false);
  });

  it("reads the live draft ref (production getCurrent) — typing/undo during the send is still detected", async () => {
    // the overlay keeps body in a ref next to the state; getCurrent reads that
    // ref, so it must reflect edits made after send() started.
    const live = { issueId: "i1", body: "hello", revision: 3 };
    const gate = deferred<typeof ok>();
    const sender = createCommentSender(() => gate.promise);
    const send = sender.send({
      draft: draft(),
      getCurrent: () => ({ ...live }),
    });
    // user types "lost word" then undoes to "hello" — different revision
    live.body = "hello lost word";
    live.revision = 4;
    live.body = "hello";
    live.revision = 5;
    gate.resolve(ok);
    expect((await send)?.current).toBe(false);
    // no edits at all: a confirmed completion may clear exactly the sent text
    const noEdit = sender.send({ draft: draft(), getCurrent: () => ({ ...live, revision: 3 }) });
    expect((await noEdit)?.current).toBe(true);
  });

  it("marks a response stale when the issue changed mid-send", async () => {
    const gate = deferred<typeof ok>();
    const sender = createCommentSender(() => gate.promise);
    const send = sender.send({
      draft: draft(),
      getCurrent: () => draft({ issueId: "i2", body: "", revision: 0 }),
    });
    gate.resolve(ok);
    const result = await send;
    expect(result?.current).toBe(false);
  });

  it("propagates refusal, unconfirmed and applied-but-unrecorded outcomes", async () => {
    const sender = createCommentSender(async () => failed);
    expect((await sender.send({ draft: draft(), getCurrent: () => draft() }))?.outcome).toEqual(failed);
    const sender2 = createCommentSender(async () => unconfirmed);
    expect((await sender2.send({ draft: draft(), getCurrent: () => draft() }))?.outcome).toEqual(unconfirmed);
    const sender3 = createCommentSender(async () => appliedUnrecorded);
    expect((await sender3.send({ draft: draft(), getCurrent: () => draft() }))?.outcome).toEqual(appliedUnrecorded);
  });

  it("rejects an empty draft and stays idle", async () => {
    const post = vi.fn();
    const sender = createCommentSender(post);
    await expect(sender.send({ draft: draft({ body: "   " }), getCurrent: () => draft() })).resolves.toBeNull();
    expect(post).not.toHaveBeenCalled();
  });
});

describe("reconcileComment", () => {
  const c = (id: string): IssueComment => ({ id, author: "x", body: "b", createdAt: "2026-01-01T00:00:00Z" });

  it("keeps the list unchanged on success, unconfirmed, and applied-unrecorded", () => {
    const list = [c("temp-1")];
    expect(reconcileComment(list, "temp-1", null)).toBe(list);
    expect(reconcileComment(list, "temp-1", { kind: "unconfirmed" })).toBe(list);
    expect(reconcileComment(list, "temp-1", { kind: "applied-unrecorded" })).toBe(list);
  });

  it("removes only the temp entry on refused, preserving a later real comment and any draft", () => {
    const list = [c("temp-1"), c("real-2")];
    const result = reconcileComment(list, "temp-1", { kind: "refused", error: "nope" });
    expect(result).toEqual([c("real-2")]);
    // a draft is separate component state, never passed to this function —
    // proven by this call's signature taking no draft argument at all.
  });
});