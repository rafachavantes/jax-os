import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Envelope } from "@/lib/api";
import type { KanbanIssue } from "@/server/collectors/linear";
import {
  nextLabelSet, parseCreateResponse, postCreateIssue, PropertiesSidebar, reconcileWrite, releaseLabelLock, reserveAndWrite, reserveField, runWriteField, sameLabelSet, write, writeGuidance,
} from "./PropertiesSidebar";

const harness = vi.hoisted(() => ({
  members: { data: undefined as Envelope<unknown[]> | undefined },
  labels: { data: undefined as Envelope<unknown[]> | undefined },
}));

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en-US",
}));
vi.mock("@tanstack/react-query", () => ({
  useQuery: ({ queryKey }: { queryKey: string[] }) =>
    queryKey.includes("members") ? harness.members : harness.labels,
  useQueryClient: () => ({ invalidateQueries: vi.fn() }),
}));

const issue: KanbanIssue = {
  id: "i1",
  identifier: "MOA-1",
  title: "t",
  priority: 2,
  url: "https://x",
  stateId: "s1",
  createdAt: "2026-01-01T00:00:00Z",
  updatedAt: "2026-01-01T00:00:00Z",
  labels: [{ id: "l1", name: "bug", color: "#f00" }],
};

function jsonResponse(body: unknown, ok = true) {
  return { ok, json: async () => body };
}

describe("write (shared mutation POST)", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("returns ok:true for a successful envelope", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ ok: true })));
    expect(await write("/api/x", { a: 1 })).toEqual({ ok: true });
  });

  it("preserves effect/audit metadata from a MutationFailure envelope", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({ ok: false, code: "audit-finalization-failed", effect: "applied", audit: "pending", error: "recording failed" }),
      ),
    );
    expect(await write("/api/x", {})).toEqual({
      ok: false,
      code: "audit-finalization-failed",
      effect: "applied",
      audit: "pending",
      error: "recording failed",
    });
  });

  it("keeps a bare route rejection as refused", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ ok: false, error: "invalid payload" })));
    expect(await write("/api/x", {})).toEqual({ ok: false, error: "invalid payload" });
  });

  it("classifies transport and JSON failures as unconfirmed, never success", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("network down")));
    expect(await write("/api/x", {})).toEqual({
      ok: false,
      code: "mutation-unconfirmed",
      effect: "unconfirmed",
      audit: "unavailable",
      error: "unavailable",
    });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => { throw new Error("bad json"); } }));
    const out = await write("/api/x", {});
    expect(out.ok).toBe(false);
    if (!out.ok) expect(writeGuidance(out)).toEqual({ kind: "unconfirmed" });
  });

  it("treats syntactically valid but malformed responses as unconfirmed, not refused", async () => {
    const malformed = [null, [1, 2], { unknown: true }, { ok: "yes" }, { ok: false }, 42, "nope"];
    for (const body of malformed) {
      vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(body)));
      const out = await write("/api/x", {});
      expect(out.ok).toBe(false);
      if (!out.ok) expect(writeGuidance(out)).toEqual({ kind: "unconfirmed" });
    }
  });

  it("treats invalid ok:false metadata as unconfirmed, not a validated refusal", async () => {
    const malformed = [
      { ok: false, error: "x", code: "bogus-code", effect: "applied", audit: "recorded" },
      { ok: false, error: "x", code: "mutation-rejected", effect: "maybe", audit: "recorded" },
      { ok: false, error: "x", code: "mutation-rejected", effect: "not-applied", audit: 7 },
      { ok: false, error: 42, code: "mutation-rejected", effect: "not-applied", audit: "recorded" },
    ];
    for (const body of malformed) {
      vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(body)));
      const out = await write("/api/x", {});
      expect(out.ok).toBe(false);
      if (!out.ok) expect(writeGuidance(out)).toEqual({ kind: "unconfirmed" });
    }
  });
});

describe("per-field write guards (production submission path)", () => {
  it("admits the first same-field call and rejects a duplicate before any request exists", async () => {
    const inflight = new Set<string>();
    const post = vi.fn().mockResolvedValue({ ok: true });
    const first = reserveAndWrite(inflight, "status", post);
    const duplicate = reserveAndWrite(inflight, "status", post);
    await expect(duplicate).resolves.toBeNull();
    await first;
    expect(post).toHaveBeenCalledTimes(1);
    expect(inflight.size).toBe(0);
    // same field is usable again once settled
    await reserveAndWrite(inflight, "status", post);
    expect(post).toHaveBeenCalledTimes(2);
  });

  it("keeps independent fields usable while one is pending", async () => {
    const inflight = new Set<string>();
    let resolve!: (v: { ok: true }) => void;
    const gate = new Promise<{ ok: true }>((r) => { resolve = r; });
    const pending = reserveAndWrite(inflight, "status", () => gate);
    await expect(reserveAndWrite(inflight, "priority", async () => ({ ok: true }))).resolves.toEqual({ ok: true });
    resolve({ ok: true });
    await expect(pending).resolves.toEqual({ ok: true });
    expect(inflight.size).toBe(0);
  });

  it("protects one request for two same-turn label toggles and derives sets from current labels", () => {
    expect(nextLabelSet(["l1"], "l2")).toEqual(["l1", "l2"]);
    expect(nextLabelSet(["l1", "l2"], "l1")).toEqual(["l2"]);
    const inflight = new Set<string>();
    expect(reserveField(inflight, "labels")).toBe(true);
    expect(reserveField(inflight, "labels")).toBe(false);
    inflight.delete("labels");
    expect(reserveField(inflight, "labels")).toBe(true);
  });

  it("releases the label reservation only on confirmed outcomes; uncertain results reconverge", () => {
    expect(releaseLabelLock({ ok: true })).toBe(false); // success alone keeps the lock until detail converges
    expect(releaseLabelLock({ ok: false, error: "invalid payload" })).toBe(true); // confirmed refusal
    expect(
      releaseLabelLock({ ok: false, code: "mutation-unconfirmed", effect: "unconfirmed", audit: "recorded", error: "t" }),
    ).toBe(false);
    expect(
      releaseLabelLock({ ok: false, code: "audit-finalization-failed", effect: "applied", audit: "pending", error: "t" }),
    ).toBe(false);
  });

  it("production label-take sequence: success keeps the lock until detail converges; next toggle sends from the confirmed set", async () => {
    const inflight = new Set<string>();
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ ok: true }));
    vi.stubGlobal("fetch", fetchMock);
    // toggle #1, exactly as toggleLabel does: synchronous reservation, then derive
    expect(reserveField(inflight, "labels")).toBe(true);
    const sent = nextLabelSet(["l1"], "l2");
    const outcome = await write("/api/kanban/update", { issueId: "i1", patch: { labelIds: sent } });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    // successful POST is NOT a release: the lock holds until a refetch converges
    expect(releaseLabelLock(outcome)).toBe(false);
    // stale detail still shows the old set: the convergence effect keeps the lock
    expect(sameLabelSet(sent, ["l1"])).toBe(false);
    // second toggle must not send while the lock holds
    expect(reserveField(inflight, "labels")).toBe(false);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    // matching detail arrives: the confirmed set converges and releases the lock
    expect(sameLabelSet(sent, ["l2", "l1"])).toBe(true);
    inflight.delete("labels");
    // next toggle derives from the confirmed set, not a stale copy
    expect(reserveField(inflight, "labels")).toBe(true);
    expect(nextLabelSet(sent, "l3")).toEqual(["l1", "l2", "l3"]);
  });

  it("production refusal path: a confirmed refusal releases the lock immediately (nothing changed)", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ ok: false, error: "invalid payload" })));
    const inflight = new Set<string>();
    expect(reserveField(inflight, "labels")).toBe(true);
    const sent = nextLabelSet([], "l1");
    const outcome = await write("/api/kanban/update", { issueId: "i1", patch: { labelIds: sent } });
    expect(releaseLabelLock(outcome)).toBe(true);
    inflight.delete("labels");
    expect(reserveField(inflight, "labels")).toBe(true);
  });
});

describe("writeGuidance", () => {
  it("maps refusal, unconfirmed and applied-unrecorded distinctly", () => {
    expect(writeGuidance({ ok: true })).toBeNull();
    expect(writeGuidance({ ok: false, error: "invalid payload" })).toEqual({ kind: "refused", error: "invalid payload" });
    expect(
      writeGuidance({ ok: false, code: "mutation-rejected", effect: "not-applied", audit: "recorded", error: "nope" }),
    ).toEqual({ kind: "refused", error: "nope" });
    expect(
      writeGuidance({ ok: false, code: "mutation-unconfirmed", effect: "unconfirmed", audit: "recorded", error: "t" }),
    ).toEqual({ kind: "unconfirmed" });
    expect(
      writeGuidance({ ok: false, code: "audit-finalization-failed", effect: "applied", audit: "pending", error: "t" }),
    ).toEqual({ kind: "applied-unrecorded" });
  });
});

describe("reconcileWrite", () => {
  it("keeps applied on success (null outcome)", () => {
    expect(reconcileWrite("snap", "applied", null)).toBe("applied");
  });
  it("returns the snapshot on refused", () => {
    expect(reconcileWrite("snap", "applied", { kind: "refused", error: "nope" })).toBe("snap");
  });
  it("keeps applied on unconfirmed and applied-unrecorded", () => {
    expect(reconcileWrite("snap", "applied", { kind: "unconfirmed" })).toBe("applied");
    expect(reconcileWrite("snap", "applied", { kind: "applied-unrecorded" })).toBe("applied");
  });
});

describe("parseCreateResponse", () => {
  it("maps a successful create response through unchanged", () => {
    expect(parseCreateResponse({ ok: true, id: "i9", identifier: "MOA-9" })).toEqual({ ok: true, id: "i9", identifier: "MOA-9" });
  });
  it("maps a MutationFailure shape to its typed failure", () => {
    const failure = { ok: false, code: "mutation-rejected", effect: "not-applied", audit: "recorded", error: "nope" };
    expect(parseCreateResponse(failure)).toEqual(failure);
  });
  it("maps the legacy {ok:false,error} shape and malformed responses to unconfirmed/refused correctly", () => {
    expect(parseCreateResponse({ ok: false, error: "invalid payload" })).toEqual({ ok: false, error: "invalid payload" });
    expect(parseCreateResponse(null)).toEqual({ ok: false, code: "mutation-unconfirmed", effect: "unconfirmed", audit: "unavailable", error: "unavailable" });
    expect(parseCreateResponse({ ok: true })).toEqual({ ok: false, code: "mutation-unconfirmed", effect: "unconfirmed", audit: "unavailable", error: "unavailable" });
  });
});

describe("postCreateIssue", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });
  const UNCONFIRMED_CREATE = { ok: false, code: "mutation-unconfirmed", effect: "unconfirmed", audit: "unavailable", error: "unavailable" };

  it("POSTs the exact endpoint, method, headers and serialized body, returning the parsed success shape (F7)", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ json: async () => ({ ok: true, id: "i9", identifier: "MOA-9" }) });
    vi.stubGlobal("fetch", fetchMock);
    const body = { teamId: "t1", title: "New issue" };
    await expect(postCreateIssue(body)).resolves.toEqual({ ok: true, id: "i9", identifier: "MOA-9" });
    expect(fetchMock).toHaveBeenCalledWith("/api/kanban/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  });

  it("maps a transport failure (rejected fetch) to the unconfirmed outcome", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("network down")));
    await expect(postCreateIssue({ teamId: "t1", title: "x" })).resolves.toEqual(UNCONFIRMED_CREATE);
  });

  it("maps malformed JSON (a response whose .json() throws) to the unconfirmed outcome", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ json: async () => { throw new Error("bad json"); } }));
    await expect(postCreateIssue({ teamId: "t1", title: "x" })).resolves.toEqual(UNCONFIRMED_CREATE);
  });
});

describe("runWriteField", () => {
  it("reserves, posts, invalidates both caches, and returns the outcome", async () => {
    const qc = { invalidateQueries: vi.fn() };
    const inflight = new Set<string>();
    const outcome = await runWriteField(inflight, "title", async () => ({ ok: true }), qc as never, "i1");
    expect(outcome).toEqual({ ok: true });
    expect(qc.invalidateQueries).toHaveBeenCalledWith({ queryKey: ["kanban", "board"] });
    expect(qc.invalidateQueries).toHaveBeenCalledWith({ queryKey: ["kanban", "issue", "i1"] });
    expect(inflight.size).toBe(0);
  });
  it("returns null and skips the network for a duplicate same-field call", async () => {
    const qc = { invalidateQueries: vi.fn() };
    const inflight = new Set<string>();
    const post = vi.fn().mockResolvedValue({ ok: true });
    const [first, second] = await Promise.all([
      runWriteField(inflight, "title", post, qc as never, "i1"),
      runWriteField(inflight, "title", post, qc as never, "i1"),
    ]);
    expect([first, second].filter((r) => r === null)).toHaveLength(1);
    expect(post).toHaveBeenCalledTimes(1);
  });
});

describe("PropertiesSidebar", () => {
  it("renders labeled selects and no fabricated values before data", () => {
    harness.members = { data: { ok: true, data: [{ id: "m1", displayName: "Rafa" }] } };
    harness.labels = { data: { ok: true, data: [{ id: "l1", name: "bug", color: "#f00" }] } };
    const html = renderToStaticMarkup(
      createElement(PropertiesSidebar, { issue, states: [{ id: "s1", name: "Done", color: "#0f0", position: 1, type: "completed" }], teamId: "t1" }),
    );
    expect(html).toContain('aria-label="propStatus"');
    expect(html).toContain('aria-label="propPriority"');
    expect(html).toContain('aria-label="propAssignee"');
    expect(html).toContain("propLabels");
    expect(html).toContain('value="s1"');
  });
});