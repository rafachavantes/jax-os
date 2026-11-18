import { describe, expect, it } from "vitest";
import {
  buildExecutionRow, decodeExecutionCursor, encodeExecutionCursor, executionDurationMs,
  executionFilterIdentity, executionPeriodAnchor, executionStateMatches, parseExecutionSearch,
  serializeExecutionSearch, type ExecutionCursor, type ExecutionFacts,
} from "./executions";

const DEST = "jax-os";
const RUN = "0123456789ab";

function facts(overrides: Partial<ExecutionFacts> = {}): ExecutionFacts {
  return {
    project: DEST,
    runId: RUN,
    latestEventId: 2,
    start: { id: 1, role: "builder", ts: "2026-11-18T10:00:00.000Z", payload: { kind: "build", runtime: "codex", caller: "claude", target: "feat/x" } },
    finish: { id: 2, role: "builder", ts: "2026-11-18T10:05:00.000Z", payload: { contract_status: "ok", result: "success" } },
    firstStartTs: "2026-11-18T10:00:00.000Z",
    lastFinishTs: "2026-11-18T10:05:00.000Z",
    ...overrides,
  };
}

describe("parseExecutionSearch", () => {
  it("defaults to all projects / all kinds / all states / 30d / 20 rows", () => {
    expect(parseExecutionSearch(new URLSearchParams())).toEqual({
      ok: true,
      filters: { project: null, kind: "all", state: "all", period: "30d", run: null, limit: 20 },
    });
  });

  it("accepts every bounded legal filter", () => {
    const parsed = parseExecutionSearch(new URLSearchParams("project=jax-os&kind=review&state=approve&period=7d&limit=20"));
    expect(parsed.ok && parsed.filters).toMatchObject({ project: "jax-os", kind: "review", state: "approve", period: "7d", limit: 20 });
  });

  it("refuses malformed values instead of widening the scope", () => {
    for (const qs of [
      "project=", "project=%00bad", `project=${"a".repeat(65)}`, "kind=nope", "state=nope",
      "period=1d", "limit=0", "limit=21", "limit=x", `run=${"A".repeat(12)}`, "run=0123456789ab",
      "run=0123456789ab&period=7d", "run=0123456789ab&project=jax-os&period=7d",
    ]) {
      expect(parseExecutionSearch(new URLSearchParams(qs)).ok, qs).toBe(false);
    }
  });

  it("refuses duplicate and unknown control parameters", () => {
    expect(parseExecutionSearch(new URLSearchParams("kind=build&kind=review")).ok).toBe(false);
    expect(parseExecutionSearch(new URLSearchParams("offset=20")).ok).toBe(false);
    expect(parseExecutionSearch(new URLSearchParams("sort=asc")).ok).toBe(false);
    expect(parseExecutionSearch(new URLSearchParams("cursor=x&project=jax-os")).ok).toBe(true);
  });

  it("requires destination + run together and forces period all for run context", () => {
    const parsed = parseExecutionSearch(new URLSearchParams(`project=jax-os&run=${RUN}`));
    expect(parsed.ok && parsed.filters).toMatchObject({ period: "all", run: { project: "jax-os", runId: RUN } });
  });

  it("round-trips through the URL serializer", () => {
    const params = new URLSearchParams("project=jax-os&kind=build&state=failure&period=7d");
    expect(serializeExecutionSearch(params, { project: "jax-os", kind: "build", state: "failure", period: "7d", run: null, limit: 20 }).toString())
      .toBe("project=jax-os&kind=build&state=failure&period=7d");
  });
});

describe("execution filter identity and cursor", () => {
  const filters = { project: null, kind: "all" as const, state: "all" as const, period: "30d" as const, run: null, limit: 20 };

  it("changes with each scope field", () => {
    const base = executionFilterIdentity(filters);
    expect(executionFilterIdentity({ ...filters, project: DEST })).not.toBe(base);
    expect(executionFilterIdentity({ ...filters, kind: "build" })).not.toBe(base);
    expect(executionFilterIdentity({ ...filters, state: "open" })).not.toBe(base);
    expect(executionFilterIdentity({ ...filters, period: "all" })).not.toBe(base);
    expect(executionFilterIdentity({ ...filters, run: { project: DEST, runId: RUN } })).not.toBe(base);
  });

  it("round-trips an opaque bounded cursor and rejects malformed shapes", () => {
    const cursor: ExecutionCursor = {
      v: 1, fence: 99, lastId: 42, project: DEST, runId: RUN,
      cutoff: "2026-10-19T00:00:00.000Z", identity: executionFilterIdentity(filters),
    };
    expect(decodeExecutionCursor(encodeExecutionCursor(cursor))).toEqual(cursor);
    expect(decodeExecutionCursor("")).toBeNull();
    expect(decodeExecutionCursor("a".repeat(1025))).toBeNull();
    expect(decodeExecutionCursor("!!!")).toBeNull();
    expect(decodeExecutionCursor(Buffer.from("{}").toString("base64url"))).toBeNull();
    expect(decodeExecutionCursor(Buffer.from(JSON.stringify({ ...cursor, v: 2 })).toString("base64url"))).toBeNull();
    expect(decodeExecutionCursor(Buffer.from(JSON.stringify({ ...cursor, lastId: -1 })).toString("base64url"))).toBeNull();
    expect(decodeExecutionCursor(Buffer.from(JSON.stringify({ ...cursor, runId: "r1" })).toString("base64url"))).toBeNull();
    expect(decodeExecutionCursor(Buffer.from(JSON.stringify({ ...cursor, cutoff: "nope" })).toString("base64url"))).toBeNull();
  });
});

describe("buildExecutionRow", () => {
  it("projects a terminal builder success with recorded metadata only", () => {
    const row = buildExecutionRow(facts())!;
    expect(row).toMatchObject({
      destination: DEST, runId: RUN, role: "builder", kind: "build", runtime: "codex",
      caller: "claude", target: "feat/x", outcome: "success", terminal: true,
      contractStatus: "ok", durationMs: 300_000,
    });
    expect(row.warnings).toEqual([]);
  });

  it("a later start reopens the same destination/run (compact-card difference pinned)", () => {
    const row = buildExecutionRow(facts({
      latestEventId: 3,
      start: { id: 3, role: "builder", ts: "2026-11-18T11:00:00.000Z", payload: { kind: "build", runtime: "codex" } },
    }))!;
    expect(row.outcome).toBe("open");
    expect(row.terminal).toBe(false);
    expect(row.contractStatus).toBeNull();
    expect(row.finishEventId).toBe(2); // the older terminal fact is still visible, not the state
    expect(row.durationMs).toBeNull();
  });

  it("keeps an orphan finish's actual state and warns", () => {
    const row = buildExecutionRow(facts({ start: null, firstStartTs: null }))!;
    expect(row.outcome).toBe("success");
    expect(row.role).toBe("builder");
    expect(row.warnings).toContain("orphan");
    expect(row.durationMs).toBeNull();
    expect(row.warnings).toContain("duration-unknown");
  });

  it("accepts a jaxos finish (grouping already supplies the latest finish)", () => {
    expect(buildExecutionRow(facts())!.outcome).toBe("success");
  });

  it("cancelled/interrupted contract status wins over any recorded result", () => {
    const cancelled = buildExecutionRow(facts({ finish: { id: 2, role: "builder", ts: "2026-11-18T10:05:00.000Z", payload: { contract_status: "cancelled", result: "success" } } }))!;
    expect(cancelled.outcome).toBe("cancelled");
    expect(cancelled.contractStatus).toBe("cancelled");
  });

  it("keeps a builder failure visible even with invalid contract metadata, with a warning", () => {
    const row = buildExecutionRow(facts({ finish: { id: 2, role: "builder", ts: "2026-11-18T10:05:00.000Z", payload: { contract_status: "bogus", result: "failure" } } }))!;
    expect(row.outcome).toBe("failure");
    expect(row.warnings).toContain("contract-invalid");
  });

  it("a reviewer verdict counts only on a valid contract, otherwise unknown", () => {
    const ok = buildExecutionRow(facts({
      start: { id: 1, role: "reviewer", ts: "2026-11-18T10:00:00.000Z", payload: { kind: "diff" } },
      finish: { id: 2, role: "reviewer", ts: "2026-11-18T10:05:00.000Z", payload: { contract_status: "ok", verdict: "approve-with-changes" } },
    }))!;
    expect(ok.outcome).toBe("approve-with-changes");
    const invalid = buildExecutionRow(facts({
      start: { id: 1, role: "reviewer", ts: "2026-11-18T10:00:00.000Z", payload: { kind: "diff" } },
      finish: { id: 2, role: "reviewer", ts: "2026-11-18T10:05:00.000Z", payload: { contract_status: "missing", verdict: "approve" } },
    }))!;
    expect(invalid.outcome).toBe("unknown");
    expect(invalid.warnings).toContain("contract-missing");
    expect(invalid.warnings).toContain("verdict-unknown");
  });

  it("excludes a known merge-only run and illegal identities", () => {
    expect(buildExecutionRow(facts({ start: { id: 1, role: "builder", ts: "2026-11-18T10:00:00.000Z", payload: { kind: "merge" } } }))).toBeNull();
    expect(buildExecutionRow(facts({ project: "" }))).toBeNull();
    expect(buildExecutionRow(facts({ runId: "r1" }))).toBeNull();
  });

  it("warns on role and kind conflicts while keeping the row", () => {
    const row = buildExecutionRow(facts({
      start: { id: 1, role: "reviewer", ts: "2026-11-18T10:00:00.000Z", payload: { kind: "build" } },
      finish: { id: 2, role: "builder", ts: "2026-11-18T10:05:00.000Z", payload: { contract_status: "ok", result: "success" } },
    }))!;
    expect(row.warnings).toContain("role-conflict");
    expect(row.warnings).toContain("kind-role-conflict");
  });

  it("corrupt payloads make fields unknown without removing the event", () => {
    const row = buildExecutionRow(facts({
      start: { id: 1, role: "builder", ts: "2026-11-18T10:00:00.000Z", payload: null },
      finish: { id: 2, role: "builder", ts: "2026-11-18T10:05:00.000Z", payload: null },
    }))!;
    expect(row.kind).toBeNull();
    expect(row.outcome).toBe("unknown");
    expect(row.warnings).toContain("result-unknown");
  });

  it("invalid attempt timestamps are unknown, never borrowed from an earlier attempt", () => {
    const row = buildExecutionRow(facts({
      start: { id: 3, role: "builder", ts: "not-a-date", payload: { kind: "build" } },
      finish: null, latestEventId: 3,
      firstStartTs: "2026-11-18T10:00:00.000Z",
    }))!;
    expect(row.attemptStartedAt).toBeNull();
    expect(row.warnings).toContain("attempt-timestamp-unknown");
    expect(row.firstDispatchedAt).toBe("2026-11-18T10:00:00.000Z");
  });
});

describe("duration and period provenance", () => {
  it("uses finite ordered timestamps only", () => {
    expect(executionDurationMs("2026-11-18T10:00:00.000Z", "2026-11-18T10:00:05.000Z")).toBe(5_000);
    expect(executionDurationMs("2026-11-18T10:00:05.000Z", "2026-11-18T10:00:00.000Z")).toBeNull();
    expect(executionDurationMs(null, "2026-11-18T10:00:00.000Z")).toBeNull();
    expect(executionDurationMs("nope", "2026-11-18T10:00:00.000Z")).toBeNull();
  });

  it("anchors a period on the first finite start, then the latest finite finish", () => {
    expect(executionPeriodAnchor("2026-11-18T10:00:00.000Z", null)).toBe("2026-11-18T10:00:00.000Z");
    expect(executionPeriodAnchor("nope", "2026-11-18T11:00:00.000Z")).toBe("2026-11-18T11:00:00.000Z");
    expect(executionPeriodAnchor(null, null)).toBeNull();
  });

  it("matches durable lifecycle state filters, not process observation", () => {
    expect(executionStateMatches("open", { terminal: false, outcome: "open" })).toBe(true);
    expect(executionStateMatches("finished", { terminal: true, outcome: "failure" })).toBe(true);
    expect(executionStateMatches("success", { terminal: true, outcome: "success" })).toBe(true);
    expect(executionStateMatches("success", { terminal: false, outcome: "open" })).toBe(false);
    expect(executionStateMatches("approve", { terminal: true, outcome: "approve" })).toBe(true);
  });
});
