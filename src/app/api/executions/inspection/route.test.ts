import { beforeEach, describe, expect, it, vi } from "vitest";

const harness = vi.hoisted(() => ({ getDb: vi.fn(), inspectExecutionRows: vi.fn() }));
vi.mock("@/server/db", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/server/db")>();
  return { ...actual, getDb: () => harness.getDb() };
});
vi.mock("@/server/collectors/execution-inspection", () => ({
  inspectExecutionRows: harness.inspectExecutionRows,
}));

import { parseInspectionSearch, GET } from "./route";

const RUN = "0123456789ab";

function req(qs: string) {
  return new Request(`http://127.0.0.1/api/executions/inspection${qs}`);
}

function selector(destination = "jax-os", runId = RUN, start = "12"): string {
  return encodeURIComponent(`${encodeURIComponent(destination)}|${encodeURIComponent(runId)}|${start}`);
}

describe("parseInspectionSearch", () => {
  it("parses bounded destination/run/start selectors", () => {
    const parsed = parseInspectionSearch(`?r=${selector()}&r=${selector("other", "ffffffffffff", "")}`);
    expect(parsed.ok && parsed.selectors).toEqual([
      { destination: "jax-os", runId: RUN, startEventId: 12 },
      { destination: "other", runId: "ffffffffffff", startEventId: null },
    ]);
  });

  it("refuses arbitrary paths, commands, unknown keys, duplicate fields and too many selectors", () => {
    for (const qs of [
      "?path=/etc/passwd",
      "?r=not-three-parts",
      "?r=a|b|c|d",
      `?r=${encodeURIComponent("../etc|${RUN}|1")}`,
      `?r=${encodeURIComponent(`jax-os|${RUN}|0`)}`,
      `?r=${encodeURIComponent(`jax-os|${RUN}|nope`)}`,
      `?r=${encodeURIComponent(`jax-os|${RUN}|1`)}&limit=30`,
    ]) {
      expect(parseInspectionSearch(qs).ok, qs).toBe(false);
    }
    const many = Array.from({ length: 21 }, () => `r=${selector()}`).join("&");
    expect(parseInspectionSearch(`?${many}`).ok).toBe(false);
  });
});

describe("GET /api/executions/inspection", () => {
  beforeEach(() => {
    harness.getDb.mockReset();
    harness.getDb.mockReturnValue({});
    harness.inspectExecutionRows.mockReset();
  });

  it("returns resolved rows through the existing envelope contract", async () => {
    harness.inspectExecutionRows.mockResolvedValue([{ destination: "jax-os", runId: RUN }]);
    const body = await (await GET(req(`?r=${selector()}`))).json();
    expect(body).toEqual({ ok: true, data: [{ destination: "jax-os", runId: RUN }] });
    expect(harness.inspectExecutionRows).toHaveBeenCalledWith({}, [{ destination: "jax-os", runId: RUN, startEventId: 12 }]);
  });

  it("turns a source failure into {ok:false} without a 5xx", async () => {
    harness.inspectExecutionRows.mockRejectedValue(new Error("inspection failed"));
    const res = await GET(req(`?r=${selector()}`));
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: false, error: "inspection failed" });
  });

  it("rejects an oversized query before touching the db", async () => {
    const res = await GET(req(`?r=${"x".repeat(9 * 1024)}`));
    expect(await res.json()).toEqual({ ok: false, error: "request too large" });
    expect(harness.getDb).not.toHaveBeenCalled();
  });
});
