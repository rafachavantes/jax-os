import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("next-intl", () => ({ useTranslations: () => (key: string) => key }));
vi.mock("@/components/RelativeTime", () => ({ RelativeTime: ({ epochMs }: { epochMs: number }) => `RT(${epochMs})` }));

import { ExecutionRows, executionRunHref, formatExecutionDuration } from "./ExecutionRows";
import type { ExecutionRow } from "@/lib/executions";
import type { InspectionRow } from "@/server/collectors/execution-inspection";

const RUN = "0123456789ab";

function row(overrides: Partial<ExecutionRow> = {}): ExecutionRow {
  return {
    destination: "jax-os", runId: RUN, latestEventId: 2, startEventId: 1, finishEventId: 2,
    role: "builder", kind: "build", runtime: "codex", model: "gpt-5.6", caller: "claude",
    profile: "Default", phase: "M3a", target: "feat/x", firstDispatchedAt: "2026-11-18T10:00:00.000Z",
    attemptStartedAt: "2026-11-18T10:00:00.000Z", finishedAt: "2026-11-18T10:05:00.000Z",
    outcome: "success", terminal: true, contractStatus: "ok", durationMs: 300_000, warnings: [],
    ...overrides,
  };
}

function inspection(overrides: Partial<InspectionRow> = {}): InspectionRow {
  return {
    destination: "jax-os", runId: RUN, branch: "feat/x", branchSource: "manifest",
    report: { rel: `jax-os/.local/reports/${RUN}.md` }, tests: { rel: `jax-os/.local/reports/${RUN}.tests.txt` },
    observation: { status: "not-checked" }, refreshNeeded: false,
    current: { outcome: "success", terminal: true, finishEventId: 2, finishedAt: "2026-11-18T10:05:00.000Z", contractStatus: "ok", durationMs: 300_000, warnings: [] },
    warnings: [], ...overrides,
  };
}

function render(rows: ExecutionRow[], inspections: Record<string, InspectionRow | undefined> = {}, initialExpanded: string | null = null) {
  return renderToStaticMarkup(createElement(ExecutionRows, { rows, inspections, initialExpanded }));
}

describe("ExecutionRows", () => {
  it("renders destination, kind, run, outcome, executor and dispatch facts", () => {
    const html = render([row()]);
    expect(html).toContain("jax-os");
    expect(html).toContain(RUN);
    expect(html).toContain("kind.build");
    expect(html).toContain("outcome.success");
    expect(html).toContain("executor.label");
    expect(html).toContain("codex");
    expect(html).toContain("gpt-5.6");
    expect(html).toContain(`RT(${Date.parse("2026-11-18T10:00:00.000Z")})`);
    expect(html).toContain("5m 0s");
  });

  it("shows a worker observation only for an open, current attempt", () => {
    const open = row({ outcome: "open", terminal: false, finishEventId: null, finishedAt: null, contractStatus: null, durationMs: null });
    const observed = render([open], {
      [`jax-os/${RUN}`]: inspection({
        branch: null, branchSource: null, report: null, tests: null,
        observation: { status: "observed", pid: 100 },
        current: { outcome: "open", terminal: false, finishEventId: null, finishedAt: null, contractStatus: null, durationMs: null, warnings: [] },
      }),
    });
    expect(observed).toContain("worker.observed");

    const refresh = render([open], {
      [`jax-os/${RUN}`]: inspection({ observation: { status: "observed" }, refreshNeeded: true }),
    });
    expect(refresh).toContain("worker.refresh");
    expect(refresh).not.toContain("worker.observed");

    const terminal = render([row()], { [`jax-os/${RUN}`]: inspection({ observation: { status: "observed" } }) });
    expect(terminal).not.toContain("worker.observed");
  });

  it("overlays a terminal recorded after the durable page read", () => {
    const open = row({ outcome: "open", terminal: false, contractStatus: null, durationMs: null, finishedAt: null });
    const html = render([open], {
      [`jax-os/${RUN}`]: inspection({
        current: { outcome: "failure", terminal: true, finishEventId: 9, finishedAt: "2026-11-18T10:06:00.000Z", contractStatus: "ok", durationMs: 360_000, warnings: [] },
      }),
    });
    expect(html).toContain("outcome.failure");
    expect(html).toContain("6m 0s");
  });

  it("keeps a durable warning when inspection returns an empty warning list", () => {
    const html = render([row({ warnings: ["contract-invalid"] })], {
      [`jax-os/${RUN}`]: inspection({ warnings: [] }),
    }, `jax-os/${RUN}`);
    expect(html).toContain("contract-invalid");
  });

  it("combines same-attempt row, current and inspection warnings", () => {
    const open = row({
      outcome: "open", terminal: false, finishEventId: null, finishedAt: null,
      contractStatus: null, durationMs: null, warnings: ["orphan"],
    });
    const html = render([open], {
      [`jax-os/${RUN}`]: inspection({
        warnings: ["report-missing"],
        current: {
          outcome: "failure", terminal: true, finishEventId: 9, finishedAt: "2026-11-18T10:06:00.000Z",
          contractStatus: "invalid", durationMs: 360_000, warnings: ["contract-invalid"],
        },
      }),
    }, `jax-os/${RUN}`);
    expect(html).toContain("orphan");
    expect(html).toContain("contract-invalid");
    expect(html).toContain("report-missing");
  });

  it("keeps the frozen attempt's facts and warnings while a newer attempt asks for refresh", () => {
    const html = render([row({ warnings: ["period-unknown"] })], {
      [`jax-os/${RUN}`]: inspection({
        refreshNeeded: true,
        current: {
          outcome: "failure", terminal: true, finishEventId: 99, finishedAt: "2026-11-18T11:06:00.000Z",
          contractStatus: "ok", durationMs: 60_000, warnings: ["finish-conflict"],
        },
      }),
    }, `jax-os/${RUN}`);
    expect(html).toContain("worker.refresh");
    expect(html).toContain("outcome.success");
    expect(html).toContain("5m 0s");
    expect(html).not.toContain("outcome.failure");
    expect(html).toContain("period-unknown");
    expect(html).not.toContain("finish-conflict");
  });

  it("expands details with recorded context and retained links", () => {
    const html = render([row()], { [`jax-os/${RUN}`]: inspection() }, `jax-os/${RUN}`);
    expect(html).toContain("phase");
    expect(html).toContain("M3a");
    expect(html).toContain("target");
    expect(html).toContain("feat/x");
    expect(html).toContain("caller");
    expect(html).toContain("branch");
    expect(html).toContain("report");
    expect(html).toContain("tests");
    expect(html).toContain(encodeURIComponent(`jax-os/.local/reports/${RUN}.md`).replace(/%2F/g, "%2F").slice(0, 10));
    expect(html).toContain("[overflow-wrap:anywhere]");
  });

  it("labels a skipped artifact inspection instead of unavailable artifacts", () => {
    const html = render([row()], {
      [`jax-os/${RUN}`]: inspection({ report: null, tests: null, warnings: [], artifactsChecked: false }),
    }, `jax-os/${RUN}`);
    expect(html).toContain("worker.notChecked");
    expect(html).toContain("report");
    expect(html).toContain("tests");
    expect(html).not.toContain("reportUnavailable");
    expect(html).not.toContain("testsUnavailable");
    expect(html).not.toContain("root-unavailable");
  });

  it("marks unavailable artifacts instead of linking a guessed path", () => {
    const html = render([row()], { [`jax-os/${RUN}`]: inspection({ report: null, tests: null, warnings: ["report-missing", "tests-missing"] }) }, `jax-os/${RUN}`);
    expect(html).toContain("reportUnavailable");
    expect(html).toContain("testsUnavailable");
    expect(html).toContain("report-missing");
  });

  it("labels unknown outcomes and unknown durations honestly", () => {
    const html = render([row({ outcome: "unknown", durationMs: null })], { [`jax-os/${RUN}`]: inspection({ current: null }) });
    expect(html).toContain("outcome.unknown");
    expect(html).toContain("durationUnknown");
  });
});

describe("execution helpers", () => {
  it("formats durations without negative or invented values", () => {
    expect(formatExecutionDuration(4_000)).toBe("4s");
    expect(formatExecutionDuration(125_000)).toBe("2m 5s");
    expect(formatExecutionDuration(3_750_000)).toBe("1h 2m");
    expect(formatExecutionDuration(null)).toBeNull();
    expect(formatExecutionDuration(-5)).toBeNull();
  });

  it("builds a validated destination/run context URL with period all", () => {
    const href = executionRunHref("jax-os", RUN);
    expect(href).toBe(`/executions?project=jax-os&run=${RUN}&period=all`);
  });
});
