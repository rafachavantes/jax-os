import { describe, expect, it, vi } from "vitest";
import type { WriteGuidance, WriteOutcome } from "@/components/kanban/PropertiesSidebar";
import { moveIssue } from "./KanbanColumn";

describe("moveIssue (production drop handler)", () => {
  const baseOpts = (overrides: Partial<Parameters<typeof moveIssue>[0]> = {}) => ({
    issueId: "i1",
    issue: { stateId: "s1" },
    targetStateId: "s2",
    reservation: new Set<string>(),
    post: vi.fn().mockResolvedValue({ ok: true }) as () => Promise<WriteOutcome>,
    onPending: vi.fn(),
    onGuidance: vi.fn(),
    onInvalidate: vi.fn(),
    ...overrides,
  });

  it("two synchronous drops of the same issue produce a single POST", async () => {
    const opts = baseOpts();
    const first = moveIssue(opts);
    const second = moveIssue(opts); // same-turn second drop while the first is in flight
    await Promise.all([first, second]);
    expect(opts.post).toHaveBeenCalledTimes(1);
    expect(opts.onGuidance).toHaveBeenNthCalledWith(1, null); // pending resets guidance
    expect(opts.onGuidance).toHaveBeenNthCalledWith(2, null); // ok outcome has no guidance
    expect(opts.onPending).toHaveBeenNthCalledWith(1, true);
    expect(opts.onPending).toHaveBeenNthCalledWith(2, false);
    expect(opts.onInvalidate).toHaveBeenCalledTimes(1);
    expect(opts.reservation.size).toBe(0);
  });

  it("a same-column drop is a silent no-op with no POST and no invalidation", async () => {
    const opts = baseOpts({ targetStateId: "s1" });
    await moveIssue(opts);
    expect(opts.post).not.toHaveBeenCalled();
    expect(opts.onPending).not.toHaveBeenCalled();
    expect(opts.onGuidance).not.toHaveBeenCalled();
    expect(opts.onInvalidate).not.toHaveBeenCalled();
  });

  it("classifies refused, unconfirmed and applied-unrecorded feedback distinctly", async () => {
    const cases: Array<{ outcome: WriteOutcome; expected: WriteGuidance | null }> = [
      { outcome: { ok: true }, expected: null },
      { outcome: { ok: false, error: "invalid payload" }, expected: { kind: "refused", error: "invalid payload" } },
      {
        outcome: { ok: false, code: "mutation-unconfirmed", effect: "unconfirmed", audit: "recorded", error: "t" },
        expected: { kind: "unconfirmed" },
      },
      {
        outcome: { ok: false, code: "audit-finalization-failed", effect: "applied", audit: "pending", error: "t" },
        expected: { kind: "applied-unrecorded" },
      },
    ];
    for (const { outcome, expected } of cases) {
      const opts = baseOpts({ post: vi.fn().mockResolvedValue(outcome) });
      await moveIssue(opts);
      expect(opts.post).toHaveBeenCalledTimes(1);
      // every settled POST invalidates the board so the UI reconverges on Linear truth
      expect(opts.onInvalidate).toHaveBeenCalledTimes(1);
      expect(opts.onGuidance).toHaveBeenNthCalledWith(2, expected);
    }
  });
});