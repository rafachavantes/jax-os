import { describe, expect, it, vi } from "vitest";
import { listLivePanes } from "../../../../server/collectors/workflow-tmux";
import { handleStalenessCheckPost } from "./handler";

vi.mock("../../../../server/db", () => ({ getDb: () => ({}) }));
vi.mock("../../../../server/collectors/workflow-tmux", () => ({
  listLivePanes: vi.fn(async () => ({ paneIds: new Set(["%1", "%2"]), incarnation: "1:1", paneCommands: new Map() })),
}));
vi.mock("../../../../server/collectors/workflow-codex", () => ({
  createCodexConnector: vi.fn(),
  collectCodexSnapshot: vi.fn(async () => ({ ok: true as const, ts: "2026-09-10T12:00:00.000Z", threads: [], truncated: false })),
}));
vi.mock("../../../../server/db/workflows", () => ({
  evaluateStaleness: vi.fn(() => ({
    candidates: [{ transport: "codex", threadId: "0191f0aa-ffff-7000-8000-00000000000f", project: "x", role: "lead", reason: "no-signal", referenceEventId: 1, payload: {} }],
    nullIncarnation: 3,
    checked: 5, // deliberately != candidates.length (1), so the test can only pass if the route reports `checked` verbatim (Findings 9/30)
    codexSourceOk: true,
    tmuxSourceOk: true,
  })),
  maybeInsertStalenessAlert: vi.fn(() => true),
}));

describe("POST /api/workflow/staleness-check", () => {
  it("reports checked/alerted/nullIncarnation and both liveness source states", async () => {
    const res = await handleStalenessCheckPost(new Request("http://x", { method: "POST", headers: { "sec-fetch-site": "none" } }));
    expect(await res.json()).toEqual({ ok: true, data: { checked: 5, alerted: 1, nullIncarnation: 3, codexSourceOk: true, tmuxSourceOk: true } });
  });

  it("a tmux collector failure degrades to a warning while native evaluation continues (C4)", async () => {
    vi.mocked(listLivePanes).mockRejectedValueOnce(new Error("spawn tmux ENOENT"));
    const res = await handleStalenessCheckPost(new Request("http://x", { method: "POST", headers: { "sec-fetch-site": "none" } }));
    const body = await res.json();
    expect(body.ok).toBe(true);
    expect(body.data.tmuxSourceOk).toBe(false);
    expect(body.data.codexSourceOk).toBe(true);
    expect(body.data.checked).toBe(5);
  });
});
