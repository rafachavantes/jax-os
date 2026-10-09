import { beforeEach, describe, expect, it, vi } from "vitest";

const settingsState = vi.hoisted(() => ({ vault: true, vaultPath: "/tmp/vault" as string | null }));
vi.mock("../../../../server/settings", () => ({
  readGeneralSettings: () => ({
    ok: true,
    data: { vaultPath: settingsState.vaultPath, integrations: { vault: settingsState.vault } },
  }),
}));

vi.mock("../../../../server/collectors/files", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../../server/collectors/files")>();
  return {
    ...actual,
    ROOTS: { repos: "/tmp/repos", vault: "/tmp/vault" },
    trashEntry: vi.fn(() => ({ trashRel: ".jax-trash/T/a.txt" })),
    sweepTrash: vi.fn(() => []),
    removeExpiredTrashEntryAt: vi.fn(),
  };
});
vi.mock("../../../../server/mutations", () => ({
  runMutation: vi.fn(),
  fileEffect: (fn: () => unknown) => fn(),
}));

import { removeExpiredTrashEntryAt, sweepTrash, trashEntry } from "../../../../server/collectors/files";
import { runMutation } from "../../../../server/mutations";
import { POST } from "./route";

function req(body: unknown) {
  return new Request("http://127.0.0.1/api/files/delete", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
}

describe("POST /api/files/delete", () => {
  beforeEach(() => {
    vi.mocked(trashEntry).mockClear();
    vi.mocked(sweepTrash).mockReset().mockReturnValue([]);
    vi.mocked(removeExpiredTrashEntryAt).mockReset();
    vi.mocked(runMutation).mockReset();
    settingsState.vault = true;
    settingsState.vaultPath = "/tmp/vault";
  });

  it.each([
    { label: "root=vault refuses when vault is off, never auditing or trashing", vault: false, vaultPath: "/tmp/vault", root: "vault", refused: true },
    { label: "root=vault refuses when vault is enabled but unconfigured (null vaultPath)", vault: true, vaultPath: null, root: "vault", refused: true },
    { label: "root=repos still deletes when vault is off", vault: false, vaultPath: "/tmp/vault", root: "repos", refused: false },
  ])("$label (spec per-integration table, decision 16)", async ({ vault, vaultPath, root, refused }) => {
    settingsState.vault = vault;
    settingsState.vaultPath = vaultPath;
    if (!refused) vi.mocked(runMutation).mockImplementationOnce(async (_entry, effect) => ({ ok: true, value: await effect() }));
    const res = await POST(req({ root, rel: "a.txt" }));
    expect(res.status).toBe(200);
    if (refused) {
      expect(await res.json()).toEqual({ ok: false, error: "disabled" });
      expect(runMutation).not.toHaveBeenCalled();
      expect(trashEntry).not.toHaveBeenCalled();
    } else {
      await res.json();
      expect(trashEntry).toHaveBeenCalledWith("repos", "a.txt");
    }
  });

  it("refuses invalid input without invoking the collector, and never sweeps when the delete is rejected", async () => {
    const res = await POST(req({ root: "x", rel: "a.txt" }));
    expect(await res.json()).toEqual({ ok: false, error: "invalid payload" });
    expect(runMutation).not.toHaveBeenCalled();
    // round 1 plan review F2: strictRel runs for real here (importOriginal above), before trashEntry.
    for (const bad of ["a/../b", "/abs", "a\\b", "a//b", "."]) {
      const badRel = await POST(req({ root: "repos", rel: bad }));
      expect(await badRel.json()).toEqual({ ok: false, error: "invalid payload" });
    }
    expect(trashEntry).not.toHaveBeenCalled();
    vi.mocked(runMutation).mockResolvedValueOnce({
      ok: false, code: "mutation-rejected", effect: "not-applied", audit: "recorded", error: "symlink source",
    });
    const rejected = await POST(req({ root: "repos", rel: "a.txt" }));
    expect(await rejected.json()).toMatchObject({ ok: false, error: "symlink source" });
    expect(sweepTrash).not.toHaveBeenCalled();
  });

  it("preserves trashRel on an applied-unrecorded response so Undo can still recover it (diff review 617d73a5a89b F2)", async () => {
    vi.mocked(runMutation).mockResolvedValueOnce({
      ok: false, code: "audit-finalization-failed", effect: "applied", audit: "pending",
      error: "action applied but audit recording failed; check current state before retrying",
      value: { trashRel: ".jax-trash/T/a.txt" },
    });
    const res = await POST(req({ root: "repos", rel: "a.txt" }));
    expect(await res.json()).toMatchObject({ ok: false, effect: "applied", trashRel: ".jax-trash/T/a.txt" });
  });

  it("calls trashEntry with root+rel through the mutation effect and returns trashRel", async () => {
    vi.mocked(runMutation).mockImplementationOnce(async (_entry, effect) => ({ ok: true, value: await effect() }));
    const res = await POST(req({ root: "repos", rel: "a.txt" }));
    expect(trashEntry).toHaveBeenCalledWith("repos", "a.txt");
    expect(await res.json()).toEqual({ ok: true, trashRel: ".jax-trash/T/a.txt" });
    expect(sweepTrash).toHaveBeenCalledWith("repos");
    expect(sweepTrash).toHaveBeenCalledOnce();
  });

  it("runs each swept entry's audit-then-rm through its own runMutation call, stopping at the first failure (round 1 plan review F1)", async () => {
    const order: string[] = [];
    vi.mocked(removeExpiredTrashEntryAt).mockImplementation((absPath) => { order.push(`rm:${absPath}`); });
    vi.mocked(runMutation)
      .mockResolvedValueOnce({ ok: true, value: { trashRel: ".jax-trash/T/a.txt" } }) // primary delete
      .mockImplementationOnce(async (entry, effect) => {
        order.push(`pending:${(entry as { trashRel: string }).trashRel}`); // audit row lands before the rm
        await effect();
        return { ok: true, value: undefined };
      })
      .mockResolvedValueOnce({ // entry y's audit insert fails — its effect (and entry z) never runs
        ok: false, code: "audit-unavailable", effect: "not-applied", audit: "unavailable",
        error: "audit unavailable; action not performed",
      });
    vi.mocked(sweepTrash).mockReturnValueOnce([
      { trashRel: ".jax-trash/OLD/x.txt", absPath: "/tmp/repos/.jax-trash/OLD/x.txt" },
      { trashRel: ".jax-trash/OLD/y.txt", absPath: "/tmp/repos/.jax-trash/OLD/y.txt" },
      { trashRel: ".jax-trash/OLD/z.txt", absPath: "/tmp/repos/.jax-trash/OLD/z.txt" },
    ]);
    const res = await POST(req({ root: "repos", rel: "a.txt" }));
    expect(await res.json()).toEqual({ ok: true, trashRel: ".jax-trash/T/a.txt" }); // sweep failure never taints the response
    expect(order).toEqual(["pending:.jax-trash/OLD/x.txt", "rm:/tmp/repos/.jax-trash/OLD/x.txt"]);
    expect(runMutation).toHaveBeenCalledTimes(3); // primary + x (succeeds) + y (fails) — z never attempted
  });

});
