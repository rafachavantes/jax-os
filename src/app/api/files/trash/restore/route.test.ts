import { beforeEach, describe, expect, it, vi } from "vitest";

const settingsState = vi.hoisted(() => ({ vault: true, vaultPath: "/tmp/vault" as string | null }));
vi.mock("../../../../../server/settings", () => ({
  readGeneralSettings: () => ({
    ok: true,
    data: { vaultPath: settingsState.vaultPath, integrations: { vault: settingsState.vault } },
  }),
}));

vi.mock("../../../../../server/collectors/files", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../../../server/collectors/files")>();
  return {
    ROOTS: { repos: "/tmp/repos", vault: "/tmp/vault" },
    gateVaultRoot: actual.gateVaultRoot, // real gate — the route boundary depends on it
    parseTrashRel: actual.parseTrashRel, // real shape validator — the route boundary depends on it (F3)
    restoreTrashEntry: vi.fn(() => ({ restoredRel: "docs/plans/old-plan.md" })),
  };
});
vi.mock("../../../../../server/mutations", () => ({
  runMutation: vi.fn(),
  fileEffect: (fn: () => unknown) => fn(),
}));

import { restoreTrashEntry } from "../../../../../server/collectors/files";
import { runMutation } from "../../../../../server/mutations";
import { POST } from "./route";

function req(body: unknown) {
  return new Request("http://127.0.0.1/api/files/trash/restore", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
}

describe("POST /api/files/trash/restore", () => {
  beforeEach(() => {
    vi.mocked(restoreTrashEntry).mockClear();
    vi.mocked(runMutation).mockReset();
    settingsState.vault = true;
    settingsState.vaultPath = "/tmp/vault";
  });

  it.each([
    { label: "root=vault refuses when vault is off, never auditing or restoring", vault: false, vaultPath: "/tmp/vault", root: "vault", refused: true },
    { label: "root=vault refuses when vault is enabled but unconfigured (null vaultPath)", vault: true, vaultPath: null, root: "vault", refused: true },
    { label: "root=repos still restores when vault is off", vault: false, vaultPath: "/tmp/vault", root: "repos", refused: false },
  ])("$label (spec per-integration table, decision 16)", async ({ vault, vaultPath, root, refused }) => {
    settingsState.vault = vault;
    settingsState.vaultPath = vaultPath;
    if (!refused) vi.mocked(runMutation).mockImplementationOnce(async (_entry, effect) => ({ ok: true, value: await effect() }));
    const res = await POST(req({ root, trashRel: ".jax-trash/20260918T183000Z/docs/plans/old-plan.md" }));
    expect(res.status).toBe(200);
    if (refused) {
      expect(await res.json()).toEqual({ ok: false, error: "disabled" });
      expect(runMutation).not.toHaveBeenCalled();
      expect(restoreTrashEntry).not.toHaveBeenCalled();
    } else {
      await res.json();
      expect(restoreTrashEntry).toHaveBeenCalledWith("repos", ".jax-trash/20260918T183000Z/docs/plans/old-plan.md");
    }
  });

  it("refuses invalid payloads without invoking the collector", async () => {
    const badRoot = await POST(req({ root: "x", trashRel: ".jax-trash/T/a.txt" }));
    expect(await badRoot.json()).toEqual({ ok: false, error: "invalid payload" });
    const badRel = await POST(req({ root: "repos", trashRel: "\0" }));
    expect(await badRel.json()).toEqual({ ok: false, error: "invalid payload" });
    expect(runMutation).not.toHaveBeenCalled();
  });

  it("rejects a traversal-shaped trashRel at the route boundary without invoking runMutation or the collector (round 4 F3)", async () => {
    const res = await POST(req({ root: "repos", trashRel: ".jax-trash/20260918T183000Z/../../etc/passwd" }));
    expect(await res.json()).toEqual({ ok: false, error: "invalid payload" });
    expect(runMutation).not.toHaveBeenCalled();
    expect(restoreTrashEntry).not.toHaveBeenCalled();
  });

  it("calls restoreTrashEntry with root+trashRel through the mutation effect and returns restoredRel", async () => {
    vi.mocked(runMutation).mockImplementationOnce(async (_entry, effect) => ({ ok: true, value: await effect() }));
    const res = await POST(req({ root: "repos", trashRel: ".jax-trash/20260918T183000Z/docs/plans/old-plan.md" }));
    expect(restoreTrashEntry).toHaveBeenCalledWith("repos", ".jax-trash/20260918T183000Z/docs/plans/old-plan.md");
    expect(vi.mocked(runMutation).mock.calls[0][0]).toMatchObject({
      kind: "file-trash-restore", root: "repos", trashRel: ".jax-trash/20260918T183000Z/docs/plans/old-plan.md",
    });
    expect(await res.json()).toEqual({ ok: true, restoredRel: "docs/plans/old-plan.md" });
  });

  it("propagates a mutation failure", async () => {
    vi.mocked(runMutation).mockResolvedValueOnce({
      ok: false, code: "mutation-rejected", effect: "not-applied", audit: "recorded", error: "already exists",
    });
    const res = await POST(req({ root: "repos", trashRel: ".jax-trash/20260918T183000Z/a.txt" }));
    expect(await res.json()).toMatchObject({ ok: false, error: "already exists" });
  });

  it("bounds the length on the parsed original rel, not the .jax-trash/<timestamp>/ envelope (diff review 617d73a5a89b F1)", async () => {
    const rel4096 = "a".repeat(4096);
    vi.mocked(runMutation).mockImplementationOnce(async (_entry, effect) => ({ ok: true, value: await effect() }));
    const ok = await POST(req({ root: "repos", trashRel: `.jax-trash/20260918T183000Z/${rel4096}` }));
    expect(await ok.json()).toEqual({ ok: true, restoredRel: "docs/plans/old-plan.md" });
    expect(runMutation).toHaveBeenCalledTimes(1);

    const rel4097 = "a".repeat(4097);
    const bad = await POST(req({ root: "repos", trashRel: `.jax-trash/20260918T183000Z/${rel4097}` }));
    expect(await bad.json()).toEqual({ ok: false, error: "invalid payload" });
    expect(runMutation).toHaveBeenCalledTimes(1); // still 1 — the oversized rel never reaches it
  });

  it("preserves restoredRel on an applied-unrecorded response (diff review 617d73a5a89b F2)", async () => {
    vi.mocked(runMutation).mockResolvedValueOnce({
      ok: false, code: "audit-finalization-failed", effect: "applied", audit: "pending",
      error: "action applied but audit recording failed; check current state before retrying",
      value: { restoredRel: "docs/plans/old-plan.md" },
    });
    const res = await POST(req({ root: "repos", trashRel: ".jax-trash/20260918T183000Z/docs/plans/old-plan.md" }));
    expect(await res.json()).toMatchObject({ ok: false, effect: "applied", restoredRel: "docs/plans/old-plan.md" });
  });
});
