import { createHash } from "node:crypto";
import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from "vitest";
import type Database from "better-sqlite3";

const HASH = "ab".repeat(32);
const MOCK_TRASH_REL = ".jax-trash/20260101T000000Z/a.txt";
const rev = (s: string) => createHash("sha256").update(s, "utf8").digest("hex");

const settingsHarness = vi.hoisted(() => ({ vault: true, vaultPath: "/tmp/vault" as string | null, linear: true }));
vi.mock("../../server/settings", () => ({
  readGeneralSettings: () => ({
    ok: true,
    data: {
      vaultPath: settingsHarness.vaultPath,
      integrations: { vault: settingsHarness.vault, linear: settingsHarness.linear, agents: { claude: true, codex: true, opencode: true } },
    },
  }),
}));

vi.mock("../../server/collectors/files", () => ({
  ROOTS: { repos: "/tmp/repos", vault: "/tmp/vault" },
  READ_CAP: 2 * 1024 * 1024,
  UPLOAD_CAP: 5 * 1024 * 1024,
  hashBytes: (buf: Buffer) => createHash("sha256").update(buf).digest("hex"),
  createEntry: vi.fn(),
  strictRel: vi.fn(() => true),
  trashEntry: vi.fn(() => ({ trashRel: MOCK_TRASH_REL })),
  sweepTrash: vi.fn(() => []),
  removeExpiredTrashEntryAt: vi.fn(),
  renameEntry: vi.fn(),
  writeFile: vi.fn(() => ({ hash: HASH })),
  uploadFile: vi.fn(),
  gateVaultRoot: (root: string, vaultUsable: boolean) => root === "vault" && !vaultUsable,
}));

vi.mock("../../server/collectors/tmux", () => ({
  getAgents: vi.fn(() => [{ session: "jax-test" }]),
}));

vi.mock("../../server/collectors/linear", () => ({
  updateIssue: vi.fn(),
  moveIssue: vi.fn(),
  createComment: vi.fn(),
  createIssue: vi.fn(),
}));

vi.mock("../../server/collectors/rules", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../server/collectors/rules")>();
  return { ...actual, applyRuleFile: vi.fn(() => ({ ok: true, bytes: 4, backup: ".bak-z" })) };
});

vi.mock("../../server/db/rules", () => ({
  getRuleDocs: vi.fn(() => ({ global: "g\n", claude: "c\n", codex: "x\n", opencode: "o\n" })),
}));

let testDb: Database.Database;
vi.mock("../../server/db", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../server/db")>();
  return { ...actual, getDb: () => testDb };
});


import { openDb } from "../../server/db";
import { createEntry, renameEntry, trashEntry, uploadFile, writeFile } from "../../server/collectors/files";
import { getAgents } from "../../server/collectors/tmux";
import { createComment, createIssue, moveIssue, updateIssue } from "../../server/collectors/linear";
import { MutationRejected } from "../../lib/mutationOutcome";
import { POST as createPost } from "./files/create/route";
import { POST as deletePost } from "./files/delete/route";
import { POST as renamePost } from "./files/rename/route";
import { POST as writePost } from "./files/write/route";
import { POST as uploadPost } from "./files/upload/route";
import { POST as takePost } from "./tmux/take-control/route";
import { POST as releasePost } from "./tmux/release-control/route";
import { POST as movePost } from "./kanban/move/route";
import { POST as issueCreatePost } from "./kanban/create/route";
import { POST as updatePost } from "./kanban/update/route";
import { POST as commentPost } from "./kanban/comment/route";
import { POST as applyPost } from "./rules/apply/route";
import { applyRuleFile, rulePaths } from "../../server/collectors/rules";
import { getRuleDocs } from "../../server/db/rules";
import sample from "../../../workflow/fixtures/agent-settings-v1.json";
import { POST as settingsRoutePost } from "./agent-settings/route";
import { POST as providersRoutePost } from "./opencode-providers/route";

function settingsPost(req: Request, deps: unknown) {
  Object.defineProperty(req, "jaxDeps", { value: deps });
  return settingsRoutePost(req);
}
function providersPost(req: Request, deps: unknown) {
  Object.defineProperty(req, "jaxDeps", { value: deps });
  return providersRoutePost(req);
}

const request = (payload: unknown) =>
  new Request("http://127.0.0.1/api/test", {
    method: "POST",
    body: JSON.stringify(payload),
    headers: { "content-type": "application/json" },
  });

async function multipart(file: File, fields: Record<string, string> = { root: "repos" }) {
  const form = new FormData();
  for (const [k, v] of Object.entries(fields)) form.set(k, v);
  form.set("file", file);
  const tmp = new Request("http://127.0.0.1/upload", { method: "POST", body: form });
  const bytes = await tmp.arrayBuffer();
  return new Request("http://127.0.0.1/upload", {
    method: "POST",
    body: bytes,
    headers: {
      "content-type": tmp.headers.get("content-type") ?? "",
      "content-length": String(bytes.byteLength),
    },
  });
}

function errno(code: string, message = "err"): NodeJS.ErrnoException {
  const e = new Error(message) as NodeJS.ErrnoException;
  e.code = code;
  return e;
}

type Row = { id: number; ok: number | null; payload: string };
function rows(): Row[] {
  return testDb.prepare("SELECT id, ok, payload FROM mutations").all() as Row[];
}
function failInsert() {
  testDb.exec("CREATE TRIGGER fail_ins BEFORE INSERT ON mutations BEGIN SELECT RAISE(FAIL, 'fixture'); END");
}
function failUpdate() {
  testDb.exec("CREATE TRIGGER fail_upd BEFORE UPDATE ON mutations BEGIN SELECT RAISE(FAIL, 'fixture'); END");
}

describe("file and tmux mutation lifecycle", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    settingsHarness.vault = true;
    settingsHarness.vaultPath = "/tmp/vault";
    settingsHarness.linear = true;
    testDb = openDb(":memory:");
  });
  afterEach(() => {
    testDb.close();
  });

  const createBody = { root: "repos", relParentDir: "", basename: "a.txt", kind: "file" };
  const deleteBody = { root: "repos", rel: "a.txt" };
  const renameBody = { root: "repos", relFrom: "a.txt", relTo: "b.txt" };
  const writeBody = { root: "repos", rel: "x.txt", content: "x", baseHash: HASH };

  it("create: pending is visible inside the collector, then the same row is done", async () => {
    vi.mocked(createEntry).mockImplementation(() => {
      const seen = rows();
      expect(seen).toHaveLength(1);
      expect(seen[0].ok).toBeNull();
      expect(JSON.parse(seen[0].payload)).toMatchObject({
        kind: "file-create", root: "repos", relParentDir: "", basename: "a.txt", entryKind: "file", outcome: "pending",
      });
    });
    const res = await createPost(request(createBody));
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: true });
    expect(createEntry).toHaveBeenCalledOnce();
    const done = rows();
    expect(done).toHaveLength(1);
    expect(done[0].ok).toBe(1);
    expect(JSON.parse(done[0].payload).outcome).toBe("done");
  });

  it("create: insert failure skips the collector", async () => {
    failInsert();
    const res = await createPost(request(createBody));
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({
      ok: false, code: "audit-unavailable", effect: "not-applied", audit: "unavailable",
      error: "audit unavailable; action not performed",
    });
    expect(createEntry).not.toHaveBeenCalled();
    expect(rows()).toEqual([]);
  });

  it("create: EEXIST is a confirmed 200 rejection", async () => {
    vi.mocked(createEntry).mockImplementation(() => { throw errno("EEXIST"); });
    const res = await createPost(request(createBody));
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({
      ok: false, code: "mutation-rejected", effect: "not-applied", audit: "recorded", error: "already exists",
    });
    expect(JSON.parse(rows()[0].payload).outcome).toBe("failed");
  });

  it("create: finalization failure after success stays pending", async () => {
    failUpdate();
    const res = await createPost(request(createBody));
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body).toEqual({
      ok: false, code: "audit-finalization-failed", effect: "applied", audit: "pending",
      error: "action applied but audit recording failed; check current state before retrying",
    });
    expect(body).not.toHaveProperty("status");
    expect(body).not.toHaveProperty("value");
    expect(rows()[0].ok).toBeNull();
  });

  it("delete: pending, skip on insert fail, symlink source, finalization fail", async () => {
    vi.mocked(trashEntry).mockImplementation(() => {
      expect(rows()).toHaveLength(1);
      expect(rows()[0].ok).toBeNull();
      return { trashRel: MOCK_TRASH_REL };
    });
    expect(await (await deletePost(request(deleteBody))).json()).toEqual({ ok: true, trashRel: MOCK_TRASH_REL });
    expect(JSON.parse(rows()[0].payload)).toMatchObject({ kind: "file-delete", outcome: "done" });

    vi.clearAllMocks();
    testDb.close();
    testDb = openDb(":memory:");
    failInsert();
    expect(await (await deletePost(request(deleteBody))).json()).toMatchObject({ code: "audit-unavailable" });
    expect(trashEntry).not.toHaveBeenCalled();

    // trashEntry rejects the source when it is a symlink (Task 2) — the new contract's own
    // rejection vocabulary; the old ENOTEMPTY leg no longer applies because a rename-based trash
    // moves a non-empty directory in one shot, it never refuses one for being non-empty.
    testDb.close();
    testDb = openDb(":memory:");
    vi.mocked(trashEntry).mockImplementation(() => { throw new Error("symlink source"); });
    const refused = await deletePost(request(deleteBody));
    expect(refused.status).toBe(200);
    expect(await refused.json()).toEqual({
      ok: false, code: "mutation-rejected", effect: "not-applied", audit: "recorded", error: "symlink source",
    });

    testDb.close();
    testDb = openDb(":memory:");
    vi.mocked(trashEntry).mockReset().mockReturnValue({ trashRel: MOCK_TRASH_REL });
    failUpdate();
    expect(await (await deletePost(request(deleteBody))).json()).toMatchObject({
      code: "audit-finalization-failed", effect: "applied", audit: "pending",
    });
    expect(rows()[0].ok).toBeNull();
  });

  it("rename: pending, skip on insert fail, destination exists, finalization fail", async () => {
    vi.mocked(renameEntry).mockImplementation(() => {
      expect(JSON.parse(rows()[0].payload)).toMatchObject({
        kind: "file-rename", root: "repos", from: "a.txt", to: "b.txt", outcome: "pending",
      });
    });
    expect((await (await renamePost(request(renameBody))).json()).ok).toBe(true);
    expect(JSON.parse(rows()[0].payload).outcome).toBe("done");

    testDb.close();
    testDb = openDb(":memory:");
    failInsert();
    vi.mocked(renameEntry).mockClear();
    expect(await (await renamePost(request(renameBody))).json()).toMatchObject({ code: "audit-unavailable" });
    expect(renameEntry).not.toHaveBeenCalled();

    testDb.close();
    testDb = openDb(":memory:");
    vi.mocked(renameEntry).mockImplementation(() => { throw new Error("destination exists"); });
    const res = await renamePost(request(renameBody));
    expect(res.status).toBe(200);
    expect(await res.json()).toMatchObject({ code: "mutation-rejected", error: "already exists" });

    testDb.close();
    testDb = openDb(":memory:");
    vi.mocked(renameEntry).mockReset();
    failUpdate();
    expect(await (await renamePost(request(renameBody))).json()).toMatchObject({ effect: "applied", audit: "pending" });
  });

  it("write: pending, skip on insert fail, changed-on-disk, allowlist, bad name, ENOSPC, hash on finalization fail", async () => {
    vi.mocked(writeFile).mockImplementation(() => {
      expect(JSON.parse(rows()[0].payload)).toMatchObject({ kind: "file-edit", root: "repos", rel: "x.txt", outcome: "pending" });
      return { hash: HASH };
    });
    expect(await (await writePost(request(writeBody))).json()).toEqual({ ok: true, hash: HASH });
    expect(JSON.parse(rows()[0].payload).outcome).toBe("done");

    testDb.close();
    testDb = openDb(":memory:");
    failInsert();
    vi.mocked(writeFile).mockClear();
    expect(await (await writePost(request(writeBody))).json()).toMatchObject({ code: "audit-unavailable" });
    expect(writeFile).not.toHaveBeenCalled();

    testDb.close();
    testDb = openDb(":memory:");
    vi.mocked(writeFile).mockImplementation(() => { throw new Error("changed on disk"); });
    const stale = await writePost(request(writeBody));
    expect(stale.status).toBe(200);
    expect(await stale.json()).toEqual({
      ok: false, code: "mutation-rejected", effect: "not-applied", audit: "recorded", error: "changed on disk",
    });

    testDb.close();
    testDb = openDb(":memory:");
    vi.mocked(writeFile).mockImplementation(() => { throw new Error("outside allowlist"); });
    expect((await (await writePost(request(writeBody))).json()).error).toBe("outside allowlist");
    testDb.close();
    testDb = openDb(":memory:");
    vi.mocked(writeFile).mockImplementation(() => { throw new Error("bad name"); });
    const bad = await writePost(request(writeBody));
    expect(bad.status).toBe(200);
    expect(await bad.json()).toMatchObject({ code: "mutation-rejected", error: "bad name" });

    testDb.close();
    testDb = openDb(":memory:");
    vi.mocked(writeFile).mockImplementation(() => { throw errno("ENOSPC", "no space"); });
    const space = await writePost(request(writeBody));
    expect(space.status).toBe(200);
    expect(await space.json()).toEqual({
      ok: false, code: "mutation-unconfirmed", effect: "unconfirmed", audit: "recorded",
      error: "action outcome unconfirmed; check current state before retrying",
    });
    expect(JSON.parse(rows()[0].payload).outcome).toBe("abandoned");

    testDb.close();
    testDb = openDb(":memory:");
    vi.mocked(writeFile).mockImplementation(() => ({ hash: HASH }));
    failUpdate();
    const fin = await (await writePost(request(writeBody))).json();
    expect(fin).toMatchObject({
      ok: false, code: "audit-finalization-failed", effect: "applied", audit: "pending", hash: HASH,
    });
    expect(fin).not.toHaveProperty("status");
    expect(fin).not.toHaveProperty("value");
  });

  it("write: atomic save rejection is recorded as failed", async () => {
    vi.mocked(writeFile).mockImplementation(() => {
      throw new MutationRejected("atomic save failed; original file was not replaced");
    });
    const res = await writePost(request(writeBody));
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({
      ok: false, code: "mutation-rejected", effect: "not-applied", audit: "recorded",
      error: "atomic save failed; original file was not replaced",
    });
    expect(JSON.parse(rows()[0].payload).outcome).toBe("failed");
  });

  it("upload: parsed bytes reach the collector once; insert failure prevents create", async () => {
    const bytes = new Uint8Array([1, 2, 3]);
    vi.mocked(uploadFile).mockImplementation(() => {
      expect(rows()).toHaveLength(1);
      expect(JSON.parse(rows()[0].payload)).toMatchObject({
        kind: "file-upload", root: "repos", relParentDir: "", basename: "a.bin", outcome: "pending",
      });
    });
    const res = await uploadPost(await multipart(new File([bytes], "a.bin")));
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: true });
    expect(uploadFile).toHaveBeenCalledOnce();
    expect(Buffer.compare(vi.mocked(uploadFile).mock.calls[0][3] as Buffer, Buffer.from(bytes))).toBe(0);

    testDb.close();
    testDb = openDb(":memory:");
    failInsert();
    vi.mocked(uploadFile).mockClear();
    expect(await (await uploadPost(await multipart(new File([bytes], "a.bin")))).json()).toMatchObject({
      code: "audit-unavailable",
    });
    expect(uploadFile).not.toHaveBeenCalled();
    expect(rows()).toEqual([]);
  });

  it("upload: EEXIST 200, arrayBuffer failure has no audit, finalization fail", async () => {
    vi.mocked(uploadFile).mockImplementation(() => { throw errno("EEXIST"); });
    const exists = await uploadPost(await multipart(new File([new Uint8Array([1])], "a.bin")));
    expect(exists.status).toBe(200);
    expect(await exists.json()).toMatchObject({ code: "mutation-rejected", error: "already exists" });

    testDb.close();
    testDb = openDb(":memory:");
    vi.mocked(uploadFile).mockClear();
    const req = await multipart(new File([new Uint8Array([1])], "a.bin"));
    const spy = vi.spyOn(Blob.prototype, "arrayBuffer").mockRejectedValue(new Error("boom"));
    try {
      const bad = await uploadPost(req);
      expect((await bad.json()).ok).toBe(false);
      expect(uploadFile).not.toHaveBeenCalled();
      expect(rows()).toEqual([]);
    } finally {
      spy.mockRestore();
    }

    testDb.close();
    testDb = openDb(":memory:");
    vi.mocked(uploadFile).mockReset();
    failUpdate();
    expect(await (await uploadPost(await multipart(new File([new Uint8Array([1])], "a.bin")))).json()).toMatchObject({
      code: "audit-finalization-failed", effect: "applied",
    });
  });

  it("invalid file input inserts nothing", async () => {
    expect((await (await createPost(request({ ...createBody, kind: "symlink" }))).json()).ok).toBe(false);
    expect(createEntry).not.toHaveBeenCalled();
    expect(rows()).toEqual([]);
  });

  // Spec decision 16 / per-integration table: every root=vault mutation form refuses with the
  // distinguishable "disabled" answer BEFORE runMutation — no audit row, no collector call.
  it("refuses every root=vault mutation with disabled and no audit row when vault is off", async () => {
    settingsHarness.vault = false;
    try {
      const cases: [string, () => Promise<Response>, Mock][] = [
        ["create", () => createPost(request({ root: "vault", relParentDir: "", basename: "a.txt", kind: "file" })), vi.mocked(createEntry)],
        ["rename", () => renamePost(request({ root: "vault", relFrom: "a.txt", relTo: "b.txt" })), vi.mocked(renameEntry)],
        ["write", () => writePost(request({ root: "vault", rel: "x.txt", content: "x", baseHash: HASH })), vi.mocked(writeFile)],
        ["upload", async () => uploadPost(await multipart(new File([new Uint8Array([1])], "a.bin"), { root: "vault" })), vi.mocked(uploadFile)],
      ];
      for (const [label, run, spy] of cases) {
        expect(await (await run()).json(), label).toEqual({ ok: false, error: "disabled" });
        expect(spy, label).not.toHaveBeenCalled();
      }
      expect(rows()).toEqual([]);
    } finally {
      settingsHarness.vault = true;
    }
  });

  // Null-vault (497B): the integration is on but no path is configured — the SAME disabled
  // answer, never a collector throw the route happens to catch (tech-lead addendum).
  it("refuses root=vault when vault is enabled but unconfigured (null vaultPath)", async () => {
    settingsHarness.vaultPath = null;
    try {
      const res = await writePost(request({ root: "vault", rel: "x.txt", content: "x", baseHash: HASH }));
      expect(res.status).toBe(200);
      expect(await res.json()).toEqual({ ok: false, error: "disabled" });
      expect(writeFile).not.toHaveBeenCalled();
      expect(rows()).toEqual([]);
    } finally {
      settingsHarness.vaultPath = "/tmp/vault";
    }
  });

  it("still mutates root=repos with vault off (regression)", async () => {
    settingsHarness.vault = false;
    try {
      expect(await (await createPost(request(createBody))).json()).toEqual({ ok: true });
      expect(createEntry).toHaveBeenCalledOnce();
      expect(rows()).toHaveLength(1);
    } finally {
      settingsHarness.vault = true;
    }
  });

  it("tmux take/release insert failure never reports success; take liveness failure never inserts", async () => {
    failInsert();
    const takeFail = await takePost(request({ session: "jax-test" }));
    expect(takeFail.status).toBe(200);
    expect(await takeFail.json()).toEqual({
      ok: false, code: "audit-unavailable", effect: "not-applied", audit: "unavailable",
      error: "control record unavailable",
    });
    expect(rows()).toEqual([]);

    const releaseFail = await releasePost(request({ session: "jax-test" }));
    expect(releaseFail.status).toBe(200);
    expect(await releaseFail.json()).toEqual({
      ok: false, code: "audit-unavailable", effect: "not-applied", audit: "unavailable",
      error: "control record unavailable",
    });

    testDb.close();
    testDb = openDb(":memory:");
    vi.mocked(getAgents).mockImplementation(() => { throw new Error("no tmux"); });
    expect(await (await takePost(request({ session: "jax-test" }))).json()).toEqual({
      ok: false, error: "session not found",
    });
    expect(rows()).toEqual([]);
  });

  it("tmux success is a single insert, not runMutation", async () => {
    expect(await (await takePost(request({ session: "jax-test" }))).json()).toEqual({ ok: true });
    expect(await (await releasePost(request({ session: "jax-test" }))).json()).toEqual({ ok: true });
    const all = rows();
    expect(all).toHaveLength(2);
    expect(all.every((r) => r.ok === null)).toBe(true);
    expect(all.map((r) => JSON.parse(r.payload).outcome)).toEqual([undefined, undefined]);
    expect(all.map((r) => JSON.parse(r.payload).kind)).toEqual(["take-control", "release-control"]);
  });

  it("move: pending before network, insert failure skips network, success/false/timeout/finalization", async () => {
    vi.mocked(moveIssue).mockImplementation(async () => {
      expect(rows()).toHaveLength(1);
      expect(JSON.parse(rows()[0].payload)).toMatchObject({
        kind: "linear-move", issueId: "iss1", stateId: "st1", outcome: "pending",
      });
    });
    expect(await (await movePost(request({ issueId: "iss1", stateId: "st1" }))).json()).toEqual({ ok: true });
    expect(moveIssue).toHaveBeenCalledOnce();
    expect(JSON.parse(rows()[0].payload).outcome).toBe("done");

    testDb.close();
    testDb = openDb(":memory:");
    failInsert();
    vi.mocked(moveIssue).mockClear();
    expect(await (await movePost(request({ issueId: "iss1", stateId: "st1" }))).json()).toMatchObject({
      code: "audit-unavailable",
    });
    expect(moveIssue).not.toHaveBeenCalled();

    testDb.close();
    testDb = openDb(":memory:");
    vi.mocked(moveIssue).mockRejectedValue(new MutationRejected("issueUpdate did not succeed"));
    const denied = await movePost(request({ issueId: "iss1", stateId: "st1" }));
    expect(denied.status).toBe(200);
    expect(await denied.json()).toMatchObject({
      code: "mutation-rejected", effect: "not-applied", audit: "recorded",
    });
    expect(moveIssue).toHaveBeenCalledOnce();

    testDb.close();
    testDb = openDb(":memory:");
    vi.mocked(moveIssue).mockClear();
    vi.mocked(moveIssue).mockRejectedValue(new Error("timeout after accept"));
    const timeout = await movePost(request({ issueId: "iss1", stateId: "st1" }));
    expect(timeout.status).toBe(200);
    expect(await timeout.json()).toEqual({
      ok: false, code: "mutation-unconfirmed", effect: "unconfirmed", audit: "recorded",
      error: "action outcome unconfirmed; check current state before retrying",
    });
    expect(JSON.parse(rows()[0].payload).outcome).toBe("abandoned");
    expect(moveIssue).toHaveBeenCalledOnce();

    testDb.close();
    testDb = openDb(":memory:");
    vi.mocked(moveIssue).mockReset();
    failUpdate();
    expect(await (await movePost(request({ issueId: "iss1", stateId: "st1" }))).json()).toMatchObject({
      code: "audit-finalization-failed", effect: "applied", audit: "pending",
    });
    expect(rows()[0].ok).toBeNull();
  });

  it("update: pending, skip on insert fail, confirmed false, timeout, finalization fail", async () => {
    vi.mocked(updateIssue).mockImplementation(async () => {
      expect(JSON.parse(rows()[0].payload)).toMatchObject({
        kind: "linear-priority", issueId: "iss1", priority: 2, outcome: "pending",
      });
    });
    expect(await (await updatePost(request({ issueId: "iss1", patch: { priority: 2 } }))).json()).toEqual({ ok: true });
    expect(updateIssue).toHaveBeenCalledOnce();

    testDb.close();
    testDb = openDb(":memory:");
    failInsert();
    vi.mocked(updateIssue).mockClear();
    expect(await (await updatePost(request({ issueId: "iss1", patch: { priority: 2 } }))).json()).toMatchObject({
      code: "audit-unavailable",
    });
    expect(updateIssue).not.toHaveBeenCalled();

    testDb.close();
    testDb = openDb(":memory:");
    vi.mocked(updateIssue).mockRejectedValue(new MutationRejected("issueUpdate did not succeed"));
    expect(await (await updatePost(request({ issueId: "iss1", patch: { priority: 2 } }))).json()).toMatchObject({
      code: "mutation-rejected", effect: "not-applied",
    });

    testDb.close();
    testDb = openDb(":memory:");
    vi.mocked(updateIssue).mockRejectedValue(new Error("timeout"));
    expect(await (await updatePost(request({ issueId: "iss1", patch: { priority: 2 } }))).json()).toMatchObject({
      code: "mutation-unconfirmed", effect: "unconfirmed",
    });
    expect(JSON.parse(rows()[0].payload).outcome).toBe("abandoned");

    testDb.close();
    testDb = openDb(":memory:");
    vi.mocked(updateIssue).mockReset();
    failUpdate();
    expect(await (await updatePost(request({ issueId: "iss1", patch: { priority: 2 } }))).json()).toMatchObject({
      effect: "applied", audit: "pending",
    });
  });

  it("comment: pending, preview only, insert failure, timeout unconfirmed, finalization fail", async () => {
    const body = "x".repeat(250);
    vi.mocked(createComment).mockImplementation(async () => {
      const payload = JSON.parse(rows()[0].payload) as Record<string, unknown>;
      expect(payload).toMatchObject({ kind: "linear-comment", issueId: "iss1", outcome: "pending" });
      expect(payload.preview).toBe(`${"x".repeat(200)}…`);
      expect(JSON.stringify(payload)).not.toContain(body);
    });
    expect(await (await commentPost(request({ issueId: "iss1", body }))).json()).toEqual({ ok: true });
    expect(createComment).toHaveBeenCalledOnce();
    expect(createComment).toHaveBeenCalledWith("iss1", body);

    testDb.close();
    testDb = openDb(":memory:");
    failInsert();
    vi.mocked(createComment).mockClear();
    expect(await (await commentPost(request({ issueId: "iss1", body: "hi" }))).json()).toMatchObject({
      code: "audit-unavailable",
    });
    expect(createComment).not.toHaveBeenCalled();

    testDb.close();
    testDb = openDb(":memory:");
    vi.mocked(createComment).mockRejectedValue(new Error("timeout after accept"));
    expect(await (await commentPost(request({ issueId: "iss1", body: "hi" }))).json()).toMatchObject({
      code: "mutation-unconfirmed", effect: "unconfirmed",
    });
    expect(JSON.parse(rows()[0].payload).outcome).toBe("abandoned");
    expect(JSON.parse(rows()[0].payload).preview).toBe("hi");

    testDb.close();
    testDb = openDb(":memory:");
    vi.mocked(createComment).mockReset();
    failUpdate();
    expect(await (await commentPost(request({ issueId: "iss1", body: "hi" }))).json()).toMatchObject({
      code: "audit-finalization-failed", effect: "applied",
    });
  });

  it("create: pending visible inside the collector, then done; insert failure skips network; rejected/unconfirmed/finalization", async () => {
    vi.mocked(createIssue).mockImplementation(async () => {
      expect(rows()).toHaveLength(1);
      expect(JSON.parse(rows()[0].payload)).toMatchObject({ kind: "linear-issue-create", teamId: "t1", title: "New", outcome: "pending" });
      return { id: "i9", identifier: "MOA-9" };
    });
    const res = await issueCreatePost(request({ teamId: "t1", title: "New" }));
    expect(await res.json()).toEqual({ ok: true, id: "i9", identifier: "MOA-9" });
    expect(createIssue).toHaveBeenCalledOnce();
    expect(JSON.parse(rows()[0].payload).outcome).toBe("done");

    testDb.close();
    testDb = openDb(":memory:");
    failInsert();
    vi.mocked(createIssue).mockClear();
    expect(await (await issueCreatePost(request({ teamId: "t1", title: "New" }))).json()).toMatchObject({ code: "audit-unavailable" });
    expect(createIssue).not.toHaveBeenCalled();

    testDb.close();
    testDb = openDb(":memory:");
    vi.mocked(createIssue).mockRejectedValue(new MutationRejected("issueCreate did not succeed"));
    expect(await (await issueCreatePost(request({ teamId: "t1", title: "New" }))).json()).toMatchObject({
      code: "mutation-rejected", effect: "not-applied",
    });

    testDb.close();
    testDb = openDb(":memory:");
    vi.mocked(createIssue).mockRejectedValue(new Error("timeout"));
    expect(await (await issueCreatePost(request({ teamId: "t1", title: "New" }))).json()).toMatchObject({
      code: "mutation-unconfirmed", effect: "unconfirmed",
    });
    expect(JSON.parse(rows()[0].payload).outcome).toBe("abandoned");

    testDb.close();
    testDb = openDb(":memory:");
    vi.mocked(createIssue).mockReset();
    vi.mocked(createIssue).mockResolvedValue({ id: "i9", identifier: "MOA-9" });
    failUpdate();
    expect(await (await issueCreatePost(request({ teamId: "t1", title: "New" }))).json()).toMatchObject({
      code: "audit-finalization-failed", effect: "applied", audit: "pending",
    });
  });

  const RULE_DOCS = { global: "g\n", claude: "c\n", codex: "x\n", opencode: "o\n" };
  const REVS = { global: rev("g\n"), exception: rev("c\n") };

  it("rules apply: composition and invalid app happen before insert", async () => {
    const bad = await applyPost(request({ app: "nope", diskHash: "h", applySourceRevisions: REVS }));
    expect(bad.status).toBe(400);
    expect(await bad.json()).toEqual({ ok: false, error: "invalid payload" });
    expect(applyRuleFile).not.toHaveBeenCalled();
    expect(rows()).toEqual([]);

    vi.mocked(getRuleDocs).mockImplementation(() => { throw new Error("unseeded"); });
    const composeFail = await applyPost(request({ app: "claude", diskHash: "h", applySourceRevisions: REVS }));
    expect(composeFail.status).toBe(500);
    expect(await composeFail.json()).toMatchObject({ ok: false, code: "mutation-unconfirmed", effect: "unconfirmed" });
    expect(applyRuleFile).not.toHaveBeenCalled();
    expect(JSON.parse(rows()[0].payload).outcome).toBe("abandoned");
  });

  it("rules apply: rejects a four-key GET revision object and a stale source revision before any disk effect", async () => {
    vi.mocked(getRuleDocs).mockReturnValue(RULE_DOCS);
    const fourKey = await applyPost(request({
      app: "claude", diskHash: "h",
      applySourceRevisions: { global: REVS.global, exception: REVS.exception, claude: REVS.exception, codex: REVS.exception },
    }));
    expect(fourKey.status).toBe(400);
    expect(await fourKey.json()).toEqual({ ok: false, error: "invalid applySourceRevisions" });
    expect(applyRuleFile).not.toHaveBeenCalled();

    const staleCanon = await applyPost(request({
      app: "claude", diskHash: "h",
      applySourceRevisions: { global: rev("other\n"), exception: REVS.exception },
    }));
    expect(staleCanon.status).toBe(409);
    expect(await staleCanon.json()).toMatchObject({
      ok: false, code: "mutation-rejected", effect: "not-applied", error: "canonical source changed since preview",
    });
    expect(applyRuleFile).not.toHaveBeenCalled();

    testDb.close();
    testDb = openDb(":memory:");
    vi.mocked(getRuleDocs).mockReturnValue(RULE_DOCS);
    const staleExc = await applyPost(request({
      app: "claude", diskHash: "h",
      applySourceRevisions: { global: REVS.global, exception: rev("other\n") },
    }));
    expect(staleExc.status).toBe(409);
    expect(await staleExc.json()).toMatchObject({ error: "exception source changed since preview" });
    expect(applyRuleFile).not.toHaveBeenCalled();
  });

  it("rules apply: success records one done row with bytes/backup on rulePaths()", async () => {
    vi.mocked(getRuleDocs).mockReturnValue(RULE_DOCS);
    vi.mocked(applyRuleFile).mockImplementation(() => {
      expect(rows()).toHaveLength(1);
      expect(JSON.parse(rows()[0].payload)).toMatchObject({
        kind: "rules-apply", app: "claude", path: rulePaths().claude, outcome: "pending",
      });
      return { ok: true, bytes: 4, backup: ".bak-z" };
    });
    const res = await applyPost(request({ app: "claude", diskHash: "h", path: "/tmp/evil", applySourceRevisions: REVS }));
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: true, data: { bytes: 4, backup: ".bak-z" } });
    expect(applyRuleFile).toHaveBeenCalledOnce();
    expect(vi.mocked(applyRuleFile).mock.calls[0][0]).toBe(rulePaths().claude);
    expect(vi.mocked(applyRuleFile).mock.calls[0][1]).toBe("c\n\n\ng\n");
    expect(JSON.parse(rows()[0].payload)).toMatchObject({
      outcome: "done", bytes: 4, backup: ".bak-z", path: rulePaths().claude,
    });
  });

  it("rules apply: stale 409, symlink 400, fs-error 500", async () => {
    vi.mocked(getRuleDocs).mockReturnValue(RULE_DOCS);
    vi.mocked(applyRuleFile).mockReturnValue({ ok: false, reason: "stale" });
    const stale = await applyPost(request({ app: "claude", diskHash: "h", applySourceRevisions: REVS }));
    expect(stale.status).toBe(409);
    expect(await stale.json()).toEqual({
      ok: false, code: "mutation-rejected", effect: "not-applied", audit: "recorded",
      error: "file changed on disk since diskHash was read",
    });

    testDb.close();
    testDb = openDb(":memory:");
    vi.mocked(getRuleDocs).mockReturnValue(RULE_DOCS);
    vi.mocked(applyRuleFile).mockReturnValue({ ok: false, reason: "symlink" });
    const link = await applyPost(request({ app: "claude", diskHash: "h", applySourceRevisions: REVS }));
    expect(link.status).toBe(400);
    expect(await link.json()).toEqual({
      ok: false, code: "mutation-rejected", effect: "not-applied", audit: "recorded",
      error: "refused: symlink in target path",
    });

    testDb.close();
    testDb = openDb(":memory:");
    vi.mocked(getRuleDocs).mockReturnValue(RULE_DOCS);
    vi.mocked(applyRuleFile).mockReturnValue({ ok: false, reason: "fs-error", error: "ENOSPC raw" });
    const fsErr = await applyPost(request({ app: "claude", diskHash: "h", applySourceRevisions: REVS }));
    expect(fsErr.status).toBe(500);
    const fsBody = await fsErr.json();
    expect(fsBody).toMatchObject({
      ok: false, code: "mutation-unconfirmed", effect: "unconfirmed", audit: "recorded",
    });
    expect(JSON.stringify(fsBody)).not.toContain("ENOSPC raw");
    expect(JSON.parse(rows()[0].payload).outcome).toBe("abandoned");
  });

  it("rules apply: insert failure is 500 and does not apply; finalization failure keeps pending and backup", async () => {
    vi.mocked(getRuleDocs).mockReturnValue(RULE_DOCS);
    failInsert();
    const insertFail = await applyPost(request({ app: "claude", diskHash: "h", applySourceRevisions: REVS }));
    expect(insertFail.status).toBe(500);
    expect(await insertFail.json()).toMatchObject({
      ok: false, code: "audit-unavailable", effect: "not-applied", audit: "unavailable",
    });
    expect(applyRuleFile).not.toHaveBeenCalled();
    expect(rows()).toEqual([]);

    testDb.close();
    testDb = openDb(":memory:");
    vi.mocked(getRuleDocs).mockReturnValue(RULE_DOCS);
    vi.mocked(applyRuleFile).mockReturnValue({ ok: true, bytes: 9, backup: ".bak-keep" });
    failUpdate();
    const fin = await applyPost(request({ app: "claude", diskHash: "h", applySourceRevisions: REVS }));
    expect(fin.status).toBe(500);
    const body = await fin.json();
    expect(body).toMatchObject({
      ok: false, code: "audit-finalization-failed", effect: "applied", audit: "pending",
      data: { bytes: 9, backup: ".bak-keep" },
    });
    expect(body).not.toHaveProperty("status");
    expect(body).not.toHaveProperty("value");
    expect(rows()[0].ok).toBeNull();
  });
});

describe("agent settings mutation lifecycle", () => {
  const REV = { settings: "absent", sources: { "opencode.json": "absent", "opencode.jsonc": "absent" } };
  const OP = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee";

  beforeEach(() => {
    testDb = openDb(":memory:");
  });
  afterEach(() => {
    testDb.close();
  });

  it("settings save: insert failure skips apply; finalization keeps published effect", async () => {
    const apply = vi.fn(async () => ({ effect: "published", editor_revision: REV, settings: sample }));
    const deps = {
      snapshot: vi.fn(),
      preview: vi.fn(),
      apply,
      syncStatus: vi.fn(),
      sanitize: vi.fn(),
    };
    failInsert();
    expect(await (await settingsPost(request({
      action: "save", expected: REV, reviewers: sample.reviewers, builders: sample.builders, operation_id: OP,
    }), deps)).json()).toMatchObject({ code: "audit-unavailable", effect: "not-applied" });
    expect(apply).not.toHaveBeenCalled();

    testDb.close();
    testDb = openDb(":memory:");
    failUpdate();
    const body = await (await settingsPost(request({
      action: "save", expected: REV, reviewers: sample.reviewers, builders: sample.builders, operation_id: OP,
    }), deps)).json();
    expect(body).toMatchObject({
      ok: false, code: "audit-finalization-failed", effect: "applied", audit: "pending",
    });
    expect(body).not.toHaveProperty("status");
    expect(body).not.toHaveProperty("value");
    expect(body.data).toMatchObject({ effect: "published" });
  });

  it("providers save-connection: applied-but-audit-pending is not a generic failed save", async () => {
    failUpdate();
    const body = await (await providersPost(request({
      action: "save-connection",
      expected: REV,
      connection: { id: "acme", adapter: "openai-compatible", base_url: "https://api.example.com" },
      operation_id: OP,
    }), {
      snapshot: vi.fn(),
      preview: vi.fn(),
      apply: vi.fn(async () => ({ effect: "published", editor_revision: REV, settings: null })),
      sanitize: vi.fn(),
    })).json();
    expect(body).toMatchObject({
      ok: false, code: "audit-finalization-failed", effect: "applied", audit: "pending",
    });
    expect(body.data).toMatchObject({ effect: "published" });
  });
});
