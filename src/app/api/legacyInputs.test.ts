import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from "vitest";
import type Database from "better-sqlite3";

const HASH = "ab".repeat(32);
const MOCK_TRASH_REL = ".jax-trash/20260101T000000Z/a.txt";

// A real temp repos root with a "jax-os" dir: the search tests' real parseScope reads the disk.
const settingsHarness = vi.hoisted(() => {
  const { mkdirSync, mkdtempSync } = require("node:fs");
  const { tmpdir } = require("node:os");
  const { join } = require("node:path");
  const reposRoot = mkdtempSync(join(tmpdir(), "jaxos-legacy-repos-"));
  mkdirSync(join(reposRoot, "jax-os"));
  return { vault: true, vaultPath: "/tmp/vault" as string | null, linear: true, reposRoot };
});
vi.mock("../../server/settings", () => ({
  readGeneralSettings: () => ({
    ok: true,
    data: {
      reposRoot: settingsHarness.reposRoot,
      vaultPath: settingsHarness.vaultPath,
      integrations: { vault: settingsHarness.vault, linear: settingsHarness.linear },
    },
  }),
}));

vi.mock("../../server/collectors/files", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../server/collectors/files")>()),
  createEntry: vi.fn(),
  strictRel: vi.fn(() => true),
  trashEntry: vi.fn(() => ({ trashRel: MOCK_TRASH_REL })),
  sweepTrash: vi.fn(() => []),
  removeExpiredTrashEntryAt: vi.fn(),
  renameEntry: vi.fn(),
  writeFile: vi.fn(() => ({ hash: HASH })),
  uploadFile: vi.fn(),
  searchByName: vi.fn(async () => ({ hits: [], truncated: false })),
}));

vi.mock("../../server/collectors/linear", () => ({
  updateIssue: vi.fn(),
  moveIssue: vi.fn(),
  createComment: vi.fn(),
  createIssue: vi.fn(),
  getBoard: vi.fn(() => ({})),
  getIssueDetail: vi.fn(() => ({})),
  getLabels: vi.fn(() => []),
  getMembers: vi.fn(() => []),
  getTeams: vi.fn(() => []),
  getProjectCounts: vi.fn(() => ({})),
}));

vi.mock("../../server/collectors/tmux", () => ({
  getAgents: vi.fn(() => [{ session: "jax-test" }]),
}));

let testDb: Database.Database;
vi.mock("../../server/db", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../server/db")>();
  return { ...actual, getDb: () => testDb };
});

import {
  COMMENT_CAP,
  COMMENT_JSON_CAP,
  SMALL_JSON_CAP,
  WRITE_JSON_CAP,
} from "../../server/inputLimits";
import { READ_CAP, UPLOAD_CAP, createEntry, renameEntry, searchByName, trashEntry, uploadFile, writeFile } from "../../server/collectors/files";
import { createComment, createIssue, getBoard, getIssueDetail, getLabels, getMembers, getProjectCounts, getTeams, moveIssue, updateIssue } from "../../server/collectors/linear";
import { openDb } from "../../server/db";
import { getAgents } from "../../server/collectors/tmux";
import { POST as createPost } from "./files/create/route";
import { POST as deletePost } from "./files/delete/route";
import { POST as renamePost } from "./files/rename/route";
import { POST as writePost } from "./files/write/route";
import { GET as searchGet } from "./files/search/route";
import { POST as movePost } from "./kanban/move/route";
import { POST as issueCreatePost } from "./kanban/create/route";
import { POST as updatePost } from "./kanban/update/route";
import { POST as commentPost } from "./kanban/comment/route";
import { POST as takePost } from "./tmux/take-control/route";
import { POST as releasePost } from "./tmux/release-control/route";
import { POST as uploadPost } from "./files/upload/route";
import { GET as boardGet } from "./kanban/board/route";
import { GET as issueGet } from "./kanban/issue/route";
import { GET as labelsGet } from "./kanban/labels/route";
import { GET as membersGet } from "./kanban/members/route";
import { GET as teamsGet } from "./kanban/teams/route";
import { GET as projectCountsGet } from "./kanban/project-counts/route";

const request = (payload: unknown, headers: Record<string, string> = {}) =>
  new Request("http://127.0.0.1/api/test", {
    method: "POST", body: JSON.stringify(payload),
    headers: { "content-type": "application/json", ...headers },
  });

const rawPost = (body: string, headers: Record<string, string> = {}) =>
  new Request("http://127.0.0.1/api/test", {
    method: "POST",
    body,
    headers: { "content-type": "application/json", ...headers },
  });

const dishonestPost = (byteLength: number) =>
  new Request("http://127.0.0.1/api/test", {
    method: "POST",
    body: new ReadableStream<Uint8Array>({
      start(c) {
        c.enqueue(new Uint8Array(byteLength).fill(120));
        c.close();
      },
    }),
    headers: { "content-type": "application/json", "content-length": "8" },
    duplex: "half",
  } as RequestInit);

function mutationCount() {
  return (testDb.prepare("SELECT COUNT(*) AS n FROM mutations").get() as { n: number }).n;
}

function lastMutationKind(): string {
  const row = testDb.prepare("SELECT payload FROM mutations ORDER BY id DESC LIMIT 1").get() as { payload: string };
  return JSON.parse(row.payload).kind;
}

function resetEffects() {
  vi.clearAllMocks();
  testDb.close();
  testDb = openDb(":memory:");
}

async function refused(res: Response, effect?: object) {
  expect(res.status).toBe(200);
  expect((await res.json()).ok).toBe(false);
  if (effect) expect(effect as Mock).not.toHaveBeenCalled();
  expect(mutationCount()).toBe(0);
}

describe("legacy mutation input caps", () => {
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

  describe("files/create", () => {
    const valid = { root: "repos", relParentDir: "", basename: "a.txt", kind: "file" };
    it("creates and audits once", async () => {
      const res = await createPost(request(valid));
      expect(res.status).toBe(200);
      expect(await res.json()).toEqual({ ok: true });
      expect(createEntry).toHaveBeenCalledOnce();
      expect(createEntry).toHaveBeenCalledWith("repos", "", "a.txt", "file");
      expect(mutationCount()).toBe(1);
    });
    it("rejects invalid schema before any effect", async () => {
      await refused(await createPost(request({ ...valid, kind: "symlink" })), createEntry);
    });
    it("rejects a 256-byte basename and accepts 255", async () => {
      await refused(await createPost(request({ ...valid, basename: "a".repeat(256) })), createEntry);
      resetEffects();
      expect((await (await createPost(request({ ...valid, basename: "a".repeat(255) }))).json()).ok).toBe(true);
      expect(createEntry).toHaveBeenCalledOnce();
    });
    it("rejects an oversized body and a dishonest Content-Length", async () => {
      await refused(await createPost(rawPost("x".repeat(SMALL_JSON_CAP + 1))), createEntry);
      await refused(await createPost(dishonestPost(SMALL_JSON_CAP + 1)), createEntry);
    });
    it("rejects same-site", async () => {
      await refused(await createPost(request(valid, { "sec-fetch-site": "same-site" })), createEntry);
    });
  });

  describe("files/delete", () => {
    const valid = { root: "repos", rel: "a.txt" };
    it("trashes and audits once", async () => {
      expect(await (await deletePost(request(valid))).json()).toEqual({ ok: true, trashRel: MOCK_TRASH_REL });
      expect(trashEntry).toHaveBeenCalledWith("repos", "a.txt");
      expect(mutationCount()).toBe(1);
    });
    it("rejects empty rel and NUL", async () => {
      await refused(await deletePost(request({ ...valid, rel: "" })), trashEntry);
      await refused(await deletePost(request({ ...valid, rel: "a\0b" })), trashEntry);
    });
    it("accepts a 4096-byte rel and rejects 4097", async () => {
      expect((await (await deletePost(request({ ...valid, rel: "a".repeat(4096) }))).json()).ok).toBe(true);
      resetEffects();
      await refused(await deletePost(request({ ...valid, rel: "a".repeat(4097) })), trashEntry);
    });
    it("rejects an oversized body, dishonest Content-Length, and same-site", async () => {
      await refused(await deletePost(rawPost("x".repeat(SMALL_JSON_CAP + 1))), trashEntry);
      await refused(await deletePost(dishonestPost(SMALL_JSON_CAP + 1)), trashEntry);
      await refused(await deletePost(request(valid, { "sec-fetch-site": "same-site" })), trashEntry);
    });
  });

  describe("files/rename", () => {
    const valid = { root: "repos", relFrom: "a.txt", relTo: "b.txt" };
    it("renames and audits once", async () => {
      expect((await (await renamePost(request(valid))).json()).ok).toBe(true);
      expect(renameEntry).toHaveBeenCalledWith("repos", "a.txt", "b.txt");
      expect(mutationCount()).toBe(1);
    });
    it("rejects a traversal basename", async () => {
      await refused(await renamePost(request({ ...valid, relTo: "dir/.." })), renameEntry);
    });
    it("rejects an oversized body, dishonest Content-Length, and same-site", async () => {
      await refused(await renamePost(rawPost("x".repeat(SMALL_JSON_CAP + 1))), renameEntry);
      await refused(await renamePost(dishonestPost(SMALL_JSON_CAP + 1)), renameEntry);
      await refused(await renamePost(request(valid, { "sec-fetch-site": "same-site" })), renameEntry);
    });
  });

  describe("files/write", () => {
    const valid = { root: "repos", rel: "x.txt", content: "x", baseHash: HASH };
    it("writes empty content and audits once", async () => {
      const res = await writePost(request({ ...valid, content: "" }));
      expect(await res.json()).toEqual({ ok: true, hash: HASH });
      expect(writeFile).toHaveBeenCalledWith("repos", "x.txt", "", HASH);
      expect(mutationCount()).toBe(1);
    });
    it("refuses a malformed save hash before any effect", async () => {
      const response = await writePost(request({ root: "repos", rel: "x.txt", content: "x", baseHash: "bad" }));
      expect(response.status).toBe(200);
      expect((await response.json()).ok).toBe(false);
      expect(writeFile).not.toHaveBeenCalled();
      expect(mutationCount()).toBe(0);
    });
    it("lowercases an accepted uppercase hash", async () => {
      expect((await (await writePost(request({ ...valid, baseHash: "AB".repeat(32) }))).json()).ok).toBe(true);
      expect(writeFile).toHaveBeenCalledWith("repos", "x.txt", "x", HASH);
    });
    it("dispatches maximum-valid content and rejects one extra byte", async () => {
      const content = "\u0001".repeat(READ_CAP);
      const rel = "a".repeat(4096);
      expect(Buffer.byteLength(content)).toBe(READ_CAP);
      const payload = { root: "repos", rel, content, baseHash: HASH };
      const serialized = JSON.stringify(payload);
      expect(Buffer.byteLength(serialized)).toBeLessThan(WRITE_JSON_CAP);
      const res = await writePost(request(payload));
      expect((await res.json()).ok).toBe(true);
      expect(writeFile).toHaveBeenCalledWith("repos", rel, content, HASH);
      resetEffects();
      await refused(await writePost(request({ ...valid, content: content + "x" })), writeFile);
    });
    it("rejects an oversized body, dishonest Content-Length, and same-site", async () => {
      await refused(await writePost(dishonestPost(WRITE_JSON_CAP + 1)), writeFile);
      await refused(await writePost(request(valid, { "sec-fetch-site": "same-site" })), writeFile);
    });
  });

  describe("files/search", () => {
    it("searches a valid query", async () => {
      const req = new Request("http://127.0.0.1/api/files/search?q=foo");
      const res = await searchGet(req);
      expect(await res.json()).toEqual({ ok: true, data: [], truncated: false });
      expect(searchByName).toHaveBeenCalledWith("foo", { kind: "all" }, req.signal);
    });
    it("keeps an empty query as quiet empty", async () => {
      const req = new Request("http://127.0.0.1/api/files/search?q=");
      const res = await searchGet(req);
      expect(await res.json()).toEqual({ ok: true, data: [], truncated: false });
      expect(searchByName).toHaveBeenCalledWith("", { kind: "all" }, req.signal);
    });
    it("rejects a 257-byte query before search", async () => {
      const res = await searchGet(new Request(`http://127.0.0.1/api/files/search?q=${"a".repeat(257)}`));
      expect(res.status).toBe(200);
      expect(await res.json()).toEqual({ ok: false, error: "invalid payload" });
      expect(searchByName).not.toHaveBeenCalled();
    });
  });

  describe("kanban/move", () => {
    const valid = { issueId: "iss1", stateId: "st1" };
    it("moves and audits once", async () => {
      expect((await (await movePost(request(valid))).json()).ok).toBe(true);
      expect(moveIssue).toHaveBeenCalledWith("iss1", "st1");
      expect(mutationCount()).toBe(1);
    });
    it("rejects a missing stateId", async () => {
      await refused(await movePost(request({ issueId: "iss1" })), moveIssue);
    });
    it("rejects an oversized body, dishonest Content-Length, and same-site", async () => {
      await refused(await movePost(rawPost("x".repeat(SMALL_JSON_CAP + 1))), moveIssue);
      await refused(await movePost(dishonestPost(SMALL_JSON_CAP + 1)), moveIssue);
      await refused(await movePost(request(valid, { "sec-fetch-site": "same-site" })), moveIssue);
    });
  });

  describe("kanban/update", () => {
    it("accepts each allowed field", async () => {
      expect((await (await updatePost(request({ issueId: "iss1", patch: { priority: 0 } }))).json()).ok).toBe(true);
      expect(updateIssue).toHaveBeenCalledWith("iss1", { priority: 0 });
      resetEffects();
      expect((await (await updatePost(request({ issueId: "iss1", patch: { assigneeId: null } }))).json()).ok).toBe(true);
      expect(updateIssue).toHaveBeenCalledWith("iss1", { assigneeId: null });
      resetEffects();
      expect((await (await updatePost(request({ issueId: "iss1", patch: { labelIds: [] } }))).json()).ok).toBe(true);
      expect(updateIssue).toHaveBeenCalledWith("iss1", { labelIds: [] });
      expect(mutationCount()).toBe(1);
    });
    it("accepts 100 labels and rejects 101", async () => {
      const hundred = Array.from({ length: 100 }, (_, i) => `id${i}`);
      expect((await (await updatePost(request({ issueId: "iss1", patch: { labelIds: hundred } }))).json()).ok).toBe(true);
      resetEffects();
      await refused(await updatePost(request({ issueId: "iss1", patch: { labelIds: [...hundred, "x"] } })), updateIssue);
    });
    it("accepts title and description as sole fields, audited under the matching mutation kind", async () => {
      expect((await (await updatePost(request({ issueId: "iss1", patch: { title: "New title" } }))).json()).ok).toBe(true);
      expect(updateIssue).toHaveBeenCalledWith("iss1", { title: "New title" });
      expect(lastMutationKind()).toBe("linear-title"); // F6
      resetEffects();
      expect((await (await updatePost(request({ issueId: "iss1", patch: { description: "" } }))).json()).ok).toBe(true);
      expect(updateIssue).toHaveBeenCalledWith("iss1", { description: "" });
      expect(lastMutationKind()).toBe("linear-description"); // F6
    });
    it("rejects an empty title and a 256-char title, accepts 255", async () => {
      await refused(await updatePost(request({ issueId: "iss1", patch: { title: "" } })), updateIssue);
      await refused(await updatePost(request({ issueId: "iss1", patch: { title: "a".repeat(256) } })), updateIssue);
      expect((await (await updatePost(request({ issueId: "iss1", patch: { title: "a".repeat(255) } }))).json()).ok).toBe(true);
    });
    it("rejects an over-cap description", async () => {
      await refused(await updatePost(request({ issueId: "iss1", patch: { description: "x".repeat(COMMENT_CAP + 1) } })), updateIssue);
    });
    it("rejects unsupported, multiple, array, and bad values", async () => {
      await refused(await updatePost(request({ issueId: "iss1", patch: { labels: [] } })), updateIssue);
      await refused(await updatePost(request({ issueId: "iss1", patch: { priority: 1, assigneeId: null } })), updateIssue);
      await refused(await updatePost(request({ issueId: "iss1", patch: [] })), updateIssue);
      await refused(await updatePost(request({ issueId: "iss1", patch: { priority: 1.5 } })), updateIssue);
      await refused(await updatePost(request({ issueId: "iss1", patch: { priority: 5 } })), updateIssue);
      await refused(await updatePost(request({ issueId: "iss1", patch: { labelIds: [{}] } })), updateIssue);
    });
    it("rejects an oversized body, dishonest Content-Length, and same-site", async () => {
      await refused(await updatePost(rawPost("x".repeat(SMALL_JSON_CAP + 1))), updateIssue);
      await refused(await updatePost(dishonestPost(SMALL_JSON_CAP + 1)), updateIssue);
      await refused(await updatePost(request({ issueId: "iss1", patch: { priority: 1 } }, { "sec-fetch-site": "same-site" })), updateIssue);
    });
  });

  describe("kanban/comment", () => {
    const valid = { issueId: "iss1", body: "  hello" };
    it("comments without trimming submitted text", async () => {
      expect((await (await commentPost(request(valid))).json()).ok).toBe(true);
      expect(createComment).toHaveBeenCalledWith("iss1", "  hello");
      expect(mutationCount()).toBe(1);
    });
    it("accepts a 32 KiB multibyte comment and rejects one extra byte", async () => {
      const body = "é".repeat(COMMENT_CAP / 2);
      expect((await (await commentPost(request({ issueId: "iss1", body }))).json()).ok).toBe(true);
      expect(createComment).toHaveBeenCalledWith("iss1", body);
      resetEffects();
      await refused(await commentPost(request({ issueId: "iss1", body: body + "x" })), createComment);
    });
    it("rejects whitespace-only text", async () => {
      await refused(await commentPost(request({ issueId: "iss1", body: "  " })), createComment);
    });
    it("rejects an oversized JSON body, dishonest Content-Length, and same-site", async () => {
      await refused(await commentPost(rawPost("x".repeat(COMMENT_JSON_CAP + 1))), createComment);
      await refused(await commentPost(dishonestPost(COMMENT_JSON_CAP + 1)), createComment);
      await refused(await commentPost(request(valid, { "sec-fetch-site": "same-site" })), createComment);
    });
  });

  describe("kanban/create", () => {
    const valid = { teamId: "t1", title: "New issue" };
    it("creates with only the required fields and audits once", async () => {
      vi.mocked(createIssue).mockResolvedValue({ id: "i9", identifier: "MOA-9" });
      const res = await issueCreatePost(request(valid));
      expect(await res.json()).toEqual({ ok: true, id: "i9", identifier: "MOA-9" });
      expect(createIssue).toHaveBeenCalledWith(valid);
      expect(mutationCount()).toBe(1);
    });
    it("passes every optional field through untouched", async () => {
      vi.mocked(createIssue).mockResolvedValue({ id: "i9", identifier: "MOA-9" });
      const full = { ...valid, description: "d", projectId: "p1", stateId: "s1", priority: 2, assigneeId: "u1", labelIds: ["l1"] };
      await issueCreatePost(request(full));
      expect(createIssue).toHaveBeenCalledWith(full);
    });
    it("rejects a missing title or teamId before any effect", async () => {
      await refused(await issueCreatePost(request({ teamId: "t1" })), createIssue);
      await refused(await issueCreatePost(request({ title: "x" })), createIssue);
    });
    it("rejects an empty title and a 256-char title, accepts 255", async () => {
      await refused(await issueCreatePost(request({ ...valid, title: "" })), createIssue);
      await refused(await issueCreatePost(request({ ...valid, title: "a".repeat(256) })), createIssue);
      vi.mocked(createIssue).mockResolvedValue({ id: "i9", identifier: "MOA-9" });
      expect((await (await issueCreatePost(request({ ...valid, title: "a".repeat(255) }))).json()).ok).toBe(true);
    });
    it("rejects an over-cap description, bad projectId/stateId/assigneeId types, out-of-range priority, and bad labelIds", async () => {
      await refused(await issueCreatePost(request({ ...valid, description: "x".repeat(COMMENT_CAP + 1) })), createIssue);
      await refused(await issueCreatePost(request({ ...valid, projectId: 5 })), createIssue);
      await refused(await issueCreatePost(request({ ...valid, stateId: 5 })), createIssue);
      await refused(await issueCreatePost(request({ ...valid, assigneeId: 5 })), createIssue);
      await refused(await issueCreatePost(request({ ...valid, priority: 5 })), createIssue);
      await refused(await issueCreatePost(request({ ...valid, priority: 1.5 })), createIssue);
      await refused(await issueCreatePost(request({ ...valid, labelIds: [...Array.from({ length: 100 }, (_, i) => `id${i}`), "x"] })), createIssue);
    });
    it("accepts null projectId/assigneeId (clears, does not reject)", async () => {
      vi.mocked(createIssue).mockResolvedValue({ id: "i9", identifier: "MOA-9" });
      const res = await issueCreatePost(request({ ...valid, projectId: null, assigneeId: null }));
      expect((await res.json()).ok).toBe(true);
      expect(createIssue).toHaveBeenCalledWith({ ...valid, projectId: null, assigneeId: null });
    });
    it("rejects an oversized body, dishonest Content-Length, and same-site", async () => {
      await refused(await issueCreatePost(rawPost("x".repeat(COMMENT_JSON_CAP + 1))), createIssue);
      await refused(await issueCreatePost(dishonestPost(COMMENT_JSON_CAP + 1)), createIssue);
      await refused(await issueCreatePost(request(valid, { "sec-fetch-site": "same-site" })), createIssue);
    });
  });

  describe("tmux/take-control", () => {
    it("grants control after getAgents", async () => {
      expect((await (await takePost(request({ session: "jax-test" }))).json()).ok).toBe(true);
      expect(getAgents).toHaveBeenCalledOnce();
      expect(mutationCount()).toBe(1);
    });
    it("rejects an invalid session before getAgents", async () => {
      await refused(await takePost(request({ session: "" })), getAgents);
    });
    it("rejects an oversized body, dishonest Content-Length, and same-site", async () => {
      await refused(await takePost(rawPost("x".repeat(SMALL_JSON_CAP + 1))), getAgents);
      await refused(await takePost(dishonestPost(SMALL_JSON_CAP + 1)), getAgents);
      await refused(await takePost(request({ session: "jax-test" }, { "sec-fetch-site": "same-site" })), getAgents);
    });
  });

  describe("tmux/release-control", () => {
    it("releases and audits once", async () => {
      expect((await (await releasePost(request({ session: "jax-test" }))).json()).ok).toBe(true);
      expect(mutationCount()).toBe(1);
    });
    it("rejects a missing session before audit", async () => {
      await refused(await releasePost(request({})));
    });
    it("rejects an oversized body, dishonest Content-Length, and same-site", async () => {
      const res = await releasePost(rawPost("x".repeat(SMALL_JSON_CAP + 1)));
      expect(res.status).toBe(200);
      expect((await res.json()).ok).toBe(false);
      expect(mutationCount()).toBe(0);
      await refused(await releasePost(dishonestPost(SMALL_JSON_CAP + 1)));
      await refused(await releasePost(request({ session: "jax-test" }, { "sec-fetch-site": "same-site" })));
    });
  });

  describe("files/upload", () => {
    const cap = UPLOAD_CAP + 64 * 1024;

    async function multipartRequest(fields: Record<string, string | File>, headers: Record<string, string> = {}) {
      const form = new FormData();
      for (const [k, v] of Object.entries(fields)) form.set(k, v);
      const tmp = new Request("http://127.0.0.1/upload", { method: "POST", body: form });
      const bytes = await tmp.arrayBuffer();
      return new Request("http://127.0.0.1/upload", {
        method: "POST",
        body: bytes,
        headers: {
          "content-type": tmp.headers.get("content-type") ?? "",
          "content-length": String(bytes.byteLength),
          ...headers,
        },
      });
    }

    async function noReadNoParse(req: Request) {
      const readSpy = req.body ? vi.spyOn(req.body, "getReader") : null;
      const parseSpy = vi.spyOn(Response.prototype, "formData");
      try {
        const res = await uploadPost(req);
        expect(res.status).toBe(200);
        expect((await res.json()).ok).toBe(false);
        expect(readSpy?.mock.calls ?? []).toHaveLength(0);
        expect(parseSpy).not.toHaveBeenCalled();
        expect(uploadFile).not.toHaveBeenCalled();
        expect(mutationCount()).toBe(0);
      } finally {
        readSpy?.mockRestore();
        parseSpy.mockRestore();
      }
    }

    it("uploads exact small bytes once", async () => {
      const bytes = new Uint8Array([1, 2, 3, 4]);
      const res = await uploadPost(await multipartRequest({
        root: "repos",
        relParentDir: "",
        file: new File([bytes], "a.bin"),
      }));
      expect(res.status).toBe(200);
      expect(await res.json()).toEqual({ ok: true });
      expect(uploadFile).toHaveBeenCalledOnce();
      const passed = vi.mocked(uploadFile).mock.calls[0][3] as Buffer;
      expect(Buffer.compare(passed, Buffer.from(bytes))).toBe(0);
      expect(mutationCount()).toBe(1);
    });

    it("accepts an empty file and a maximum-size file", async () => {
      expect((await (await uploadPost(await multipartRequest({
        root: "repos",
        file: new File([], "empty.txt"),
      }))).json()).ok).toBe(true);
      resetEffects();
      const max = new Uint8Array(UPLOAD_CAP).fill(7);
      expect((await (await uploadPost(await multipartRequest({
        root: "repos",
        file: new File([max], "max.bin"),
      }))).json()).ok).toBe(true);
      expect(uploadFile).toHaveBeenCalledOnce();
      expect((vi.mocked(uploadFile).mock.calls[0][3] as Buffer).length).toBe(UPLOAD_CAP);
    });

    it("rejects an over-limit file inside the multipart body cap", async () => {
      const parseSpy = vi.spyOn(Response.prototype, "formData");
      try {
        const res = await uploadPost(await multipartRequest({
          root: "repos",
          file: new File([new Uint8Array(UPLOAD_CAP + 1)], "big.bin"),
        }));
        expect((await res.json()).ok).toBe(false);
        expect(parseSpy).toHaveBeenCalled();
        expect(uploadFile).not.toHaveBeenCalled();
        expect(mutationCount()).toBe(0);
      } finally {
        parseSpy.mockRestore();
      }
    });

    it("rejects missing, zero, negative, fractional, non-numeric, and over-cap Content-Length without reading", async () => {
      const fileReq = await multipartRequest({ root: "repos", file: new File([new Uint8Array([1])], "a.bin") });
      const type = fileReq.headers.get("content-type") ?? "";
      const body = await fileReq.arrayBuffer();
      await noReadNoParse(new Request("http://127.0.0.1/upload", { method: "POST", body, headers: { "content-type": type } }));
      await noReadNoParse(new Request("http://127.0.0.1/upload", { method: "POST", body, headers: { "content-type": type, "content-length": "0" } }));
      await noReadNoParse(new Request("http://127.0.0.1/upload", { method: "POST", body, headers: { "content-type": type, "content-length": "-1" } }));
      await noReadNoParse(new Request("http://127.0.0.1/upload", { method: "POST", body, headers: { "content-type": type, "content-length": "1.5" } }));
      await noReadNoParse(new Request("http://127.0.0.1/upload", { method: "POST", body, headers: { "content-type": type, "content-length": "abc" } }));
      await noReadNoParse(new Request("http://127.0.0.1/upload", { method: "POST", body, headers: { "content-type": type, "content-length": String(cap + 1) } }));
    });

    it("cancels a dishonest small header plus over-cap bytes and does not parse", async () => {
      let cancelled = false;
      const parseSpy = vi.spyOn(Response.prototype, "formData");
      const stream = new ReadableStream<Uint8Array>(
        {
          pull(controller) {
            controller.enqueue(new Uint8Array(cap + 1).fill(120));
          },
          cancel() {
            cancelled = true;
          },
        },
        { highWaterMark: 0 },
      );
      try {
        const res = await uploadPost(new Request("http://127.0.0.1/upload", {
          method: "POST",
          body: stream,
          headers: { "content-type": "multipart/form-data; boundary=x", "content-length": "100" },
          duplex: "half",
        } as RequestInit));
        expect((await res.json()).ok).toBe(false);
        expect(cancelled).toBe(true);
        expect(parseSpy).not.toHaveBeenCalled();
        expect(uploadFile).not.toHaveBeenCalled();
        expect(mutationCount()).toBe(0);
      } finally {
        parseSpy.mockRestore();
      }
    });

    it("lets an exact total cap through the reader and rejects cap+1", async () => {
      const parseSpy = vi.spyOn(Response.prototype, "formData");
      try {
        const exact = await uploadPost(new Request("http://127.0.0.1/upload", {
          method: "POST",
          body: new Uint8Array(cap),
          headers: { "content-type": "multipart/form-data; boundary=x", "content-length": String(cap) },
        }));
        expect(await exact.json()).toEqual({ ok: false, error: "invalid payload" });
        expect(parseSpy).toHaveBeenCalled();
        parseSpy.mockClear();
        await noReadNoParse(new Request("http://127.0.0.1/upload", {
          method: "POST",
          body: new Uint8Array(cap + 1),
          headers: { "content-type": "multipart/form-data; boundary=x", "content-length": String(cap + 1) },
        }));
      } finally {
        parseSpy.mockRestore();
      }
    });

    it("rejects malformed multipart, file-valued fields, invalid basename, and missing file", async () => {
      const malformed = await uploadPost(new Request("http://127.0.0.1/upload", {
        method: "POST",
        body: "not-multipart",
        headers: { "content-type": "multipart/form-data; boundary=x", "content-length": "13" },
      }));
      expect(await malformed.json()).toEqual({ ok: false, error: "invalid payload" });
      expect(uploadFile).not.toHaveBeenCalled();

      const fileAsRoot = new File([new Uint8Array([1])], "a.bin");
      await refused(await uploadPost(await multipartRequest({ root: fileAsRoot, file: fileAsRoot })), uploadFile);
      await refused(await uploadPost(await multipartRequest({ root: "repos", relParentDir: fileAsRoot, file: fileAsRoot })), uploadFile);
      await refused(await uploadPost(await multipartRequest({ root: "repos", file: new File([new Uint8Array([1])], "a/b.bin") })), uploadFile);

      const form = new FormData();
      form.set("root", "repos");
      const tmp = new Request("http://127.0.0.1/upload", { method: "POST", body: form });
      const bytes = await tmp.arrayBuffer();
      const missing = await uploadPost(new Request("http://127.0.0.1/upload", {
        method: "POST",
        body: bytes,
        headers: {
          "content-type": tmp.headers.get("content-type") ?? "",
          "content-length": String(bytes.byteLength),
        },
      }));
      expect((await missing.json()).ok).toBe(false);
      expect(uploadFile).not.toHaveBeenCalled();
    });

    it("maps collector EEXIST", async () => {
      vi.mocked(uploadFile).mockImplementationOnce(() => {
        const err = new Error("exists") as NodeJS.ErrnoException;
        err.code = "EEXIST";
        throw err;
      });
      const res = await uploadPost(await multipartRequest({
        root: "repos",
        file: new File([new Uint8Array([1])], "a.bin"),
      }));
        expect(await res.json()).toEqual({
          ok: false, code: "mutation-rejected", effect: "not-applied", audit: "recorded", error: "already exists",
        });
        expect(mutationCount()).toBe(1);
    });

    it("rejects same-site", async () => {
      await refused(await uploadPost(await multipartRequest(
        { root: "repos", file: new File([new Uint8Array([1])], "a.bin") },
        { "sec-fetch-site": "same-site" },
      )), uploadFile);
    });
  });
});

describe("kanban routes — linear gate (spec per-integration table)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    settingsHarness.vault = true;
    settingsHarness.vaultPath = "/tmp/vault";
    settingsHarness.linear = true;
    testDb = openDb(":memory:");
  });
  afterEach(() => {
    testDb.close();
  });

  it("refuses every kanban route with disabled and never calls its collector", async () => {
    settingsHarness.linear = false;
    try {
      const cases: [string, Promise<Response>, Mock][] = [
        ["board", boardGet(new Request("http://127.0.0.1/api/kanban/board?team=t1")), vi.mocked(getBoard)],
        ["issue", issueGet(new Request("http://127.0.0.1/api/kanban/issue?id=i1")), vi.mocked(getIssueDetail)],
        ["move", movePost(request({ issueId: "iss1", stateId: "st1" })), vi.mocked(moveIssue)],
        ["comment", commentPost(request({ issueId: "iss1", body: "hello" })), vi.mocked(createComment)],
        ["create", issueCreatePost(request({ teamId: "t1", title: "T" })), vi.mocked(createIssue)],
        ["labels", labelsGet(new Request("http://127.0.0.1/api/kanban/labels?team=t1")), vi.mocked(getLabels)],
        ["members", membersGet(new Request("http://127.0.0.1/api/kanban/members?team=t1")), vi.mocked(getMembers)],
        ["teams", teamsGet(), vi.mocked(getTeams)],
        ["project-counts", projectCountsGet(), vi.mocked(getProjectCounts)],
        ["update", updatePost(request({ issueId: "iss1", patch: { title: "T" } })), vi.mocked(updateIssue)],
      ];
      for (const [label, res, spy] of cases) {
        expect(await (await res).json(), label).toEqual({ ok: false, error: "disabled" });
        expect(spy, label).not.toHaveBeenCalled();
      }
      expect(mutationCount()).toBe(0);
    } finally {
      settingsHarness.linear = true;
    }
  });

  it("still serves a kanban GET with linear on (regression)", async () => {
    const res = await boardGet(new Request("http://127.0.0.1/api/kanban/board?team=t1"));
    expect((await res.json()).ok).toBe(true);
    expect(getBoard).toHaveBeenCalledWith("t1");
  });
});

describe("bounded GET ids", () => {
  beforeEach(() => {
    vi.mocked(getBoard).mockClear();
    vi.mocked(getIssueDetail).mockClear();
    vi.mocked(getLabels).mockClear();
    vi.mocked(getMembers).mockClear();
    settingsHarness.linear = true;
  });

  const teamRoutes = [
    ["board", boardGet, getBoard],
    ["labels", labelsGet, getLabels],
    ["members", membersGet, getMembers],
  ] as const;

  it.each(teamRoutes)("%s rejects over-cap multibyte team before the collector", async (_name, get, collector) => {
    const res = await get(new Request(`http://127.0.0.1/api?team=${encodeURIComponent("é".repeat(129))}`));
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: false, error: "invalid payload" });
    expect(collector).not.toHaveBeenCalled();
  });

  it.each(teamRoutes)("%s accepts a 256-byte multibyte team", async (_name, get, collector) => {
    const team = "é".repeat(128);
    const res = await get(new Request(`http://127.0.0.1/api?team=${encodeURIComponent(team)}`));
    expect((await res.json()).ok).toBe(true);
    expect(collector).toHaveBeenCalledWith(team);
  });

  it("rejects an over-cap issue id and accepts the 256-byte boundary", async () => {
    const bad = await issueGet(new Request(`http://127.0.0.1/api?id=${encodeURIComponent("é".repeat(129))}`));
    expect(await bad.json()).toEqual({ ok: false, error: "invalid payload" });
    expect(getIssueDetail).not.toHaveBeenCalled();
    const id = "é".repeat(128);
    const ok = await issueGet(new Request(`http://127.0.0.1/api?id=${encodeURIComponent(id)}`));
    expect((await ok.json()).ok).toBe(true);
    expect(getIssueDetail).toHaveBeenCalledWith(id);
  });
});
