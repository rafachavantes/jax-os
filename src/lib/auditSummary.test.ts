import { describe, expect, it } from "vitest";
import { auditStatusKey, mutationSummary } from "./auditSummary";

describe("mutationSummary", () => {
  it("maps each kind family to its real payload shape", () => {
    expect(mutationSummary("file-edit", { root: "repos", rel: "jax-os/CLAUDE.md" })).toBe("repos/jax-os/CLAUDE.md");
    expect(mutationSummary("file-delete", { root: "vault", rel: "note.md" })).toBe("vault/note.md");
    expect(mutationSummary("file-create", { root: "repos", relParentDir: "x", basename: "new.md", entryKind: "file" })).toBe("repos/x/new.md");
    expect(mutationSummary("file-upload", { root: "repos", relParentDir: "", basename: "img.png" })).toBe("repos/img.png");
    expect(mutationSummary("file-rename", { root: "repos", from: "a.md", to: "b.md" })).toBe("a.md → b.md");
    expect(mutationSummary("approval", { project: "jax-os", gate: "build" })).toBe("jax-os · build");
    expect(mutationSummary("take-control", { session: "jax-scratch" })).toBe("jax-scratch");
    expect(mutationSummary("release-control", { session: "jax-scratch" })).toBe("jax-scratch");
  });
  it("keys linear kinds off issueId presence (uuid or identifier)", () => {
    expect(mutationSummary("linear-move", { issueId: "MOA-264", stateId: "s1" })).toBe("MOA-264");
    expect(mutationSummary("linear-comment", { issueId: "0c9e8f2a-uuid" })).toBe("0c9e8f2a-uuid");
  });
  it("falls back to the first payload fields and never throws on junk", () => {
    expect(mutationSummary("weird", { ts: "x", kind: "weird", foo: 1, bar: "b" })).toBe("foo=1 bar=b");
    expect(mutationSummary("weird", null)).toBe("");
    expect(mutationSummary("weird", "raw-string-payload")).toBe("");
  });
});

describe("auditStatusKey", () => {
  it("prefers payload outcome over nullable ok", () => {
    expect(auditStatusKey({ ok: true, payload: { outcome: "pending" } })).toBe("uncertain");
    expect(auditStatusKey({ ok: false, payload: { outcome: "abandoned" } })).toBe("uncertain");
    expect(auditStatusKey({ ok: null, payload: { outcome: "done" } })).toBe("success");
    expect(auditStatusKey({ ok: true, payload: { outcome: "failed" } })).toBe("failure");
  });

  it("keeps the legacy nullable fallback when outcome is absent", () => {
    expect(auditStatusKey({ ok: null, payload: { kind: "take-control" } })).toBe("legacy-none");
    expect(auditStatusKey({ ok: true, payload: { kind: "approval" } })).toBe("legacy-ok");
    expect(auditStatusKey({ ok: false, payload: { error: "nope" } })).toBe("legacy-error");
  });
});
