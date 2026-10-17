import { createElement } from "react";
import { readFileSync } from "node:fs";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { createEdit, tabKey, type EditState } from "@/lib/filesState";

const harness = vi.hoisted(() => ({
  query: {
    data: undefined as unknown,
    isError: false,
    isLoading: false,
    isFetching: false,
    refetch: vi.fn(),
  },
  queryOpts: undefined as unknown,
  edits: {} as Record<string, EditState>,
  locked: false,
  retiredModels: [] as string[],
  expanded: new Set<string>(),
  toggle: vi.fn(),
}));

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en-US",
}));

vi.mock("@tanstack/react-query", () => ({
  useQuery: (opts: unknown) => {
    harness.queryOpts = opts;
    return harness.query;
  },
  useQueryClient: () => ({ setQueryData: vi.fn() }),
}));

vi.mock("next/dynamic", () => ({
  default: () => () => null,
}));

vi.mock("@/components/files/monacoLoader", () => ({}));
vi.mock("@/components/kanban/Markdown", () => ({ Markdown: () => null }));

vi.mock("./FilesWorkspaceProvider", () => ({
  useFilesWorkspace: () => ({
    getEdit: (key: string) => harness.edits[key],
    isLocked: () => harness.locked,
    onSeed: () => {},
    onEdit: () => {},
    onUserEdit: () => {},
    retiredModels: harness.retiredModels,
    expanded: harness.expanded,
    toggle: harness.toggle,
  }),
}));

import { ancestorKeys, applyPollOutcome, csvDelimiterFor, disposeUnattachedModels, FileEditor, pollOutcome, statusChip, syncReadCacheAfterSave, viewerFor, wrapHtmlForPreview } from "./FileEditor";

const file = { root: "repos" as const, rel: "data.bin" };

function render(active = file) {
  return renderToStaticMarkup(createElement(FileEditor, { file: active }));
}

describe("disposeUnattachedModels", () => {
  it("keeps the attached model and still-open tabs, disposing only detached retired paths", () => {
    const disposed: string[] = [];
    const models = new Map([
      ["repos/a.ts", { dispose: () => disposed.push("repos/a.ts") }],
      ["repos/b.ts", { dispose: () => disposed.push("repos/b.ts") }],
      ["repos/old/c.ts", { dispose: () => disposed.push("repos/old/c.ts") }],
    ]);
    disposeUnattachedModels(
      (path) => models.get(path) ?? null,
      ["repos/old/c.ts"],
      "repos/a.ts",
    );
    expect(disposed).toEqual(["repos/old/c.ts"]);
    disposeUnattachedModels(
      (path) => models.get(path) ?? null,
      ["repos/a.ts", "repos/b.ts"],
      "repos/a.ts",
    );
    expect(disposed).toEqual(["repos/old/c.ts", "repos/b.ts"]);
  });
});

// createEdit(savedContent, baseHash) starts clean (content === savedContent);
// override .content after to build a dirty fixture. Mirrors this file's own
// existing convention (see the "shows SVG preview control" test below).
function editAt(content: string, savedContent: string, baseHash: string): EditState {
  return { ...createEdit(savedContent, baseHash), content };
}

describe("pollOutcome", () => {
  it("does nothing when the disk hash matches the edit's baseline", () => {
    expect(pollOutcome("h", editAt("a", "a", "h"))).toEqual({ action: "none" });
    expect(pollOutcome("h", editAt("b", "a", "h"))).toEqual({ action: "none" });
  });

  it("adopts silently when the buffer is clean", () => {
    expect(pollOutcome("h2", editAt("a", "a", "h"))).toEqual({ action: "adopt" });
  });

  it("flags stale without touching the buffer when dirty", () => {
    expect(pollOutcome("h2", editAt("b", "a", "h"))).toEqual({ action: "stale" });
  });
});

// Cold review round 1 F3/F4: pollOutcome alone proves the DECISION; these
// prove the WIRING — the exact patch shape each branch calls, via a vi.fn()
// patch fake, without needing to mount the component's effect.
describe("applyPollOutcome", () => {
  it("adopts silently clearing stale/justSaved/reloadKept (F4), flags stale exactly once, and makes no call when the hash matches", () => {
    const patch = vi.fn();
    const clean = { ...editAt("a", "a", "h"), justSaved: true, reloadKept: true };
    expect(applyPollOutcome({ hash: "h2", content: "b" }, clean, patch)).toEqual({ action: "adopt" });
    expect(patch).toHaveBeenCalledWith({
      content: "b", savedContent: "b", baseHash: "h2", stale: false, justSaved: false, reloadKept: false,
    });
    patch.mockClear();
    const dirty = editAt("b", "a", "h");
    expect(applyPollOutcome({ hash: "h2", content: "z" }, dirty, patch)).toEqual({ action: "stale" });
    expect(patch).toHaveBeenCalledWith({ stale: true });
    patch.mockClear();
    expect(applyPollOutcome({ hash: "h2", content: "z" }, { ...dirty, stale: true }, patch)).toEqual({ action: "stale" });
    expect(patch).not.toHaveBeenCalled();
    expect(applyPollOutcome({ hash: "h", content: "a" }, editAt("a", "a", "h"), patch)).toEqual({ action: "none" });
    expect(patch).not.toHaveBeenCalled();
  });
});

// Diff review 13b2b3d5fb4c F1 (MEDIUM): save() never touched the read query
// cache, so a poll landing right after a save could still compare against the
// pre-save disk snapshot and show a stale conflict diff. Proves both ends of
// the fix: syncReadCacheAfterSave writes the exact saved content/hash on a
// successful save, and feeding that back through applyPollOutcome against the
// post-save edit state yields "none" — no adoption, no staleness.
describe("syncReadCacheAfterSave", () => {
  it("writes the read cache with the saved content/hash on success, so the next poll sees no drift", () => {
    const setReadCache = vi.fn();
    const successPatch = { savedContent: "new content", baseHash: "newhash", justSaved: true, unconfirmed: false, writeErr: false, auditUnavailable: false };
    syncReadCacheAfterSave(successPatch, "new content", setReadCache);
    expect(setReadCache).toHaveBeenCalledWith({
      ok: true,
      data: { binary: false, content: "new content", hash: "newhash", size: 11 },
    });
    const disk = setReadCache.mock.calls[0][0].data;
    const editAfterSave = editAt("new content", "new content", "newhash");
    expect(applyPollOutcome({ hash: disk.hash, content: disk.content }, editAfterSave, vi.fn())).toEqual({ action: "none" });
  });
});

describe("FileEditor query wiring and the removed confirm", () => {
  it("polls every 5s enabled only while a file is open (F3), and no longer references window.confirm anywhere", () => {
    harness.edits = {};
    harness.locked = false;
    harness.query = { data: undefined, isError: false, isLoading: false, isFetching: false, refetch: vi.fn() };
    render({ root: "repos", rel: "a.ts" });
    expect(harness.queryOpts).toMatchObject({ refetchInterval: 5000, enabled: true });
    renderToStaticMarkup(createElement(FileEditor, { file: null })); // render()'s own param type excludes null
    expect(harness.queryOpts).toMatchObject({ enabled: false });
    const source = readFileSync(new URL("./FileEditor.tsx", import.meta.url), "utf8");
    expect(source).not.toContain("window.confirm");
  });
});

describe("ancestorKeys", () => {
  it("computes root plus every ancestor directory, including the target itself when it is a directory", () => {
    expect(ancestorKeys("repos", "", true)).toEqual(["repos:"]);
    expect(ancestorKeys("repos", "src/components", true))
      .toEqual(["repos:", "repos:src", "repos:src/components"]);
    expect(ancestorKeys("repos", "src/components/files/FileEditor.tsx", false))
      .toEqual(["repos:", "repos:src", "repos:src/components", "repos:src/components/files"]);
  });
});

// Full priority chain, high to low (F5): each row also sets every lower-
// priority flag, proving statusChip picks the right one out of the whole chain.
describe("statusChip", () => {
  const t = (key: string) => key;
  type Flag = "stale" | "unconfirmed" | "auditWarning" | "auditUnavailable" | "writeErr" | "reloadKept" | "reloading" | "saving" | "justSaved";
  const on = (...flags: Flag[]): EditState => flags.reduce((m, f) => ({ ...m, [f]: true }), createEdit("x", "h"));

  it("shows nothing without an edit, or with a clean edit and no flag set", () => {
    expect(statusChip(undefined, false, t)).toBeNull();
    expect(statusChip(createEdit("x", "h"), false, t)).toBeNull();
  });

  const rows: Array<[string, EditState, boolean, ReturnType<typeof statusChip>]> = [
    ["stale", on("stale", "unconfirmed", "auditWarning", "auditUnavailable", "writeErr", "reloadKept", "reloading", "saving", "justSaved"), true, { kind: "stale", tone: "warning", text: "staleReload" }],
    ["unconfirmed", on("unconfirmed", "auditWarning", "auditUnavailable", "writeErr", "reloadKept", "reloading", "saving", "justSaved"), true, { kind: "text", tone: "warning", text: "checkCurrentState" }],
    ["auditWarning", on("auditWarning", "auditUnavailable", "writeErr", "reloadKept", "reloading", "saving", "justSaved"), true, { kind: "text", tone: "warning", text: "savedAuditPending" }],
    ["auditUnavailable", on("auditUnavailable", "writeErr", "reloadKept", "reloading", "saving", "justSaved"), true, { kind: "text", tone: "danger", text: "auditUnavailable" }],
    ["writeErr", on("writeErr", "reloadKept", "reloading", "saving", "justSaved"), true, { kind: "text", tone: "danger", text: "writeError" }],
    ["reloadKept", on("reloadKept", "reloading", "saving", "justSaved"), true, { kind: "text", tone: "warning", text: "reloadKept" }],
    ["reloading", on("reloading", "saving", "justSaved"), true, { kind: "text", tone: "muted", text: "reloading" }],
    ["saving", on("saving", "justSaved"), true, { kind: "text", tone: "muted", text: "saving" }],
    ["justSaved while clean", on("justSaved"), false, { kind: "text", tone: "muted", text: "saved" }],
    ["justSaved suppressed while dirty", on("justSaved"), true, null],
  ];

  it.each(rows)("%s outranks every lower-priority flag", (_name, m, dirty, expected) => {
    expect(statusChip(m, dirty, t)).toEqual(expected);
  });

  it("end-to-end F4: after a save, a clean poll adoption clears justSaved so the chip stops showing 'saved'", () => {
    let patched = { ...createEdit("a", "h"), justSaved: true };
    applyPollOutcome({ hash: "h2", content: "b" }, patched, (p) => { patched = { ...patched, ...p }; });
    expect(statusChip(patched, false, t)).toBeNull();
  });
});

describe("FileEditor", () => {
  it("offers a same-origin download of saved bytes for binary files", () => {
    harness.edits = {};
    harness.locked = false;
    harness.query = {
      data: { ok: true, data: { binary: true, content: null, hash: "abc", size: 12 } },
      isError: false, isLoading: false, isFetching: false, refetch: vi.fn(),
    };
    const html = render();
    expect(html).toContain("/api/files/download?root=repos&amp;rel=data.bin");
    expect(html).toContain("downloadSavedHint");
    expect(html).toContain("binaryPlaceholder");
  });

  it("shows SVG preview control and saving status", () => {
    const edit = createEdit("<svg></svg>", "h");
    edit.saving = true;
    harness.edits = { "repos:icon.svg": edit };
    harness.locked = false;
    harness.query = {
      data: { ok: true, data: { binary: false, content: "<svg></svg>", hash: "h", size: 11 } },
      isError: false, isLoading: false, isFetching: false, refetch: vi.fn(),
    };
    const html = render({ root: "repos", rel: "icon.svg" });
    expect(html).toContain("preview");
    expect(html).toContain("saving");
    expect(html).toContain("/api/files/download?root=repos&amp;rel=icon.svg");
    expect(tabKey({ root: "repos", rel: "icon.svg" })).toBe("repos:icon.svg");
  });

  it("cache-busts image preview with the saved hash and skips download when locked without EditState", () => {
    harness.edits = {};
    harness.locked = true;
    harness.query = {
      data: { ok: true, data: { binary: true, content: null, hash: "deadbeef", size: 2048 } },
      isError: false, isLoading: false, isFetching: false, refetch: vi.fn(),
    };
    const html = render({ root: "repos", rel: "shot.png" });
    expect(html).toContain("h=deadbeef");
    expect(html).toContain("aria-disabled=\"true\"");
    expect(html).not.toContain('href="/api/files/download?root=repos&amp;rel=shot.png"');
  });

  it("defaults a markdown file to the rendered preview, not Monaco source", () => {
    harness.edits = {};
    harness.locked = false;
    harness.query = {
      data: { ok: true, data: { binary: false, content: "# Hello", hash: "h", size: 7 } },
      isError: false, isLoading: false, isFetching: false, refetch: vi.fn(),
    };
    const html = render({ root: "repos", rel: "notes.md" });
    // preview=true shows the "Editar" toggle (t() mock returns the raw key "edit")
    expect(html).toContain(">edit<");
  });

  it("routes a .markdown alias to the rendered preview, same as .md (F3)", () => {
    harness.edits = {};
    harness.locked = false;
    harness.query = {
      data: { ok: true, data: { binary: false, content: "# Hello", hash: "h", size: 7 } },
      isError: false, isLoading: false, isFetching: false, refetch: vi.fn(),
    };
    const html = render({ root: "repos", rel: "notes.markdown" });
    expect(html).toContain(">edit<");
  });

  it("routes a .htm alias to the html preview, same as .html (F3)", () => {
    harness.edits = {};
    harness.locked = false;
    harness.query = {
      data: { ok: true, data: { binary: false, content: "<p>hi</p>", hash: "h", size: 9 } },
      isError: false, isLoading: false, isFetching: false, refetch: vi.fn(),
    };
    const html = render({ root: "repos", rel: "page.htm" });
    // HTML preview toggle uses the htmlCode label ("Código"/"Code"), not the generic "edit" (F2)
    expect(html).toContain(">htmlCode<");
    expect(html).not.toContain(">edit<");
  });

  it("renders the pdf overlay for a text-classified pdf, not the plain binary placeholder (F4)", () => {
    harness.edits = {};
    harness.locked = false;
    harness.query = {
      data: { ok: true, data: { binary: false, content: "%PDF-1.4 ascii-only stream", hash: "h", size: 26 } },
      isError: false, isLoading: false, isFetching: false, refetch: vi.fn(),
    };
    const html = render({ root: "repos", rel: "doc.pdf" });
    expect(html).toContain('src="/api/files/raw?root=repos&amp;rel=doc.pdf&amp;h=h"');
    expect(html).not.toContain("binaryPlaceholder");
  });

  it("falls back to the placeholder for oversized markdown text, not the rendered preview (round 2 MEDIUM)", () => {
    harness.edits = {};
    harness.locked = false;
    harness.query = {
      data: { ok: true, data: { binary: false, content: "# too big", hash: "h", size: 3 * 1024 * 1024 } },
      isError: false, isLoading: false, isFetching: false, refetch: vi.fn(),
    };
    const html = render({ root: "repos", rel: "notes.md" });
    expect(html).toContain("binaryPlaceholder");
    expect(html).not.toContain(">edit<");
  });

  it("swaps the breadcrumb for a back arrow + basename when onClose is set, keeping Save/Download (spec §4.6)", () => {
    harness.edits = {};
    harness.locked = false;
    harness.query = { data: { ok: true, data: { binary: false, content: "# hi", hash: "h", size: 4 } }, isError: false, isLoading: false, isFetching: false, refetch: vi.fn() };
    const html = renderToStaticMarkup(createElement(FileEditor, { file: { root: "repos", rel: "src/notes.md" }, onClose: () => {} }));
    expect(html).toContain("backToFolders");
    expect(html).toContain(">notes.md<");
    expect(html).not.toContain("breadcrumbLabel");
    expect(html).toContain("downloadSavedHint");
  });
});

describe("viewerFor", () => {
  const MB = 1024 * 1024;
  it("covers every row of the spec §8 decision table", () => {
    expect(viewerFor("md", 100)).toBe("markdown");
    expect(viewerFor("MD", 3 * MB)).toBe("placeholder");
    expect(viewerFor("html", 100)).toBe("html");
    expect(viewerFor("htm", 3 * MB)).toBe("placeholder");
    expect(viewerFor("pdf", 10 * MB)).toBe("pdf");
    expect(viewerFor("pdf", 21 * MB)).toBe("placeholder");
    expect(viewerFor("png", 4 * MB)).toBe("image");
    expect(viewerFor("png", 6 * MB)).toBe("placeholder");
    expect(viewerFor("csv", 100, { rows: 10 })).toBe("csv");
    expect(viewerFor("csv", 3 * MB, { rows: 10 })).toBe("placeholder");
    expect(viewerFor("tsv", 100, { rows: 6000 })).toBe("csvTooBig");
    expect(viewerFor("mp3", 10 * MB)).toBe("audio");
    expect(viewerFor("mp3", 65 * MB)).toBe("placeholder");
    expect(viewerFor("mp4", 10 * MB)).toBe("video");
    expect(viewerFor("mov", 65 * MB)).toBe("placeholder");
    expect(viewerFor("json", 100)).toBe("text");
    expect(viewerFor("yaml", 100)).toBe("text");
    expect(viewerFor("toml", 3 * MB)).toBe("placeholder");
  });

  it("holds the exact boundary for every capped kind (at-cap passes, cap+1 fails)", () => {
    expect(viewerFor("png", 5 * MB)).toBe("image");
    expect(viewerFor("png", 5 * MB + 1)).toBe("placeholder");
    expect(viewerFor("pdf", 20 * MB)).toBe("pdf");
    expect(viewerFor("pdf", 20 * MB + 1)).toBe("placeholder");
    expect(viewerFor("mp3", 64 * MB)).toBe("audio");
    expect(viewerFor("mp3", 64 * MB + 1)).toBe("placeholder");
    expect(viewerFor("mp4", 64 * MB)).toBe("video");
    expect(viewerFor("mp4", 64 * MB + 1)).toBe("placeholder");
    expect(viewerFor("csv", 100, { rows: 5000 })).toBe("csv");
    expect(viewerFor("csv", 100, { rows: 5001 })).toBe("csvTooBig");
  });
});

describe("wrapHtmlForPreview", () => {
  it("puts the CSP meta as the first child of the real head, exactly once", () => {
    const out = wrapHtmlForPreview("<p>hi</p>");
    const headOpen = out.indexOf("<head>") + "<head>".length;
    expect(out.startsWith("<meta http-equiv=\"Content-Security-Policy\"", headOpen)).toBe(true);
    expect(out.split("Content-Security-Policy")).toHaveLength(2);
  });

  it("still lands the CSP outside a commented-out fake <head>", () => {
    const out = wrapHtmlForPreview("<!-- <head> -->");
    const realHeadOpen = out.indexOf("<head>") + "<head>".length;
    expect(out.startsWith("<meta http-equiv=\"Content-Security-Policy\"", realHeadOpen)).toBe(true);
    expect(out.indexOf("Content-Security-Policy")).toBeLessThan(out.indexOf("<body>"));
  });

  it("wraps a full document as body content instead of parsing it", () => {
    const source = "<html><head><title>x</title></head><body>hi</body></html>";
    const out = wrapHtmlForPreview(source);
    expect(out.startsWith(`<!DOCTYPE html><html><head>`)).toBe(true);
    expect(out).toContain(`<body>${source}</body>`);
  });
});

describe("csvDelimiterFor", () => {
  it("returns a tab for .tsv and a comma for everything else (round 1 F4)", () => {
    expect(csvDelimiterFor("tsv")).toBe("\t");
    expect(csvDelimiterFor("TSV")).toBe("\t");
    expect(csvDelimiterFor("csv")).toBe(",");
  });
});
