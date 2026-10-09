import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";

const harness = vi.hoisted(() => ({
  term: "", mode: "name" as "name" | "content",
  query: { data: undefined as unknown, isError: false, isFetching: false },
  lastOptions: undefined as { queryKey: unknown; retry: unknown; staleTime: unknown; queryFn: (ctx: { signal: AbortSignal }) => Promise<unknown> } | undefined,
  callIndex: 0,
  ws: { selected: null as { root: "repos" | "vault"; rel: string; isDir: boolean } | null, scope: { kind: "all" } as { kind: "all" } | { kind: "repo"; name: string } | { kind: "vault" } },
}));

vi.mock("react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react")>();
  return {
    ...actual,
    useState: (initial: unknown) => {
      const values = [harness.term, harness.term, harness.mode]; // [input, q, mode] — FileSearch's own hook order
      const i = harness.callIndex++;
      return i < values.length ? [values[i], vi.fn()] : actual.useState(initial);
    },
  };
});
vi.mock("next-intl", () => ({ useTranslations: () => (key: string) => key }));
vi.mock("@tanstack/react-query", () => ({
  useQuery: (options: typeof harness.lastOptions) => { harness.lastOptions = options; return harness.query; },
}));
vi.mock("../../lib/fileSearch", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../lib/fileSearch")>();
  return { fetchFileSearch: vi.fn(), fetchContentSearch: vi.fn(), groupContentHits: actual.groupContentHits };
});
vi.mock("./ScopeChip", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./ScopeChip")>();
  return { ...actual, ScopeChip: () => null };
});
vi.mock("./FilesWorkspaceProvider", () => ({ useFilesTreeWorkspace: () => harness.ws }));
vi.mock("../mission/SourceWarning", async () => {
  const { createElement } = await import("react");
  return { SourceWarning: ({ label, detail }: { label: string; detail?: string }) => createElement("div", null, label, detail ?? null) };
});

import { fetchContentSearch, fetchFileSearch } from "../../lib/fileSearch";
import { FileSearch } from "./FileSearch";

const hit = { root: "repos" as const, rel: "a.ts", name: "a.ts" };
const contentHit = { root: "vault" as const, rel: "daily/a.md", line: 3, snippet: "const needle = 1;" };

function render() {
  harness.callIndex = 0;
  return renderToStaticMarkup(createElement(FileSearch, {
    onOpenFile: () => {}, onOpenFilePinned: () => {}, activeFile: null, scopeOverride: null, onScopeChange: () => {},
  }));
}

describe("FileSearch — name mode (default, unchanged behavior)", () => {
  beforeEach(() => {
    harness.term = ""; harness.mode = "name";
    harness.query = { data: undefined, isError: false, isFetching: false };
    harness.lastOptions = undefined;
    harness.ws = { selected: null, scope: { kind: "all" } };
    vi.mocked(fetchFileSearch).mockReset();
    vi.mocked(fetchContentSearch).mockReset();
  });

  it("shows no result/loading/error/empty message when q is empty; shows searchFailed on transport error or ok:false", () => {
    expect(render()).toContain("searchLabel");
    harness.term = "foo";
    harness.query = { data: undefined, isError: true, isFetching: false };
    expect(render()).toContain("searchFailed");
    harness.query = { data: { ok: false, error: "file search unavailable" }, isError: false, isFetching: false };
    expect(render()).toContain("searchFailed");
  });

  it("shows searchNoResults for a complete empty result, searchIncomplete (not searchNoResults) for an incomplete one; renders a name hit as a plain path button", () => {
    harness.term = "foo";
    harness.query = { data: { ok: true, truncated: false, hits: [] }, isError: false, isFetching: false };
    expect(render()).toContain("searchNoResults");
    harness.query = { data: { ok: true, truncated: true, hits: [] }, isError: false, isFetching: false };
    const incomplete = render();
    expect(incomplete).toContain("searchIncomplete");
    expect(incomplete).not.toContain("searchNoResults");
    harness.query = { data: { ok: true, truncated: false, hits: [{ kind: "name", ...hit }] }, isError: false, isFetching: false };
    const html = render();
    expect(html).toContain(">a.ts<");
    expect(html).toContain(">repos/<");
  });

  it("wires the query key with mode+scope and the queryFn to fetchFileSearch, never fetchContentSearch", async () => {
    harness.term = "foo";
    render();
    expect(harness.lastOptions?.queryKey).toEqual(["files", "search", "name", "foo", "all"]);
    const signal = new AbortController().signal;
    // Round-2 (d4876572f596) F1: queryFn dereferences the fetcher's resolved value (`r.ok`) —
    // an unconfigured mock resolves `undefined` and this await throws before the assertion runs.
    vi.mocked(fetchFileSearch).mockResolvedValueOnce({ ok: true, data: [], truncated: false });
    await harness.lastOptions?.queryFn({ signal });
    expect(fetchFileSearch).toHaveBeenCalledWith("foo", { kind: "all" }, signal);
    expect(fetchContentSearch).not.toHaveBeenCalled();
  });

  it("falls back to ws.selected's own repo, then ws.scope, when no override is given (spec §8)", async () => {
    harness.term = "foo";
    harness.ws = { selected: { root: "repos", rel: "jax-os/a.ts", isDir: false }, scope: { kind: "all" } };
    render();
    expect(harness.lastOptions?.queryKey).toEqual(["files", "search", "name", "foo", "repo:jax-os"]);
    harness.ws = { selected: null, scope: { kind: "vault" } };
    render();
    expect(harness.lastOptions?.queryKey).toEqual(["files", "search", "name", "foo", "vault"]);
  });
});

describe("FileSearch — content mode", () => {
  beforeEach(() => {
    harness.term = "needle"; harness.mode = "content";
    harness.query = { data: undefined, isError: false, isFetching: false };
    harness.lastOptions = undefined;
    harness.ws = { selected: null, scope: { kind: "all" } };
    vi.mocked(fetchFileSearch).mockReset();
    vi.mocked(fetchContentSearch).mockReset();
  });

  it("wires the queryFn to fetchContentSearch, never fetchFileSearch", async () => {
    render();
    expect(harness.lastOptions?.queryKey).toEqual(["files", "search", "content", "needle", "all"]);
    const signal = new AbortController().signal;
    // Round-2 (d4876572f596) F1: same reason as the name-mode test above — configure the mock
    // before the await, not after.
    vi.mocked(fetchContentSearch).mockResolvedValueOnce({ ok: true, data: [], truncated: false });
    await harness.lastOptions?.queryFn({ signal });
    expect(fetchContentSearch).toHaveBeenCalledWith("needle", { kind: "all" }, signal);
    expect(fetchFileSearch).not.toHaveBeenCalled();
  });

  it("shows contentSearch*-prefixed labels, never the name-mode ones; renders a content hit's path, line, and snippet", () => {
    harness.query = { data: { ok: false, error: "content search unavailable" }, isError: false, isFetching: false };
    let html = render();
    expect(html).toContain("contentSearchFailed");
    expect(html).not.toContain("searchFailed");
    harness.query = { data: { ok: true, truncated: true, hits: [] }, isError: false, isFetching: false };
    expect(render()).toContain("contentSearchIncomplete");
    harness.query = { data: { ok: true, truncated: false, hits: [{ kind: "content", ...contentHit }] }, isError: false, isFetching: false };
    html = render();
    expect(html).toContain("a.md");
    expect(html).toContain("vault/daily/");
    expect(html).toContain(":3");
    expect(html).toMatch(/const <mark[^>]*>needle<\/mark> = 1;/);
  });

  // Review F2: content hits sharing root+rel collapse into ONE file heading (name + dim parent
  // path + match count), each match still its own clickable line row.
  it("groups multiple hits for the same file under a single heading with a match count", () => {
    const secondHit = { root: "vault" as const, rel: "daily/a.md", line: 9, snippet: "another needle here" };
    harness.query = {
      data: { ok: true, truncated: false, hits: [{ kind: "content", ...contentHit }, { kind: "content", ...secondHit }] },
      isError: false,
      isFetching: false,
    };
    const html = render();
    expect((html.match(/vault\/daily\//g) ?? []).length).toBe(1);
    expect(html).toContain("contentMatchCount");
    expect(html).toContain(":3");
    expect(html).toContain(":9");
    expect(html).toMatch(/const <mark[^>]*>needle<\/mark> = 1;/);
    expect(html).toMatch(/another <mark[^>]*>needle<\/mark> here/);
  });
});
