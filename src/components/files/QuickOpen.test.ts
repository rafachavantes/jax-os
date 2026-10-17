import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";

// F1-7: render tests need `open`/`query`/`selected`/`index` state forced open, so useState
// is stubbed to return fixed values in call order (same technique as FileSearch.test.tsx's
// single-useState override, extended to QuickOpen's four calls).
const harness = vi.hoisted(() => ({
  open: true,
  query: "test",
  selected: -1,
  index: {
    ok: true as const,
    data: [{ root: "repos" as const, rel: "src/test.ts", name: "test.ts" }],
    truncated: false,
    truncatedRoots: [],
    builtAt: "",
  } as unknown,
  callIndex: 0,
  ws: { selected: null as { root: "repos" | "vault"; rel: string; isDir: boolean } | null, scope: { kind: "all" } as { kind: "all" } | { kind: "repo"; name: string } | { kind: "vault" } },
}));

vi.mock("react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react")>();
  return {
    ...actual,
    useState: (initial: unknown) => {
      const values = [harness.open, harness.query, harness.selected, harness.index];
      const i = harness.callIndex++;
      return i < values.length ? [values[i], vi.fn()] : actual.useState(initial);
    },
  };
});

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
}));

vi.mock("./ScopeChip", () => ({ ScopeChip: () => null }));
vi.mock("./FilesWorkspaceProvider", () => ({ useFilesTreeWorkspace: () => harness.ws }));

import { fuzzyScore, indexFetchUrl, quickOpenEmptyState, QuickOpen, rankHits, shouldApplyIndexFetch, type IndexHit } from "./QuickOpen";

function render() {
  harness.callIndex = 0;
  return renderToStaticMarkup(createElement(QuickOpen, {
    onOpenFilePinned: () => {}, activeFile: null, scopeOverride: null, onScopeChange: () => {},
  }));
}

describe("fuzzyScore", () => {
  it("ranks exact > prefix > scattered subsequence > no match (null)", () => {
    const exact = fuzzyScore("editor.tsx", "editor.tsx")!;
    const prefix = fuzzyScore("edit", "editor.tsx")!;
    const scattered = fuzzyScore("edtr", "editor.tsx")!;
    expect(exact).toBeGreaterThan(prefix);
    expect(prefix).toBeGreaterThan(scattered);
    expect(scattered).not.toBeNull();
    expect(fuzzyScore("zzz", "editor.tsx")).toBeNull();
  });

  it("is case-insensitive", () => {
    expect(fuzzyScore("EDITOR", "editor.tsx")).not.toBeNull();
  });

  it("treats an empty query as a universal, lowest-effort match", () => {
    expect(fuzzyScore("", "anything")).toBe(0);
  });

  it("ranks an earlier substring match above a later one", () => {
    const early = fuzzyScore("edit", "src/edit/a.ts")!;
    const late = fuzzyScore("edit", "src/aaaaaa/edit.ts")!;
    expect(early).toBeGreaterThan(late);
  });

  it("fails a subsequence that isn't in order", () => {
    expect(fuzzyScore("tide", "editor.tsx")).toBeNull(); // t-i-d-e not a subsequence of editor.tsx
  });
});

describe("rankHits", () => {
  it("filters non-matches, sorts by score descending, caps the result at 50", () => {
    const hits: IndexHit[] = [
      { root: "repos", rel: "src/FileEditor.tsx", name: "FileEditor.tsx" },
      { root: "repos", rel: "src/nope.ts", name: "nope.ts" },
      { root: "repos", rel: "src/edit.ts", name: "edit.ts" },
    ];
    const ranked = rankHits("edit", hits);
    expect(ranked.map((h) => h.name)).toEqual(["edit.ts", "FileEditor.tsx"]);
  });

  it("caps at 50 results even with more matches available", () => {
    const hits: IndexHit[] = Array.from({ length: 80 }, (_, i) => ({
      root: "repos" as const, rel: `f${i}.ts`, name: `f${i}.ts`,
    }));
    expect(rankHits("f", hits)).toHaveLength(50);
  });
});

describe("quickOpenEmptyState (F3 error contract)", () => {
  const okIndex = { ok: true as const, data: [], truncated: false, truncatedRoots: [], builtAt: "" };
  const failedIndex = { ok: false as const, error: "unavailable" };

  it("is unavailable on a failed index, regardless of query", () => {
    expect(quickOpenEmptyState(failedIndex, "", 0)).toBe("unavailable");
    expect(quickOpenEmptyState(failedIndex, "abc", 0)).toBe("unavailable");
  });

  it("is noRecents for an empty query with no results and no index failure", () => {
    expect(quickOpenEmptyState(null, "", 0)).toBe("noRecents");
    expect(quickOpenEmptyState(okIndex, "", 0)).toBe("noRecents");
  });

  it("is noResults for a non-empty query with no results and no index failure", () => {
    expect(quickOpenEmptyState(null, "abc", 0)).toBe("noResults");
    expect(quickOpenEmptyState(okIndex, "abc", 0)).toBe("noResults");
  });

  it("is null whenever there are results, even with a failed index", () => {
    expect(quickOpenEmptyState(failedIndex, "abc", 3)).toBeNull();
    expect(quickOpenEmptyState(okIndex, "", 1)).toBeNull();
  });
});

describe("QuickOpen rows (FI-7)", () => {
  beforeEach(() => {
    harness.open = true;
    harness.query = "test";
    harness.selected = -1;
    harness.index = {
      ok: true,
      data: [{ root: "repos", rel: "src/test.ts", name: "test.ts" }],
      truncated: false,
      truncatedRoots: [],
      builtAt: "",
    };
  });

  it("splits a result row into a bold name and a dimmed parent path", () => {
    const html = render();
    expect(html).toContain('<span class="text-ink font-semibold">test.ts</span>');
    expect(html).toContain('<span class="text-muted"> repos/src/</span>');
    expect(html).not.toContain("repos/src/test.ts</button>");
  });

  it("shows a static keyboard-hint footer under the results", () => {
    const html = render();
    expect(html).toContain("quickOpenHint");
  });
});

describe("indexFetchUrl (spec §7)", () => {
  it("serializes each scope kind into the query string", () => {
    expect(indexFetchUrl({ kind: "all" })).toBe("/api/files/index?scope=all");
    expect(indexFetchUrl({ kind: "vault" })).toBe("/api/files/index?scope=vault");
    expect(indexFetchUrl({ kind: "repo", name: "jax-os" })).toBe("/api/files/index?scope=repo%3Ajax-os");
  });
  it("encodes a repo name containing reserved URL characters so the scope param round-trips (round-2 F4)", () => {
    const url = indexFetchUrl({ kind: "repo", name: "weird#&?%name" });
    // searchParams.get already percent-decodes; the raw value must round-trip byte-for-byte.
    expect(new URL(url, "http://127.0.0.1").searchParams.get("scope")).toBe("repo:weird#&?%name");
  });
});

describe("shouldApplyIndexFetch (round-2 F3 — stale-scope response guard)", () => {
  it("applies only the latest request's result; an older, later-resolving one is ignored, even if a scope is revisited (repeat seq still ties to its OWN request, never an earlier one)", () => {
    expect(shouldApplyIndexFetch(1, 1)).toBe(true);
    expect(shouldApplyIndexFetch(1, 2)).toBe(false); // scope changed again before request 1's response arrived
    expect(shouldApplyIndexFetch(2, 2)).toBe(true);
  });
});
