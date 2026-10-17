import { describe, expect, it } from "vitest";
import { breadcrumbSegments } from "./breadcrumb";

describe("breadcrumbSegments (spec §5, moved from FileEditor.test.tsx's JSX-string assertions)", () => {
  it("returns one entry per segment when 3 or fewer, every non-final segment isDir:true by construction", () => {
    expect(breadcrumbSegments("src/notes.bin", false)).toEqual([
      { label: "src", rel: "src", isDir: true, collapsed: false },
      { label: "notes.bin", rel: "src/notes.bin", isDir: false, collapsed: false },
    ]);
  });

  it("collapses more than 3 segments into root-adjacent '…' + the last two, the collapsed entry's rel spanning every hidden segment (FI-3)", () => {
    expect(breadcrumbSegments(".local/docs/audits/report.md", false)).toEqual([
      { label: "…", rel: ".local/docs", isDir: true, collapsed: true },
      { label: "audits", rel: ".local/docs/audits", isDir: true, collapsed: false },
      { label: "report.md", rel: ".local/docs/audits/report.md", isDir: false, collapsed: false },
    ]);
  });

  it("a folder target (finalIsDir:true) marks the LAST segment isDir:true too", () => {
    expect(breadcrumbSegments("src/lib", true)).toEqual([
      { label: "src", rel: "src", isDir: true, collapsed: false },
      { label: "lib", rel: "src/lib", isDir: true, collapsed: false },
    ]);
  });

  it("a single-segment rel returns just the final entry, honoring finalIsDir; an empty rel returns no segments (nothing selected — just the root crumb is shown)", () => {
    expect(breadcrumbSegments("a.ts", false)).toEqual([{ label: "a.ts", rel: "a.ts", isDir: false, collapsed: false }]);
    expect(breadcrumbSegments("", true)).toEqual([]);
  });
});
