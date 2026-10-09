import { describe, expect, it } from "vitest";
import { mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const SRC_ROOT = fileURLToPath(new URL(".", import.meta.url));

const HEX_RE = /#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{4}|[0-9a-fA-F]{3})(?![0-9a-fA-F])/;
const FN_DECL_RE = /function\s+(\w+)\s*\(/;

// Paths this rule does not govern:
// - styles/ds/**: the token DEFINITIONS themselves (spec MOA-499), not a consumer of them.
// - *.test.ts / *.test.tsx: fixture data mimicking an EXTERNAL system's own color field
//   (Linear issue/label colors) -- never DS-token-governed UI. This is this plan's OWN 4th
//   exclusion, beyond the spec's literal 3-item list (see Global Constraints) -- flagged for
//   Rafa, not silently invented.
function isExcludedPath(relPath: string): boolean {
  return relPath.startsWith("styles/ds/") || /\.test\.tsx?$/.test(relPath);
}

// Two documented single-line exceptions (AGENTS.md): --body-ink's two theme values, and
// FileEditor.tsx's resolveHex() "#000000" fallback -- the one sanctioned Monaco chrome
// color-conversion path. `fnName` is the innermost enclosing `function NAME(` at that line,
// tracked by findUnscopedHex below -- narrows the FileEditor exception to resolveHex() itself,
// never any future unrelated "#000000" return elsewhere in the same file (cold review F3).
function isExcludedLine(relPath: string, line: string, fnName = ""): boolean {
  if (relPath === "app/globals.css" && line.includes("--body-ink:")) return true;
  if (
    relPath === "components/files/FileEditor.tsx" &&
    fnName === "resolveHex" &&
    /return "#000000"/.test(line)
  ) {
    return true;
  }
  return false;
}

function listSourceFiles(root: string): string[] {
  return (readdirSync(root, { recursive: true }) as string[])
    .filter((p) => /\.(ts|tsx|css)$/.test(p))
    .filter((p) => statSync(`${root}/${p}`).isFile());
}

function findUnscopedHex(root: string): string[] {
  const violations: string[] = [];
  for (const rel of listSourceFiles(root)) {
    if (isExcludedPath(rel)) continue;
    const content = readFileSync(`${root}/${rel}`, "utf-8");
    let currentFn = "";
    content.split("\n").forEach((line, i) => {
      const fnMatch = line.match(FN_DECL_RE);
      if (fnMatch) currentFn = fnMatch[1];
      if (isExcludedLine(rel, line, currentFn)) return;
      if (HEX_RE.test(line)) violations.push(`${rel}:${i + 1}`);
    });
  }
  return violations;
}

describe("no unscoped raw hex in src/ (spec MOA-499 Raw-hex scope)", () => {
  it("the live tree has zero unscoped hex literals today", () => {
    expect(findUnscopedHex(SRC_ROOT)).toEqual([]);
  });

  it("a DS token file (colors.css) is excluded by path, not by accident", () => {
    expect(isExcludedPath("styles/ds/colors.css")).toBe(true);
  });

  it("globals.css's --body-ink line is excluded; an unrelated hex line in the same file is not", () => {
    expect(isExcludedLine("app/globals.css", "  --body-ink: #c9bcaf;")).toBe(true);
    expect(isExcludedLine("app/globals.css", "  --something-else: #123456;")).toBe(false);
  });

  it("FileEditor.tsx's resolveHex() fallback is excluded only inside resolveHex() itself; the identical line text under another function name is not", () => {
    expect(isExcludedLine("components/files/FileEditor.tsx", '  return "#000000";', "resolveHex")).toBe(true);
    expect(isExcludedLine("components/files/FileEditor.tsx", '  return "#000000";', "otherFn")).toBe(false);
    expect(isExcludedLine("components/files/FileEditor.tsx", '  const x = "#abcdef";', "resolveHex")).toBe(false);
  });

  it("a *.test.ts(x) fixture file is excluded by path", () => {
    expect(isExcludedPath("lib/boardView.test.ts")).toBe(true);
    expect(isExcludedPath("components/kanban/ListView.test.tsx")).toBe(true);
  });

  it("findUnscopedHex reports an ordinary source file with a raw hex literal (proves the check can fail — cold review F2)", () => {
    const dir = mkdtempSync(join(tmpdir(), "hex-scope-fixture-"));
    try {
      writeFileSync(join(dir, "Widget.tsx"), 'const bg = "#ABCDEF";\n');
      expect(findUnscopedHex(dir)).toEqual(["Widget.tsx:1"]);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("findUnscopedHex exempts only the resolveHex() fallback in a FileEditor.tsx-shaped fixture, not a later unrelated return (cold review F3)", () => {
    const dir = mkdtempSync(join(tmpdir(), "hex-scope-fixture-"));
    try {
      mkdirSync(join(dir, "components", "files"), { recursive: true });
      writeFileSync(
        join(dir, "components", "files", "FileEditor.tsx"),
        [
          "function resolveHex(varName: string): string {",
          '  if (typeof document === "undefined") return "#000000";',
          "  return String(varName);",
          "}",
          "",
          "function otherFn(): string {",
          '  return "#111111";',
          "}",
        ].join("\n"),
      );
      expect(findUnscopedHex(dir)).toEqual(["components/files/FileEditor.tsx:7"]);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
