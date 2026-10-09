import { describe, expect, it } from "vitest";
import { parseLogLines } from "./import";

describe("parseLogLines", () => {
  it("parses valid JSONL, skips corrupt/non-object lines, ignores blanks", () => {
    const raw = [
      '{"ts":"2026-07-06T22:23:22.226Z","kind":"approval","ok":true}',
      "not json at all",
      "",
      '"a bare string"',
      "[1,2]",
      '{"ts":"2026-07-07T00:36:27.391Z","kind":"take-control","session":"s"}',
    ].join("\n");
    const r = parseLogLines(raw);
    expect(r.entries).toHaveLength(2);
    expect(r.entries[1]).toMatchObject({ kind: "take-control" });
    expect(r.skipped).toBe(3); // corrupt, bare string, array — blanks don't count
  });
});
