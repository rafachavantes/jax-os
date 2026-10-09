import { describe, expect, it } from "vitest";
import { diffLines } from "./rulesDiff";

describe("diffLines", () => {
  it("identical text: every line marked same", () => {
    expect(diffLines("a\nb\n", "a\nb\n")).toEqual([
      { type: "same", text: "a" },
      { type: "same", text: "b" },
      { type: "same", text: "" },
    ]);
  });
  it("one changed line: a remove/add pair around unchanged context", () => {
    expect(diffLines("a\nb\nc", "a\nX\nc")).toEqual([
      { type: "same", text: "a" },
      { type: "remove", text: "b" },
      { type: "add", text: "X" },
      { type: "same", text: "c" },
    ]);
  });
  it("empty vs content: all additions", () => {
    expect(diffLines("", "a\nb")).toEqual([
      { type: "add", text: "a" },
      { type: "add", text: "b" },
    ]);
  });
  it("content vs empty: all removals (missing-file diff case, spec §6.3)", () => {
    expect(diffLines("a\nb", "")).toEqual([
      { type: "remove", text: "a" },
      { type: "remove", text: "b" },
    ]);
  });
  it("both empty: no lines at all", () => {
    expect(diffLines("", "")).toEqual([]);
  });
});
