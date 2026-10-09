import { describe, expect, it } from "vitest";
import { parseCsv, sortRows } from "./CsvTable";

describe("parseCsv", () => {
  it("splits a simple comma-delimited body into rows of fields", () => {
    expect(parseCsv("a,b\n1,2\n3,4")).toEqual([["a", "b"], ["1", "2"], ["3", "4"]]);
  });

  it("honors quoted fields containing the delimiter and escaped quotes", () => {
    expect(parseCsv('name,note\n"Doe, Jane","said ""hi"""')).toEqual([
      ["name", "note"],
      ["Doe, Jane", 'said "hi"'],
    ]);
  });

  it("accepts a custom delimiter (tsv)", () => {
    expect(parseCsv("a\tb\n1\t2", "\t")).toEqual([["a", "b"], ["1", "2"]]);
  });
});

describe("sortRows", () => {
  it("sorts numerically-aware, ascending then descending", () => {
    const rows = [["10"], ["2"], ["1"]];
    expect(sortRows(rows, 0, "asc")).toEqual([["1"], ["2"], ["10"]]);
    expect(sortRows(rows, 0, "desc")).toEqual([["10"], ["2"], ["1"]]);
  });
});
