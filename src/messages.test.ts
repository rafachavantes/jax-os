import { describe, expect, it } from "vitest";
import en from "../messages/en-US.json";
import pt from "../messages/pt-BR.json";

function flattenKeys(obj: unknown, prefix = ""): Set<string> {
  const keys = new Set<string>();
  if (obj !== null && typeof obj === "object" && !Array.isArray(obj)) {
    for (const [k, v] of Object.entries(obj as Record<string, unknown>)) {
      const path = prefix ? `${prefix}.${k}` : k;
      if (v !== null && typeof v === "object" && !Array.isArray(v)) {
        for (const child of flattenKeys(v, path)) keys.add(child);
      } else {
        keys.add(path);
      }
    }
  }
  return keys;
}

describe("en-US / pt-BR key parity (spec MOA-499)", () => {
  it("both locale files have exactly the same set of dotted keys", () => {
    const enKeys = flattenKeys(en);
    const ptKeys = flattenKeys(pt);
    const onlyInEn = [...enKeys].filter((k) => !ptKeys.has(k)).sort();
    const onlyInPt = [...ptKeys].filter((k) => !enKeys.has(k)).sort();
    expect({ onlyInEn, onlyInPt }).toEqual({ onlyInEn: [], onlyInPt: [] });
  });

  it("flattenKeys reports the exact missing key on a drift (proves the check can fail)", () => {
    const a = { a: { b: "x", c: "y" } };
    const b = { a: { b: "x" } };
    expect([...flattenKeys(a)].filter((k) => !flattenKeys(b).has(k))).toEqual(["a.c"]);
  });
});
