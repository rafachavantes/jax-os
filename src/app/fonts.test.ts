import { createHash } from "node:crypto";
import { readdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const at = (rel: string) => fileURLToPath(new URL(rel, import.meta.url));
const FONTS = at("../fonts/");
const sha256 = (file: string) => createHash("sha256").update(readFileSync(file)).digest("hex");
// the retired CSS variable, built so the literal never appears in the tracked tree
const OLD_VAR = ["font", "good", "times"].join("-");

// fvar axes as [tag, min, default, max], read straight from the TTF table directory (16.16 fixed point)
function fvarAxes(file: string): [string, number, number, number][] {
  const b = readFileSync(file);
  const tables = b.readUInt16BE(4);
  let fvar = -1;
  for (let i = 0; i < tables; i++) {
    const rec = 12 + i * 16;
    if (b.toString("latin1", rec, rec + 4) === "fvar") fvar = b.readUInt32BE(rec + 8);
  }
  if (fvar < 0) return [];
  const axesAt = fvar + b.readUInt16BE(fvar + 4);
  const count = b.readUInt16BE(fvar + 8);
  const size = b.readUInt16BE(fvar + 10);
  return Array.from({ length: count }, (_, i) => {
    const o = axesAt + i * size;
    return [b.toString("latin1", o, o + 4), b.readInt32BE(o + 4) / 65536, b.readInt32BE(o + 8) / 65536, b.readInt32BE(o + 12) / 65536];
  });
}

describe("self-hosted display font (MOA-501 D2)", () => {
  it("ships exactly the variable Orbitron TTF and its OFL notice", () => {
    expect(readdirSync(FONTS).sort()).toEqual(["OFL.txt", "Orbitron-VariableFont_wght.ttf"]);
  });
  it("keeps both files byte-identical to the staged upstream copies", () => {
    expect(sha256(`${FONTS}Orbitron-VariableFont_wght.ttf`)).toBe("f42db2dd16e642258e35782916eceb1dcdbea06fb958d77ad71dc5963587e8fd");
    expect(sha256(`${FONTS}OFL.txt`)).toBe("ab609b0e110d622435ff337cdf233288556e011bbf9bd0550be98846c0630819");
  });
  it("is variable over exactly one wght axis, 400 to 900", () => {
    expect(fvarAxes(`${FONTS}Orbitron-VariableFont_wght.ttf`)).toEqual([["wght", 400, 400, 900]]);
  });
  it("carries the upstream copyright line first", () => {
    expect(readFileSync(`${FONTS}OFL.txt`, "utf8").split("\n")[0]).toMatch(/^Copyright 2018 The Orbitron Project Authors/);
  });
  it("loads the font through next/font/local with one src entry and a weight range, no remote URL", () => {
    const src = readFileSync(at("./fonts.ts"), "utf8");
    expect(src.match(/path:/g)?.length).toBe(1);
    expect(src).toContain('path: "../fonts/Orbitron-VariableFont_wght.ttf"');
    expect(src).toContain('weight: "400 900"');
    expect(src).toContain('variable: "--font-orbitron"');
    expect(src).not.toMatch(/https?:/);
    expect(src).not.toContain(OLD_VAR);
  });
  it("points --font-display at Orbitron in both globals.css declarations", () => {
    const css = readFileSync(at("./globals.css"), "utf8");
    const chain = "--font-display: var(--font-orbitron), var(--font-geist-sans), sans-serif;";
    expect(css.split(chain).length - 1).toBe(2);
    expect(css).not.toContain(OLD_VAR);
  });
  it("names Orbitron in the vendored DS typography tokens", () => {
    const css = readFileSync(at("../styles/ds/typography.css"), "utf8");
    expect(css).toContain('--font-display: "Orbitron", "Geist"');
    expect(css).toContain("Display: Orbitron");
  });
});
