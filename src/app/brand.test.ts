import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const file = (rel: string) => fileURLToPath(new URL(`../../${rel}`, import.meta.url));
const text = (rel: string) => readFileSync(file(rel), "utf8");

const LOGO = ["public/brand/logo-dark.svg", "public/brand/logo-light.svg"];
const MARK = ["public/brand/mark-dark.svg", "public/brand/mark-light.svg"];
const ALL_SVG = [...LOGO, ...MARK, "src/app/icon.svg"];
const paths = (svg: string) => [...svg.matchAll(/<path [^>]*\bd="([^"]+)"/g)].map((m) => m[1]);
const hexes = (svg: string) => [...new Set([...svg.matchAll(/#[0-9A-Fa-f]{6}\b/g)].map((m) => m[0].toUpperCase()))].sort();

describe("brand SVG assets (MOA-511 Deliverables)", () => {
  it.each(ALL_SVG)("%s is a self-contained SVG with a viewBox", (rel) => {
    const svg = text(rel);
    expect(svg.startsWith('<svg xmlns="http://www.w3.org/2000/svg" viewBox="')).toBe(true);
    expect(svg.trimEnd().endsWith("</svg>")).toBe(true);
    expect(svg.match(/viewBox="([\d. ]+)"/)?.[1].trim().split(/\s+/).map(Number).every((n) => n >= 0)).toBe(true);
    expect(svg).not.toMatch(/<script|<image|<text|<style|<foreignObject|href=|url\(|data:|@import|font-family/);
    // every <path .../> is self-closing and carries d=
    expect([...svg.matchAll(/<path\b[^>]*>/g)].every((m) => m[0].endsWith("/>") && /\bd="/.test(m[0]))).toBe(true);
    expect(svg.match(/<svg\b/g)?.length).toBe(1);
  });
  it("fixes the lockup and mark aspect ratios", () => {
    for (const rel of LOGO) expect(text(rel)).toContain('viewBox="0 0 1519 305"');
    for (const rel of MARK) expect(text(rel)).toContain('viewBox="0 0 335 305"');
  });
  it("embeds only the named palette values per theme", () => {
    for (const rel of ["public/brand/logo-dark.svg", "public/brand/mark-dark.svg"]) {
      expect(hexes(text(rel)).every((h) => ["#E46C54", "#A6BCA6", "#F7F1EA"].includes(h))).toBe(true);
    }
    for (const rel of ["public/brand/logo-light.svg", "public/brand/mark-light.svg"]) {
      expect(hexes(text(rel)).every((h) => ["#C8543D", "#557355", "#1E1915"].includes(h))).toBe(true);
    }
    expect(hexes(text("src/app/icon.svg"))).toEqual(["#16120F", "#A6BCA6", "#E46C54", "#F7F1EA"]);
  });
  it("shares one geometry across themes, the mark and the lockup; the icon keeps the same three-part structure", () => {
    const logoDark = paths(text(LOGO[0]));
    expect(logoDark.length).toBe(6);
    expect(paths(text(LOGO[1]))).toEqual(logoDark);
    const markDark = paths(text(MARK[0]));
    expect(markDark.length).toBe(3);
    expect(paths(text(MARK[1]))).toEqual(markDark);
    expect(logoDark.slice(0, 3)).toEqual(markDark);
    // The icon is the small-size-optimized source: step 2.6 permits limited optical adjustment
    // (wider aperture, simplified tiny vertices), so exact path equality is NOT required. Pin the
    // structure instead: three filled paths in coral / sage / neutral order. Bounds and the
    // aperture pixel are gated by the tile test below and by the step 2.6 raster script.
    const icon = text("src/app/icon.svg");
    expect(paths(icon).length).toBe(3);
    expect([...icon.matchAll(/<path fill="(#[0-9A-F]{6})"/g)].map((m) => m[1])).toEqual(["#E46C54", "#A6BCA6", "#F7F1EA"]);
  });
  it("uses the fixed browser tile: 32x32, radius 6, Relay fitted inside a centered 28x28 box", () => {
    const svg = text("src/app/icon.svg");
    expect(svg).toContain('viewBox="0 0 32 32"');
    expect(svg).toContain('<rect width="32" height="32" rx="6" fill="#16120F"/>');
    const m = svg.match(/translate\(([\d.]+) ([\d.]+)\) scale\(([\d.]+)\)/);
    expect(m).not.toBeNull();
    const [tx, ty, s] = [Number(m![1]), Number(m![2]), Number(m![3])];
    const [w, h] = [335 * s, 305 * s];
    expect(w).toBeLessThanOrEqual(28);
    expect(w).toBeGreaterThan(27.9); // fitted to the box width, aspect ratio preserved
    expect(h).toBeLessThanOrEqual(28);
    expect(tx).toBeGreaterThanOrEqual(2); // at least 2 units of padding
    expect(ty).toBeGreaterThanOrEqual(2);
    expect(tx + w / 2).toBeCloseTo(16, 1); // centered
    expect(ty + h / 2).toBeCloseTo(16, 1);
  });
});

describe("favicon.ico (MOA-511 acceptance 2)", () => {
  const ico = readFileSync(file("src/app/favicon.ico"));
  it("is a genuine ICO directory with 16, 32 and 48 px PNG frames", () => {
    expect([ico.readUInt16LE(0), ico.readUInt16LE(2), ico.readUInt16LE(4)]).toEqual([0, 1, 3]);
    const sizes: number[] = [];
    for (let i = 0; i < 3; i++) {
      const entry = 6 + i * 16;
      const size = ico.readUInt32LE(entry + 8);
      const offset = ico.readUInt32LE(entry + 12);
      const png = ico.subarray(offset, offset + size);
      expect([...png.subarray(0, 4)]).toEqual([0x89, 0x50, 0x4e, 0x47]); // not an extension-renamed non-PNG
      expect(png.readUInt32BE(16)).toBe(ico[entry]); // IHDR width matches the directory
      expect(png.readUInt32BE(20)).toBe(ico[entry + 1]);
      sizes.push(ico[entry]);
    }
    expect(sizes).toEqual([16, 32, 48]);
  });
});

describe("icon registration", () => {
  it("registers icons by the Next.js file convention only: no manual icon links or metadata.icons", () => {
    expect(existsSync(file("src/app/icon.svg"))).toBe(true);
    expect(existsSync(file("src/app/favicon.ico"))).toBe(true);
    const layout = text("src/app/layout.tsx");
    expect(layout).not.toMatch(/\bicons\s*:/);
    expect(layout).not.toContain('rel="icon"');
  });
});

describe("README header and docs/brand.md", () => {
  it("shows the full lockup as a theme-aware picture with a light fallback, 320 px max", () => {
    const readme = text("README.md");
    expect(readme).toContain('srcset="public/brand/logo-dark.svg"');
    expect(readme).toContain('src="public/brand/logo-light.svg"');
    expect(readme).toContain('width="320"');
    expect(readme).not.toContain(".local/");
    expect(readme.startsWith("# Jax OS\n")).toBe(true);
  });
  it("documents usage, palette mapping, provenance and ICO export without private paths", () => {
    const doc = text("docs/brand.md");
    for (const heading of ["## Files", "## Palette", "## Provenance", "## Exporting the browser icons"]) expect(doc).toContain(heading);
    expect(doc).not.toMatch(/\/home\/|\.local\//);
  });
});
