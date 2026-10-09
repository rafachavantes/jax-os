import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { switcherCurrentLabel, toScope } from "./RepoSwitcher";
import type { ScopeChipRow } from "./ScopeChip";

describe("toScope (spec §4.2)", () => {
  it("maps 'all' to {kind:'all'}, keeping vault/repo:<name> unchanged", () => {
    expect(toScope("all")).toEqual({ kind: "all" });
    expect(toScope("vault")).toEqual({ kind: "vault" });
    expect(toScope("repo:jax-os")).toEqual({ kind: "repo", name: "jax-os" });
  });
});

describe("switcherCurrentLabel (spec §4.2)", () => {
  const rows: ScopeChipRow[] = [{ value: "repo:jax-os", label: "jax-os" }, { value: "vault", label: "obsidian-vault" }];
  it("shows the all-locations label for {kind:'all'} without consulting rows", () => {
    expect(switcherCurrentLabel(rows, { kind: "all" }, "Todos os locais", "obsidian-vault")).toBe("Todos os locais");
  });
  it("still resolves a known repo/vault scope from rows", () => {
    expect(switcherCurrentLabel(rows, { kind: "repo", name: "jax-os" }, "Todos os locais", "obsidian-vault")).toBe("jax-os");
    expect(switcherCurrentLabel(rows, { kind: "vault" }, "Todos os locais", "obsidian-vault")).toBe("obsidian-vault");
  });
  it("falls back to the scope's own name before rows has loaded", () => {
    expect(switcherCurrentLabel([], { kind: "repo", name: "other" }, "Todos os locais", "obsidian-vault")).toBe("other");
  });
});

describe("RepoSwitcher source wiring (spec §4.2)", () => {
  it("renders the leading all-locations row before the RECENT header in source order", () => {
    const src = readFileSync(fileURLToPath(new URL("./RepoSwitcher.tsx", import.meta.url)), "utf8");
    const leadingIdx = src.indexOf("allLocationsRow");
    const recentIdx = src.indexOf("switcherRecentHeader");
    expect(leadingIdx).toBeGreaterThan(-1);
    expect(leadingIdx).toBeLessThan(recentIdx);
  });
});
