import { describe, expect, it } from "vitest";
import { scopeChipFallbackLabel, scopeChipRows } from "./ScopeChip";

describe("scopeChipRows (spec §9)", () => {
  const t = (key: string) => key;
  it("orders valid recents first, then repos+vault alphabetically, 'All repos' last; drops a stale or duplicate recent", () => {
    const rows = scopeChipRows(["zeta", "alpha"], ["repo:zeta", "repo:gone", "repo:zeta"], t, true);
    expect(rows.map((r) => r.value)).toEqual(["repo:zeta", "repo:alpha", "vault", "all"]);
    expect(rows.map((r) => r.label)).toEqual(["zeta", "alpha", "scopeVault", "scopeAll"]);
  });
  it("with no recents at all, lists repos+vault alphabetically then 'All repos'", () => {
    expect(scopeChipRows(["b", "a"], [], t, true).map((r) => r.value)).toEqual(["repo:a", "repo:b", "vault", "all"]);
  });
  it("omits vault from the row list when vaultEnabled is false, keeps it when true", () => {
    const off = scopeChipRows(["jax-os"], [], t, false);
    expect(off.some((r) => r.value === "vault")).toBe(false);
    const on = scopeChipRows(["jax-os"], [], t, true);
    expect(on.some((r) => r.value === "vault")).toBe(true);
  });
  it("drops a stale vault recent when vaultEnabled is false", () => {
    const off = scopeChipRows(["jax-os"], ["vault"], t, false);
    expect(off.some((r) => r.value === "vault")).toBe(false);
  });
});

// F3: before the repo list has loaded, `rows` has no matching entry for a repo scope — the
// fallback must derive the label from `scope` itself instead of defaulting to "All repos".
describe("scopeChipFallbackLabel (review F3)", () => {
  const t = (key: string) => key;
  it("derives a repo scope's label from its own name", () => {
    expect(scopeChipFallbackLabel({ kind: "repo", name: "jax-os" }, t)).toBe("jax-os");
  });
  it("derives a vault scope's label from the scopeVault translation", () => {
    expect(scopeChipFallbackLabel({ kind: "vault" }, t)).toBe("scopeVault");
  });
  it("falls back to the scopeAll translation only for {kind: 'all'}", () => {
    expect(scopeChipFallbackLabel({ kind: "all" }, t)).toBe("scopeAll");
  });
});
