import { describe, expect, it } from "vitest";
import { navItemFor, NAV_ITEMS } from "./nav";

describe("navItemFor — Decision 19 regression pin", () => {
  it("still resolves /health and /audit to their own NavItem once both carry hidden: true", () => {
    expect(navItemFor("/health").key).toBe("server");
    expect(navItemFor("/audit").key).toBe("audit");
  });

  it("hidden items stay present in NAV_ITEMS for navItemFor to scan", () => {
    expect(NAV_ITEMS.find((i) => i.key === "server")?.hidden).toBe(true);
    expect(NAV_ITEMS.find((i) => i.key === "audit")?.hidden).toBe(true);
    expect(NAV_ITEMS.find((i) => i.key === "mission")?.hidden).toBeUndefined();
  });
});
