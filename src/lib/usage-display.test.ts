import { describe, expect, it } from "vitest";
import { isCollectionStale, isUnresolvedModel, STALE_AFTER_SECONDS } from "./usage-display";

describe("isCollectionStale", () => {
  it("is stale only after the five-minute window", () => {
    expect(STALE_AFTER_SECONDS).toBe(300);
    expect(isCollectionStale({ state: "available", lastRefresh: 1000 }, 1301)).toBe(true);
    expect(isCollectionStale({ state: "available", lastRefresh: 1000 }, 1300)).toBe(false);
    expect(isCollectionStale({ state: "available", lastRefresh: null }, 99_999)).toBe(false);
    expect(isCollectionStale({ state: "disabled", lastRefresh: 1000 }, 99_999)).toBe(false);
  });
});

describe("isUnresolvedModel", () => {
  it("matches only the builder alias suffixes", () => {
    expect(isUnresolvedModel("opencode-go/jaxflow-builder-default")).toBe(true);
    expect(isUnresolvedModel("xai/jaxflow-builder-fallback")).toBe(true);
    expect(isUnresolvedModel("xai/grok-4.7")).toBe(false);
  });
});
