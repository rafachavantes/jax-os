import { describe, expect, it } from "vitest";
import { resolveLocale } from "./request";

describe("resolveLocale (decision 12)", () => {
  it("a valid cookie always wins over settings", () => {
    expect(resolveLocale("en-US", "pt-BR")).toBe("en-US");
  });
  it("falls back to settings.locale when the cookie is absent", () => {
    expect(resolveLocale(undefined, "pt-BR")).toBe("pt-BR");
  });
  it("falls back to settings.locale when the cookie is present but invalid", () => {
    expect(resolveLocale("fr-FR", "pt-BR")).toBe("pt-BR");
  });
  it("falls back to en-US when both cookie and settings locale are absent/invalid (497A's own default)", () => {
    expect(resolveLocale(undefined, undefined)).toBe("en-US");
    expect(resolveLocale(undefined, "fr-FR")).toBe("en-US");
  });
});
