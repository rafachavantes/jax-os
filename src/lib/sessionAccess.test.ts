import { describe, expect, it } from "vitest";
import {
  buildTtydFrameUrl,
  parseExpectedBinding,
  parseSessionAccessOrigin,
  parseSessionAccessSelector,
  sessionAccessHref,
  type SessionAccessSelector,
} from "./sessionAccess";

const card: SessionAccessSelector = {
  kind: "card", project: "jax-os", harnessSession: "ab12", pane: "%7", incarnation: "1234:1700000000",
};
const unidentified: SessionAccessSelector = { ...card, harnessSession: null };

function queryOf(href: string): URLSearchParams {
  return new URL(href, "http://127.0.0.1").searchParams;
}

describe("parseSessionAccessSelector", () => {
  it("parses the exact card and generic shapes", () => {
    expect(parseSessionAccessSelector({ kind: "card", project: "p", harnessSession: null, pane: "%7", incarnation: "1234:1700000000" }))
      .toEqual({ kind: "card", project: "p", harnessSession: null, pane: "%7", incarnation: "1234:1700000000" });
    expect(parseSessionAccessSelector({ kind: "tmux", session: "project agent" }))
      .toEqual({ kind: "tmux", session: "project agent" });
  });

  it("rejects controls, NUL, empty values, unknown kind and wrong types", () => {
    expect(parseSessionAccessSelector({ kind: "tmux", session: "bad\u0000name" })).toBeNull();
    expect(parseSessionAccessSelector({ kind: "tmux", session: "bad\nname" })).toBeNull();
    expect(parseSessionAccessSelector({ kind: "tmux", session: "" })).toBeNull();
    expect(parseSessionAccessSelector({ kind: "other", session: "x" })).toBeNull();
    expect(parseSessionAccessSelector(null)).toBeNull();
    expect(parseSessionAccessSelector("x")).toBeNull();
    expect(parseSessionAccessSelector({ kind: "card", project: 1, harnessSession: null, pane: "%7", incarnation: "1:1" })).toBeNull();
    expect(parseSessionAccessSelector({ kind: "tmux", session: "x".repeat(257) })).toBeNull();
  });

  it("enforces pane, incarnation and harness caps", () => {
    expect(parseSessionAccessSelector({ kind: "card", project: "p", harnessSession: null, pane: "%12345678901", incarnation: "1:1" })).toBeNull();
    expect(parseSessionAccessSelector({ kind: "card", project: "p", harnessSession: null, pane: "%1", incarnation: "1" })).toBeNull();
    expect(parseSessionAccessSelector({ kind: "card", project: "p", harnessSession: "x".repeat(513), pane: "%1", incarnation: "1:1" })).toBeNull();
  });

  it("rejects duplicate and oversized query fields", () => {
    const dup = new URLSearchParams([["kind", "tmux"], ["session", "a"], ["session", "b"]]);
    expect(parseSessionAccessSelector(dup)).toBeNull();
    const oversized = new URLSearchParams([["kind", "tmux"], ["session", "x".repeat(2100)]]);
    expect(parseSessionAccessSelector(oversized)).toBeNull();
  });

  it("round-trips every href field, including encoded natives and a nullable harness", () => {
    for (const selector of [card, unidentified, { kind: "tmux", session: "project agent" } as const]) {
      const href = sessionAccessHref(selector, "mosaic");
      expect(href.startsWith("/tmux?")).toBe(true);
      expect(parseSessionAccessSelector(queryOf(href))).toEqual(selector);
    }
  });
});

describe("sessionAccessOrigin", () => {
  it("accepts only the fixed enum", () => {
    expect(parseSessionAccessOrigin(queryOf(sessionAccessHref(card, "home")))).toBe("home");
    const params = new URLSearchParams([["from", "https://evil.example"]]);
    expect(parseSessionAccessOrigin(params)).toBeNull();
  });
});

describe("parseExpectedBinding", () => {
  it("distinguishes absent, valid and malformed", () => {
    expect(parseExpectedBinding(new URLSearchParams())).toEqual({ kind: "absent" });
    expect(parseExpectedBinding({ expectedTarget: "$2:@3.%7", expectedIncarnation: "1234:1700000000" }))
      .toEqual({ kind: "valid", expected: { target: "$2:@3.%7", incarnation: "1234:1700000000" } });
    expect(parseExpectedBinding({ expectedTarget: "$2:@3.%7" })).toEqual({ kind: "invalid" });
    expect(parseExpectedBinding({ expectedTarget: "attach -t x", expectedIncarnation: "1:1" })).toEqual({ kind: "invalid" });
    expect(parseExpectedBinding({ expectedTarget: "$2:@3.%7", expectedIncarnation: "1:1", expectedExtra: "x" }))
      .toEqual({ kind: "invalid" });
  });
});

describe("buildTtydFrameUrl", () => {
  it("builds one bounded ID-only arg with automatic reconnect disabled", () => {
    const url = new URL(buildTtydFrameUrl("https://tail.example:10000/rw", "$2:@3.%7"));
    expect(url.pathname).toBe("/rw/");
    expect(url.searchParams.getAll("arg")).toEqual(["$2:@3.%7"]);
    expect(url.searchParams.get("disableReconnect")).toBe("true");
    expect(url.searchParams.toString()).not.toContain("session");
  });

  it("rejects a non-grammar target and any base carrying userinfo, query or fragment", () => {
    expect(() => buildTtydFrameUrl("http://127.0.0.1:1/ro", "project agent")).toThrow();
    expect(() => buildTtydFrameUrl("ftp://host/ro", "$2:@3.%7")).toThrow();
    expect(() => buildTtydFrameUrl("http://u:p@host/ro", "$2:@3.%7")).toThrow();
    expect(() => buildTtydFrameUrl("http://host/ro?x=1", "$2:@3.%7")).toThrow();
    expect(() => buildTtydFrameUrl("http://host/ro#frag", "$2:@3.%7")).toThrow();
  });
});
