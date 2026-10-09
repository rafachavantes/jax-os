import { describe, expect, it } from "vitest";
import { compose, computeStatus, norm, replaceRuleSpan, ruleOutline, slotVisible, type RuleSpan } from "./rules";

describe("norm (§7 test 1)", () => {
  it("strips all trailing LF and appends exactly one", () => {
    expect(norm("foo\n\n\n")).toBe("foo\n");
    expect(norm("foo")).toBe("foo\n");
  });
  it("strips trailing CRLF too, collapsing a mixed run", () => {
    expect(norm("foo\r\n\r\n")).toBe("foo\n");
  });
  it("preserves interior line endings verbatim", () => {
    expect(norm("a\r\nb\n\n\n")).toBe("a\r\nb\n");
  });
  it("is idempotent", () => {
    const once = norm("foo\n\n");
    expect(norm(once)).toBe(once);
  });
  it("normalizes an already-clean single-LF string to itself", () => {
    expect(norm("foo\n")).toBe("foo\n");
  });
});

describe("compose (§7 test 1)", () => {
  it("no exception (empty string): canonical alone, normalized", () => {
    expect(compose("", "canon\n\n\n")).toBe("canon\n");
  });
  it("with exception: the \"\\n\\n\" joint — an empty-line gap matching the real codex layout", () => {
    expect(compose("exc\n\n", "canon\n")).toBe("exc\n\n\ncanon\n");
  });
  it("both sides get normalized independently before composing", () => {
    expect(compose("exc\r\n\r\n\r\n", "canon")).toBe("exc\n\n\ncanon\n");
  });
  // Codex-SHAPED fixture, 3-LF trailing prefix (same SHAPE as ~/.codex/AGENTS.md's
  // COMPOUND CODEX TOOL MAP + Honcho Memory block — separator mechanics only; the
  // "..." line is a stand-in, so this is NOT the real bytes. The real byte-exact
  // proof is the live probe recorded in the plan-header note).
  it("codex-shaped fixture (separator mechanics, 3-LF trailing prefix)", () => {
    const exception = [
      "<!-- BEGIN COMPOUND CODEX TOOL MAP -->",
      "## Compound Codex Tool Mapping (Claude Compatibility)",
      "...",
      "<!-- END COMPOUND CODEX TOOL MAP -->",
      "",
      "## Honcho Memory",
      "Use the `honcho-codex` skill for persistent memory via Honcho MCP tools.",
      "",
      "",
      "",
    ].join("\n"); // ends with exactly 3 raw LFs: "...tools.\n" + "\n" + "\n"
    expect(exception.endsWith("\n\n\n")).toBe(true); // sanity: fixture really has the real file's 3-LF shape
    const canonical = "<!-- BEGIN JAX RULES -->\n## Jax OS\ncontent\n<!-- END JAX RULES -->\n";
    expect(compose(exception, canonical)).toBe(norm(exception) + "\n\n" + norm(canonical));
    expect(compose(exception, canonical)).toBe(
      "<!-- BEGIN COMPOUND CODEX TOOL MAP -->\n## Compound Codex Tool Mapping (Claude Compatibility)\n...\n" +
        "<!-- END COMPOUND CODEX TOOL MAP -->\n\n## Honcho Memory\n" +
        "Use the `honcho-codex` skill for persistent memory via Honcho MCP tools.\n\n\n" +
        "<!-- BEGIN JAX RULES -->\n## Jax OS\ncontent\n<!-- END JAX RULES -->\n",
    );
  });
});

describe("computeStatus (§7 test 2)", () => {
  const expected = "canon\n";
  it("synced: disk normalizes to exactly the expected text", () => {
    expect(computeStatus("canon\n\n", expected)).toBe("synced");
    expect(computeStatus("canon\n", expected)).toBe("synced");
  });
  it("drifted: one byte different", () => {
    expect(computeStatus("canoX\n", expected)).toBe("drifted");
  });
  it("missing: null disk (the impure read layer turns ENOENT into null — this function never touches fs)", () => {
    expect(computeStatus(null, expected)).toBe("missing");
  });
  it("empty disk content is drifted (or synced if it happens to equal expected), never missing — null is the only missing signal", () => {
    expect(computeStatus("", "canon\n")).toBe("drifted");
    expect(computeStatus("", "\n")).toBe("synced"); // norm("") === "\n"
  });
});

describe("ruleOutline (Task 1 — conservative H2 outline)", () => {
  it("finds column-zero H2 sections; H1 and the preamble are not spans", () => {
    const src = "# Title\n\npreamble\n\n## A\nalpha\n\n## B\nbeta\n";
    const { spans, fullDocumentOnly } = ruleOutline(src);
    expect(fullDocumentOnly).toBe(false);
    expect(spans.map((s) => s.label)).toEqual(["A", "B"]);
    expect(spans[0]).toEqual({ start: 19, end: 31, original: "## A\nalpha\n\n", label: "A" });
    expect(spans[1]).toEqual({ start: 31, end: src.length, original: "## B\nbeta\n", label: "B" });
    expect(spans[0].original).toBe(src.slice(spans[0].start, spans[0].end));
    expect(spans[1].original).toBe(src.slice(spans[1].start, spans[1].end));
  });

  it("keeps nested H3+ headings inside the H2 span", () => {
    const src = "## A\ntext\n### nested\nmore\n## B\nx\n";
    const { spans, fullDocumentOnly } = ruleOutline(src);
    expect(fullDocumentOnly).toBe(false);
    expect(spans).toHaveLength(2);
    expect(spans[0].original).toBe("## A\ntext\n### nested\nmore\n");
    expect(spans[0].original).toContain("### nested");
  });

  it("an H1 closes an H2 span but starts no span of its own", () => {
    const src = "## A\nalpha\n# Reset\nmiddle\n## B\nbeta\n";
    const { spans } = ruleOutline(src);
    expect(spans.map((s) => s.label)).toEqual(["A", "B"]);
    expect(spans[0].original).toBe("## A\nalpha\n");
    expect(spans[1].original).toBe("## B\nbeta\n");
  });

  it("ignores H2-looking lines inside HTML comments (multi-line)", () => {
    const src = "## Real\nok\n<!--\n## Fake\n-->\n## Second\nx\n";
    const { spans, fullDocumentOnly } = ruleOutline(src);
    expect(fullDocumentOnly).toBe(false);
    expect(spans.map((s) => s.label)).toEqual(["Real", "Second"]);
    expect(spans[0].original).toContain("## Fake");
  });

  it("keeps duplicate labels as distinct spans (identity is offset, not label)", () => {
    const src = "## Dup\na\n## Dup\nb\n";
    const { spans } = ruleOutline(src);
    expect(spans).toHaveLength(2);
    expect(spans.map((s) => s.label)).toEqual(["Dup", "Dup"]);
    expect(spans[0].start).not.toBe(spans[1].start);
  });

  it("ignores headings inside backtick and tilde fences", () => {
    const src = "## A\n```\n## NotHeading\n```\n## B\n~~~\n## AlsoNot\n~~~\nend\n";
    const { spans, fullDocumentOnly } = ruleOutline(src);
    expect(fullDocumentOnly).toBe(false);
    expect(spans.map((s) => s.label)).toEqual(["A", "B"]);
    expect(spans[0].original).toContain("## NotHeading");
    expect(spans[1].original).toContain("## AlsoNot");
  });

  it("a closing fence must match the opener's character and be at least as long", () => {
    const src = "## A\n```\n## still content\n~~~~\n```\n## B\nx\n";
    const { spans } = ruleOutline(src);
    expect(spans.map((s) => s.label)).toEqual(["A", "B"]);
    expect(spans[0].original).toContain("## still content");
    expect(spans[0].original).toContain("~~~~");
  });

  it("handles CRLF line endings and offsets exactly", () => {
    const src = "## A\r\nalpha\r\n\r\n## B\r\nbeta\r\n";
    const { spans, fullDocumentOnly } = ruleOutline(src);
    expect(fullDocumentOnly).toBe(false);
    expect(spans.map((s) => s.label)).toEqual(["A", "B"]);
    for (const s of spans) expect(s.original).toBe(src.slice(s.start, s.end));
    expect(spans[0].original).toBe("## A\r\nalpha\r\n\r\n");
  });

  it("strips ATX closing hashes from the label", () => {
    const src = "## A ##\nx\n";
    expect(ruleOutline(src).spans[0].label).toBe("A");
  });

  it("empty and no-heading text are full-document only", () => {
    expect(ruleOutline("")).toEqual({ spans: [], fullDocumentOnly: true });
    expect(ruleOutline("just text\nnothing here\n")).toEqual({ spans: [], fullDocumentOnly: true });
    expect(ruleOutline("# only an h1\nbody\n")).toEqual({ spans: [], fullDocumentOnly: true });
  });

  it("an unclosed fence forces full-document editing", () => {
    const { fullDocumentOnly } = ruleOutline("## A\ntext\n```\nstill fenced\n");
    expect(fullDocumentOnly).toBe(true);
  });

  it("an unclosed HTML comment forces full-document editing", () => {
    const { fullDocumentOnly } = ruleOutline("## A\n<!-- never closed\n");
    expect(fullDocumentOnly).toBe(true);
  });

  it("setext heading underlines are ambiguous and force full-document editing", () => {
    expect(ruleOutline("Title\n=====\ncontent\n").fullDocumentOnly).toBe(true);
    expect(ruleOutline("Title\n-----\ncontent\n").fullDocumentOnly).toBe(true);
  });

  it("a # at column zero without a space is not a heading (heading text stays inside its span)", () => {
    const src = "## A\n#notaheading\n#alsonot\nx\n";
    const { spans } = ruleOutline(src);
    // neither "#notaheading" nor "#alsonot" is an ATX heading (no space after #),
    // so both lines stay inside the A span
    expect(spans).toHaveLength(1);
    expect(spans[0].original).toBe(src);
  });
});

describe("replaceRuleSpan (Task 1 — equality-checked splice)", () => {
  it("preserves the exact prefix and suffix outside the span", () => {
    const src = "pre\n\n## A\naaa\n\n## B\nbbb\n";
    const span = ruleOutline(src).spans[0];
    const replacement = span.original.replace("aaa", "aaa edited");
    const out = replaceRuleSpan(src, span, replacement);
    expect(out.slice(0, span.start)).toBe(src.slice(0, span.start));
    expect(out.slice(span.start + replacement.length)).toBe(src.slice(span.end));
    expect(out).toBe("pre\n\n## A\naaa edited\n\n## B\nbbb\n");
  });

  it("throws on a stale span whose original no longer matches", () => {
    const stale: RuleSpan = { start: 0, end: 3, original: "xyz", label: "X" };
    expect(() => replaceRuleSpan("abc", stale, "q")).toThrow("stale rule span");
  });

  it("throws on out-of-range or non-integer offsets", () => {
    const src = "abc";
    const base: RuleSpan = { start: 0, end: 3, original: "abc", label: "A" };
    expect(() => replaceRuleSpan(src, { ...base, end: 4 }, "q")).toThrow("stale rule span");
    expect(() => replaceRuleSpan(src, { ...base, start: 2, end: 1, original: "" }, "q")).toThrow("stale rule span");
    expect(() => replaceRuleSpan(src, { ...base, start: 0.5 }, "q")).toThrow("stale rule span");
  });
});

describe("slotVisible (MOA-504 D10)", () => {
  const agents = { claude: false, codex: true, opencode: true };
  it("global is always visible; an app slot follows its agent", () => {
    expect(slotVisible("global", { claude: false, codex: false, opencode: false })).toBe(true);
    expect(slotVisible("claude", agents)).toBe(false);
    expect(slotVisible("codex", agents)).toBe(true);
  });
});
