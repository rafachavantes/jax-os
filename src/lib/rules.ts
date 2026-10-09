// Shared pure core for the Jax Rules feature (spec C.2 §3). Zero Node
// built-ins — this file is imported both by "use client" UI code AND by
// src/server/collectors/rules.ts, exactly the split src/lib/tools.ts /
// src/server/collectors/tools.ts already uses (a client component may never
// value-import a module that pulls in node:fs/os/crypto at module scope).

export type AppKey = "claude" | "codex" | "opencode";
export const APP_KEYS: readonly AppKey[] = ["claude", "codex", "opencode"];

// MOA-504 D10: a rule slot / destination exists only while its agent is on ("global" always does).
export function slotVisible(slot: "global" | AppKey, agents: Record<AppKey, boolean>): boolean {
  return slot === "global" || agents[slot];
}

export type RuleStatus = "synced" | "drifted" | "missing";
export type RuleDocs = { global: string; claude: string; codex: string; opencode: string };

// One SHA-256 per stored source slot (Task 1 §4). The GET response carries
// these; canonical/exception POSTs must echo the revision they read. Distinct
// from the per-app disk hash, which hashes raw destination bytes.
export type RuleSourceRevisions = Record<keyof RuleDocs, string>;

// Sentinel for "no disk file at all" (spec §4: "diskHash ... Absent file →
// the literal sentinel 'absent' (so empty ≠ missing)").
export const ABSENT_HASH = "absent";

// Normalization (spec §3, exact): strip ALL trailing \n/\r\n, then append
// exactly one \n. Interior line endings are preserved verbatim.
export function norm(t: string): string {
  return t.replace(/(\r?\n)+$/, "") + "\n";
}

// Composition (spec §3, exact): exception-first, "\n\n" joint (an empty-line
// gap — the constant that reproduces the real codex file byte-exact; spec
// amendment 2026-08-30). Empty exception ("" — the "no exception" sentinel,
// spec §3) composes to the canonical alone.
export function compose(exception: string, canonical: string): string {
  if (exception === "") return norm(canonical);
  return norm(exception) + "\n\n" + norm(canonical);
}

// Pure status decision (spec §4). diskRaw === null means "the file is
// absent" — the impure read layer (collectors/rules.ts, Task 3) is what
// turns an lstat ENOENT into null; this function never touches fs.
export function computeStatus(diskRaw: string | null, expected: string): RuleStatus {
  if (diskRaw === null) return "missing";
  return norm(diskRaw) === expected ? "synced" : "drifted";
}

// ---- Task 1: lossless H2 outline (conservative, no Markdown parser) ----

// A verified span of the ORIGINAL source string. Identity is (start, end),
// never the label: duplicate headings stay distinct sections.
export type RuleSpan = { start: number; end: number; original: string; label: string };
export type RuleOutline = { spans: RuleSpan[]; fullDocumentOnly: boolean };

const ATX_RE = /^(#{1,6})(?:[ \t]+(.*?))?[ \t]*$/;
const FENCE_OPEN_RE = /^(`{3,}|~{3,})/;
const FENCE_CLOSE_RE = /^(`{3,}|~{3,})[ \t]*$/;
const SETEXT_UNDERLINE_RE = /^(=+|-+)[ \t]*$/;

function headingLabel(line: string): string {
  return line.replace(/^#{1,6}[ \t]*/, "").replace(/[ \t]+#+[ \t]*$/, "").trim();
}

// Scans line by line, tracking fenced code and HTML comments so their
// H2-looking lines never become sections. Any structure this scanner cannot
// disambiguate (unclosed fence/comment, setext underline, no headings at all)
// sets fullDocumentOnly, and the UI falls back to whole-document editing.
export function ruleOutline(source: string): RuleOutline {
  const spans: RuleSpan[] = [];
  let fullDocumentOnly = false;
  let open: { start: number; label: string } | null = null;
  let fence: { char: string; len: number } | null = null;
  let inComment = false;
  let prevLineNonBlank = false;

  const closeOpen = (end: number) => {
    if (!open) return;
    spans.push({ start: open.start, end, original: source.slice(open.start, end), label: open.label });
    open = null;
  };

  let i = 0;
  while (i < source.length) {
    const nl = source.indexOf("\n", i);
    const lineEnd = nl === -1 ? source.length : nl + 1;
    let content = source.slice(i, lineEnd);
    if (content.endsWith("\n")) content = content.slice(0, -1);
    if (content.endsWith("\r")) content = content.slice(0, -1);

    if (fence) {
      const close = content.match(FENCE_CLOSE_RE);
      if (close && close[1][0] === fence.char && close[1].length >= fence.len) fence = null;
      prevLineNonBlank = content.trim() !== "";
      i = lineEnd;
      continue;
    }

    if (inComment) {
      if (content.includes("-->")) inComment = false;
      prevLineNonBlank = content.trim() !== "";
      i = lineEnd;
      continue;
    }

    const fenceOpen = content.match(FENCE_OPEN_RE);
    if (fenceOpen) {
      fence = { char: fenceOpen[1][0], len: fenceOpen[1].length };
      prevLineNonBlank = true;
      i = lineEnd;
      continue;
    }

    const commentAt = content.indexOf("<!--");
    if (commentAt !== -1) {
      if (!content.includes("-->", commentAt + 4)) inComment = true;
      prevLineNonBlank = content.trim() !== "";
      i = lineEnd;
      continue;
    }

    // A "---"/"===" line directly under a paragraph is a setext heading (or an
    // ambiguous thematic break) — not something this scanner can safely outline.
    if (SETEXT_UNDERLINE_RE.test(content) && prevLineNonBlank) fullDocumentOnly = true;

    if (content[0] === "#") {
      const atx = content.match(ATX_RE);
      if (atx) {
        const level = atx[1].length;
        if (level === 2) {
          closeOpen(i);
          open = { start: i, label: headingLabel(content) };
        } else if (level === 1) {
          closeOpen(i);
        }
      }
    }

    prevLineNonBlank = content.trim() !== "";
    i = lineEnd;
  }

  if (fence || inComment) fullDocumentOnly = true;
  closeOpen(source.length);
  if (spans.length === 0) fullDocumentOnly = true;
  return { spans, fullDocumentOnly };
}

// Pure splice guarded by an exact equality check on the ORIGINAL bytes. The
// client holds the full base document and its revision, so a mismatch here
// means the buffer is stale, never that a different section should be guessed.
export function replaceRuleSpan(source: string, span: RuleSpan, replacement: string): string {
  if (
    !Number.isInteger(span.start) ||
    !Number.isInteger(span.end) ||
    span.start < 0 ||
    span.end < span.start ||
    span.end > source.length ||
    source.slice(span.start, span.end) !== span.original
  ) {
    throw new Error("stale rule span");
  }
  return source.slice(0, span.start) + replacement + source.slice(span.end);
}
