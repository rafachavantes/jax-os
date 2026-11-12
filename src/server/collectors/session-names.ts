import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

// Session cards (spec §9): the display name of an agent conversation, from the runtime's own
// local registry. Both sources are undocumented local files (spec §16) — every read is
// best-effort and silent: missing, unreadable or unparseable means "no name", never an error.

const CONTROL_CHARS = /[\x00-\x1f\x7f]/g;
const NAME_MAX = 80;

export function sanitizeSessionName(raw: unknown): string | null {
  if (typeof raw !== "string") return null;
  const clean = raw.replace(CONTROL_CHARS, "").trim();
  if (!clean) return null;
  return [...clean].slice(0, NAME_MAX).join("");
}

// One Claude registry file (~/.claude/sessions/<pid>.json) — read only sessionId and name.
export function parseClaudeSessionRecord(text: string): { sessionId: string; name: string } | null {
  try {
    const value: unknown = JSON.parse(text);
    if (!value || typeof value !== "object" || Array.isArray(value)) return null;
    const record = value as { sessionId?: unknown; name?: unknown };
    if (typeof record.sessionId !== "string" || !record.sessionId) return null;
    const name = sanitizeSessionName(record.name);
    return name ? { sessionId: record.sessionId, name } : null;
  } catch {
    return null;
  }
}

// ~/.codex/session_index.jsonl — one JSON object per line; the LAST line per id wins, and a
// line whose name is unusable removes the id (a rename can be followed by a failed one).
export function parseCodexSessionIndex(text: string): Map<string, string> {
  const names = new Map<string, string>();
  for (const line of text.split("\n")) {
    try {
      const value: unknown = JSON.parse(line);
      if (!value || typeof value !== "object" || Array.isArray(value)) continue;
      const record = value as { id?: unknown; thread_name?: unknown };
      if (typeof record.id !== "string" || !record.id) continue;
      const name = sanitizeSessionName(record.thread_name);
      if (name) names.set(record.id, name);
      else names.delete(record.id);
    } catch {
      // A garbage line is skipped; other valid ids on other lines stay.
    }
  }
  return names;
}

export function readSessionNames(homeDir: string): Map<string, string> {
  const names = new Map<string, string>();

  // Claude: one small JSON file per running session, under ~/.claude and under each claude-switch
  // account (sessions started with that account's CLAUDE_CONFIG_DIR). Only regular files ending in
  // ".json" are read — never .key files (they hold credentials), symlinks or directories.
  const claudeDirs = [join(homeDir, ".claude", "sessions")];
  try {
    const accounts = join(homeDir, ".claude-switch", "accounts");
    for (const a of readdirSync(accounts, { withFileTypes: true })) {
      if (a.isDirectory()) claudeDirs.push(join(accounts, a.name, "sessions"));
    }
  } catch {
    // No claude-switch: only the default registry.
  }
  for (const dir of claudeDirs) {
    try {
      for (const entry of readdirSync(dir, { withFileTypes: true })) {
        if (!entry.isFile() || !entry.name.endsWith(".json")) continue;
        try {
          const record = parseClaudeSessionRecord(readFileSync(join(dir, entry.name), "utf8"));
          if (record) names.set(record.sessionId, record.name);
        } catch {
          // One unreadable file never hides the rest.
        }
      }
    } catch {
      // No registry (no Claude session ever ran here): no names, no warning.
    }
  }

  // Codex: independent source — its failure never suppresses the Claude names above.
  try {
    const text = readFileSync(join(homeDir, ".codex", "session_index.jsonl"), "utf8");
    for (const [id, name] of parseCodexSessionIndex(text)) names.set(id, name);
  } catch {
    // No index yet: no names, no warning.
  }

  return names;
}
