// The ONE pure rule binding a native Codex thread to its existing physical tmux pane (M3b
// correction). The title segment is only a discriminator inside an otherwise validated native
// snapshot: a healthy complete loaded-UUID discovery, a canonical owner and exactly one eligible
// physical pane are all required. Raw title text never leaves this boundary.
import { PANE_RE, UUID_RE } from "../../lib/workflow";
import { INCARNATION_RE } from "../../lib/sessionAccess";
import { isCodexCommand } from "./workflow-tmux";
import type { CodexSnapshot } from "./workflow-codex";

export type CodexTerminal = { pane: string; tmuxIncarnation: string; session: string };
export type CodexTerminalObservation = CodexTerminal & {
  command: string; dead: boolean; eligible: boolean; title: string;
};
export type CodexTerminalBinding = { project: string; terminal: CodexTerminal };

// The native rightmost-project render: 24 graphemes max, first 21 plus "..." when longer.
function projectTitle(owner: string): string {
  const parts = Array.from(new Intl.Segmenter(undefined, { granularity: "grapheme" }).segment(owner), (s) => s.segment);
  return parts.length <= 24 ? owner : parts.slice(0, 21).join("") + "...";
}

function boundedTitle(value: string): boolean {
  return value.length > 0 && Buffer.byteLength(value, "utf8") <= 1024 && !/[\u0000-\u001f\u007f]/.test(value);
}

export function codexTerminalBindings(
  snapshot: CodexSnapshot | null,
  rows: CodexTerminalObservation[],
): Map<string, CodexTerminalBinding> {
  const result = new Map<string, CodexTerminalBinding>();
  if (!snapshot?.ok || snapshot.truncated || !snapshot.loadedThreadIds) return result;
  const ids = snapshot.loadedThreadIds;
  if (ids.some((id) => !UUID_RE.test(id)) || new Set(ids).size !== ids.length) return result;
  const incarnations = new Set(rows.map((r) => r.tmuxIncarnation));
  if (incarnations.size !== 1 || !INCARNATION_RE.test(rows[0].tmuxIncarnation)) return result;
  const counts = new Map<string, number>();
  for (const row of rows) counts.set(row.pane, (counts.get(row.pane) ?? 0) + 1);
  const candidates = new Map<string, CodexTerminalBinding[]>();
  for (const row of rows) {
    if (!row.eligible || row.dead || counts.get(row.pane) !== 1 || !PANE_RE.test(row.pane) || !isCodexCommand(row.command)) continue;
    if (!row.session || Buffer.byteLength(row.session, "utf8") > 256 || /[\u0000-\u001f\u007f]/.test(row.session)) continue;
    if (!boundedTitle(row.title)) continue;
    const parts = row.title.split(" | ");
    if (parts.length < 2) continue;
    const segment = parts[parts.length - 1];
    const matches = ids.filter((id) => segment === id || segment === `${id.slice(0, 29)}...`);
    if (matches.length !== 1) continue;
    const id = matches[0];
    const roots = snapshot.threads.filter((t) => t.threadId === id);
    if (roots.length !== 1) continue;
    const root = roots[0];
    if (!root.owner || !["cli", "vscode", "appServer"].includes(root.source) || !["active", "idle"].includes(root.status)) continue;
    if (parts[parts.length - 2] !== projectTitle(root.owner)) continue;
    const list = candidates.get(id) ?? [];
    list.push({ project: root.owner, terminal: { pane: row.pane, tmuxIncarnation: row.tmuxIncarnation, session: row.session } });
    candidates.set(id, list);
  }
  for (const [id, bindings] of candidates) if (bindings.length === 1) result.set(id, bindings[0]);
  return result;
}
