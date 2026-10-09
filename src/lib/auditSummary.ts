// Human summary per mutation kind, mapped to the REAL payload shapes found in
// the live log (create/upload have relParentDir+basename, not rel; rename has
// from/to). Pure + client-safe.
export function mutationSummary(kind: string, payload: unknown): string {
  if (typeof payload !== "object" || payload === null) return "";
  const p = payload as Record<string, unknown>;
  const s = (k: string): string | null => (typeof p[k] === "string" ? (p[k] as string) : null);

  if (kind === "file-edit" || kind === "file-delete") return [s("root"), s("rel")].filter(Boolean).join("/");
  if (kind === "file-create" || kind === "file-upload")
    return [s("root"), s("relParentDir"), s("basename")].filter(Boolean).join("/");
  if (kind === "file-rename") return `${s("from") ?? ""} → ${s("to") ?? ""}`;
  if (kind === "approval") return [s("project"), s("gate")].filter(Boolean).join(" · ");
  if (kind === "take-control" || kind === "release-control") return s("session") ?? "";
  const issueId = s("issueId"); // covers linear-move|priority|assignee|label|comment, present and future
  if (issueId) return issueId;
  return Object.entries(p)
    .filter(([k]) => k !== "ts" && k !== "kind")
    .slice(0, 3)
    .map(([k, v]) => `${k}=${String(v)}`)
    .join(" ");
}

export type AuditStatusKey = "uncertain" | "success" | "failure" | "legacy-ok" | "legacy-error" | "legacy-none";

export function auditStatusKey(row: { ok: boolean | null; payload: unknown }): AuditStatusKey {
  const payload = row.payload;
  const outcome = typeof payload === "object" && payload !== null
    ? (payload as { outcome?: unknown }).outcome
    : undefined;
  if (outcome === "pending" || outcome === "abandoned") return "uncertain";
  if (outcome === "done") return "success";
  if (outcome === "failed") return "failure";
  if (row.ok === null) return "legacy-none";
  return row.ok ? "legacy-ok" : "legacy-error";
}
