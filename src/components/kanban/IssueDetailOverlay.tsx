"use client";

import { useQuery, useQueryClient, type QueryClient } from "@tanstack/react-query";
import { Pencil, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import type { IssueComment, IssueDetail } from "@/server/collectors/linear";
import { RelativeTime } from "@/components/RelativeTime";
import { SourceWarning } from "@/components/mission/SourceWarning";
import { Markdown } from "./Markdown";
import { guidanceText, PropertiesSidebar, runWriteField, write, writeGuidance, type Translator, type WriteGuidance } from "./PropertiesSidebar";
import { createCommentSender, reconcileComment } from "@/lib/commentSubmit";

// envelope with the route's stable not-found code, absent on plain unavailability
type DetailEnvelope = { ok: true; data: IssueDetail } | { ok: false; error: string; code?: string };

function DiscardBar({ t, onKeepWriting, onDiscard }: { t: Translator; onKeepWriting: () => void; onDiscard: () => void }) {
  return (
    <div role="status" aria-live="polite" className="flex items-center gap-3 rounded-md border border-line bg-surface-2 px-3 py-2 text-[12px] text-ink">
      <span>{t("commentDiscardConfirm")}</span>
      <button type="button" onClick={onKeepWriting} className="font-semibold text-brand hover:underline">{t("discardKeepWriting")}</button>
      <button type="button" onClick={onDiscard} className="font-semibold text-danger hover:underline">{t("discardConfirm")}</button>
    </div>
  );
}

// Shared by the comment-draft close flow and the description editor's
// Esc-discard: a value differing from its saved baseline (default "",
// matching a fresh comment draft) needs a confirm before being thrown away.
// round-2 F3: trimming `current` against an untrimmed `baseline`
// false-positived on a baseline that itself carries whitespace (an
// unchanged description) — trim only for the fresh-draft (empty) baseline.
export function shouldConfirmDiscard(current: string, baseline = ""): boolean {
  return baseline !== "" ? current !== baseline : current.trim() !== "";
}

// Comment-echo cache transaction (phase 6 spec §6.4, F2/F5): both functions
// preserve the real DetailEnvelope shape end to end — no bare IssueDetail
// ever touches the cache — and no-op on a non-ok envelope, the same rule
// moveTo applies to the board cache.
export function applyCommentOptimistic(
  queryClient: QueryClient,
  key: unknown[],
  optimistic: IssueComment,
): void {
  const snapshot = queryClient.getQueryData<DetailEnvelope>(key);
  if (!snapshot?.ok) return;
  queryClient.setQueryData(key, { ...snapshot, data: { ...snapshot.data, comments: [...snapshot.data.comments, optimistic] } });
}

// Success (guidance === null, per writeGuidance), unconfirmed, and
// applied-unrecorded are all non-refusals, so reconcileComment leaves the
// temp echo in place (Part 1, Task 9) — reconcileComment itself never
// receives a server comment to swap in (diff review a5910333016f F2). This
// invalidate is what eventually replaces the temp entry via refetch in every
// one of those three cases, since the detail query has no polling interval
// of its own to catch it. Only a proven `refused` skips it — the entry was
// already removed above, nothing left to reconcile.
export function reconcileCommentCache(
  queryClient: QueryClient,
  key: unknown[],
  tempId: string,
  guidance: WriteGuidance | null,
): void {
  queryClient.setQueryData<DetailEnvelope | undefined>(key, (current) =>
    current?.ok ? { ...current, data: { ...current.data, comments: reconcileComment(current.data.comments, tempId, guidance) } } : current);
  if (guidance === null || guidance.kind === "unconfirmed" || guidance.kind === "applied-unrecorded") {
    queryClient.invalidateQueries({ queryKey: key });
  }
}

export function IssueDetailOverlay({
  issueId, onClose, boardIssueIds, onOpenParent,
}: {
  issueId: string; onClose: () => void; boardIssueIds?: Set<string>; onOpenParent?: (id: string) => void;
}) {
  const t = useTranslations("kanban");
  const queryClient = useQueryClient();
  const [commentBody, setCommentBody] = useState("");
  const [commentError, setCommentError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const dialogRef = useRef<HTMLDialogElement>(null);
  // live draft ref: getCurrent must read the CURRENT composer text, not the
  // closure captured when sendComment started — a newer draft typed during the
  // send has to survive the older completion (revision also catches it).
  const bodyRef = useRef("");
  // increments on every keystroke so a success can only clear the exact text
  // that was sent — a newer draft typed during submission always survives.
  const revisionRef = useRef(0);
  const sender = useRef(createCommentSender((issueId, body) => write("/api/kanban/comment", { issueId, body }))).current;

  const [editing, setEditing] = useState<"title" | "description" | null>(null);
  const [draft, setDraft] = useState("");
  const [fieldPending, setFieldPending] = useState(false);
  const [fieldError, setFieldError] = useState<string | null>(null);
  const [descDirty, setDescDirty] = useState(false);
  const [pendingDiscard, setPendingDiscard] = useState<"close" | "description" | null>(null);
  const fieldInflight = useRef(new Set<string>()).current;

  const detail = useQuery<DetailEnvelope>({
    queryKey: ["kanban", "issue", issueId],
    queryFn: async ({ signal }) =>
      (await fetch(`/api/kanban/issue?id=${encodeURIComponent(issueId)}`, { signal })).json(),
  });
  const d = detail.data?.ok ? detail.data.data : null;
  // the overlay must mount even when the board lacks the issue or is
  // unavailable; every editable/property source of truth lives in the detail
  // response, so nothing is fabricated before it arrives.
  const missing = detail.data && !detail.data.ok && detail.data.code === "issue-not-found";
  const failed = detail.isError || (detail.data !== undefined && !detail.data.ok && !missing);

  useEffect(() => {
    const dlg = dialogRef.current;
    if (dlg && typeof dlg.showModal === "function") dlg.showModal();
  }, []);

  // Escape (native cancel) and the close button share the dirty-comment check
  function requestClose() {
    const draftDirty = editing === "title" ? shouldConfirmDiscard(draft, d?.issue.title ?? "")
      : editing === "description" ? shouldConfirmDiscard(draft, d?.description ?? "") : false;
    if (shouldConfirmDiscard(commentBody) || draftDirty) { setPendingDiscard("close"); return; }
    closeNow();
  }
  function closeNow() {
    const dlg = dialogRef.current;
    if (dlg && typeof dlg.close === "function" && "open" in dlg && dlg.open) dlg.close();
    else onClose();
  }

  async function sendComment() {
    if (!d || !commentBody.trim() || sending || sender.isPending()) return;
    const draftMsg = { issueId: d.issue.id, body: commentBody, revision: revisionRef.current };
    const tempId = `temp-${Date.now()}`;
    const key = ["kanban", "issue", issueId];
    applyCommentOptimistic(queryClient, key, { id: tempId, author: t("you"), body: commentBody, createdAt: new Date().toISOString(), pending: true });
    setSending(true);
    setCommentError(null);
    try {
      const result = await sender.send({ draft: draftMsg, getCurrent: () => ({ issueId, body: bodyRef.current, revision: revisionRef.current }) });
      if (!result) return;
      const guidance = writeGuidance(result.outcome);
      reconcileCommentCache(queryClient, key, tempId, guidance); // invalidates on success too (diff review a5910333016f F2) — no separate invalidate needed here
      if (result.outcome.ok) {
        if (result.current) setCommentBody("");
      } else {
        setCommentError(guidance ? guidanceText(t, guidance) : t("writeError", { error: "unavailable" }));
      }
    } finally {
      setSending(false);
    }
  }

  function startEditTitle() {
    if (!d || fieldPending) return;
    setEditing("title"); setDraft(d.issue.title); setFieldError(null);
  }
  async function saveTitle() {
    if (!d || !draft.trim() || fieldPending) return;
    setFieldPending(true);
    const outcome = await runWriteField(fieldInflight, "title", () => write("/api/kanban/update", { issueId: d.issue.id, patch: { title: draft } }), queryClient, d.issue.id);
    setFieldPending(false);
    if (outcome) { const g = writeGuidance(outcome); if (g) setFieldError(guidanceText(t, g)); }
    setEditing(null);
  }
  function startEditDescription() {
    if (!d || fieldPending) return;
    setEditing("description"); setDraft(d.description); setDescDirty(false); setFieldError(null);
  }
  function handleDescriptionBlur() {
    if (!d) return;
    if (draft === d.description) { setEditing(null); return; }
    setDescDirty(true);
  }
  async function saveDescription() {
    if (!d || fieldPending) return;
    setFieldPending(true);
    const outcome = await runWriteField(fieldInflight, "description", () => write("/api/kanban/update", { issueId: d.issue.id, patch: { description: draft } }), queryClient, d.issue.id);
    setFieldPending(false);
    if (outcome) { const g = writeGuidance(outcome); if (g) setFieldError(guidanceText(t, g)); }
    setEditing(null); setDescDirty(false);
  }
  function requestDescriptionDiscard() {
    if (!d) return;
    if (shouldConfirmDiscard(draft, d.description)) { setPendingDiscard("description"); return; }
    setEditing(null); setDescDirty(false);
  }

  return (
    <dialog
      ref={dialogRef}
      aria-labelledby="issue-detail-title"
      onClose={onClose}
      onCancel={(e) => {
        e.preventDefault();
        requestClose();
      }}
      className="flex h-full max-h-full w-full max-w-none flex-col rounded-none border-0 bg-base p-0"
    >
      <div className="flex h-[46px] flex-none items-center gap-3 border-b border-line px-4">
        {d ? (
          <>
            <span className="font-mono text-xs text-muted">{d.issue.identifier}</span>
            <a
              href={d.issue.url}
              target="_blank"
              rel="noreferrer"
              className="text-xs text-brand hover:underline"
            >
              {t("detailOpenLinear")}
            </a>
          </>
        ) : (
          <span className="text-xs text-muted">{t("detailLoading")}</span>
        )}
        <button
          type="button"
          onClick={requestClose}
          autoFocus
          aria-label={t("detailClose")}
          className="ml-auto rounded p-1 text-muted hover:text-ink"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
      {pendingDiscard === "close" ? (
        <div className="border-b border-line px-4 py-2">
          <DiscardBar t={t} onKeepWriting={() => setPendingDiscard(null)} onDiscard={() => { setPendingDiscard(null); setCommentBody(""); setEditing(null); setDraft(""); setDescDirty(false); closeNow(); }} />
        </div>
      ) : null}
      <div className="grid min-h-0 flex-1 grid-cols-1 overflow-hidden md:grid-cols-[1fr_300px]">
        <div className="jax-scroll overflow-y-auto p-6">
          {d?.parent ? (
            <button
              type="button"
              onClick={() => {
                if (boardIssueIds?.has(d.parent!.id)) onOpenParent?.(d.parent!.id);
                else window.open(d.parent!.url, "_blank", "noreferrer");
              }}
              className="mb-1 block text-[11px] text-muted hover:text-brand"
            >
              {t("detailParent")}: {d.parent.identifier}
            </button>
          ) : null}
          {editing === "title" ? (
            <h1 className="mb-4 text-xl font-bold text-ink">
              <input
                id="issue-detail-title"
                autoFocus
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") void saveTitle(); else if (e.key === "Escape") setEditing(null); }}
                onBlur={() => setEditing(null)}
                disabled={fieldPending}
                aria-label={t("titlePlaceholder")}
                className="w-full rounded-md border border-line bg-surface-2 px-2 py-1 text-xl font-bold text-ink outline-none focus:border-line-strong"
              />
            </h1>
          ) : (
            <h1 className="mb-4 text-xl font-bold text-ink">
              <button type="button" id="issue-detail-title" onClick={startEditTitle} className="cursor-text text-left">
                {d ? d.issue.title : t("detailLoading")}
              </button>
            </h1>
          )}
          {missing ? (
            <SourceWarning label={t("detailMissing")} />
          ) : failed ? (
            <SourceWarning
              label={t("detailUnavailable")}
              detail={detail.data && !detail.data.ok ? detail.data.error : undefined}
              aside={detail.dataUpdatedAt ? <RelativeTime epochMs={detail.dataUpdatedAt} /> : undefined}
            />
          ) : null}
          {d && editing === "description" ? (
            <div className="flex flex-col gap-2">
              <textarea
                autoFocus
                value={draft}
                onChange={(e) => { setDraft(e.target.value); setDescDirty(false); }}
                onBlur={handleDescriptionBlur}
                onKeyDown={(e) => { if (e.key === "Escape") requestDescriptionDiscard(); }}
                disabled={fieldPending}
                aria-label={t("descriptionPlaceholder")}
                rows={6}
                className="jax-scroll w-full resize-y rounded-md border border-line bg-surface-2 p-2.5 text-[13px] text-ink outline-none focus:border-line-strong"
              />
              <div className="flex items-center gap-3">
                <button type="button" onClick={() => void saveDescription()} disabled={fieldPending} className="rounded-md bg-brand px-3 py-1.5 text-[12px] font-semibold text-on-brand disabled:cursor-not-allowed disabled:opacity-50">
                  {t("commentSend")}
                </button>
                {descDirty ? <span className="rounded-full border border-line bg-surface-2 px-2 py-0.5 text-[11px] text-muted">{t("descriptionUnsaved")}</span> : null}
                {fieldPending ? <span className="text-[11px] text-muted">{t("saving")}</span> : null}
              </div>
              {pendingDiscard === "description" ? (
                <DiscardBar t={t} onKeepWriting={() => setPendingDiscard(null)} onDiscard={() => { setPendingDiscard(null); setEditing(null); setDescDirty(false); }} />
              ) : null}
            </div>
          ) : d ? (
            // ponytail: whole-block click-to-edit stays mouse-only (a rare
            // click on a description link also opens edit mode — acceptable
            // for a single-user tool); the pencil button is the one keyboard
            // path in, and it doesn't sit inside the Markdown so it never
            // nests inside a link.
            <div className="flex flex-col gap-1.5">
              <button
                type="button"
                onClick={startEditDescription}
                aria-label={t("descriptionEdit")}
                className="self-start rounded-md p-1 text-muted hover:text-ink"
              >
                <Pencil className="h-3.5 w-3.5" />
              </button>
              <div onClick={startEditDescription} className="cursor-text">
                <Markdown>{d.description}</Markdown>
              </div>
            </div>
          ) : !missing && !failed ? (
            <div aria-busy="true" role="status" className="flex flex-col gap-2">
              <span className="text-xs text-muted">{t("detailLoading")}</span>
              <div className="h-24 animate-pulse rounded bg-surface-2" />
            </div>
          ) : null}
          {fieldError ? <span role="status" aria-live="polite" className="text-[11px] text-danger">{fieldError}</span> : null}

          {d && (d.subIssue.total > 0 || d.childrenTruncated || d.childrenIncomplete) ? (
            <div className="mt-6">
              <div className="mb-2 flex items-center gap-2">
                <h2 className="text-[13px] font-bold text-ink">{t("detailSubIssues")}</h2>
                <span className="rounded-full border border-line bg-surface-2 px-2 py-px text-[11px] text-muted">
                  {d.subIssue.done}/{d.subIssue.total}
                </span>
                {d.childrenIncomplete ? (
                  <a href={d.issue.url} target="_blank" rel="noreferrer" className="text-[10px] text-muted hover:text-brand">{t("detailIncomplete")}</a>
                ) : d.childrenTruncated ? (
                  <a href={d.issue.url} target="_blank" rel="noreferrer" className="text-[10px] text-muted hover:text-brand">{t("detailTruncated")}</a>
                ) : null}
              </div>
              <div className="flex flex-col gap-1.5">
                {d.children.map((c) => {
                  const cs = d.states.find((s) => s.name === c.stateName);
                  return (
                    <div
                      key={c.identifier}
                      className="flex items-center gap-2.5 rounded-md border border-line bg-surface px-3 py-2"
                    >
                      <span
                        className={`h-2 w-2 flex-none rounded-full ${cs ? "" : "bg-line"}`}
                        // Linear state color is runtime data, not a design token
                        style={cs ? { background: cs.color } : undefined}
                      />
                      <span className="flex-none font-mono text-[10.5px] text-muted">{c.identifier}</span>
                      <span className="truncate text-[12px] text-ink">{c.title}</span>
                      {c.assignee ? (
                        <span className="ml-auto flex-none text-[11px] text-muted">{c.assignee}</span>
                      ) : null}
                    </div>
                  );
                })}
              </div>
            </div>
          ) : null}

          {d ? (
            <div className="mt-6">
              <h2 className="mb-2 text-[13px] font-bold text-ink">{t("detailAttachments")}</h2>
              {d.attachments.length === 0 ? (
                <p className="text-[12px] text-muted">{t("detailNoAttachments")}</p>
              ) : (
                <div className="flex flex-col gap-1.5">
                  {d.attachments.map((a) => (
                    <a key={a.id} href={a.url} target="_blank" rel="noreferrer" className="truncate text-[12px] text-brand hover:underline">{a.title}</a>
                  ))}
                </div>
              )}
            </div>
          ) : null}

          {d ? (
            <div className="mt-6">
              <div className="mb-2 flex items-center gap-2">
                <h2 className="text-[13px] font-bold text-ink">{t("detailComments")}</h2>
                {d.commentsIncomplete ? (
                  <a href={d.issue.url} target="_blank" rel="noreferrer" className="text-[10px] text-muted hover:text-brand">{t("detailIncomplete")}</a>
                ) : d.commentsTruncated ? (
                  <a href={d.issue.url} target="_blank" rel="noreferrer" className="text-[10px] text-muted hover:text-brand">{t("detailTruncated")}</a>
                ) : null}
              </div>
              {d.comments.length === 0 ? (
                <p className="text-[12px] text-muted">{t("detailNoComments")}</p>
              ) : (
                <div className="flex flex-col gap-3">
                  {d.comments.map((c) => {
                    const createdMs = Date.parse(c.createdAt);
                    return (
                      <div key={c.id} className={`rounded-md border border-line bg-surface p-3 ${c.pending ? "opacity-60" : ""}`}>
                        <div className="mb-1.5 flex items-center gap-2">
                          <span className="flex h-5 w-5 items-center justify-center rounded-full bg-accent-soft text-[10px] font-bold text-accent">
                            {c.author.charAt(0).toUpperCase()}
                          </span>
                          <span className="text-[12px] font-semibold text-ink">{c.author}</span>
                          {c.pending ? (
                            <span className="text-[10px] text-muted">{t("commentSending")}</span>
                          ) : Number.isFinite(createdMs) ? (
                            <span className="ml-auto text-[11px] text-muted"><RelativeTime epochMs={createdMs} /></span>
                          ) : null}
                        </div>
                        <Markdown>{c.body}</Markdown>
                      </div>
                    );
                  })}
                </div>
              )}
              <div className="mt-3">
                <textarea
                  value={commentBody}
                  onChange={(e) => {
                    setCommentBody(e.target.value);
                    bodyRef.current = e.target.value;
                    revisionRef.current += 1;
                  }}
                  aria-label={t("commentPlaceholder")}
                  placeholder={t("commentPlaceholder")}
                  rows={3}
                  className="jax-scroll w-full resize-y rounded-md border border-line bg-surface-2 p-2.5 text-[12px] text-ink outline-none placeholder:text-muted focus:border-line-strong"
                />
                <div className="mt-2 flex items-center gap-3">
                  <button
                    onClick={() => void sendComment()}
                    disabled={!commentBody.trim() || sending}
                    className="rounded-md bg-brand px-3 py-1.5 text-[12px] font-semibold text-on-brand transition-transform active:scale-95 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {sending ? t("commentSending") : t("commentSend")}
                  </button>
                  {commentError ? (
                    <span role="status" aria-live="polite" className="text-[11px] text-danger">
                      {commentError}
                    </span>
                  ) : null}
                </div>
              </div>
            </div>
          ) : null}
        </div>
        {d ? (
          <div className="jax-scroll max-h-full overflow-y-auto border-t border-line p-4 md:max-h-none md:border-l md:border-t-0">
            <PropertiesSidebar issue={d.issue} states={d.states} teamId={d.teamId} />
          </div>
        ) : null}
      </div>
    </dialog>
  );
}
