"use client";

import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import type { Envelope } from "@/lib/api";
import type { KanbanIssue, KanbanLabel, KanbanState, LinearMember } from "@/server/collectors/linear";
import { PRIORITY_LABEL, PRIORITY_ORDER, postCreateIssue } from "./PropertiesSidebar";

const FIELD_CLASS =
  "w-full rounded-md border border-line bg-surface-2 px-2 py-1.5 text-[13px] text-ink outline-none focus:border-line-strong";

export function CreateIssueOverlay({
  teamId, states, issues, defaultProjectName, onClose, onCreated,
}: {
  teamId: string; states: KanbanState[]; issues: KanbanIssue[]; defaultProjectName: string | null;
  onClose: () => void; onCreated: (id: string) => void;
}) {
  const t = useTranslations("kanban");
  const dialogRef = useRef<HTMLDialogElement>(null);
  // {id,name} pairs derived from the board's own issues — no new "list all
  // projects" endpoint (Decision 7).
  const projects = [...new Map(issues.filter((i) => i.projectId && i.project).map((i) => [i.projectId as string, i.project as string]))]
    .map(([id, name]) => ({ id, name }))
    .sort((a, b) => a.name.localeCompare(b.name));
  const defaultProjectId = projects.find((p) => p.name === defaultProjectName)?.id ?? "";

  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [projectId, setProjectId] = useState(defaultProjectId);
  const [stateId, setStateId] = useState(states[0]?.id ?? "");
  const [priority, setPriority] = useState(0);
  const [assigneeId, setAssigneeId] = useState("");
  const [labelIds, setLabelIds] = useState<string[]>([]);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const membersQuery = useQuery<Envelope<LinearMember[]>>({
    queryKey: ["kanban", "members", teamId],
    queryFn: async () => (await fetch(`/api/kanban/members?team=${teamId}`)).json(),
  });
  const members = membersQuery.data?.ok ? membersQuery.data.data : [];
  // error contract: a rejected fetch (isError) or an {ok:false} envelope are
  // both source-unavailable — either one must surface a warning, not a
  // silently empty option list.
  const membersUnavailable = membersQuery.isError || membersQuery.data?.ok === false;
  const labelsQuery = useQuery<Envelope<KanbanLabel[]>>({
    queryKey: ["kanban", "labels", teamId],
    queryFn: async () => (await fetch(`/api/kanban/labels?team=${teamId}`)).json(),
  });
  const labels = labelsQuery.data?.ok ? labelsQuery.data.data : [];
  const labelsUnavailable = labelsQuery.isError || labelsQuery.data?.ok === false;

  useEffect(() => {
    if (dialogRef.current?.showModal) dialogRef.current.showModal();
  }, []);

  async function submit() {
    if (!title.trim() || pending) return;
    setPending(true);
    setError(null);
    const outcome = await postCreateIssue({
      teamId, title, description: description || undefined, projectId: projectId || null,
      stateId: stateId || undefined, priority, assigneeId: assigneeId || null, labelIds,
    });
    setPending(false);
    if (outcome.ok) onCreated(outcome.id);
    else setError(outcome.error);
  }

  return (
    <dialog
      ref={dialogRef}
      aria-labelledby="create-issue-title"
      onClose={onClose}
      onCancel={onClose}
      className="w-full max-w-md rounded-lg border border-line bg-base p-5"
    >
      <h2 id="create-issue-title" className="mb-4 text-base font-bold text-ink">{t("createButton")}</h2>
      <div className="flex flex-col gap-3">
        <input
          autoFocus
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder={t("createTitlePlaceholder")}
          aria-label={t("createTitlePlaceholder")}
          className={FIELD_CLASS}
        />
        <textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder={t("createDescriptionPlaceholder")}
          aria-label={t("createDescriptionPlaceholder")}
          rows={3}
          className={`${FIELD_CLASS} resize-y`}
        />
        <select value={projectId} onChange={(e) => setProjectId(e.target.value)} aria-label={t("propProject")} className={FIELD_CLASS}>
          <option value="">{t("unassigned")}</option>
          {projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
        </select>
        <select value={stateId} onChange={(e) => setStateId(e.target.value)} aria-label={t("propStatus")} className={FIELD_CLASS}>
          {states.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
        </select>
        <select value={priority} onChange={(e) => setPriority(Number(e.target.value))} aria-label={t("propPriority")} className={FIELD_CLASS}>
          {PRIORITY_ORDER.map((p) => <option key={p} value={p}>{t(PRIORITY_LABEL[p])}</option>)}
        </select>
        <select value={assigneeId} onChange={(e) => setAssigneeId(e.target.value)} aria-label={t("propAssignee")} className={FIELD_CLASS}>
          <option value="">{t("unassigned")}</option>
          {members.map((m) => <option key={m.id} value={m.id}>{m.displayName}</option>)}
        </select>
        {membersUnavailable ? (
          <span role="status" aria-live="polite" className="text-[11px] text-warning">{t("createMembersUnavailable")}</span>
        ) : null}
        {labelsUnavailable ? (
          <span role="status" aria-live="polite" className="text-[11px] text-warning">{t("createLabelsUnavailable")}</span>
        ) : null}
        {labels.length > 0 ? (
          <div className="flex flex-wrap items-center gap-1.5">
            {labels.map((l) => {
              const on = labelIds.includes(l.id);
              return (
                <button
                  key={l.id}
                  type="button"
                  onClick={() => setLabelIds(on ? labelIds.filter((id) => id !== l.id) : [...labelIds, l.id])}
                  className="rounded-full border px-2 py-0.5 text-[10px] font-semibold transition-opacity"
                  style={{ color: l.color, borderColor: l.color, opacity: on ? 1 : 0.4 }}
                >
                  {l.name}
                </button>
              );
            })}
          </div>
        ) : null}
        {error ? <span role="status" aria-live="polite" className="text-[11px] text-danger">{error}</span> : null}
        <div className="mt-1 flex justify-end gap-2">
          <button type="button" onClick={onClose} className="rounded-md px-3 py-1.5 text-[12px] font-semibold text-muted hover:text-ink">
            {t("createCancel")}
          </button>
          <button
            type="button"
            onClick={() => void submit()}
            disabled={!title.trim() || pending}
            className="rounded-md bg-brand px-3 py-1.5 text-[12px] font-semibold text-on-brand disabled:cursor-not-allowed disabled:opacity-50"
          >
            {t("createSubmit")}
          </button>
        </div>
      </div>
    </dialog>
  );
}
