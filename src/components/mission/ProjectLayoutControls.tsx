"use client";

import { GripVertical } from "lucide-react";
import { useTranslations } from "next-intl";
import type { LayoutEntry, PrScope, ProjectWidth } from "@/lib/cockpitLayout";
import type { OrganizeState } from "@/lib/cockpitPrefsQuery";

export const PROJECT_DRAG_MIME = "application/x-jax-project";

// Only the dedicated handle starts a drag; the card's own buttons, selects and text are never
// draggable (spec: "Only edit mode enables project drag handles ... preventing accidental
// movement during normal interaction or text selection").
export function ProjectOrganizeHandle({
  dir, name, width, index, count, onMove, onWidth,
}: {
  dir: string; name: string; width: ProjectWidth; index: number; count: number;
  onMove: (dir: string, delta: -1 | 1) => void;
  onWidth: (dir: string, width: ProjectWidth) => void;
}) {
  const t = useTranslations("mission.projects.organize");
  return (
    <div data-testid="organize-handle-row" className="flex min-w-0 flex-wrap items-center gap-2 rounded-md border border-dashed border-line-strong bg-surface-2 px-2 py-1">
      <span
        draggable
        role="button"
        tabIndex={0}
        aria-label={t("drag", { name })}
        data-project-dir={dir}
        onDragStart={(e) => {
          e.dataTransfer.setData(PROJECT_DRAG_MIME, dir);
          e.dataTransfer.setData("text/plain", dir);
          e.dataTransfer.effectAllowed = "move";
        }}
        className="flex cursor-grab items-center rounded px-1 text-muted focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand"
      >
        <GripVertical className="h-4 w-4" aria-hidden />
      </span>
      <label className="flex items-center gap-1 text-[11px] text-muted">
        <span>{t("widthLabel")}</span>
        <select
          value={width}
          aria-label={t("width", { name })}
          onChange={(e) => onWidth(dir, e.target.value as ProjectWidth)}
          className="rounded border border-line bg-surface px-1.5 py-0.5 text-[11px] text-body-ink"
        >
          <option value="third">{t("third")}</option>
          <option value="half">{t("half")}</option>
          <option value="full">{t("full")}</option>
        </select>
      </label>
      <button
        type="button"
        disabled={index === 0}
        onClick={() => onMove(dir, -1)}
        className="rounded border border-line px-1.5 py-0.5 text-[11px] font-semibold text-body-ink disabled:cursor-not-allowed disabled:text-muted"
        aria-label={t("moveEarlier", { name })}
      >
        ↑
      </button>
      <button
        type="button"
        disabled={index === count - 1}
        onClick={() => onMove(dir, 1)}
        className="rounded border border-line px-1.5 py-0.5 text-[11px] font-semibold text-body-ink disabled:cursor-not-allowed disabled:text-muted"
        aria-label={t("moveLater", { name })}
      >
        ↓
      </button>
    </div>
  );
}

export function readDraggedProject(dataTransfer: Pick<DataTransfer, "getData">, draft: LayoutEntry[]): string | null {
  const dir = dataTransfer.getData(PROJECT_DRAG_MIME) || dataTransfer.getData("text/plain");
  if (!dir || !/^[\w./-]+$/.test(dir)) return null;
  return draft.some((e) => e.project === dir) ? dir : null;
}

export function ProjectLayoutControls({
  unavailable, organize, onEnter, onSave, onCancel, onReset, onReload,
  prScope, scopeBusy, scopeError, onScope,
}: {
  unavailable: boolean;
  organize: OrganizeState;
  onEnter: () => void;
  onSave: () => void;
  onCancel: () => void;
  onReset: () => void;
  onReload: () => void;
  prScope: PrScope;
  scopeBusy: boolean;
  scopeError: string | null;
  onScope: (scope: PrScope) => void;
}) {
  const t = useTranslations("mission.projects");
  const editing = organize.draft !== null;
  const saving = organize.status === "saving";
  const btn = "rounded-md border border-line px-2.5 py-1 text-[11px] font-semibold text-body-ink disabled:cursor-not-allowed disabled:border-line-subtle disabled:text-muted";
  const scopeBtn = (active: boolean) =>
    `rounded-md border px-2 py-0.5 text-[11px] font-semibold ${active ? "border-brand-soft-border bg-brand-soft text-brand" : "border-line text-muted hover:text-body-ink"}`;
  return (
    <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1.5" data-testid="projects-controls">
      {unavailable ? <span className="text-[11.5px] text-warning">{t("controls.unavailable")}</span> : null}
      {editing ? (
        <>
          <button type="button" className={btn} disabled={saving} onClick={onSave}>{t("controls.save")}</button>
          <button type="button" className={btn} disabled={saving} onClick={onCancel}>{t("controls.cancel")}</button>
        </>
      ) : (
        <button type="button" className={btn} disabled={unavailable} onClick={onEnter}>{t("controls.organize")}</button>
      )}
      <button type="button" className={btn} disabled={unavailable || saving} onClick={onReset}>{t("controls.reset")}</button>
      {organize.status === "saving" ? <span className="text-[11px] text-muted">{t("controls.saving")}</span> : null}
      {organize.status === "error" && organize.error ? (
        <span className="text-[11.5px] text-danger">{t("controls.saveError", { error: organize.error })}</span>
      ) : null}
      {organize.status === "conflict" ? (
        <span className="flex items-center gap-1.5 text-[11.5px] text-warning">
          {t("controls.conflict")}
          <button type="button" className="font-semibold text-body-ink underline" onClick={onReload}>{t("controls.reload")}</button>
        </span>
      ) : null}
      <span className="ml-auto flex flex-none items-center gap-1.5 text-[11px] text-muted">
        {t("controls.prScope")}
        <button type="button" disabled={unavailable || scopeBusy} aria-pressed={prScope === "mine"} className={scopeBtn(prScope === "mine")} onClick={() => onScope("mine")}>
          {t("controls.mine")}
        </button>
        <button type="button" disabled={unavailable || scopeBusy} aria-pressed={prScope === "all"} className={scopeBtn(prScope === "all")} onClick={() => onScope("all")}>
          {t("controls.all")}
        </button>
      </span>
      {scopeError ? <span className="basis-full text-[11.5px] text-danger">{t("controls.scopeError")}</span> : null}
    </div>
  );
}
