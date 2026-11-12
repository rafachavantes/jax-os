"use client";

import { FolderGit2 } from "lucide-react";
import { useTranslations } from "next-intl";
import { useState } from "react";
import { postWorkflowAction, projectSectionKey, type MissionCard, type MissionGroupFooter, type MissionLastRun } from "@/lib/mission";
import { useGeneralSettings } from "@/lib/settingsQuery";
import { RelativeTime } from "@/components/RelativeTime";
import { DispatchForm, PrefsButtons, PrSummary } from "./ProjectCard";

// The footer's two segments, exposed pure for testing (spec §7.3).
export function footerSegments(footer: MissionGroupFooter): { worktrees: number | null; lastRun: MissionLastRun | null } {
  return { worktrees: footer.retainedWorktrees, lastRun: footer.unownedLastRun };
}

// The project header (+ its expandable project section), spec §7.1–§7.2. Rendered into the
// board entry's header slot; the project stays the unit for prefs, PR/CI, issues and Dispatch.
export function GroupHeader({
  project, expanded, expandedKey, issues, onToggle,
}: {
  project: MissionCard; expanded: boolean; expandedKey: string | null;
  issues?: number; onToggle: () => void;
}) {
  const t = useTranslations("mission");
  const { data: settings } = useGeneralSettings();
  const githubEnabled = settings?.ok === true ? settings.data.integrations.github : true;
  const group = project.group!;
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sectionKey = projectSectionKey(project.dir);
  const sectionOpen = expanded || expandedKey === sectionKey;
  const sessionCount = group.cards.filter((c) => c.scope?.kind === "session").length;
  const loop = project.loopSummary;
  const hasLoop = loop !== null && loop !== undefined && (loop.buildCount > 0 || loop.diffCount > 0 || loop.specCount > 0 || loop.planCount > 0);

  async function runAction(url: string, body: unknown) {
    setBusy(true);
    setError(null);
    const result = await postWorkflowAction<unknown>(url, body);
    setBusy(false);
    if (!result.ok) setError(result.error);
  }

  return (
    <>
      <div data-testid="group-header" className="flex min-w-0 flex-wrap items-center gap-3 text-left">
        <span className="flex h-9 w-9 flex-none items-center justify-center rounded-md bg-brand-soft text-brand">
          <FolderGit2 className="h-[18px] w-[18px]" />
        </span>
        <span className="min-w-0 flex-1 truncate text-[15px] font-semibold text-ink">{project.name}</span>
        {project.flag ? (
          <span className="flex-none rounded-full border border-brand-soft-border bg-brand-soft px-[7px] py-0.5 text-[10px] font-bold uppercase tracking-[.06em] text-brand">
            {project.flag}
          </span>
        ) : null}
        <span className="flex-none rounded-full border border-line px-2 py-0.5 text-[10.5px] font-semibold text-muted">
          {t("board.sessionsChip", { count: sessionCount })}
        </span>
        <span className="flex flex-none items-center gap-1.5 text-xs text-muted">
          <PrSummary pr={project.pr} githubEnabled={githubEnabled} />
        </span>
        {typeof issues === "number" ? <span className="flex-none text-xs text-muted">{t("projects.issueCount", { count: issues })}</span> : null}
        <button type="button" aria-expanded={sectionOpen} onClick={onToggle}
          className="flex-none rounded-md border border-line px-2.5 py-1 text-[11px] font-semibold text-body-ink">
          {t("board.projectToggle")}
        </button>
        <PrefsButtons dir={project.dir} prefs={project.prefs} visibility={project.visibility} />
      </div>

      {sectionOpen ? (
        <div data-testid="group-project-section" className="flex flex-col gap-3 border-t border-line-subtle pt-3">
          {hasLoop ? (
            <span className="text-[11px] text-muted">
              {t("expanded.loopSummary", { specCount: loop!.specCount, planCount: loop!.planCount,
                buildCount: loop!.buildCount, diffCount: loop!.diffCount, rounds: loop!.rounds })}
              {loop!.wallDays !== null ? ` · ${t("expanded.loopWallTime", { days: Math.round(loop!.wallDays) })}` : ""}
            </span>
          ) : null}
          <section className="flex flex-col gap-1.5">
            <div className="flex items-center gap-2">
              <h3 className="text-[11px] font-bold uppercase tracking-[.08em] text-muted">{t("expanded.residuals")}</h3>
              {project.residuals.length > 0 ? <span className="text-[11px] text-muted">{project.residuals.length}</span> : null}
            </div>
            {project.residuals.length === 0 ? (
              <span className="text-[12.5px] text-muted">{t("expanded.noResiduals")}</span>
            ) : (
              <ul className="flex flex-col gap-1 text-[12.5px] text-body-ink">
                {project.residuals.map((r, i) => <li key={i}>• {r}</li>)}
              </ul>
            )}
          </section>
          <DispatchForm card={project} busy={busy} onAction={runAction} />
          <span className="text-[11px] text-muted">{t("board.dispatchHint")}</span>
          {error ? <span className="text-[11.5px] text-danger">{t("actions.actionFailed", { error })}</span> : null}
        </div>
      ) : null}
    </>
  );
}

// The one-line footer (spec §7.3), rendered into the board entry's footer slot.
export function GroupFooter({ project }: { project: MissionCard }) {
  const t = useTranslations("mission");
  const footer = project.group?.footer ? footerSegments(project.group.footer) : null;
  return (
    <>
      {footer && (footer.worktrees !== null || footer.lastRun !== null) ? (
        <div data-testid="group-footer" className="flex flex-wrap items-center gap-x-2 border-t border-line-subtle pt-2 text-[11px] text-muted">
          <span>{t("board.footerNoSession")}</span>
          {footer.worktrees !== null ? <span>· {t("board.footerWorktrees", { count: footer.worktrees })}</span> : null}
          {footer.lastRun ? (
            <span>
              · {t("board.footerLastRun", { run: t(`lastRun.${footer.lastRun.labelKey}`) })}
              {footer.lastRun.finishedAt && Number.isFinite(Date.parse(footer.lastRun.finishedAt))
                ? <> · <RelativeTime epochMs={Date.parse(footer.lastRun.finishedAt)} /></> : null}
            </span>
          ) : null}
        </div>
      ) : null}
    </>
  );
}
