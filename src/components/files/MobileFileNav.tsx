"use client";

import { useQuery } from "@tanstack/react-query";
import { useLocale, useTranslations } from "next-intl";
import { ChevronLeft, ChevronRight, Folder } from "lucide-react";
import type { DirListing, RepoRef } from "@/server/collectors/files";
import { breadcrumbSegments } from "@/lib/breadcrumb";
import { fetchRepoNames, repoNamesFromListing } from "@/lib/fileSearch";
import { DEFAULT_SCOPE } from "@/lib/filesState";
import { displayedTarget, relativeToScope, scopedRel } from "@/lib/filesWorkspace";
import { iconForEntry } from "./FileTree";
import { fmtSize } from "./FileEditor";
import { Breadcrumb } from "./Breadcrumb";
import { RelativeTime } from "../RelativeTime";
import { useFilesTreeWorkspace } from "./FilesWorkspaceProvider";
import { useGeneralSettings, vaultUsable } from "@/lib/settingsQuery";
import { SourceWarning } from "../mission/SourceWarning";

// Phone one-folder-at-a-time nav (spec §7): a plain useQuery against the EXISTING tree route, one
// directory at a time — deliberately NOT @headless-tree (a flat list needs none of its expand/
// collapse/selection machinery). Reuses ws.scope/ws.selected exactly as the desktop tree does; the
// transition logic itself (Back, tap folder, tap file) is the already-tested controller methods
// (goBack/setSelected/openFile, Task 2) — this component only renders the current listing.
export function MobileFileNav({ onOpenFile }: { onOpenFile?: () => void } = {}) {
  const t = useTranslations("files");
  const locale = useLocale();
  const { data: settings } = useGeneralSettings();
  const vaultEnabled = vaultUsable(settings);
  const ws = useFilesTreeWorkspace();
  // Global Constraints move 3: the directory readers below assume a concrete repo/vault; at the
  // "all" scope the body renders the flat picker instead, so this fallback is never observed.
  const treeScope: RepoRef = ws.scope.kind === "all" ? DEFAULT_SCOPE : ws.scope;
  const target = displayedTarget(treeScope, ws.selected);
  const canGoBack = ws.scope.kind !== "all";
  const repoNamesQuery = useQuery({ queryKey: ["files", "repoNames"], queryFn: () => fetchRepoNames(), staleTime: 60_000 });
  const repoNames = repoNamesQuery.data?.ok ? repoNamesFromListing(repoNamesQuery.data.data) : [];
  const repoNamesFailed = repoNamesQuery.isError || (repoNamesQuery.data != null && !repoNamesQuery.data.ok);
  const active = ws.activeIdx !== null ? (ws.tabs[ws.activeIdx] ?? null) : null;

  const query = useQuery<{ ok: true; data: DirListing } | { ok: false; error: string }>({
    queryKey: ["files", "tree", target.root, target.rel],
    queryFn: async () => (await fetch(`/api/files/tree?root=${target.root}&rel=${encodeURIComponent(target.rel)}`)).json(),
  });
  const listing = query.data?.ok ? query.data.data : null;
  const failed = query.isError || (query.data != null && !query.data.ok);

  function join(name: string): string {
    return target.rel ? `${target.rel}/${name}` : name;
  }
  function goTo(rel: string) {
    ws.setSelected(target.root, rel, true);
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b border-line px-1 pb-2">
        <button type="button" onClick={() => ws.goBack()} disabled={!canGoBack} aria-label={t("backToFolders")}
          className="flex-none rounded p-1 text-muted transition hover:bg-surface-3 hover:text-body-ink disabled:cursor-not-allowed disabled:opacity-30">
          <ChevronLeft className="h-4 w-4" />
        </button>
        <Breadcrumb rootLabel={t("filesRoot")} rootClickable={ws.scope.kind !== "all"} ariaLabel={t("breadcrumbLabel")}
          segments={ws.scope.kind === "all" ? [] : [
            { label: ws.scope.kind === "vault" ? t("scopeVault") : ws.scope.name, rel: "", isDir: true, collapsed: false },
            ...breadcrumbSegments(relativeToScope(ws.scope, target.rel), true),
          ]}
          onRootClick={() => ws.setScope({ kind: "all" })} onSegmentClick={(rel) => goTo(scopedRel(treeScope, rel))} />
      </div>
      <div className="jax-scroll flex-1 overflow-y-auto">
        {ws.scope.kind === "all" ? (
          repoNamesFailed ? (
            <div className="p-2"><SourceWarning label={t("unavailable")} /></div>
          ) : (
            <>
              {repoNames.map((name) => (
                <button key={name} type="button" onClick={() => ws.setScope({ kind: "repo", name })}
                  className="flex min-h-[60px] w-full items-center gap-3 border-b border-line-subtle border-l-[3px] border-transparent px-3 text-left text-[15px]">
                  <Folder className="h-5 w-5 flex-none text-warning" /><span className="truncate text-ink">{name}</span>
                  <ChevronRight className="ml-auto h-4 w-4 flex-none text-line-strong" />
                </button>
              ))}
              {vaultEnabled ? (
                <button type="button" onClick={() => ws.setScope({ kind: "vault" })}
                  className="flex min-h-[60px] w-full items-center gap-3 border-b border-line-subtle border-l-[3px] border-transparent px-3 text-left text-[15px]">
                  <Folder className="h-5 w-5 flex-none text-warning" /><span className="truncate text-ink">{t("scopeVault")}</span>
                  <ChevronRight className="ml-auto h-4 w-4 flex-none text-line-strong" />
                </button>
              ) : null}
            </>
          )
        ) : failed ? (
          <div className="p-2"><SourceWarning label={t("unavailable")} /></div>
        ) : query.isLoading ? (
          <div className="p-2"><div className="h-4 animate-pulse rounded bg-surface-2" /></div>
        ) : listing ? (
          [...listing.dirs.map((d) => ({ ...d, isDir: true as const })), ...listing.files.map((f) => ({ ...f, isDir: false as const }))].map((entry) => {
            const Icon = iconForEntry(entry.name, entry.isDir, false);
            const rel = join(entry.name);
            const open = !entry.isDir && !!active && active.root === target.root && active.rel === rel;
            return (
              <button
                key={entry.name}
                type="button"
                onClick={() => { if (entry.isDir) { goTo(rel); return; } ws.openFile(target.root, rel); onOpenFile?.(); }}
                className={`flex min-h-[60px] w-full items-center gap-3 border-b border-line-subtle border-l-[3px] px-3 text-left text-[15px] ${
                  open ? "border-accent bg-surface-3" : "border-transparent"
                }`}
              >
                <Icon className={`h-5 w-5 flex-none ${entry.isDir ? "text-warning" : "text-muted"}`} />
                <span className="flex min-w-0 flex-1 flex-col gap-0.5">
                  <span className={`truncate text-ink ${open ? "font-semibold" : ""}`}>{entry.name}</span>
                  {!entry.isDir ? (
                    <span className="truncate text-[12px] text-muted">
                      {entry.size !== undefined ? `${fmtSize(entry.size, locale)} · ` : null}
                      <RelativeTime epochMs={new Date(entry.mtime).getTime()} />
                    </span>
                  ) : null}
                </span>
                {open ? <span className="flex-none text-[11px] font-semibold text-brand">{t("openTag")}</span> : null}
                <ChevronRight className="h-4 w-4 flex-none text-line-strong" />
              </button>
            );
          })
        ) : null}
      </div>
    </div>
  );
}
