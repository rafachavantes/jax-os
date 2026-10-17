"use client";

import { useQuery } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";
import { File as FileIcon, Search as SearchIcon } from "lucide-react";
import type { Root, SearchScope } from "../../server/collectors/files";
import { SourceWarning } from "../mission/SourceWarning";
import { fetchContentSearch, fetchFileSearch, groupContentHits } from "../../lib/fileSearch";
import { resolveCurrentRepo, repoOf, serializeScope } from "../../lib/filesWorkspace";
import { useFilesTreeWorkspace } from "./FilesWorkspaceProvider";
import { ScopeChip, scopeChipFallbackLabel } from "./ScopeChip";

type SearchMode = "name" | "content";
type Hit = { kind: "name"; root: Root; rel: string } | { kind: "content"; root: Root; rel: string; line: number; snippet: string };
type SearchOutcome = { ok: true; hits: Hit[]; truncated: boolean } | { ok: false; error: string };

// F2: cheap query highlighting — a single case-insensitive substring wrap, no fuzzy matching.
function highlightSnippet(snippet: string, query: string) {
  const trimmed = snippet.trim().slice(0, 120);
  if (!query) return trimmed;
  const idx = trimmed.toLowerCase().indexOf(query.toLowerCase());
  if (idx === -1) return trimmed;
  return (
    <>
      {trimmed.slice(0, idx)}
      <mark className="rounded-sm bg-brand/30 text-ink">{trimmed.slice(idx, idx + query.length)}</mark>
      {trimmed.slice(idx + query.length)}
    </>
  );
}

async function runSearch(mode: SearchMode, q: string, scope: SearchScope, signal: AbortSignal): Promise<SearchOutcome> {
  if (mode === "name") {
    const r = await fetchFileSearch(q, scope, signal);
    return r.ok ? { ok: true, truncated: r.truncated, hits: r.data.map((h) => ({ kind: "name" as const, root: h.root, rel: h.rel })) } : r;
  }
  const r = await fetchContentSearch(q, scope, signal);
  return r.ok ? { ok: true, truncated: r.truncated, hits: r.data.map((h) => ({ kind: "content" as const, ...h })) } : r;
}

export function FileSearch({
  onOpenFile, onOpenFilePinned, activeFile, scopeOverride, onScopeChange, onActiveChange,
}: {
  onOpenFile: (root: Root, rel: string, line?: number) => void;
  onOpenFilePinned: (root: Root, rel: string, line?: number) => void;
  activeFile: { root: Root; rel: string } | null;
  scopeOverride: SearchScope | null;
  onScopeChange: (scope: SearchScope | null) => void;
  // Item 6 (spec): the desktop bar lives above the editor/tabs — results replace them while a
  // query is active. Purely a layout signal for the parent; no search behavior changes.
  onActiveChange?: (active: boolean) => void;
}) {
  const t = useTranslations("files");
  const ws = useFilesTreeWorkspace();
  const [input, setInput] = useState("");
  const [q, setQ] = useState("");
  const [mode, setMode] = useState<SearchMode>("name");
  const scope = scopeOverride ?? resolveCurrentRepo(activeFile, ws.selected ? repoOf(ws.selected) : null, ws.scope);
  const scopeLabel = scopeChipFallbackLabel(scope, t);

  useEffect(() => {
    const id = setTimeout(() => setQ(input.trim()), 200);
    return () => clearTimeout(id);
  }, [input]);

  useEffect(() => {
    onActiveChange?.(q.length > 0);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- onActiveChange is a setState setter from the caller, stable across renders
  }, [q]);

  // Opening a hit ends the search so the editor comes back (desktop hides it while a query is active).
  const pick = (open: (root: Root, rel: string, line?: number) => void) => (root: Root, rel: string, line?: number) => {
    open(root, rel, line);
    setInput("");
    setQ("");
  };
  const openHit = pick(onOpenFile);
  const openHitPinned = pick(onOpenFilePinned);

  const query = useQuery({
    queryKey: ["files", "search", mode, q, serializeScope(scope)],
    queryFn: ({ signal }) => runSearch(mode, q, scope, signal),
    enabled: q.length > 0, staleTime: 10000, retry: false,
  });

  const hits = query.data?.ok ? query.data.hits : null;
  const truncated = query.data?.ok ? query.data.truncated : false;
  const failed = query.isError || (query.data != null && !query.data.ok);
  const label = (name: string, contentName: string) => (mode === "name" ? t(name) : t(contentName));

  let body = null;
  if (q.length === 0) {
    body = null;
  } else if (failed) {
    body = <SourceWarning label={label("searchFailed", "contentSearchFailed")} detail={query.data && !query.data.ok ? query.data.error : undefined} />;
  } else {
    const loading = query.isFetching ? <p aria-live="polite" className="px-1 py-1 text-[12px] text-muted">{t("searchLoading")}</p> : null;
    const incomplete = <p aria-live="polite" className="px-1 py-1 text-[12px] text-muted">{label("searchIncomplete", "contentSearchIncomplete")}</p>;
    let results = null;
    if (hits && hits.length > 0) {
      results = (
        <div className="jax-scroll max-h-[60vh] overflow-y-auto">
          {mode === "name" ? (
            <div className="flex flex-col gap-0.5">
              {hits.map((h) => {
                const full = `${h.root}/${h.rel}`;
                const slash = full.lastIndexOf("/");
                const name = full.slice(slash + 1);
                const dir = full.slice(0, slash + 1);
                return (
                  // Item 10 (spec): two lines, name then dim dir — same shape on desktop and phone.
                  <button key={`${h.root}:${h.rel}`} onClick={() => openHit(h.root, h.rel)} onDoubleClick={() => openHitPinned(h.root, h.rel)}
                    className="flex flex-col gap-0.5 rounded-md px-2 py-1.5 text-left font-mono transition-colors hover:bg-surface-2">
                    <span className="truncate text-[13.5px] text-ink">{name}</span>
                    <span className="truncate text-[12px] text-muted">{dir}</span>
                  </button>
                );
              })}
            </div>
          ) : (
            <div className="flex flex-col gap-3">
              {groupContentHits(hits.filter((h): h is Extract<Hit, { kind: "content" }> => h.kind === "content")).map((group) => {
                const full = `${group.root}/${group.rel}`;
                const slash = full.lastIndexOf("/");
                const name = full.slice(slash + 1);
                const parent = full.slice(0, slash + 1);
                return (
                  <div key={`${group.root}:${group.rel}`} className="overflow-hidden rounded-lg border border-line bg-surface">
                    <div className="flex items-center gap-2 border-b border-line bg-surface-2 px-3 py-2">
                      <FileIcon className="h-3.5 w-3.5 flex-none text-muted" />
                      <span className="truncate font-mono text-[13px] font-semibold text-ink">{name}</span>
                      <span className="truncate font-mono text-[12px] text-muted">{parent}</span>
                      <span className="ml-auto shrink-0 text-[11px] text-muted">{t("contentMatchCount", { count: group.hits.length })}</span>
                    </div>
                    {group.hits.map((hit) => (
                      <button key={`${group.root}:${group.rel}:${hit.line}`} onClick={() => openHit(group.root, group.rel, hit.line)} onDoubleClick={() => openHitPinned(group.root, group.rel, hit.line)}
                        className="flex w-full gap-3 border-t border-line-subtle px-3 py-1.5 text-left font-mono text-[12.5px] transition-colors hover:bg-surface-2">
                        <span className="w-9 flex-none text-right text-muted">{`:${hit.line}`}</span>
                        <span className="min-w-0 flex-1 truncate text-body-ink">{highlightSnippet(hit.snippet, q)}</span>
                      </button>
                    ))}
                  </div>
                );
              })}
            </div>
          )}
          {truncated ? incomplete : null}
        </div>
      );
    } else if (hits && truncated) {
      results = incomplete;
    } else if (hits) {
      results = <p className="px-1 py-1 text-[12px] text-muted">{label("searchNoResults", "contentSearchNoResults")}</p>;
    }
    body = <>{loading}{results}</>;
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative flex h-10 min-w-0 basis-full items-center gap-2 rounded-lg border border-line-strong bg-surface px-3 focus-within:border-brand md:basis-0 md:flex-1">
          <SearchIcon className="h-4 w-4 flex-none text-muted" />
          <input
            type="search" value={input} onChange={(e) => setInput(e.target.value)}
            placeholder={t(mode === "name" ? "searchPlaceholder" : "contentSearchPlaceholder", { scope: scopeLabel })}
            aria-label={label("searchLabel", "contentSearchLabel")}
            className="h-full min-w-0 flex-1 bg-transparent text-[13.5px] text-ink outline-none placeholder:text-muted"
          />
          {input.length === 0 ? (
            <span className="flex-none rounded border border-line-strong px-1.5 py-0.5 font-mono text-[11px] text-muted">{t("searchKeyHint")}</span>
          ) : null}
        </div>
        <div role="group" aria-label={t("searchModeGroupLabel")} className="flex flex-1 items-center gap-0.5 rounded-lg border border-line-strong bg-surface p-[3px] md:flex-none">
          <button type="button" aria-pressed={mode === "name"} onClick={() => setMode("name")} className={`h-8 flex-1 rounded-md px-3 text-[12.5px] md:flex-none ${mode === "name" ? "bg-surface-3 font-semibold text-ink" : "text-muted"}`}>{t("nameSearchLabel")}</button>
          <button type="button" aria-pressed={mode === "content"} onClick={() => setMode("content")} className={`h-8 flex-1 rounded-md px-3 text-[12.5px] md:flex-none ${mode === "content" ? "bg-surface-3 font-semibold text-ink" : "text-muted"}`}>{t("contentSearchLabel")}</button>
        </div>
        <ScopeChip scope={scope} onChange={onScopeChange} />
      </div>
      {q.length > 0 ? (
        <div className="flex min-h-0 flex-1 flex-col">
          {body}
          {scope.kind !== "all" ? (
            <p className="px-1 pt-3 text-[12px] text-muted">
              {t(mode === "name" ? "nameSearchHint" : "contentSearchHint", { scope: scopeLabel })}
            </p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
