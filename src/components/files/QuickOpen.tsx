"use client";

import { useEffect, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import type { Root, SearchScope } from "@/server/collectors/files";
import { parseRecents, RECENTS_KEY, type RecentEntry } from "@/lib/filesState";
import { resolveCurrentRepo, repoOf, serializeScope } from "@/lib/filesWorkspace";
import { SourceWarning } from "../mission/SourceWarning";
import { useFilesTreeWorkspace } from "./FilesWorkspaceProvider";
import { ScopeChip } from "./ScopeChip";

export type IndexHit = { root: Root; rel: string; name: string };
type IndexResponse =
  | { ok: true; data: IndexHit[]; truncated: boolean; truncatedRoots: Root[]; builtAt: string }
  | { ok: false; error: string };

export type QuickOpenEmptyState = "unavailable" | "noRecents" | "noResults" | null;

// F3: an index fetch that finished with `{ok:false}` (or the network/JSON catch below that
// synthesizes one) is an outage, not zero hits — it must never render as the same quiet
// "no results"/"no recent files" copy a genuinely empty, successful response gets. Pure so
// it's testable without jsdom, same as fuzzyScore/rankHits above.
export function quickOpenEmptyState(index: IndexResponse | null, query: string, resultsLength: number): QuickOpenEmptyState {
  if (resultsLength > 0) return null;
  if (index !== null && !index.ok) return "unavailable";
  return query === "" ? "noRecents" : "noResults";
}

// Exact match ranks above a prefix match, a prefix match ranks above a scattered subsequence
// match, no match returns null (spec §11 "fuzzyScore" bullet). Matched against "root/rel".
export function fuzzyScore(query: string, target: string): number | null {
  const q = query.toLowerCase();
  const t = target.toLowerCase();
  if (q === "") return 0;
  if (t === q) return 1000;
  if (t.startsWith(q)) return 500;
  const idx = t.indexOf(q);
  if (idx >= 0) return 300 - idx;
  let ti = 0;
  let first = -1;
  let last = -1;
  for (let qi = 0; qi < q.length; qi++) {
    const found = t.indexOf(q[qi], ti);
    if (found === -1) return null;
    if (first === -1) first = found;
    last = found;
    ti = found + 1;
  }
  return 100 - (last - first);
}

const RESULTS_CAP = 50;

// ponytail: no separate ranking service — a few thousand paths score in well under a frame.
export function rankHits(query: string, hits: IndexHit[]): IndexHit[] {
  return hits
    .map((hit) => ({ hit, score: fuzzyScore(query, `${hit.root}/${hit.rel}`) }))
    .filter((x): x is { hit: IndexHit; score: number } => x.score !== null)
    .sort((a, b) => b.score - a.score)
    .slice(0, RESULTS_CAP)
    .map((x) => x.hit);
}

export function indexFetchUrl(scope: SearchScope): string {
  // Round-2 (d4876572f596) F4: same reason as fetchFileSearch/fetchContentSearch in fileSearch.ts.
  return `/api/files/index?scope=${encodeURIComponent(serializeScope(scope))}`;
}

// Round-2 F3: a scope change starts a NEW fetch without cancelling the old one — this is the
// active-request guard that keeps a late, stale-scope response from overwriting the current index.
// A plain `let cancelled` closure flag (the other common shape for this) would work too, but
// wouldn't be independently testable; a monotonic sequence ref is, and is exercised directly above.
export function shouldApplyIndexFetch(requestSeq: number, currentSeq: number): boolean {
  return requestSeq === currentSeq;
}

// ⌘K/Ctrl+K overlay (spec §5 item 8, §8). Reads recents directly off localStorage on render, the
// same inline pattern FilesWorkspaceProvider.tsx already uses for tabs/expanded — this component is
// never server-rendered with `open: true` (it starts closed and only opens from a client keydown),
// so there is no SSR hazard; not component-tested for the same reason page.tsx's effects aren't
// (this repo has no jsdom to mount a real DOM/localStorage — verified instead by pnpm build + Rafa's
// visual pass). fuzzyScore/rankHits above carry the actual test coverage for this file.
export function QuickOpen({
  onOpenFilePinned, activeFile, scopeOverride, onScopeChange,
}: {
  onOpenFilePinned: (root: Root, rel: string) => void;
  activeFile: { root: Root; rel: string } | null;
  scopeOverride: SearchScope | null;
  onScopeChange: (scope: SearchScope | null) => void;
}) {
  const t = useTranslations("files");
  const ws = useFilesTreeWorkspace();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState(0);
  const [index, setIndex] = useState<IndexResponse | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const fetchSeqRef = useRef(0);
  const scope = scopeOverride ?? resolveCurrentRepo(activeFile, ws.selected ? repoOf(ws.selected) : null, ws.scope);
  const scopeKey = serializeScope(scope);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((o) => !o);
      } else if (e.key === "Escape") {
        setOpen(false);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    if (!open) return;
    setQuery("");
    setSelected(0);
    setIndex(null);
    inputRef.current?.focus();
    const seq = ++fetchSeqRef.current;
    fetch(indexFetchUrl(scope))
      .then((r) => r.json())
      .then((data) => { if (shouldApplyIndexFetch(seq, fetchSeqRef.current)) setIndex(data); })
      .catch(() => { if (shouldApplyIndexFetch(seq, fetchSeqRef.current)) setIndex({ ok: false, error: "unavailable" }); });
  }, [open, scopeKey]);

  if (!open) return null;

  let recents: RecentEntry[] = [];
  if (query === "") {
    try {
      recents = parseRecents(JSON.parse(localStorage.getItem(RECENTS_KEY) ?? "null"));
    } catch {
      recents = [];
    }
  }
  const hits = index?.ok ? rankHits(query, index.data) : [];
  const results: { root: Root; rel: string }[] = query === "" ? recents : hits;
  const emptyState = quickOpenEmptyState(index, query, results.length);

  function openSelected(i: number) {
    const target = results[i];
    if (!target) return;
    onOpenFilePinned(target.root, target.rel);
    setOpen(false);
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-base/60 pt-24"
      onClick={() => setOpen(false)}
    >
      <div
        role="dialog"
        aria-label={t("quickOpenLabel")}
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-lg rounded-lg border border-line bg-surface-2 shadow-lg"
      >
        <div className="flex items-center gap-2 rounded-t-lg border-b border-line px-3 py-1">
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => { setQuery(e.target.value); setSelected(0); }}
            onKeyDown={(e) => {
              if (e.key === "ArrowDown") { e.preventDefault(); setSelected((s) => Math.min(s + 1, results.length - 1)); }
              else if (e.key === "ArrowUp") { e.preventDefault(); setSelected((s) => Math.max(s - 1, 0)); }
              else if (e.key === "Enter") { e.preventDefault(); openSelected(selected); }
            }}
            placeholder={t("quickOpenPlaceholder")}
            aria-label={t("quickOpenLabel")}
            className="min-w-0 flex-1 bg-transparent py-1 text-[14px] text-body-ink outline-none"
          />
          <ScopeChip scope={scope} onChange={onScopeChange} />
        </div>
        {index?.ok && index.truncated ? (
          <p className="px-3 py-1 text-[11px] text-warning">{t("indexTruncated")}</p>
        ) : null}
        <div className="jax-scroll max-h-80 overflow-y-auto py-1">
          {emptyState === "unavailable" ? (
            <div className="px-3 py-2">
              <SourceWarning label={t("unavailable")} />
            </div>
          ) : emptyState ? (
            <p className="px-3 py-2 text-[12px] text-muted">
              {emptyState === "noRecents" ? t("quickOpenNoRecents") : t("searchNoResults")}
            </p>
          ) : (
            results.map((r, i) => {
              const full = `${r.root}/${r.rel}`;
              const slash = full.lastIndexOf("/");
              const name = full.slice(slash + 1);
              const path = full.slice(0, slash + 1);
              const rowSelected = i === selected;
              return (
                <button
                  key={`${r.root}:${r.rel}`}
                  type="button"
                  onClick={() => openSelected(i)}
                  className={`block w-full truncate px-3 py-1.5 text-left font-mono text-[12px] ${
                    rowSelected ? "bg-brand text-on-brand" : "text-body-ink hover:bg-surface-3"
                  }`}
                >
                  <span className={rowSelected ? "font-semibold" : "text-ink font-semibold"}>{name}</span>
                  <span className={rowSelected ? "" : "text-muted"}> {path}</span>
                </button>
              );
            })
          )}
        </div>
        <div className="border-t border-line px-3 py-1.5 text-[11px] text-muted">{t("quickOpenHint")}</div>
      </div>
    </div>
  );
}
