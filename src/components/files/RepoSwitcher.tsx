"use client";

import { useQuery } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import { useEffect, useRef, useState } from "react";
import { ChevronDown, FolderGit2 } from "lucide-react";
import type { SearchScope } from "@/server/collectors/files";
import { fetchRepoNames, repoNamesFromListing } from "@/lib/fileSearch";
import { parseRecentRepos, RECENT_REPOS_KEY } from "@/lib/filesState";
import { serializeScope } from "@/lib/filesWorkspace";
import { useGeneralSettings, vaultUsable } from "@/lib/settingsQuery";
import { scopeChipRows, type ScopeChipRow } from "./ScopeChip";
import { SourceWarning } from "../mission/SourceWarning";

function readRecentScopes(): string[] {
  try { return parseRecentRepos(JSON.parse(localStorage.getItem(RECENT_REPOS_KEY) ?? "null")); }
  catch { return []; }
}
export function toScope(value: string): SearchScope {
  return value === "all" ? { kind: "all" } : value === "vault" ? { kind: "vault" } : { kind: "repo", name: value.slice(5) };
}

// §4.2: the "all" scope has no row to resolve against — its label comes from i18n directly.
export function switcherCurrentLabel(rows: ScopeChipRow[], scope: SearchScope, allLabel: string, vaultLabel: string): string {
  if (scope.kind === "all") return allLabel;
  const value = serializeScope(scope);
  return rows.find((r) => r.value === value)?.label ?? (scope.kind === "vault" ? vaultLabel : scope.name);
}

// Splits an already-ordered (recents-first) row list back into its two sections for the
// RECENTES/TODOS headers (Switcher.dc.html) — pure so the partition itself stays easy to reason
// about; ordering within each half is left exactly as scopeChipRows produced it.
export function splitSwitcherRows(rows: ScopeChipRow[], recentScopes: string[]): { recent: ScopeChipRow[]; rest: ScopeChipRow[] } {
  const recentSet = new Set(recentScopes);
  return { recent: rows.filter((r) => recentSet.has(r.value)), rest: rows.filter((r) => !recentSet.has(r.value)) };
}

// Left-panel repo/vault switcher (spec §5 item 1): a collapsed "📁 <repo> ▾" button opening a
// popover — filter field, RECENTES first then TODOS, ↑↓/Enter/Esc — replacing the old always-open
// filter+flat-list. Reuses A1's ScopeChip row ordering (scopeChipRows) minus its trailing "all"
// entry, same as before this restyle.
export function RepoSwitcher({ scope, onChange }: { scope: SearchScope; onChange: (scope: SearchScope) => void }) {
  const t = useTranslations("files");
  const { data: settings } = useGeneralSettings();
  const vaultEnabled = vaultUsable(settings);
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState("");
  const [active, setActive] = useState(0);
  const filterRef = useRef<HTMLInputElement>(null);
  const query = useQuery({ queryKey: ["files", "repoNames"], queryFn: () => fetchRepoNames(), staleTime: 60_000 });
  const failed = query.isError || (query.data != null && !query.data.ok);
  const repoNames = query.data?.ok ? repoNamesFromListing(query.data.data) : [];
  const recentScopes = readRecentScopes();
  const rows = scopeChipRows(repoNames, recentScopes, t, vaultEnabled).filter((r) => r.value !== "all");
  const visible = filter ? rows.filter((r) => r.label.toLowerCase().includes(filter.toLowerCase())) : rows;
  const { recent, rest } = splitSwitcherRows(visible, recentScopes);
  const currentValue = serializeScope(scope);
  const currentLabel = switcherCurrentLabel(rows, scope, t("switcherAllLocations"), t("scopeVault"));
  // Spec §4.2: one standalone leading row above RECENT/ALL, still matched by the filter box.
  const allLocationsRow: ScopeChipRow = { value: "all", label: t("switcherAllLocations") };
  const showAllRow = !filter || allLocationsRow.label.toLowerCase().includes(filter.toLowerCase());
  const flat = [...(showAllRow ? [allLocationsRow] : []), ...recent, ...rest];
  const highlightOffset = showAllRow ? 1 : 0;

  useEffect(() => {
    if (open) { setFilter(""); setActive(0); filterRef.current?.focus(); }
  }, [open]);
  useEffect(() => { setActive(0); }, [filter]);

  function pick(value: string) {
    onChange(toScope(value));
    setOpen(false);
  }

  return (
    <div className="relative flex-1 min-w-0">
      <button
        type="button"
        aria-label={t("repoSwitcherLabel")}
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className={`flex h-10 w-full items-center gap-2 rounded-lg border px-3 text-[14px] font-semibold text-ink ${open ? "border-brand bg-surface-2" : "border-line-strong bg-surface-2"}`}
      >
        <FolderGit2 className="h-4 w-4 flex-none text-brand" />
        <span className="flex-1 truncate text-left">{currentLabel}</span>
        <ChevronDown className={`h-3.5 w-3.5 flex-none text-muted transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open ? (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
          <div role="listbox" aria-label={t("repoSwitcherLabel")} className="absolute left-0 top-11 z-50 flex max-h-96 w-72 flex-col overflow-hidden rounded-lg border border-line-strong bg-surface-2 shadow-lg">
            <div className="border-b border-line p-2">
              <input
                ref={filterRef}
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "ArrowDown") { e.preventDefault(); setActive((i) => Math.min(i + 1, flat.length - 1)); }
                  else if (e.key === "ArrowUp") { e.preventDefault(); setActive((i) => Math.max(i - 1, 0)); }
                  else if (e.key === "Enter") { e.preventDefault(); if (flat[active]) pick(flat[active].value); }
                  else if (e.key === "Escape") { e.preventDefault(); setOpen(false); }
                }}
                placeholder={t("repoSwitcherPlaceholder")}
                aria-label={t("repoSwitcherPlaceholder")}
                className="h-8 w-full rounded-md border border-line-strong bg-base px-2 text-[13px] text-ink outline-none placeholder:text-muted"
              />
            </div>
            <div className="jax-scroll flex-1 overflow-y-auto py-1">
              {showAllRow ? <SwitcherRow row={allLocationsRow} current={scope.kind === "all"} highlighted={active === 0} onPick={pick} t={t} /> : null}
              {failed ? (
                <div className="p-2"><SourceWarning label={t("unavailable")} /></div>
              ) : visible.length === 0 && !showAllRow ? (
                <p className="px-3 py-2 text-[12px] text-muted">{t("noRepoMatch")}</p>
              ) : visible.length > 0 ? (
                <>
                  {recent.length > 0 ? (
                    <>
                      <div className="px-3 pb-1 pt-2 text-[10.5px] font-bold tracking-wide text-muted">{t("switcherRecentHeader")}</div>
                      {recent.map((r, i) => (
                        <SwitcherRow key={r.value} row={r} current={r.value === currentValue} highlighted={i + highlightOffset === active} onPick={pick} t={t} />
                      ))}
                    </>
                  ) : null}
                  {rest.length > 0 ? (
                    <>
                      <div className="border-t border-line px-3 pb-1 pt-2 text-[10.5px] font-bold tracking-wide text-muted">
                        {t("switcherAllHeader", { count: repoNames.length })}
                      </div>
                      {rest.map((r, i) => (
                        <SwitcherRow key={r.value} row={r} current={r.value === currentValue} highlighted={highlightOffset + recent.length + i === active} onPick={pick} t={t} />
                      ))}
                    </>
                  ) : null}
                </>
              ) : null}
            </div>
            <div className="border-t border-line px-3 py-1.5 text-[11px] text-muted">{t("switcherHint")}</div>
          </div>
        </>
      ) : null}
    </div>
  );
}

function SwitcherRow({ row, current, highlighted, onPick, t }: {
  row: ScopeChipRow; current: boolean; highlighted: boolean; onPick: (value: string) => void; t: (k: string) => string;
}) {
  return (
    <button
      type="button"
      role="option"
      aria-selected={current}
      onClick={() => onPick(row.value)}
      className={`flex h-9 w-full items-center gap-2 px-3 text-left text-[13.5px] ${
        current ? "bg-brand-soft font-semibold text-ink" : highlighted ? "bg-surface-3 text-body-ink" : "text-body-ink hover:bg-surface-3"
      }`}
    >
      <span className="flex-1 truncate">{row.label}</span>
      {current ? <span className="text-[11px] text-muted">{t("switcherCurrent")}</span> : null}
    </button>
  );
}
