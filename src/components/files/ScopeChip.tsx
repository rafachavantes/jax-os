"use client";

import { useQuery } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import { useState } from "react";
import { ChevronDown } from "lucide-react";
import type { SearchScope } from "@/server/collectors/files";
import { fetchRepoNames, repoNamesFromListing } from "@/lib/fileSearch";
import { parseRecentRepos, pushRecentRepo, RECENT_REPOS_KEY } from "@/lib/filesState";
import { serializeScope } from "@/lib/filesWorkspace";
import { useGeneralSettings, vaultUsable } from "@/lib/settingsQuery";

export type ScopeChipRow = { value: string; label: string };

// Pure ordering (§9): recents first (valid, deduped), then every repo + "vault" alphabetically,
// "All repos" always last. `vaultEnabled` (MOA-496) drops the vault row entirely when the
// integration is off — the ONE function both ScopeChip and RepoSwitcher build their lists with.
export function scopeChipRows(repoNames: string[], recentScopes: string[], t: (key: string) => string, vaultEnabled: boolean): ScopeChipRow[] {
  const toValue = (name: string) => (name === "vault" ? "vault" : `repo:${name}`);
  const labelFor = (value: string) => (value === "vault" ? t("scopeVault") : value.slice(5));
  const names = vaultEnabled ? [...repoNames, "vault"] : repoNames;
  const validValues = new Set(names.map(toValue));
  const seen = new Set<string>();
  const rows: ScopeChipRow[] = [];
  for (const value of recentScopes) {
    if (!validValues.has(value) || seen.has(value)) continue;
    seen.add(value);
    rows.push({ value, label: labelFor(value) });
  }
  for (const name of [...names].sort((a, b) => a.localeCompare(b))) {
    const value = toValue(name);
    if (seen.has(value)) continue;
    seen.add(value);
    rows.push({ value, label: labelFor(value) });
  }
  rows.push({ value: "all", label: t("scopeAll") });
  return rows;
}

// F3: while the repo list hasn't loaded, `rows` (built from it) has no matching entry for a
// repo scope — derive the label from `scope` itself instead of defaulting to "All repos".
export function scopeChipFallbackLabel(scope: SearchScope, t: (key: string) => string): string {
  if (scope.kind === "repo") return scope.name;
  if (scope.kind === "vault") return t("scopeVault");
  return t("scopeAll");
}

function readRecentScopes(): string[] {
  try { return parseRecentRepos(JSON.parse(localStorage.getItem(RECENT_REPOS_KEY) ?? "null")); }
  catch { return []; }
}
function toScope(value: string): SearchScope {
  return value === "all" ? { kind: "all" } : value === "vault" ? { kind: "vault" } : { kind: "repo", name: value.slice(5) };
}

// Filter field + recents + repo/vault/all list (§9). No dedicated interaction test — this file's
// JSX is verified by `pnpm build` + Rafa's live check, matching this codebase's existing
// convention for QuickOpen/FileSearch (only their pure helpers are unit-tested).
export function ScopeChip({ scope, onChange }: { scope: SearchScope; onChange: (scope: SearchScope) => void }) {
  const t = useTranslations("files");
  const { data: settings } = useGeneralSettings();
  const vaultEnabled = vaultUsable(settings);
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState("");
  const query = useQuery({ queryKey: ["files", "repoNames"], queryFn: () => fetchRepoNames(), staleTime: 60_000 });
  const repoNames = query.data?.ok ? repoNamesFromListing(query.data.data) : [];
  const rows = scopeChipRows(repoNames, readRecentScopes(), t, vaultEnabled);
  const visible = filter ? rows.filter((r) => r.label.toLowerCase().includes(filter.toLowerCase())) : rows;
  const currentLabel = rows.find((r) => r.value === serializeScope(scope))?.label ?? scopeChipFallbackLabel(scope, t);

  function pick(value: string) {
    if (value !== "all") {
      try { localStorage.setItem(RECENT_REPOS_KEY, JSON.stringify(pushRecentRepo(readRecentScopes(), value))); }
      catch { /* quota/private mode — best-effort, same as file recents */ }
    }
    onChange(toScope(value));
    setOpen(false);
    setFilter("");
  }

  return (
    <div className="relative">
      <button type="button" aria-label={t("scopeChipLabel")} onClick={() => setOpen((o) => !o)}
        className="flex h-8 flex-none items-center gap-1 truncate rounded-md border border-brand-soft-border bg-brand-soft px-2 text-[12px] text-ink">
        {t("scopeChipPrefix")} <b>{currentLabel}</b>
        <ChevronDown className="h-3 w-3 text-body-ink" />
      </button>
      {open ? (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
          <div className="absolute right-0 top-9 z-50 w-48 rounded-md border border-line bg-surface-2 shadow-lg">
            <input
              autoFocus value={filter} onChange={(e) => setFilter(e.target.value)} placeholder={t("scopeChipPlaceholder")}
              className="w-full border-b border-line bg-transparent px-2 py-1.5 text-[12px] text-ink outline-none"
            />
            <div className="jax-scroll max-h-56 overflow-y-auto py-1">
              {visible.length === 0 ? (
                <p className="px-2 py-1 text-[12px] text-muted">{t("noRepoMatch")}</p>
              ) : visible.map((r) => (
                <button key={r.value} type="button" onClick={() => pick(r.value)}
                  className="block w-full truncate px-2 py-1 text-left text-[12px] text-body-ink hover:bg-surface-3">
                  {r.label}
                </button>
              ))}
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
}
