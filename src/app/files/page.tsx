"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useTranslations } from "next-intl";
import { Search, X } from "lucide-react";
import { FileTree } from "@/components/files/FileTree";
import { FileSearch } from "@/components/files/FileSearch";
import { FileEditor } from "@/components/files/FileEditor";
import { TabBar } from "@/components/files/TabBar";
import { QuickOpen } from "@/components/files/QuickOpen";
import { RepoSwitcher } from "@/components/files/RepoSwitcher";
import { MobileFileNav } from "@/components/files/MobileFileNav";
import { useFilesWorkspace } from "@/components/files/FilesWorkspaceProvider";
import { syncFromUrl, syncToUrl } from "@/lib/filesUrl";
import { openFromSearch, repoOf, serializeScope } from "@/lib/filesWorkspace";
import type { Root, SearchScope } from "@/server/collectors/files"; // type-only: erased at build, safe in client code

function FilesPageInner() {
  const ws = useFilesWorkspace();
  const t = useTranslations("files");
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const [urlError, setUrlError] = useState(false);
  const [searchScopeOverride, setSearchScopeOverride] = useState<SearchScope | null>(null);
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchActive, setSearchActive] = useState(false);
  // Spec §4.6: one history entry pushed on open; the back gesture pops it (popstate closes the
  // overlay without a second history.back()), arrow/Escape pop it exactly once via this guard.
  const [phoneEditorOpen, setPhoneEditorOpen] = useState(false);
  const phoneEditorHistoryPushed = useRef(false);
  function openPhoneEditor() {
    history.pushState({ phoneEditorOverlay: true }, "");
    phoneEditorHistoryPushed.current = true;
    setPhoneEditorOpen(true);
  }
  function closePhoneEditor() {
    setPhoneEditorOpen(false);
    if (phoneEditorHistoryPushed.current) { phoneEditorHistoryPushed.current = false; history.back(); }
  }
  const wsRef = useRef(ws);
  wsRef.current = ws;
  const urlApplied = useRef(false);
  const prevActiveRef = useRef<{ root: Root; rel: string; pinned: boolean } | null>(null);
  const lastWrittenUrlRef = useRef<string | null>(null);

  useEffect(() => {
    const current = wsRef.current;
    if (!current.hydrated) return;
    const active = current.activeIdx !== null ? current.tabs[current.activeIdx] : null;
    // Deep-link open routes through openFromSearch (round-2 F1): reveal/scope-switch MUST run
    // before open, or requestReveal's scope switch clears `selected` right after open set it.
    const opened = syncFromUrl(new URLSearchParams(params.toString()), active, lastWrittenUrlRef.current, {
      setUrlError,
      openFilePinned: (root, rel) => openFromSearch(current.requestReveal, current.openFilePinned, root, rel),
    });
    if (opened) {
      urlApplied.current = true;
      lastWrittenUrlRef.current = null;
    }
  }, [params, ws.hydrated]);

  useEffect(() => {
    if (!ws.hydrated) return;
    if (urlApplied.current) {
      urlApplied.current = false;
      return;
    }
    const active = ws.activeIdx !== null ? (ws.tabs[ws.activeIdx] ?? null) : null;
    lastWrittenUrlRef.current = syncToUrl(new URLSearchParams(params.toString()), pathname, active, prevActiveRef.current, {
      push: (href) => router.push(href, { scroll: false }),
      replace: (href) => router.replace(href, { scroll: false }),
    });
    prevActiveRef.current = active;
  }, [ws.hydrated, ws.tabs, ws.activeIdx, params, pathname, router]);

  const active = ws.activeIdx !== null ? (ws.tabs[ws.activeIdx] ?? null) : null;

  // Restore on return (spec §6): runs ONCE after hydration. If the persisted selection's repo
  // matches the restored scope, request the SAME reveal "Reveal active" uses — FileTree's own
  // reveal effect (Task 6) already scrolls the result into view, no separate wiring needed here.
  const restoredRef = useRef(false);
  useEffect(() => {
    if (restoredRef.current || !ws.hydrated) return;
    restoredRef.current = true;
    if (ws.selected && serializeScope(repoOf(ws.selected)) === serializeScope(ws.scope)) {
      ws.requestReveal(ws.selected.root, ws.selected.rel, ws.selected.isDir);
    }
  }, [ws.hydrated, ws.selected, ws.scope, ws.requestReveal]);

  useEffect(() => {
    if (!searchOpen) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setSearchOpen(false); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [searchOpen]);

  useEffect(() => {
    if (!phoneEditorOpen) return;
    const onPopState = () => { phoneEditorHistoryPushed.current = false; setPhoneEditorOpen(false); };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") closePhoneEditor(); };
    window.addEventListener("popstate", onPopState);
    document.addEventListener("keydown", onKey);
    return () => { window.removeEventListener("popstate", onPopState); document.removeEventListener("keydown", onKey); };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- closePhoneEditor is a stable-enough call target; only the open state should re-trigger
  }, [phoneEditorOpen]);

  return (
    <>
      {searchOpen ? (
        <div role="dialog" aria-label={t("searchScreenTitle")} className="fixed inset-0 z-50 flex flex-col bg-base md:hidden">
          <div className="flex items-center justify-between border-b border-line px-3 py-2">
            <span className="text-[13px] font-semibold text-ink">{t("searchScreenTitle")}</span>
            <button type="button" onClick={() => setSearchOpen(false)} className="text-muted"><X className="h-4 w-4" /></button>
          </div>
          <div className="flex-1 overflow-y-auto p-2">
            <FileSearch
              onOpenFile={(root, rel, line) => { openFromSearch(ws.requestReveal, ws.openFile, root, rel, line); setSearchOpen(false); openPhoneEditor(); }}
              onOpenFilePinned={(root, rel, line) => { openFromSearch(ws.requestReveal, ws.openFilePinned, root, rel, line); setSearchOpen(false); openPhoneEditor(); }}
              activeFile={active} scopeOverride={searchScopeOverride} onScopeChange={setSearchScopeOverride}
            />
          </div>
        </div>
      ) : null}
      {phoneEditorOpen ? (
        <div className="fixed inset-0 z-50 flex flex-col bg-base md:hidden">
          <FileEditor file={active} onClose={closePhoneEditor} />
        </div>
      ) : null}
      <QuickOpen onOpenFilePinned={(root, rel) => openFromSearch(ws.requestReveal, ws.openFilePinned, root, rel)} activeFile={active} scopeOverride={searchScopeOverride} onScopeChange={setSearchScopeOverride} />
      <div className="grid h-full gap-4 lg:grid-cols-[320px_1fr] [animation:jax-rise_.4s_ease]">
        <div className="flex h-full flex-col overflow-hidden rounded-lg border border-line bg-surface p-2">
          {urlError ? (
            <p role="status" aria-live="polite" className="px-1 py-1 text-[12px] text-danger">
              {t("invalidUrl")}
            </p>
          ) : null}
          <div className="hidden min-h-0 flex-1 md:flex md:flex-col">
            <FileTree
              expanded={ws.expanded}
              onToggle={ws.toggle}
              onOpenFile={ws.openFile}
              onOpenFilePinned={ws.openFilePinned}
              activeFile={active}
            />
          </div>
          <div className="flex min-h-0 flex-1 flex-col gap-2 md:hidden">
            <div className="flex gap-2">
              <RepoSwitcher scope={ws.scope} onChange={ws.setScope} />
              <button type="button" aria-label={t("searchScreenTitle")} onClick={() => setSearchOpen(true)}
                className="flex h-10 w-10 flex-none items-center justify-center rounded-lg border border-line-strong bg-surface-2 text-body-ink">
                <Search className="h-4 w-4" />
              </button>
            </div>
            <MobileFileNav onOpenFile={openPhoneEditor} />
          </div>
        </div>

        <div className="hidden h-full flex-col overflow-hidden rounded-lg border border-line bg-surface-inset md:flex">
          <div className="hidden border-b border-line px-3 py-2 md:block">
            <FileSearch
              onOpenFile={(root, rel, line) => openFromSearch(ws.requestReveal, ws.openFile, root, rel, line)}
              onOpenFilePinned={(root, rel, line) => openFromSearch(ws.requestReveal, ws.openFilePinned, root, rel, line)}
              activeFile={active} scopeOverride={searchScopeOverride} onScopeChange={setSearchScopeOverride}
              onActiveChange={setSearchActive}
            />
          </div>
          {!searchActive ? (
            <>
              {ws.tabs.length > 0 ? (
                <TabBar
                  tabs={ws.tabs}
                  activeIdx={ws.activeIdx}
                  dirtyKeys={ws.dirtyKeys}
                  onSelect={ws.setActiveIdx}
                  onPin={ws.pin}
                  onClose={ws.close}
                />
              ) : null}
              <FileEditor file={active} />
            </>
          ) : null}
        </div>
      </div>
    </>
  );
}

export default function FilesPage() {
  return (
    <Suspense fallback={null}>
      <FilesPageInner />
    </Suspense>
  );
}
