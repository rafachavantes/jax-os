"use client";

import { useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import type { SidebarMode } from "@/lib/nav";
import type { Theme } from "@/components/ThemeToggle";
import { UnsavedFilesNotice, useFilesEditWorkspace } from "./files/FilesWorkspaceProvider";
import { Sidebar } from "./Sidebar";
import { Topbar } from "./Topbar";

const SIDEBAR_COOKIE = "sidebar";
const MAX_AGE = 31536000;

function nextMode(mode: SidebarMode): SidebarMode {
  if (mode === "expanded") return "collapsed";
  if (mode === "collapsed") return "hidden";
  return "expanded";
}

export function Shell({
  initialSidebar,
  initialTheme,
  children,
}: {
  initialSidebar: SidebarMode;
  initialTheme: Theme;
  children: React.ReactNode;
}) {
  const [sidebar, setSidebar] = useState<SidebarMode>(initialSidebar);
  const files = useFilesEditWorkspace();
  const tShell = useTranslations("shell");

  function setSidebarMode(next: SidebarMode) {
    setSidebar(next);
    document.cookie = `${SIDEBAR_COOKIE}=${next};path=/;max-age=${MAX_AGE};SameSite=Lax`;
  }

  function cycleSidebar() {
    setSidebarMode(nextMode(sidebar));
  }

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (!(e.metaKey || e.ctrlKey) || e.key !== "b") return;
      const el = e.target as HTMLElement | null;
      if (el?.closest("input, textarea, .monaco-editor")) return;
      e.preventDefault();
      setSidebarMode(nextMode(sidebar));
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [sidebar]);

  return (
    <div className="flex h-screen w-full overflow-hidden">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-[100] focus:rounded-md focus:bg-brand focus:px-3 focus:py-1.5 focus:text-sm focus:font-semibold focus:text-on-brand"
      >
        {tShell("skipToContent")}
      </a>
      <Sidebar mode={sidebar} onToggle={cycleSidebar} />
      <div className="flex h-full min-w-0 flex-1 flex-col">
        <Topbar initialTheme={initialTheme} sidebar={sidebar} onToggleSidebar={cycleSidebar} />
        {files.dirty ? <UnsavedFilesNotice /> : null}
        <main id="main-content" tabIndex={-1} className="jax-scroll min-h-0 flex-1 overflow-y-auto p-6">
          {children}
        </main>
      </div>
    </div>
  );
}
