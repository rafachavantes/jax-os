"use client";

import { Menu, X } from "lucide-react";
import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";
import { BrandImage } from "./BrandImage";
import { NavList } from "./NavList";
import { LanguageSelector } from "./LanguageSelector";
import { ThemeToggle, type Theme } from "./ThemeToggle";
import { AfkToggle } from "./mission/AfkToggle";

// Exported for testability — same two controls the desktop Topbar hides below md
// (Topbar.tsx), rendered here instead so the drawer can be asserted directly without
// simulating the open click (MobileNav's `open` state has no test seam).
export function MobileNavControls({ initialTheme }: { initialTheme: Theme }) {
  return (
    <div className="flex items-center gap-3 border-t border-line px-2 pt-4">
      <LanguageSelector />
      <ThemeToggle initial={initialTheme} />
    </div>
  );
}

// Exported for the same testability reason as MobileNavControls above — the mission-only
// AFK row that now sits in the drawer instead of InboxStrip.
export function MobileNavAfkRow() {
  return (
    <div className="flex items-center border-t border-line px-2 pt-4">
      <AfkToggle />
    </div>
  );
}

// Exported for the same testability reason — the drawer header (full lockup + close control).
export function MobileNavBrand({ onClose }: { onClose: () => void }) {
  const t = useTranslations("shell");
  return (
    <div className="flex items-center gap-3 px-2 pb-4">
      <BrandImage variant="logo" width={149} height={30} />
      <button
        onClick={onClose}
        aria-label={t("closeMenu")}
        className="ml-auto flex h-8 w-8 items-center justify-center rounded-full text-muted"
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  );
}

export function MobileNav({ initialTheme }: { initialTheme: Theme }) {
  const [open, setOpen] = useState(false);
  const t = useTranslations("shell");

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [open]);

  return (
    <div className="md:hidden">
      <button
        onClick={() => setOpen(true)}
        aria-label={t("openMenu")}
        className="flex h-9 w-9 items-center justify-center rounded-full border border-line bg-surface-2 text-body-ink"
      >
        <Menu className="h-4 w-4" />
      </button>

      {open ? (
        <div className="fixed inset-0 z-50 flex">
          <div
            className="absolute inset-0 bg-black/60"
            onClick={() => setOpen(false)}
            aria-hidden
          />
          <div className="relative flex h-full w-[248px] flex-col border-r border-line bg-surface px-3.5 py-5 [animation:jax-rise_.2s_ease]">
            <MobileNavBrand onClose={() => setOpen(false)} />
            <NavList onNavigate={() => setOpen(false)} />
            <div className="mt-auto flex flex-col">
              <MobileNavAfkRow />
              <MobileNavControls initialTheme={initialTheme} />
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
