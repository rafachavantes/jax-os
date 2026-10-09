"use client";

import { PanelLeftClose } from "lucide-react";
import { useTranslations } from "next-intl";
import { BrandImage } from "./BrandImage";
import { NavList } from "./NavList";
import type { SidebarMode } from "@/lib/nav";

export function Sidebar({
  mode,
  onToggle,
}: {
  mode: SidebarMode;
  onToggle: () => void;
}) {
  const t = useTranslations("shell");

  if (mode === "hidden") return null;

  const collapsed = mode === "collapsed";

  return (
    <aside
      className={`hidden h-full flex-none flex-col border-r border-line bg-surface py-5 transition-[width] duration-200 ease md:flex ${
        collapsed ? "w-[64px] px-2" : "w-[248px] px-3.5"
      }`}
    >
      {/* logo */}
      {collapsed ? (
        <div className="flex justify-center pb-4">
          <span className="flex h-[38px] w-[38px] flex-none items-center justify-center">
            <BrandImage variant="mark" width={29} height={26} />
          </span>
        </div>
      ) : (
        <div className="flex items-center gap-3 px-2 pb-4">
          <div className="flex flex-col gap-1">
            <BrandImage variant="logo" width={149} height={30} />
            <span className="text-[10px] uppercase tracking-[.16em] text-muted">
              {t("tagline")}
            </span>
          </div>
          <button
            onClick={onToggle}
            aria-label={t("collapse")}
            className="ml-auto flex h-9 w-9 items-center justify-center rounded-full border border-line bg-surface-2 text-body-ink transition-transform hover:bg-surface-3 hover:text-ink active:scale-95"
          >
            <PanelLeftClose className="h-4 w-4" />
          </button>
        </div>
      )}

      <NavList mode={collapsed ? "collapsed" : "expanded"} />

      {/* profile + toggle */}
      {collapsed ? (
        <div className="mt-2.5 flex flex-col items-center gap-2.5">
          <span
            title={t("profileName")}
            className="flex h-[34px] w-[34px] items-center justify-center"
          >
            <BrandImage variant="mark" width={29} height={26} alt="" />
          </span>
          <button
            onClick={onToggle}
            aria-label={t("hide")}
            className="flex h-9 w-9 items-center justify-center rounded-full border border-line bg-surface-2 text-body-ink transition-transform active:scale-95"
          >
            <PanelLeftClose className="h-4 w-4" />
          </button>
        </div>
      ) : (
        <div className="mt-2.5 flex flex-col gap-2.5 rounded-lg border border-line bg-surface-2 p-3">
          <div className="flex items-center gap-2.5">
            <span className="flex h-[34px] w-[34px] flex-none items-center justify-center">
              <BrandImage variant="mark" width={29} height={26} alt="" />
            </span>
            <div className="flex min-w-0 flex-1 flex-col gap-px">
              <span className="text-[13px] font-semibold text-ink">{t("profileName")}</span>
              <span className="flex items-center gap-1.5 text-[11px] text-accent">
                <span className="h-1.5 w-1.5 rounded-full bg-accent [animation:jax-pulse_1.6s_infinite]" />
                {t("profileStatus")}
              </span>
            </div>
          </div>
        </div>
      )}
    </aside>
  );
}
