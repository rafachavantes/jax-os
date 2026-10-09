"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useTranslations } from "next-intl";
import { NAV_ITEMS } from "@/lib/nav";
import { useFilesEditWorkspace } from "@/components/files/FilesWorkspaceProvider";
import { useToolsDraft } from "@/components/tools/ToolsDraftProvider";
import { shouldConfirmLeaveTools } from "@/lib/agent-settings";
import { shouldConfirmLeave } from "@/lib/filesWorkspace";
import { useGeneralSettings } from "@/lib/settingsQuery";

export function NavList({
  onNavigate,
  mode = "expanded",
}: {
  onNavigate?: () => void;
  mode?: "expanded" | "collapsed";
}) {
  const pathname = usePathname();
  const { data: settings } = useGeneralSettings();
  const linearEnabled = settings?.ok === true && settings.data.integrations.linear === true;
  const t = useTranslations("nav");
  const tShell = useTranslations("shell");
  const tFiles = useTranslations("files");
  const tTools = useTranslations("tools");
  const { dirty } = useFilesEditWorkspace();
  const tools = useToolsDraft();

  function handleNavigate(href: string, event: { preventDefault: () => void }) {
    if (shouldConfirmLeave(pathname, href, dirty) && !window.confirm(tFiles("leaveFilesDirty"))) {
      event.preventDefault();
      return;
    }
    if (
      shouldConfirmLeaveTools(pathname, href, tools.dirty || !!tools.credential)
      && !window.confirm(tTools("leaveToolsDirty"))
    ) {
      event.preventDefault();
      return;
    }
    onNavigate?.();
  }

  if (mode === "collapsed") {
    return (
      <nav className="flex flex-1 flex-col gap-[3px]">
        {NAV_ITEMS.filter((item) => !item.hidden && (item.key !== "kanban" || linearEnabled)).map((item) => {
          const active = pathname === item.href;
          const Icon = item.icon;
          return (
            <Link
              key={item.key}
              href={item.href}
              onNavigate={(event) => handleNavigate(item.href, event)}
              title={t(item.key)}
              aria-current={active ? "page" : undefined}
              className={`flex h-[42px] w-full items-center justify-center rounded-md transition-colors ${
                active
                  ? "bg-brand-soft text-brand"
                  : "text-muted hover:bg-surface-3 hover:text-ink"
              }`}
            >
              <Icon className="h-[18px] w-[18px]" />
            </Link>
          );
        })}
      </nav>
    );
  }

  return (
    <nav className="jax-scroll flex flex-1 flex-col gap-[3px] overflow-y-auto">
      <span className="px-2.5 pb-1.5 pt-3 text-[10px] font-bold uppercase tracking-[.14em] text-muted">
        {tShell("cockpit")}
      </span>
      {NAV_ITEMS.filter((item) => !item.hidden && (item.key !== "kanban" || linearEnabled)).map((item) => {
        const active = pathname === item.href;
        const Icon = item.icon;
        return (
          <Link
            key={item.key}
            href={item.href}
            onNavigate={(event) => handleNavigate(item.href, event)}
            aria-current={active ? "page" : undefined}
            className={`flex h-[42px] w-full items-center gap-3 rounded-md px-3 text-sm transition-colors ${
              active
                ? "bg-brand-soft font-semibold text-brand"
                : "font-medium text-muted hover:bg-surface-3 hover:text-ink"
            }`}
          >
            <Icon className="h-[18px] w-[18px]" />
            <span className="flex-1 text-left">{t(item.key)}</span>
            {item.badge ? (
              <span
                className={`rounded-full px-[7px] py-[2px] text-[10px] font-bold ${
                  active ? "bg-brand text-on-brand" : "bg-surface-3 text-muted"
                }`}
              >
                {item.badge}
              </span>
            ) : null}
          </Link>
        );
      })}
    </nav>
  );
}
