"use client";

import { X } from "lucide-react";
import { useTranslations } from "next-intl";
import { disambiguateTabLabels, tabKey, type Tab } from "@/lib/filesState";

type Props = {
  tabs: Tab[];
  activeIdx: number | null;
  dirtyKeys: Set<string>;
  onSelect: (idx: number) => void;
  onPin: (idx: number) => void;
  onClose: (idx: number) => void;
};

// Zed conventions: preview tab renders in italic; a dirty tab swaps its close
// button for a dot (hover swaps back so it stays closable).
export function TabBar({ tabs, activeIdx, dirtyKeys, onSelect, onPin, onClose }: Props) {
  const t = useTranslations("files");
  const labels = disambiguateTabLabels(tabs);
  return (
    <div className="jax-scroll flex h-[38px] flex-none items-end gap-0.5 overflow-x-auto border-b border-line bg-surface px-1.5">
      {tabs.map((tab, i) => {
        const key = tabKey(tab);
        const dirty = dirtyKeys.has(key);
        const isActive = i === activeIdx;
        const name = labels[i];
        return (
          <div
            key={key}
            title={`${tab.root} · ${tab.rel}`}
            onAuxClick={(e) => {
              if (e.button === 1) onClose(i);
            }}
            className={`group flex flex-none items-center gap-1.5 rounded-t-md border-b-2 px-3 py-1.5 text-[12px] transition-colors ${
              isActive
                ? "border-brand bg-surface-2 text-ink"
                : "border-transparent text-muted hover:bg-surface-2 hover:text-body-ink"
            }`}
          >
            <button
              onClick={() => onSelect(i)}
              onDoubleClick={() => onPin(i)}
              className={`flex max-w-48 items-baseline gap-1.5 truncate ${tab.pinned ? "" : "italic"}`}
            >
              {(() => {
                // Item 7 (spec): a collision-disambiguated label ("mission/page.tsx") splits into
                // the bold filename plus a dim parent-path suffix, matching Main.dc.html's tab —
                // an unambiguous label (no "/") renders with no suffix at all.
                const slash = name.lastIndexOf("/");
                if (slash === -1) return name;
                return (
                  <>
                    <span className="truncate">{name.slice(slash + 1)}</span>
                    <span className="shrink-0 text-[10.5px] text-muted">{name.slice(0, slash)}</span>
                  </>
                );
              })()}
            </button>
            {dirty ? (
              <span
                className="h-1.5 w-1.5 flex-none rounded-full bg-brand group-hover:hidden group-focus-within:hidden"
                aria-label={t("unsaved")}
              />
            ) : null}
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onClose(i);
              }}
              aria-label={t("closeTab")}
              className="flex-none rounded p-0.5 text-muted opacity-0 transition hover:text-body-ink group-hover:opacity-100 focus-visible:opacity-100 group-focus-within:opacity-100"
            >
              <X className="h-3 w-3" />
            </button>
          </div>
        );
      })}
    </div>
  );
}
