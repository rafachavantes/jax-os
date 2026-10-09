"use client";

import type { BreadcrumbSegment } from "@/lib/breadcrumb";

// Shared clickable/collapsing breadcrumb (spec §5): FileEditor's own per-file breadcrumb and the
// new tree-panel/phone breadcrumbs (Tasks 9/11) all render through this. A `collapsed:true` segment
// ("…") is NEVER clickable, matching FileEditor's pre-extraction plain-span behavior. The final
// segment is clickable only when `finalClickable` — the tree-panel breadcrumb passes `false` so
// its "current" crumb is a plain, non-interactive span.
type BreadcrumbProps = {
  rootLabel: string; rootTitle?: string; ariaLabel?: string; segments: BreadcrumbSegment[];
  rootClickable?: boolean; finalClickable?: boolean; onRootClick: () => void; onSegmentClick: (rel: string, isDir: boolean) => void;
};
export function Breadcrumb({
  rootLabel, rootTitle, ariaLabel, segments, rootClickable = true, finalClickable = true, onRootClick, onSegmentClick,
}: BreadcrumbProps) {
  const rootTitleValue = rootTitle ?? rootLabel;
  return (
    <nav aria-label={ariaLabel} className="flex min-w-0 items-center gap-1 truncate font-mono text-xs">
      {rootClickable ? (
        <button type="button" onClick={onRootClick} className="shrink-0 text-body-ink hover:underline" title={rootTitleValue}>
          {rootLabel}
        </button>
      ) : (
        <span className="shrink-0 text-ink font-semibold" title={rootTitleValue}>{rootLabel}</span>
      )}
      {segments.map((seg, i) => {
        const isLast = i === segments.length - 1;
        const clickable = !seg.collapsed && (!isLast || finalClickable);
        const title = `${rootTitleValue}/${seg.rel}`;
        return (
          <span key={`${seg.rel}:${seg.collapsed ? "c" : "s"}:${i}`} className="flex shrink-0 items-center gap-1">
            <span className="text-muted">/</span>
            {clickable ? (
              <button type="button" onClick={() => onSegmentClick(seg.rel, seg.isDir)} className="text-body-ink hover:underline" title={title}>
                {seg.label}
              </button>
            ) : (
              <span className={seg.collapsed ? "text-muted" : "text-ink font-semibold"} title={title}>
                {seg.label}
              </span>
            )}
          </span>
        );
      })}
    </nav>
  );
}
