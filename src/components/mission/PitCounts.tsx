"use client";

import { useTranslations } from "next-intl";
import type { MissionBucket, MissionCounts } from "@/lib/mission";

const BUCKETS: MissionBucket[] = ["needs-you", "stuck", "working", "idle"];

const BUCKET_CLASS: Record<MissionBucket, string> = {
  "needs-you": "border-danger bg-danger-soft text-danger",
  stuck: "border-warning bg-warning-soft text-warning",
  working: "border-accent-soft-border bg-accent-soft text-accent",
  idle: "border-line bg-surface-3 text-muted",
};
const BUCKET_DOT: Record<MissionBucket, string> = {
  "needs-you": "bg-danger", stuck: "bg-warning", working: "bg-accent", idle: "bg-muted",
};

// Item 6/15: count-first order ("3 precisa de você"), a leading colored dot per bucket, the
// idle pill's hidden/autoHidden suffix, and the pit-hint line underneath — all one component,
// no separate "topbar pill" exists to change (see the plan's Global Constraints note).
export function PitCounts({
  counts, active, onToggle, hiddenCount, autoHiddenCount,
}: {
  counts: MissionCounts;
  active: MissionBucket | null;
  onToggle: (bucket: MissionBucket) => void;
  hiddenCount: number;
  autoHiddenCount: number;
}) {
  const t = useTranslations("mission");
  const hidden = hiddenCount + autoHiddenCount;
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex flex-wrap gap-2.5">
        {BUCKETS.map((bucket) => (
          <button
            key={bucket}
            type="button"
            onClick={() => onToggle(bucket)}
            aria-pressed={active === bucket}
            className={`flex h-9 items-center gap-2 rounded-full border px-3.5 text-xs font-semibold transition-opacity ${BUCKET_CLASS[bucket]} ${
              active !== null && active !== bucket ? "opacity-50" : ""
            }`}
          >
            <span className={`h-1.5 w-1.5 rounded-full ${BUCKET_DOT[bucket]}`} aria-hidden />
            {counts[bucket]} {t(`states.${bucket}`)}
            {bucket === "idle" && hidden > 0 ? t("pit.hidden", { count: hidden }) : null}
          </button>
        ))}
      </div>
      <span className="text-[11px] text-muted">{t("pitHint", { workingCount: counts.working })}</span>
    </div>
  );
}
