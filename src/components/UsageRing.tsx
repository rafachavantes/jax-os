"use client";

import { useEffect, useRef } from "react";
import type { SubscriptionsPayload } from "@/server/collectors/subscriptions";
import { formatResetLine, usageWindows } from "@/lib/usageWindows";

export { formatResetLine, highestWindow } from "@/lib/usageWindows";

const RADIUS = 16;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

function ringColor(pct: number): string {
  return pct >= 90 ? "stroke-danger" : pct >= 70 ? "stroke-warning" : "stroke-accent";
}

// MOA-486 follow-up: the percent moved INSIDE the ring (an absolutely positioned
// overlay, not rotated with the svg) so it reads on a phone with no hover, and the
// ring itself is now the tap target that opens the all-providers details modal.
// Follow-up (ring labels): a short provider name (`shortName`, brand name, untranslated)
// renders stacked below the ring on every viewport, so the three rings are tellable
// apart on a phone with no `title` hover. `aria-label` stays the full provider name.
// Header cleanup: the text label + reset line that used to sit beside the ring at lg+
// is gone — the ring (percent + short name) IS the compact indicator; the exact reset
// time now lives one tap away, in ProviderUsageModal.
export function UsageRing({
  percent, tooltip, providerName, shortName, onOpenDetails,
}: { percent: number; tooltip: string; providerName: string; shortName: string; onOpenDetails: () => void }) {
  const pct = Math.min(100, Math.max(0, percent));
  const offset = CIRCUMFERENCE * (1 - pct / 100);
  return (
    <button
      type="button"
      title={tooltip}
      aria-label={providerName}
      onClick={onOpenDetails}
      className="flex flex-none flex-col items-center gap-0.5 rounded-md"
    >
      <div className="relative flex h-10 w-10 flex-none items-center justify-center">
        <svg width="40" height="40" viewBox="0 0 40 40" className="flex-none -rotate-90">
          <circle cx="20" cy="20" r={RADIUS} fill="none" strokeWidth="3" className="stroke-surface-3" />
          <circle
            cx="20" cy="20" r={RADIUS} fill="none" strokeWidth="3" strokeLinecap="round"
            strokeDasharray={CIRCUMFERENCE} strokeDashoffset={offset} className={ringColor(pct)}
          />
        </svg>
        <span className="absolute inset-0 flex items-center justify-center font-mono text-[10px] text-ink">{pct}%</span>
      </div>
      <span className="text-[9px] leading-none text-muted">{shortName}</span>
    </button>
  );
}

// round-1 F4: rendered instead of `UsageRing` when a provider has no data or no window
// — the ring's slot always renders. `caption` is already-translated by the caller.
export function UsageRingUnavailable({ label, shortName, caption, onOpenDetails }: { label: string; shortName: string; caption: string; onOpenDetails: () => void }) {
  return (
    <button
      type="button"
      title={caption}
      aria-label={label}
      onClick={onOpenDetails}
      className="flex flex-none flex-col items-center gap-0.5 rounded-md"
    >
      <div className="relative flex h-10 w-10 flex-none items-center justify-center">
        <svg width="40" height="40" viewBox="0 0 40 40" className="flex-none">
          <circle cx="20" cy="20" r={RADIUS} fill="none" strokeWidth="3" strokeDasharray="3 3" className="stroke-line" />
        </svg>
      </div>
      <span className="text-[9px] leading-none text-muted">{shortName}</span>
    </button>
  );
}

// One dialog for ALL providers (Rafa's spec: a single tap shows the full picture,
// not a per-ring modal). Close on backdrop click, Escape and a close button; `title`
// stays on each ring for desktop hover, this is the tap/click path.
export function ProviderUsageModal({
  providers,
  payload,
  locale,
  t,
  tMission,
  onClose,
}: {
  providers: readonly { key: keyof SubscriptionsPayload; name: string }[];
  payload: SubscriptionsPayload | undefined;
  locale: string;
  t: (key: string) => string;
  tMission: (key: string, values?: Record<string, string | number>) => string;
  onClose: () => void;
}) {
  const dialogRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dlg = dialogRef.current;
    if (dlg && typeof dlg.showModal === "function") dlg.showModal();
  }, []);

  return (
    <dialog
      ref={dialogRef}
      aria-label={tMission("rings.detailsTitle")}
      onClose={onClose}
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
      onClick={(e) => {
        // native <dialog>: a click on the element itself (not a child) is the backdrop.
        if (e.target === dialogRef.current) onClose();
      }}
      className="w-[min(92vw,420px)] rounded-lg border border-line bg-surface p-0 backdrop:bg-base/60"
    >
      <div className="flex items-center justify-between border-b border-line px-4 py-3">
        <h2 className="text-sm font-semibold text-ink">{tMission("rings.detailsTitle")}</h2>
        <button type="button" onClick={onClose} className="rounded-md px-2 py-1 text-xs text-muted hover:bg-surface-2">
          {tMission("rings.close")}
        </button>
      </div>
      <div className="flex max-h-[70vh] flex-col gap-4 overflow-y-auto px-4 py-4">
        {providers.map((p) => {
          const result = payload?.[p.key];
          const windows = result?.ok ? usageWindows(result.data, t) : [];
          return (
            <div key={p.key} className="flex flex-col gap-1.5">
              <div className="flex items-center gap-2">
                <span className="text-[13px] font-bold text-ink">{p.name}</span>
                {result?.ok && result.data.plan ? (
                  <span className="rounded-full border border-brand-soft-border bg-brand-soft px-[7px] py-0.5 text-[10px] font-bold uppercase tracking-[.06em] text-brand">
                    {result.data.plan}
                  </span>
                ) : null}
              </div>
              {windows.length > 0 ? (
                windows.map((w, i) => {
                  const reset = formatResetLine(w.win.resetsAt, Date.now(), locale, tMission);
                  return (
                    <div key={i} className="flex items-baseline justify-between gap-2 text-[12px]">
                      <span className="text-muted">{w.label}</span>
                      <span className="font-mono text-ink">
                        {w.win.usedPercent}%{reset ? ` · ${reset}` : ""}
                      </span>
                    </div>
                  );
                })
              ) : (
                <span className="text-[12px] text-muted">{t("unavailable")}</span>
              )}
            </div>
          );
        })}
      </div>
    </dialog>
  );
}
