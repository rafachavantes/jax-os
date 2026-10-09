import type { ProviderUsage, RateWindow } from "@/server/collectors/subscriptions";

export type UsageWindowEntry = { label: string; win: RateWindow };

// Shared by highestWindow (pick the peak) and ProviderUsageModal (list every one).
export function usageWindows(usage: ProviderUsage, t: (key: string) => string): UsageWindowEntry[] {
  const candidates: UsageWindowEntry[] = [];
  if (usage.fiveHour) candidates.push({ label: t("fiveHour"), win: usage.fiveHour });
  if (usage.weekly) candidates.push({ label: t("weekly"), win: usage.weekly });
  if (usage.monthly) candidates.push({ label: t("monthly"), win: usage.monthly });
  for (const m of usage.models) candidates.push({ label: `${m.name} · ${t("weekly")}`, win: m.weekly });
  return candidates;
}

// Rafa's rule: the ring's percent is whichever window (5h/weekly/monthly/per-model
// weekly) is closest to its cap -- any of them can be the one about to block work.
// The tooltip (native `title`) lists every window so the peak isn't the only thing visible.
export function highestWindow(usage: ProviderUsage, t: (key: string) => string): { percent: number; tooltip: string } | null {
  const candidates = usageWindows(usage, t);
  if (candidates.length === 0) return null;
  const worst = candidates.reduce((a, b) => (b.win.usedPercent > a.win.usedPercent ? b : a));
  const tooltip = candidates
    .map((c) => `${c.label}: ${c.win.usedPercent}%${c.win.resetsAt ? ` (${t("resets")} ${c.win.resetsAt})` : ""}`)
    .join("\n");
  return { percent: worst.win.usedPercent, tooltip };
}

// Exact reset time, never vague text (Rafa's rule of thumb):
// under 1h -> "in N min", under 24h -> "in Xh Ymin", otherwise weekday + local 24h clock time.
// Minutes are rounded UP (ceil) so the line never reads more generous than reality.
export function formatResetLine(resetsAt: string | null, nowMs: number, locale: string, t: (key: string, values?: Record<string, string | number>) => string): string | null {
  if (resetsAt === null) return null;
  const ms = Date.parse(resetsAt);
  if (Number.isNaN(ms)) return null;
  const msOut = ms - nowMs;
  if (msOut <= 0) return null; // already reset, stale — omit rather than show "in 0 min"
  const totalMinutes = Math.ceil(msOut / 60_000);
  if (totalMinutes < 60) return t("rings.resetInMinutes", { minutes: totalMinutes });
  if (totalMinutes < 24 * 60) {
    return t("rings.resetInHoursMinutes", { hours: Math.floor(totalMinutes / 60), minutes: totalMinutes % 60 });
  }
  // 7+ days out (monthly windows) a bare weekday is ambiguous — show the date instead.
  const weekday = new Intl.DateTimeFormat(locale, totalMinutes >= 7 * 24 * 60 ? { day: "2-digit", month: "2-digit" } : { weekday: "long" }).format(ms);
  const d = new Date(ms);
  const time = `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  return t("rings.resetOnAt", { weekday, time });
}
