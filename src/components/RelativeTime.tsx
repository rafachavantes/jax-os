"use client";

import { useLocale } from "next-intl";
import { useEffect, useState } from "react";

const UNITS: Array<[Intl.RelativeTimeFormatUnit, number]> = [
  ["year", 31536000],
  ["month", 2592000],
  ["week", 604800],
  ["day", 86400],
  ["hour", 3600],
  ["minute", 60],
];

export function formatRelative(epochMs: number, locale: string, nowMs: number): string {
  const diff = (epochMs - nowMs) / 1000;
  const abs = Math.abs(diff);
  const rtf = new Intl.RelativeTimeFormat(locale, { numeric: "auto" });
  for (const [unit, secs] of UNITS) {
    if (abs >= secs) return rtf.format(Math.round(diff / secs), unit);
  }
  return rtf.format(Math.round(diff), "second");
}

// Computed post-mount only — SSR "2m ago" vs client "3m ago" would warn on
// hydration (spec). Renders empty until the first effect tick.
export function RelativeTime({ epochMs }: { epochMs: number }) {
  const locale = useLocale();
  const [text, setText] = useState("");
  useEffect(() => {
    const update = () => setText(formatRelative(epochMs, locale, Date.now()));
    update();
    const id = setInterval(update, 30_000);
    return () => clearInterval(id);
  }, [epochMs, locale]);
  return <span>{text}</span>;
}
