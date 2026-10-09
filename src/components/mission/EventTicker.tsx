"use client";

import { useState } from "react";
import { useLocale, useTranslations } from "next-intl";
import { formatClock, pageSlice, tickerSentence, type TickerItem } from "@/lib/mission";

// Spec §9: a pure cross-project feed over already-polled data — no new API route, no
// snooze/shortcuts (D15). An "event" item renders its human ticker sentence (Decision 19)
// and a fixed HH:MM:SS clock (Decision 20) — mission.ts's buildTicker already picked/
// sorted/capped the set and spread the full entry into the item.
function TickerClock({ epochMs }: { epochMs: number }) {
  const locale = useLocale();
  return <span>{formatClock(epochMs, locale)}</span>;
}

const PAGE_SIZE = 10;
const PAGE_BTN = "font-semibold text-body-ink underline disabled:cursor-not-allowed disabled:text-muted disabled:no-underline";

// Rafa (2026-09-19): 10 items/page over buildTicker's already-capped, already-sorted set.
// Each item's (source, dir, ts) is a cheap stable-enough signature — keying TickerPage on it
// remounts (and so resets to page 1) whenever the underlying item set actually changes,
// without a manual "did the list change" effect.
function itemsKey(items: TickerItem[]): string {
  return items.map((it) => `${it.source}|${it.dir}|${it.ts}`).join(",");
}

function TickerPage({ items, t }: { items: TickerItem[]; t: ReturnType<typeof useTranslations> }) {
  const [page, setPage] = useState(1);
  const { items: pageItems, page: current, pages } = pageSlice(items, page, PAGE_SIZE);
  return (
    <>
      <div className="flex flex-col">
        {pageItems.map((item, i) => (
          <div key={i} data-testid="ticker-row" className="flex items-baseline gap-2 py-2 font-mono text-[12.5px]">
            <span className="flex-none text-[11px] text-muted"><TickerClock epochMs={Date.parse(item.ts)} /></span>
            <span className="min-w-0 flex-1 truncate text-body-ink">
              <strong className="font-semibold text-ink">{item.name}</strong> ·{" "}
              {item.source === "commit" ? `${item.author} — ${item.message}` : null}
              {item.source === "pr" ? `#${item.number} ${item.title} · ${t(`ci.${item.ci}`)}` : null}
              {item.source === "event" ? (() => {
                const sentence = tickerSentence(item);
                return <>{t(sentence.key, sentence.values)}{item.diagnostic ? ` — ${item.diagnostic}` : ""}</>;
              })() : null}
            </span>
          </div>
        ))}
      </div>
      {pages > 1 ? (
        <div className="flex items-center justify-center gap-2 pt-1 text-[11px]">
          <button type="button" className={PAGE_BTN} disabled={current === 1} onClick={() => setPage(current - 1)}>
            {t("prev")}
          </button>
          <span className="text-muted">{t("pageInfo", { page: current, pages })}</span>
          <button type="button" className={PAGE_BTN} disabled={current === pages} onClick={() => setPage(current + 1)}>
            {t("next")}
          </button>
        </div>
      ) : null}
    </>
  );
}

export function EventTicker({ items }: { items: TickerItem[] }) {
  const t = useTranslations("mission.ticker");
  return (
    <section className="flex flex-col gap-3 rounded-lg border border-line bg-surface px-5 py-[18px]">
      <div className="flex flex-col gap-0.5">
        <span className="text-sm font-bold text-ink">{t("title")}</span>
        <span className="text-[11px] text-muted">{t("subtitle")}</span>
      </div>
      {items.length === 0 ? (
        <span className="text-[12.5px] text-muted">{t("empty")}</span>
      ) : (
        <TickerPage key={itemsKey(items)} items={items} t={t} />
      )}
    </section>
  );
}
