"use client";

import { useRef } from "react";
import { useTranslations } from "next-intl";
import { SourceWarning } from "@/components/mission/SourceWarning";
import { RelativeTime } from "@/components/RelativeTime";
import { retainOkMeta } from "@/lib/api";
import { meterColorClass } from "@/lib/format";
import { useGeneralSettings } from "@/lib/settingsQuery";
import { useTokensApi } from "@/lib/useTokensApi";
import type { ProviderUsage, RateWindow, SubscriptionsPayload } from "@/server/collectors/subscriptions";

// shortName: the label under each topbar ring (MOA-486 follow-up) — brand names, untranslated.
export const PROVIDERS = [
  { key: "anthropic", name: "Anthropic", shortName: "Claude" },
  { key: "codex", name: "OpenAI Codex", shortName: "Codex" },
  { key: "opencode", name: "OpenCode GO", shortName: "Go" },
] as const;

function WindowBar({ label, win }: { label: string; win: RateWindow | null }) {
  const t = useTranslations("tokens.subscriptions");
  if (!win) return null;
  const pct = Math.min(100, Math.max(0, win.usedPercent));
  const color = meterColorClass(pct);
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-baseline gap-2">
        <span className="text-xs text-muted">{label}</span>
        <span className="ml-auto font-mono text-[11.5px] text-ink">{pct}%</span>
      </div>
      <div className="h-[7px] overflow-hidden rounded-full bg-surface-inset">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      {win.resetsAt ? (
        <span className="text-[10.5px] text-muted">
          {t("resets")} <RelativeTime epochMs={Date.parse(win.resetsAt)} />
        </span>
      ) : null}
    </div>
  );
}

function ProviderCard({
  name, providerKey, usage, error, fetchFailed, updatedAt,
}: {
  name: string;
  providerKey: string;
  usage: ProviderUsage | undefined;
  error: string | undefined;
  fetchFailed: boolean;
  updatedAt?: number;
}) {
  const t = useTranslations("tokens.subscriptions");
  return (
    <div className="flex flex-col gap-4 rounded-lg border border-line bg-surface px-5 py-[18px]">
      <div className="flex items-center gap-2">
        <span className="text-sm font-bold text-ink">{name}</span>
        {usage?.plan ? (
          <span className="rounded-full border border-brand-soft-border bg-brand-soft px-[7px] py-0.5 text-[10px] font-bold uppercase tracking-[.06em] text-brand">
            {usage.plan}
          </span>
        ) : null}
      </div>
      {fetchFailed || error ? (
        <SourceWarning
          label={t("unavailable")}
          detail={error ? errorText(t, providerKey, error) : t("errors.generic")}
          aside={usage && updatedAt ? <RelativeTime epochMs={updatedAt} /> : undefined}
        />
      ) : null}
      {!usage && !fetchFailed && !error ? (
        <span className="text-xs text-muted">{t("loading")}</span>
      ) : usage ? (
        <>
          <WindowBar label={t("fiveHour")} win={usage.fiveHour} />
          <WindowBar label={t("weekly")} win={usage.weekly} />
          <WindowBar label={t("monthly")} win={usage.monthly ?? null} />
          {usage.models.map((m) => (
            <WindowBar key={m.name} label={`${m.name} · ${t("weekly")}`} win={m.weekly} />
          ))}
        </>
      ) : null}
    </div>
  );
}

// Known error keys map to localized hints; anything else shows generic + raw key.
function errorText(t: ReturnType<typeof useTranslations>, provider: string, error: string): string {
  if (error === "expired" || error === "missing-credentials") return t(`errors.${error}.${provider}`);
  if (error === "rate-limited") return t("errors.rate-limited");
  return `${t("errors.generic")} (${error})`;
}

export function SubscriptionCards() {
  const { data: settings } = useGeneralSettings();
  const agents = settings?.ok === true ? settings.data.integrations.agents : undefined;
  const anyEnabled = Boolean(agents?.claude || agents?.codex || agents?.opencode);
  const { data, isError, dataUpdatedAt } = useTokensApi<SubscriptionsPayload>("subscriptions", 60_000, anyEnabled);
  // MOA-504: one switch per agent — a disabled agent's card is not rendered.
  const enabledKeys = new Set<string>();
  if (agents?.claude) enabledKeys.add("anthropic");
  if (agents?.codex) enabledKeys.add("codex");
  if (agents?.opencode) enabledKeys.add("opencode");
  const payload = data?.ok ? data.data : undefined;
  const envelopeFailed = isError && !payload;
  const hold = useRef<Partial<Record<(typeof PROVIDERS)[number]["key"], { data: ProviderUsage; updatedAt: number }>>>({});
  const now = dataUpdatedAt || Date.now();
  return (
    <div className="grid gap-4 lg:grid-cols-3">
      {PROVIDERS.filter((p) => enabledKeys.has(p.key)).map((p) => {
        const incoming = payload?.[p.key];
        const retained = incoming
          ? retainOkMeta(hold.current[p.key], incoming, now)
          : { data: hold.current[p.key]?.data, error: undefined as string | undefined, updatedAt: hold.current[p.key]?.updatedAt };
        if (retained.data && !retained.error) {
          hold.current[p.key] = { data: retained.data, updatedAt: retained.updatedAt ?? now };
        }
        return (
          <ProviderCard
            key={p.key}
            name={p.name}
            providerKey={p.key}
            usage={retained.data}
            error={retained.error}
            fetchFailed={envelopeFailed}
            updatedAt={retained.updatedAt}
          />
        );
      })}
    </div>
  );
}
