"use client";

import { PanelLeft } from "lucide-react";
import Link from "next/link";
import { useLocale, useTranslations } from "next-intl";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { navItemFor } from "@/lib/nav";
import type { SidebarMode } from "@/lib/nav";
import { useMission } from "@/lib/useMission";
import { useTokensApi } from "@/lib/useTokensApi";
import { buildMission, summarizeBuckets } from "@/lib/mission";
import type { MissionBucket, MissionReadiness, SourceName } from "@/lib/mission";
import type { ProjectScan } from "@/server/collectors/projects";
import type { HubEnvelopeData } from "@/server/db/workflows";
import type { ProviderResult, SubscriptionsPayload } from "@/server/collectors/subscriptions";
import { AfkToggle } from "./mission/AfkToggle";
import { PROVIDERS } from "./tokens/SubscriptionCards";
import { ProviderUsageModal, UsageRing, UsageRingUnavailable, highestWindow } from "./UsageRing";
import { LanguageSelector } from "./LanguageSelector";
import { MobileNav } from "./MobileNav";
import { ThemeToggle, type Theme } from "./ThemeToggle";

const BUCKET_TONE: Record<MissionBucket, string> = {
  "needs-you": "border-danger bg-danger-soft text-danger",
  stuck: "border-warning bg-warning-soft text-warning",
  working: "border-accent-soft-border bg-accent-soft text-accent",
  idle: "border-line bg-surface-3 text-muted",
};

// round-1 F1: `mission.readiness` (from `buildMission`) folds in `prs`, permanently
// "pending" since Topbar never fetches it — the pill would stay a skeleton forever.
// Exported so the exact predicate is unit-testable without rendering. Part 1's
// `summarizeBuckets(readiness, cards)` contract is used AS-IS — this only builds a
// readiness value from the two sources Topbar actually polls, never `prs`.
export function pillReadinessOf(projects: { ok: boolean } | undefined, hub: { ok: boolean } | undefined): MissionReadiness {
  const failed: SourceName[] = [];
  if (projects !== undefined && !projects.ok) failed.push("projects");
  if (hub !== undefined && !hub.ok) failed.push("hub");
  const pending: SourceName[] = [];
  if (projects === undefined) pending.push("projects");
  if (hub === undefined) pending.push("hub");
  if (failed.length > 0) return { kind: "failed", failed, pending };
  if (pending.length > 0) return { kind: "loading", pending };
  return { kind: "ready" };
}

// MOA-504 D6: a meter slot exists only for an agent that is on AND has a credential. "disabled" (agent
// off) and "missing-credentials" omit it; any other failure keeps the placeholder (a credential
// exists, the meter is just down); no result yet (payload loading) renders nothing.
export function meterShown(result: ProviderResult | undefined): boolean {
  if (!result) return false;
  return result.ok || (result.error !== "disabled" && result.error !== "missing-credentials");
}

// Decision 14: the 4 already-known signals Topbar's subtitle composes from — no new fetch.
export function liveSourceCount(
  projectsOk: boolean | undefined,
  hubOk: boolean | undefined,
  tmuxSource: "ok" | "failed",
  codexSource: "ok" | "failed" | "truncated" | "absent",
): number {
  return [projectsOk === true, hubOk === true, tmuxSource === "ok", codexSource !== "failed"].filter(Boolean).length;
}

function MissionClock() {
  const locale = useLocale();
  const [text, setText] = useState("");
  useEffect(() => {
    const update = () => setText(new Intl.DateTimeFormat(locale, { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }).format(Date.now()));
    update();
    const id = setInterval(update, 60_000);
    return () => clearInterval(id);
  }, [locale]);
  return <span>{text}</span>;
}

// Decision 14 (round-1 F5): tri-state hub word (loading/ok/degraded) PLUS a zero-source override — when every liveSourceCount signal is down, that beats the hub's own state.
export function hubStatusKey(hub: { ok: boolean } | undefined, liveCount: number): string {
  if (liveCount === 0) return "topbar.noSources";
  if (hub === undefined) return "topbar.hubLoading";
  return hub.ok ? "topbar.hubOk" : "topbar.hubDegraded";
}

export function Topbar({
  initialTheme,
  sidebar,
  onToggleSidebar,
}: {
  initialTheme: Theme;
  sidebar: SidebarMode;
  onToggleSidebar: () => void;
}) {
  const pathname = usePathname();
  const item = navItemFor(pathname);
  const tTitle = useTranslations("title");
  const tSub = useTranslations("subtitle");
  const tShell = useTranslations("shell");
  const tMission = useTranslations("mission");
  const tTokens = useTranslations("tokens.subscriptions");
  const locale = useLocale();
  // Decision 4: the worst-state pill reads mission/projects + mission/hub only — never
  // mission/agents (tmux session count), so it never changes when the agent count does
  // with no card-state change (§10 AC 5). `prs` is never needed for a headline.
  const projects = useMission<ProjectScan>("projects", 10_000);
  const hub = useMission<HubEnvelopeData>("hub", 10_000);
  // Decision 5: same hook, same query key ("subscriptions") the Tokens page already
  // uses — one shared /api/tokens/subscriptions request per refetch interval (§10 AC 8).
  const subscriptions = useTokensApi<SubscriptionsPayload>("subscriptions", 60_000);
  const payload = subscriptions.data?.ok ? subscriptions.data.data : undefined;
  const shownProviders = PROVIDERS.filter((p) => meterShown(payload?.[p.key]));
  const [detailsOpen, setDetailsOpen] = useState(false);

  const mission = buildMission(projects.data, hub.data, undefined, Date.now());
  const liveCount = liveSourceCount(projects.data?.ok, hub.data?.ok, mission.model.tmuxSource, mission.model.codexSource);
  const summary = summarizeBuckets(pillReadinessOf(projects.data, hub.data), mission.model.cards);

  return (
    <header className="flex h-[66px] flex-none items-center gap-4 border-b border-line bg-surface px-4 md:px-6">
      {sidebar === "hidden" ? (
        <button
          onClick={onToggleSidebar}
          aria-label={tShell("expand")}
          className="hidden h-9 w-9 items-center justify-center rounded-full border border-line bg-surface-2 text-body-ink transition-transform active:scale-95 md:flex"
        >
          <PanelLeft className="h-4 w-4" />
        </button>
      ) : null}
      <MobileNav initialTheme={initialTheme} />
      <div className="flex min-w-0 flex-col gap-px md:min-w-[190px]">
        <span className="truncate text-[17px] font-semibold text-ink">{tTitle(item.key)}</span>
        {item.key === "mission" ? (
          <span className="hidden truncate text-xs text-muted sm:block">
            <MissionClock /> · {tMission(hubStatusKey(hub.data, liveCount))} · {tMission("topbar.liveSources", { count: liveCount })}
          </span>
        ) : (
          <span className="hidden truncate text-xs text-muted sm:block">{tSub(item.key)}</span>
        )}
      </div>

      {summary.counts ? (
        <Link
          href="/"
          className={`hidden h-9 flex-none items-center gap-2 rounded-full border px-3 text-xs font-semibold sm:flex ${BUCKET_TONE[summary.worst]}`}
        >
          <span className="h-[7px] w-[7px] rounded-full bg-current [animation:jax-pulse_1.6s_infinite]" />
          {summary.counts[summary.worst]} {tMission(`states.${summary.worst}`)}
        </Link>
      ) : (
        <div className="hidden h-9 w-28 flex-none animate-pulse rounded-full bg-surface-2 sm:block" />
      )}

      <div className="ml-auto flex items-center gap-4">
        {/* MOA-504 D6: only enabled agents with a credential get a slot. */}
        {shownProviders.map((p) => {
          const result = payload?.[p.key];
          const w = result?.ok ? highestWindow(result.data, tTokens) : null;
          return w
            ? (
              <UsageRing
                key={p.key} percent={w.percent} tooltip={w.tooltip}
                providerName={p.name} shortName={p.shortName} onOpenDetails={() => setDetailsOpen(true)}
              />
            )
            : <UsageRingUnavailable key={p.key} label={p.name} shortName={p.shortName} caption={tMission("rings.unavailable")} onOpenDetails={() => setDetailsOpen(true)} />;
        })}
        {/* below md: locale + theme move into the MobileNav drawer to free header width for the title */}
        <div className="hidden items-center gap-4 md:flex">
          <AfkToggle />
          <LanguageSelector />
          <ThemeToggle initial={initialTheme} />
        </div>
      </div>
      {detailsOpen ? (
        <ProviderUsageModal
          providers={shownProviders}
          payload={payload}
          locale={locale}
          t={tTokens}
          tMission={tMission}
          onClose={() => setDetailsOpen(false)}
        />
      ) : null}
    </header>
  );
}
