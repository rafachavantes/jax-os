"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { fetchEnvelope, type Envelope, type TtydConfig } from "@/lib/api";
import { RelativeTime } from "@/components/RelativeTime";
import type { Agent } from "@/server/collectors/tmux";
import { useMission } from "@/lib/useMission";
import { buildMission, paneAccessSelector, postWorkflowAction, type Mission, type MissionCard, type MissionPane } from "@/lib/mission";
import { findMobilePane } from "@/lib/mosaicView";
import { pillReadinessOf } from "@/components/Topbar";
import type { ProjectScan } from "@/server/collectors/projects";
import type { HubEnvelopeData } from "@/server/db/workflows";
import { SessionCard } from "@/components/sessions/SessionCard";
import {
  accessFailureStatus,
  accessLifecycle,
  controlStateReducer,
  initialAccessLifecycle,
  TerminalPanel,
  type AccessStatus,
} from "@/components/sessions/TerminalPanel";
import { SourceWarning } from "@/components/mission/SourceWarning";
import { Mosaic } from "@/components/sessions/Mosaic";
import { MosaicCellDetail } from "@/components/sessions/MosaicCellDetail";
import {
  parseSessionAccessOrigin,
  parseSessionAccessSelector,
  sessionAccessHref,
  type ExpectedSessionTarget,
  type ResolvedSessionAccess,
  type SessionAccessCode,
  type SessionAccessSelector,
  type TakenSessionAccess,
} from "@/lib/sessionAccess";

type PanesTailData = { pane: string; lines: string[] | null }[];

type Focus = { selector: SessionAccessSelector; origin: "home" | "mosaic" | null };
type TargetResult = { ok: true; data: ResolvedSessionAccess } | { ok: false; code: SessionAccessCode; error: string };

export const REQUEST_TIMEOUT_MS = 5_000;
export const FRESHNESS_MS = 10_000;
export const TARGET_POLL_MS = 5_000;

export function selectorIdentity(selector: SessionAccessSelector): string {
  return selector.kind === "card"
    ? `card|${selector.project}|${selector.harnessSession ?? ""}|${selector.pane}|${selector.incarnation}`
    : `tmux|${selector.session}`;
}

// The client receipt's Data.now() drives freshness; the server's observedAt is display only.
export function targetFreshness(receivedAt: number, now: number): "fresh" | "expired" {
  const age = now - receivedAt;
  return age < 0 || age >= FRESHNESS_MS ? "expired" : "fresh";
}

export function targetQueryString(
  selector: SessionAccessSelector,
  expected: ExpectedSessionTarget | null,
): string {
  const params = new URLSearchParams();
  if (selector.kind === "card") {
    params.set("kind", "card");
    params.set("project", selector.project);
    params.set("harnessSession", selector.harnessSession ?? "");
    params.set("pane", selector.pane);
    params.set("incarnation", selector.incarnation);
  } else {
    params.set("kind", "tmux");
    params.set("session", selector.session);
  }
  if (expected) {
    params.set("expectedTarget", expected.target);
    params.set("expectedIncarnation", expected.incarnation);
  }
  return params.toString();
}

async function fetchTarget(
  selector: SessionAccessSelector,
  expected: ExpectedSessionTarget | null,
  signal: AbortSignal,
): Promise<TargetResult> {
  const timeout = AbortSignal.timeout(REQUEST_TIMEOUT_MS);
  const res = await fetch(`/api/tmux/target?${targetQueryString(selector, expected)}`, {
    signal: AbortSignal.any([signal, timeout]),
    cache: "no-store",
  });
  const json = (await res.json().catch(() => null)) as TargetResult | null;
  if (!json || typeof json !== "object" || !("ok" in json)) {
    return { ok: false, code: "source-failure", error: "unavailable" };
  }
  return json;
}

export type FocusedLocation =
  | { kind: "focus"; selector: SessionAccessSelector; origin: "home" | "mosaic" | null }
  | { kind: "invalid" }
  | { kind: "none" };

export function parseFocusedLocation(search: string): FocusedLocation {
  const params = new URLSearchParams(search);
  const selector = parseSessionAccessSelector(params);
  if (selector) return { kind: "focus", selector, origin: parseSessionAccessOrigin(params) };
  return params.has("kind") || params.has("session") || params.has("pane") ? { kind: "invalid" } : { kind: "none" };
}

// Back is decided ONLY by the fixed origin enum; no free-form return URL is ever accepted.
export function backHref(origin: "home" | "mosaic" | null): string {
  return origin === "mosaic" ? "/tmux" : "/";
}

// A new/larger window never trusts a free-form return URL; it revalidates and starts RO, and
// Back there falls back to Home because the origin enum is deliberately absent.
export function largerWindowHref(selector: SessionAccessSelector): string {
  const params = new URLSearchParams(sessionAccessHref(selector, "home").split("?")[1]);
  params.delete("from");
  return `/tmux?${params.toString()}`;
}

export default function SessionsPage() {
  const t = useTranslations("sessions");
  const tm = useTranslations("sessions.mosaic");
  const tTitle = useTranslations("title");

  const [mounted, setMounted] = useState(false);
  const [view, setView] = useState<"mosaic" | "detail">("mosaic");
  const [focus, setFocus] = useState<Focus | null>(null);
  const [resolved, setResolved] = useState<ResolvedSessionAccess | null>(null);
  const [expected, setExpected] = useState<ExpectedSessionTarget | null>(null);
  const [visible, setVisible] = useState(true);
  const [lifecycle, dispatchLifecycle] = useReducer(accessLifecycle, initialAccessLifecycle);
  const [control, dispatchControl] = useReducer(controlStateReducer, { mode: "ro", busy: false, error: null });
  const [mobilePane, setMobilePane] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const focusRef = useRef<Focus | null>(null);
  focusRef.current = focus;
  const expectedRef = useRef<ExpectedSessionTarget | null>(null);
  expectedRef.current = expected;
  const lifecycleRef = useRef(lifecycle);
  lifecycleRef.current = lifecycle;
  const visibleRef = useRef(visible);
  visibleRef.current = visible;
  const notBeforeRef = useRef(0);
  const expiryRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Synchronous duplicate guard: a second click in the same tick never starts a second POST.
  const controlBusyRef = useRef(false);

  const projects = useMission<ProjectScan>("projects", 10_000, view === "mosaic");
  const hub = useMission<HubEnvelopeData>("hub", 10_000, view === "mosaic");
  const agents = useMission<Agent[]>("agents", 5_000, view === "detail");
  const panesTail = useQuery({
    queryKey: ["tmux", "panes-tail"],
    queryFn: ({ signal }) => fetchEnvelope<PanesTailData>("/api/tmux/panes-tail", signal),
    refetchInterval: 3_000,
    enabled: view === "mosaic",
  });
  const config = useQuery({
    queryKey: ["tmux", "config"],
    queryFn: ({ signal }) => fetchEnvelope<TtydConfig>("/api/tmux/config", signal),
    refetchInterval: 30_000,
    enabled: view === "detail",
  });

  const selectorKey = focus ? selectorIdentity(focus.selector) : "";
  const expectedKey = expected ? `${expected.target}|${expected.incarnation}` : "";
  const target = useQuery({
    queryKey: ["tmux", "target", selectorKey, expectedKey],
    queryFn: ({ signal }) => fetchTarget(focus!.selector, expectedRef.current, signal),
    enabled: mounted && focus !== null && visible,
    refetchInterval: TARGET_POLL_MS,
    retry: false,
  });
  const refetchRef = useRef(target.refetch);
  refetchRef.current = target.refetch;

  const clearExpiry = useCallback(() => {
    if (expiryRef.current !== null) {
      clearTimeout(expiryRef.current);
      expiryRef.current = null;
    }
  }, []);

  const armExpiry = useCallback((receivedAt: number) => {
    clearExpiry();
    const remaining = Math.max(0, FRESHNESS_MS - (Date.now() - receivedAt));
    expiryRef.current = setTimeout(() => {
      if (!visibleRef.current) return;
      dispatchLifecycle({ type: "failure", status: "expired" });
    }, remaining);
  }, [clearExpiry]);

  // Parse a focused link after mount — no Suspense/prerender dependency, and a missing selector
  // never falls back to the first session. An invalid focused link fails visibly instead.
  useEffect(() => {
    setMounted(true);
    const location = parseFocusedLocation(window.location.search);
    if (location.kind === "focus") {
      setFocus({ selector: location.selector, origin: location.origin });
      setView("detail");
      notBeforeRef.current = Date.now();
      dispatchLifecycle({ type: "open" });
    } else if (location.kind === "invalid") {
      setView("detail");
      dispatchLifecycle({ type: "failure", status: "unavailable" });
    }
    return () => clearExpiry();
  }, [clearExpiry]);

  useEffect(() => {
    visibleRef.current = document.visibilityState !== "hidden";
    setVisible(visibleRef.current);
    const onVisibility = () => {
      const next = document.visibilityState !== "hidden";
      visibleRef.current = next;
      setVisible(next);
      if (!next) {
        clearExpiry();
        dispatchLifecycle({ type: "hide" });
      } else if (focusRef.current) {
        notBeforeRef.current = Date.now();
        dispatchLifecycle({ type: "show" });
        refetchRef.current();
      }
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => document.removeEventListener("visibilitychange", onVisibility);
  }, [clearExpiry]);

  // One validation effect: expiry, source/target failure and the gated RO attach. A healthy poll
  // only refreshes facts — the iframe src/key stay untouched.
  useEffect(() => {
    if (!mounted || !visible || !focus || !target.data) return;
    const receivedAt = target.dataUpdatedAt;
    if (receivedAt < notBeforeRef.current) return;
    if (targetFreshness(receivedAt, Date.now()) === "expired") {
      clearExpiry();
      dispatchLifecycle({ type: "failure", status: "expired" });
      return;
    }
    const result = target.data;
    if (!result.ok) {
      clearExpiry();
      setResolved(null);
      dispatchLifecycle({ type: "failure", status: accessFailureStatus(result.code) });
      return;
    }
    setResolved(result.data);
    if (expectedRef.current === null) setExpected(result.data.expected);
    const current = lifecycleRef.current;
    if (current.pendingAttach) dispatchLifecycle({ type: "validation", roUrl: result.data.roUrl });
    if (current.pendingAttach || current.frame) armExpiry(receivedAt);
  }, [target.data, target.dataUpdatedAt, visible, mounted, focus, armExpiry, clearExpiry]);

  function openSelector(selector: SessionAccessSelector, origin: Focus["origin"]) {
    clearExpiry();
    setFocus({ selector, origin });
    setResolved(null);
    setExpected(null);
    setView("detail");
    notBeforeRef.current = Date.now();
    dispatchLifecycle({ type: "open" });
    dispatchControl({ type: "reset" });
  }

  function reconnect() {
    if (!focusRef.current) return;
    clearExpiry();
    notBeforeRef.current = Date.now();
    dispatchLifecycle({ type: "reconnect" });
    refetchRef.current();
  }

  async function postJson<T>(url: string, body: unknown): Promise<{ ok: boolean; data?: T; error?: string }> {
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const json = (await res.json().catch(() => null)) as { ok?: boolean; data?: T; error?: string } | null;
      return json && typeof json === "object" ? { ok: json.ok === true, data: json.data, error: json.error } : { ok: false };
    } catch {
      return { ok: false };
    }
  }

  async function takeControl() {
    const current = focusRef.current;
    const binding = expectedRef.current;
    if (controlBusyRef.current || !current || !binding) return;
    const selector = current.selector;
    const token = lifecycleRef.current.generation + 1;
    controlBusyRef.current = true;
    clearExpiry();
    dispatchLifecycle({ type: "takeStart" });
    dispatchControl({ type: "request", action: "take" });
    try {
      const res = await postJson<TakenSessionAccess>("/api/tmux/take-control", { selector, expected: binding });
      const stillCurrent = lifecycleRef.current.generation === token
        && focusRef.current !== null && selectorIdentity(focusRef.current.selector) === selectorIdentity(selector);
      if (res.ok && res.data?.rwUrl && stillCurrent) {
        setResolved(res.data);
        dispatchLifecycle({ type: "takeOk", rwUrl: res.data.rwUrl });
        armExpiry(Date.now());
        dispatchControl({ type: "settle", action: "take", ok: true, error: null, current: true });
      } else {
        if (stillCurrent) dispatchLifecycle({ type: "takeFailed" });
        dispatchControl({ type: "settle", action: "take", ok: false, error: res.error ?? t("controlFailed"), current: stillCurrent });
      }
    } finally {
      controlBusyRef.current = false;
    }
  }

  async function releaseControl() {
    const current = focusRef.current;
    const binding = expectedRef.current;
    if (controlBusyRef.current || !current || !binding) return;
    const selector = current.selector;
    const token = lifecycleRef.current.generation + 1;
    controlBusyRef.current = true;
    clearExpiry();
    // Synchronous read-only transition BEFORE any await; the audit cannot restore RW.
    dispatchLifecycle({ type: "releaseStart" });
    dispatchControl({ type: "request", action: "release" });
    try {
      const res = await postJson("/api/tmux/release-control", { selector, expected: binding });
      const stillCurrent = lifecycleRef.current.generation === token
        && focusRef.current !== null && selectorIdentity(focusRef.current.selector) === selectorIdentity(selector);
      dispatchControl({ type: "settle", action: "release", ok: res.ok, error: res.ok ? null : (res.error ?? t("controlFailed")), current: stillCurrent });
      if (stillCurrent) {
        notBeforeRef.current = Date.now();
        refetchRef.current();
      }
    } finally {
      controlBusyRef.current = false;
    }
  }

  function selectSession(session: string) {
    openSelector({ kind: "tmux", session }, focusRef.current?.origin ?? "mosaic");
  }

  function openCard(card: MissionCard, pane: MissionPane) {
    const selector = paneAccessSelector(card.dir, pane);
    if (selector) openSelector(selector, "mosaic");
  }

  const mission: Mission = buildMission(projects.data, hub.data, undefined, Date.now());
  const readiness = pillReadinessOf(projects.data, hub.data);
  const panesTailData = panesTail.data?.ok ? panesTail.data.data : null;
  const sessions = agents.data?.ok ? agents.data.data : null;
  const ttyd = config.data?.ok ? config.data.data : null;
  const mobileTarget = findMobilePane(mission.model.cards, mobilePane);

  useEffect(() => {
    if (mobilePane && !mobileTarget) setMobilePane(null);
  }, [mobilePane, mobileTarget]);

  if (view === "detail") {
    const label = focus
      ? focus.selector.kind === "tmux"
        ? resolved?.sessionName ?? focus.selector.session
        : resolved?.sessionName ?? focus.selector.project
      : null;
    let list: React.ReactNode;
    if (agents.isError && !sessions) {
      list = <SourceWarning label={t("unavailable")} detail={agents.error instanceof Error ? agents.error.message : undefined} />;
    } else if (!sessions) {
      list = <div className="h-24 animate-pulse rounded-lg border border-line bg-surface" />;
    } else if (sessions.length === 0) {
      list = <p className="rounded-lg border border-line bg-surface p-4 text-sm text-muted">{t("empty")}</p>;
    } else {
      list = sessions.map((a) => (
        <SessionCard
          key={a.session}
          agent={a}
          selected={focus?.selector.kind === "tmux" && focus.selector.session === a.session}
          onSelect={() => selectSession(a.session)}
        />
      ));
    }
    const back = backHref(focus?.origin ?? null);
    return (
      <div data-testid="focused-view" className="flex h-full min-h-0 flex-col gap-3 [animation:jax-rise_.4s_ease]">
        <h1 className="sr-only">{tTitle("sessions")}</h1>
        <div className="flex flex-wrap items-center gap-3">
          <Link href={back} className="text-[12.5px] font-semibold text-body-ink">← {t("back")}</Link>
          <span className="min-w-0 truncate font-mono text-[12.5px] font-semibold text-ink">{label ?? t("targetUnavailable")}</span>
          <span className="text-[11px] text-muted">{focus?.origin === "mosaic" ? t("openedFromMosaic") : t("openedFromHome")}</span>
          <span className="text-[11px] text-muted">{t("sharedSelection")}</span>
        </div>
        <details className="flex-none rounded-lg border border-line bg-surface px-3 py-2">
          <summary className="cursor-pointer text-[12.5px] font-semibold text-body-ink">{t("switcher")}</summary>
          <div className="mt-2 grid gap-[11px] sm:grid-cols-2">
            {list}
          </div>
        </details>
        <TerminalPanel
          session={label}
          frameUrl={lifecycle.frame?.url ?? null}
          frameKey={lifecycle.frame?.key ?? 0}
          frameMode={lifecycle.frame?.mode ?? "ro"}
          configLoading={config.isLoading}
          configured={ttyd?.configured === true}
          mode={lifecycle.frame?.mode ?? "ro"}
          busy={control.busy}
          error={control.error}
          status={lifecycle.status as AccessStatus}
          onTakeControl={takeControl}
          onReleaseControl={releaseControl}
          onReconnect={reconnect}
          largerHref={focus ? largerWindowHref(focus.selector) : null}
        />
      </div>
    );
  }

  const mobileLines = mobilePane ? (panesTailData?.find((p) => p.pane === mobilePane)?.lines ?? null) : null;

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto jax-scroll [animation:jax-rise_.4s_ease]">
      <h1 className="sr-only">{tTitle("sessions")}</h1>
      <Mosaic
        readiness={readiness} cards={mission.model.cards} panesTail={panesTailData}
        busy={busy} onAction={async (url, body) => { setBusy(true); await postWorkflowAction(url, body); setBusy(false); }}
        onOpenDetail={openCard} onOpenMobile={setMobilePane}
      />
      {mobileTarget ? (
        <MosaicCellDetail
          card={mobileTarget.card} pane={mobileTarget.pane} session={mobileTarget.session} lines={mobileLines}
          busy={busy} onAction={async (url, body) => { setBusy(true); await postWorkflowAction(url, body); setBusy(false); }}
          onClose={() => setMobilePane(null)}
        />
      ) : null}
    </div>
  );
}
