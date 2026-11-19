"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import { Suspense, useCallback, useEffect, useReducer, useRef, useState } from "react";
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
  parseExpectedTarget,
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

// Network rejection, timeout, non-success HTTP and a malformed envelope are all the SAME bounded
// unknown validation failure — never an exception escaping into the effect.
const UNKNOWN_TARGET: TargetResult = { ok: false, code: "source-failure", error: "unavailable" };
const FAILURE_CODES = new Set<string>(["invalid-selector", "unconfigured", "unavailable", "lost-target", "source-failure"]);

export function parseTargetResult(json: unknown): TargetResult {
  if (!json || typeof json !== "object" || Array.isArray(json)) return UNKNOWN_TARGET;
  const record = json as Record<string, unknown>;
  if (record.ok === false) {
    const code = typeof record.code === "string" && FAILURE_CODES.has(record.code) ? record.code as SessionAccessCode : "source-failure";
    return { ok: false, code, error: typeof record.error === "string" ? record.error : "unavailable" };
  }
  if (record.ok !== true) return UNKNOWN_TARGET;
  const data = record.data;
  if (!data || typeof data !== "object" || Array.isArray(data)) return UNKNOWN_TARGET;
  const resolved = data as Record<string, unknown>;
  // `{ok:true}` without usable success data or binding is malformed, not a validation.
  if (!parseExpectedTarget(resolved.expected)) return UNKNOWN_TARGET;
  if (typeof resolved.roUrl !== "string" || resolved.roUrl.length === 0) return UNKNOWN_TARGET;
  if (typeof resolved.sessionName !== "string" || typeof resolved.observedAt !== "string") return UNKNOWN_TARGET;
  if (resolved.runtime !== "claude" && resolved.runtime !== "codex" && resolved.runtime !== "unknown") return UNKNOWN_TARGET;
  return { ok: true, data: data as ResolvedSessionAccess };
}

export async function fetchTarget(
  selector: SessionAccessSelector,
  expected: ExpectedSessionTarget | null,
  signal: AbortSignal,
  fetchImpl: typeof fetch = fetch,
  timeoutMs: number = REQUEST_TIMEOUT_MS,
): Promise<TargetResult> {
  try {
    const res = await fetchImpl(`/api/tmux/target?${targetQueryString(selector, expected)}`, {
      signal: AbortSignal.any([signal, AbortSignal.timeout(timeoutMs)]),
      cache: "no-store",
    });
    if (!res.ok) return UNKNOWN_TARGET;
    return parseTargetResult(await res.json().catch(() => null));
  } catch {
    return UNKNOWN_TARGET;
  }
}

// Each request is tagged with the lifetime captured when it BEGINS (synchronous token + selector +
// expected binding). A result may act only while that exact lifetime is still current; the
// completion timestamp is freshness, never proof of generation.
export type ValidationLifetime = { token: number; selector: string; expected: string };
export type TargetResponse = { lifetime: ValidationLifetime; result: TargetResult };

export function targetAppliesTo(current: ValidationLifetime, response: ValidationLifetime): boolean {
  return current.token === response.token && current.selector === response.selector
    && current.expected === response.expected;
}

// A new lifetime is a new query entry: no in-flight or cached result of the previous one exists
// under it, so an explicit revalidation always issues a real request.
export function targetQueryKey(selector: string, expected: string, token: number) {
  return ["tmux", "target", selector, expected, token] as const;
}

export type TargetApplication =
  | { kind: "ignore" }
  | { kind: "expired" }
  | { kind: "failure"; status: Extract<AccessStatus, "unknown" | "unavailable" | "lost-target" | "expired" | "unconfigured"> }
  | { kind: "ok"; data: ResolvedSessionAccess };

export function targetApplication(
  response: TargetResponse | undefined,
  current: ValidationLifetime | null,
  receivedAt: number,
  now: number,
  notBefore: number,
): TargetApplication {
  if (!current || !response || !targetAppliesTo(current, response.lifetime)) return { kind: "ignore" };
  if (receivedAt < notBefore) return { kind: "ignore" };
  if (targetFreshness(receivedAt, now) === "expired") return { kind: "expired" };
  if (!response.result.ok) return { kind: "failure", status: accessFailureStatus(response.result.code) };
  return { kind: "ok", data: response.result.data };
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

// The first render already reflects the live search string, so a same-route query navigation is
// visible immediately and a focused URL never needs an effect to become the focused view.
function pageStateFromSearch(search: string): { view: "mosaic" | "detail"; focus: Focus | null; lifecycle: ReturnType<typeof accessLifecycle> } {
  const location = parseFocusedLocation(search);
  if (location.kind === "focus") {
    return {
      view: "detail",
      focus: { selector: location.selector, origin: location.origin },
      lifecycle: accessLifecycle(initialAccessLifecycle, { type: "open" }),
    };
  }
  if (location.kind === "invalid") {
    return {
      view: "detail",
      focus: null,
      lifecycle: accessLifecycle(initialAccessLifecycle, { type: "failure", status: "unavailable" }),
    };
  }
  return { view: "mosaic", focus: null, lifecycle: initialAccessLifecycle };
}

function SessionsPageInner() {
  const t = useTranslations("sessions");
  const tm = useTranslations("sessions.mosaic");
  const tTitle = useTranslations("title");
  const router = useRouter();
  const searchParams = useSearchParams();
  const search = searchParams?.toString() ?? "";
  const initial = pageStateFromSearch(search);

  const [mounted, setMounted] = useState(false);
  const [view, setView] = useState<"mosaic" | "detail">(initial.view);
  const [focus, setFocus] = useState<Focus | null>(initial.focus);
  const [resolved, setResolved] = useState<ResolvedSessionAccess | null>(null);
  const [expected, setExpected] = useState<ExpectedSessionTarget | null>(null);
  const [visible, setVisible] = useState(true);
  const [lifecycle, dispatchLifecycle] = useReducer(accessLifecycle, initial.lifecycle);
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
  // Synchronous selection token: every open/reconnect/visibility pause/loss bumps it, so a delayed
  // response can only affect a frame whose selector AND token still match.
  const tokenRef = useRef(0);
  const bumpToken = () => { tokenRef.current += 1; return tokenRef.current; };
  // Validation lifetime: bumped ONLY when a NEW validation is required (open, reconnect, show,
  // release). It is part of the request key, so those actions issue a genuinely fresh request and
  // an in-flight/cached result from a previous lifetime can never stand in for it. A failure or
  // expiry does NOT bump it: ordinary polling continues, but the reducer latches the frame off.
  const validationRef = useRef(0);
  const bumpValidation = () => { validationRef.current += 1; return validationRef.current; };

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
  const validationToken = validationRef.current;
  const target = useQuery({
    // The lifetime token is part of the key: every explicit revalidation asks a NEW question
    // instead of being answered from an in-flight or cached result of the previous one.
    queryKey: targetQueryKey(selectorKey, expectedKey, validationToken),
    queryFn: async ({ signal }): Promise<TargetResponse> => {
      const previous = focusRef.current;
      const previousExpected = expectedRef.current;
      const lifetime: ValidationLifetime = {
        token: validationRef.current,
        selector: previous ? selectorIdentity(previous.selector) : "",
        expected: previousExpected ? `${previousExpected.target}|${previousExpected.incarnation}` : "",
      };
      if (!previous) return { lifetime, result: UNKNOWN_TARGET };
      const result = await fetchTarget(previous.selector, previousExpected, signal);
      return { lifetime, result };
    },
    enabled: mounted && focus !== null && visible,
    refetchInterval: TARGET_POLL_MS,
    retry: false,
  });

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
      bumpToken();
      dispatchLifecycle({ type: "failure", status: "expired" });
    }, remaining);
  }, [clearExpiry]);

  const appliedSearchRef = useRef(search);

  useEffect(() => {
    setMounted(true);
    return () => clearExpiry();
  }, [clearExpiry]);

  // The live location is the single source of truth for query-only client navigation and browser
  // Back/Forward: every actual changed search string is parsed with the bounded selector grammar.
  // Applying it NEVER navigates — only a user-initiated handler below owns router calls.
  useEffect(() => {
    if (appliedSearchRef.current === search) return;
    appliedSearchRef.current = search;
    applyLocation(parseFocusedLocation(search));
  }, [search]);

  useEffect(() => {
    visibleRef.current = document.visibilityState !== "hidden";
    setVisible(visibleRef.current);
    const onVisibility = () => {
      const next = document.visibilityState !== "hidden";
      visibleRef.current = next;
      setVisible(next);
      if (!next) {
        clearExpiry();
        bumpToken();
        dispatchLifecycle({ type: "hide" });
      } else if (focusRef.current) {
        notBeforeRef.current = Date.now();
        bumpToken();
        bumpValidation();
        dispatchLifecycle({ type: "show" });
      }
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => document.removeEventListener("visibilitychange", onVisibility);
  }, [clearExpiry]);

  // One validation effect: expiry, source/target failure and the gated RO attach. Both success and
  // failure act only on the exact lifetime their request began in. A healthy poll only refreshes
  // facts — the iframe src/key stay untouched — and a failure latches the frame off, so a later
  // ordinary success can update metadata but never reattach without a new explicit lifetime.
  useEffect(() => {
    if (!mounted || !visible || !focus) return;
    const receivedAt = target.dataUpdatedAt;
    const selector = focusRef.current?.selector ?? null;
    const binding = expectedRef.current;
    const current: ValidationLifetime = {
      token: validationRef.current,
      selector: selector ? selectorIdentity(selector) : "",
      expected: binding ? `${binding.target}|${binding.incarnation}` : "",
    };
    const application = targetApplication(target.data, current, receivedAt, Date.now(), notBeforeRef.current);
    if (application.kind === "ignore") return;
    if (application.kind === "expired") {
      clearExpiry();
      bumpToken();
      dispatchLifecycle({ type: "failure", status: "expired" });
      return;
    }
    if (application.kind === "failure") {
      clearExpiry();
      setResolved(null);
      bumpToken();
      dispatchControl({ type: "reset" });
      dispatchLifecycle({ type: "failure", status: application.status });
      return;
    }
    setResolved(application.data);
    if (expectedRef.current === null) setExpected(application.data.expected);
    const lifecycleNow = lifecycleRef.current;
    if (lifecycleNow.pendingAttach) dispatchLifecycle({ type: "validation", roUrl: application.data.roUrl });
    if (lifecycleNow.pendingAttach || lifecycleNow.frame) armExpiry(receivedAt);
  }, [target.data, target.dataUpdatedAt, visible, mounted, focus, armExpiry, clearExpiry]);

  function openSelector(selector: SessionAccessSelector, origin: Focus["origin"]) {
    clearExpiry();
    setFocus({ selector, origin });
    setResolved(null);
    setExpected(null);
    setView("detail");
    notBeforeRef.current = Date.now();
    bumpToken();
    bumpValidation();
    dispatchLifecycle({ type: "open" });
    dispatchControl({ type: "reset" });
  }

  // URL-driven application of the same transition openSelector performs. It never calls the
  // router; a location that merely reflects the already-applied selector/origin re-opens nothing.
  function applyLocation(location: FocusedLocation) {
    clearExpiry();
    if (location.kind === "focus") {
      const current = focusRef.current;
      if (current && current.origin === location.origin
        && selectorIdentity(current.selector) === selectorIdentity(location.selector)) return;
      openSelector(location.selector, location.origin);
      return;
    }
    // Leaving a selection invalidates outstanding Take/validation/expiry before anything else, so
    // an old result can never restore a frame or keep focused polling alive.
    setResolved(null);
    setExpected(null);
    bumpToken();
    bumpValidation();
    if (location.kind === "invalid") {
      setFocus(null);
      setView("detail");
      dispatchControl({ type: "reset" });
      dispatchLifecycle({ type: "failure", status: "unavailable" });
      return;
    }
    setFocus(null);
    setView("mosaic");
    setMobilePane(null);
    dispatchControl({ type: "reset" });
    dispatchLifecycle({ type: "hide" });
  }

  function reconnect() {
    if (!focusRef.current) return;
    clearExpiry();
    notBeforeRef.current = Date.now();
    bumpToken();
    bumpValidation();
    dispatchLifecycle({ type: "reconnect" });
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
    const token = bumpToken();
    controlBusyRef.current = true;
    clearExpiry();
    dispatchLifecycle({ type: "takeStart" });
    dispatchControl({ type: "request", action: "take" });
    try {
      const res = await postJson<TakenSessionAccess>("/api/tmux/take-control", { selector, expected: binding });
      const stillCurrent = tokenRef.current === token
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
    const token = bumpToken();
    controlBusyRef.current = true;
    clearExpiry();
    // Synchronous read-only transition BEFORE any await; the audit cannot restore RW. The release
    // itself starts a NEW validation lifetime, so the RO reattach needs its own fresh success.
    dispatchLifecycle({ type: "releaseStart" });
    dispatchControl({ type: "request", action: "release" });
    bumpValidation();
    notBeforeRef.current = Date.now();
    try {
      const res = await postJson("/api/tmux/release-control", { selector, expected: binding });
      const stillCurrent = tokenRef.current === token
        && focusRef.current !== null && selectorIdentity(focusRef.current.selector) === selectorIdentity(selector);
      dispatchControl({ type: "settle", action: "release", ok: res.ok, error: res.ok ? null : (res.error ?? t("controlFailed")), current: stillCurrent });
    } finally {
      controlBusyRef.current = false;
    }
  }

  // User-initiated selections apply the state transition FIRST (synchronous frame/control
  // invalidation), then serialize the same selection into the canonical focused URL. An explicit
  // open PUSHES so Back can return to the previous view; an in-place switch REPLACES.
  function selectSession(session: string) {
    const origin = focusRef.current?.origin ?? "mosaic";
    const selector: SessionAccessSelector = { kind: "tmux", session };
    openSelector(selector, origin);
    router.replace(sessionAccessHref(selector, origin));
  }

  function openCard(card: MissionCard, pane: MissionPane) {
    const selector = paneAccessSelector(card.dir, pane);
    if (!selector) return;
    openSelector(selector, "mosaic");
    router.push(sessionAccessHref(selector, "mosaic"));
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
          validatedAt={lifecycle.frame !== null && target.dataUpdatedAt > 0 ? target.dataUpdatedAt : null}
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

export default function SessionsPage() {
  return (
    <Suspense>
      <SessionsPageInner />
    </Suspense>
  );
}
