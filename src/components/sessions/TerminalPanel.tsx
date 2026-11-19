"use client";

import { Eye, ExternalLink, Keyboard, RotateCw } from "lucide-react";
import Link from "next/link";
import { useTranslations } from "next-intl";
import type { SessionAccessCode } from "@/lib/sessionAccess";
import { SourceWarning } from "@/components/mission/SourceWarning";

// selection-lifetime ticket for Take/Release: a request captures the session
// and a selection token; a settlement is current only when BOTH still match.
// This closes the A -> B -> A race — session-string equality alone would let
// a Take for the first A enable control after the user re-selected A —
// without adding a lock system: the page invalidates the token synchronously
// on explicit selection changes and on polling-driven active-session changes.
export type ControlTicket = { target: string; token: number };
export function captureControlTicket(active: string | null, token: number): ControlTicket | null {
  return active === null ? null : { target: active, token };
}
export function controlTicketCurrent(
  active: string | null,
  token: number,
  ticket: ControlTicket,
): boolean {
  return active === ticket.target && token === ticket.token;
}

// take enters writable only after a confirmed success on the same session;
// release drops to read-only immediately, even when recording fails (I1)
export function controlModeAfter(
  action: "take" | "release",
  outcomeOk: boolean,
  stillOnSession: boolean,
): "ro" | "rw" {
  if (action === "release") return "ro";
  return outcomeOk && stillOnSession ? "rw" : "ro";
}

// control state machine the tmux page runs: release flips the client to ro
// SYNCHRONOUSLY on request, before the audit POST is even awaiting; the ro
// state survives delayed/error responses. A settle for a session the user
// already left never enables control and never overwrites current feedback.
export type ControlState = { mode: "ro" | "rw"; busy: boolean; error: string | null };
export type ControlAction =
  | { type: "request"; action: "take" | "release" }
  | { type: "settle"; action: "take" | "release"; ok: boolean; error: string | null; current: boolean }
  | { type: "reset" };

export function controlStateReducer(state: ControlState, action: ControlAction): ControlState {
  switch (action.type) {
    case "request":
      if (state.busy) return state; // synchronous duplicate guard
      return {
        mode: action.action === "release" ? controlModeAfter("release", false, true) : state.mode,
        busy: true,
        error: null,
      };
    case "settle":
      if (!action.current) return { ...state, busy: false }; // late prior-session result: leave feedback alone
      return {
        mode: controlModeAfter(action.action, action.ok, true),
        busy: false,
        error: action.ok ? null : action.error,
      };
    case "reset":
      return { mode: "ro", busy: state.busy, error: null };
  }
}

// The focused view's frame lifecycle, kept pure so every transition is directly testable:
// a synchronous generation token, ONE frame, and a fresh-validation attach gate. A healthy
// metadata poll never changes the frame; only an explicit open/reconnect/show/release reattaches.
export type AccessStatus =
  | "pending"
  | "live"
  | "paused"
  | "unknown"
  | "unavailable"
  | "lost-target"
  | "expired"
  | "unconfigured";

export type AccessFrame = { url: string; key: number; mode: "ro" | "rw" };
export type AccessLifecycle = {
  generation: number;
  frame: AccessFrame | null;
  status: AccessStatus;
  pendingAttach: boolean;
};

export const initialAccessLifecycle: AccessLifecycle = {
  generation: 0, frame: null, status: "pending", pendingAttach: true,
};

export type AccessEvent =
  | { type: "open" }
  | { type: "hide" }
  | { type: "show" }
  | { type: "reconnect" }
  | { type: "validation"; roUrl: string }
  | { type: "failure"; status: Extract<AccessStatus, "unknown" | "unavailable" | "lost-target" | "expired" | "unconfigured"> }
  | { type: "takeStart" }
  | { type: "takeOk"; rwUrl: string }
  | { type: "takeFailed" }
  | { type: "releaseStart" };

export function accessLifecycle(state: AccessLifecycle, event: AccessEvent): AccessLifecycle {
  switch (event.type) {
    case "open":
    case "reconnect":
      return { generation: state.generation + 1, frame: null, status: "pending", pendingAttach: true };
    case "hide":
      return { generation: state.generation + 1, frame: null, status: "paused", pendingAttach: false };
    case "show":
      return { generation: state.generation + 1, frame: null, status: "pending", pendingAttach: true };
    case "validation":
      // A late validation never remounts a frame that was already removed.
      if (!state.pendingAttach) return state;
      return {
        generation: state.generation,
        frame: { url: event.roUrl, key: state.generation, mode: "ro" },
        status: "live",
        pendingAttach: false,
      };
    case "failure":
      return { generation: state.generation + 1, frame: null, status: event.status, pendingAttach: false };
    case "takeStart":
      return { generation: state.generation + 1, frame: null, status: "pending", pendingAttach: false };
    case "takeOk":
      return {
        generation: state.generation,
        frame: { url: event.rwUrl, key: state.generation, mode: "rw" },
        status: "live",
        pendingAttach: false,
      };
    case "takeFailed":
      return { generation: state.generation + 1, frame: null, status: "pending", pendingAttach: false };
    case "releaseStart":
      return { generation: state.generation + 1, frame: null, status: "pending", pendingAttach: true };
  }
}

export function accessFailureStatus(code: SessionAccessCode): Extract<AccessStatus, "unknown" | "unavailable" | "lost-target" | "expired" | "unconfigured"> {
  if (code === "lost-target") return "lost-target";
  if (code === "unavailable") return "unavailable";
  if (code === "unconfigured") return "unconfigured";
  return "unknown";
}

type Props = {
  session: string | null;
  frameUrl: string | null;
  frameKey: number;
  frameMode: "ro" | "rw";
  configLoading: boolean;
  configured: boolean;
  mode: "ro" | "rw";
  busy: boolean;
  error: string | null;
  status: AccessStatus;
  onTakeControl: () => void;
  onReleaseControl: () => void;
  onReconnect: () => void;
  largerHref: string | null;
};

// The panel renders ONLY a prepared frame URL; it never derives one from config, a mode boolean
// or a fallback. The real ttyd display stays transport authority — an iframe load is not a
// connected badge.
export function TerminalPanel({
  session,
  frameUrl,
  frameKey,
  frameMode,
  configLoading,
  configured,
  mode,
  busy,
  error,
  status,
  onTakeControl,
  onReleaseControl,
  onReconnect,
  largerHref,
}: Props) {
  const t = useTranslations("sessions");
  const writable = mode === "rw" && frameMode === "rw" && frameUrl !== null;
  const canControl = configured && !busy && frameUrl !== null && status === "live";

  let body: React.ReactNode;
  if (configLoading) {
    body = <div className="m-5 h-6 animate-pulse rounded bg-surface-2" />;
  } else if (!configured || status === "unconfigured") {
    body = <div className="p-5"><SourceWarning label={t("notConfigured")} /></div>;
  } else if (frameUrl !== null) {
    body = (
      <iframe
        key={`${frameMode}-${frameKey}`}
        src={frameUrl}
        title={session ? `tmux: ${session}` : "tmux"}
        className="h-full min-h-0 w-full flex-1 border-0"
      />
    );
  } else if (status === "paused") {
    body = <div className="p-5 text-[12.5px] text-muted">{t("paused")}</div>;
  } else if (status === "unknown") {
    body = <div className="p-5"><SourceWarning label={t("unknown")} /></div>;
  } else if (status === "lost-target") {
    body = <div className="p-5"><SourceWarning label={t("targetLost")} /></div>;
  } else if (status === "unavailable") {
    body = <div className="p-5"><SourceWarning label={t("targetUnavailable")} /></div>;
  } else if (status === "expired") {
    body = <div className="p-5"><SourceWarning label={t("expired")} /></div>;
  } else {
    body = (
      <div className="flex flex-1 items-start p-5 font-mono text-[12.5px] text-muted">
        {t("validating")}
        <span className="ml-1 inline-block h-[15px] w-2 bg-brand [animation:jax-blink_1.1s_infinite]" />
      </div>
    );
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-lg border border-line bg-surface-inset">
      <div className="flex flex-none flex-wrap items-center gap-x-3 gap-y-1 border-b border-line bg-surface px-4 py-2">
        <div className="flex gap-[7px]">
          <span className="h-[11px] w-[11px] rounded-full bg-danger" />
          <span className="h-[11px] w-[11px] rounded-full bg-warning" />
          <span className="h-[11px] w-[11px] rounded-full bg-success" />
        </div>
        <span className="min-w-0 flex-1 truncate font-mono text-xs text-body-ink">
          {session ? `tmux · ${session}` : "tmux"}
          <span className="ml-2 font-sans text-[10.5px] text-muted">{t("sharedSelection")}</span>
        </span>
        <span
          className={`flex flex-none items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[11px] ${
            writable ? "border-brand-soft-border bg-brand-soft text-brand" : "border-line text-muted"
          }`}
        >
          {writable ? <Keyboard className="h-3 w-3" /> : <Eye className="h-3 w-3" />}
          {writable ? t("writable") : t("readOnly")}
        </span>
        {busy ? (
          <span role="status" aria-live="polite" className="flex-none text-[11px] font-semibold text-muted">
            {writable ? t("releasingControl") : t("takingControl")}
          </span>
        ) : null}
        <button
          type="button"
          onClick={onReconnect}
          disabled={busy || !configured}
          aria-label={t("reconnect")}
          className="flex h-8 flex-none items-center gap-1.5 rounded-full border border-line px-3 text-xs font-semibold text-body-ink transition-transform active:scale-95 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <RotateCw className="h-3.5 w-3.5" aria-hidden />
          {t("reconnect")}
        </button>
        {largerHref ? (
          <Link
            href={largerHref}
            target="_blank"
            rel="noopener"
            className="flex h-8 flex-none items-center gap-1.5 rounded-full border border-line px-3 text-xs font-semibold text-body-ink"
          >
            <ExternalLink className="h-3.5 w-3.5" aria-hidden />
            {t("openLarger")}
          </Link>
        ) : null}
        <button
          type="button"
          onClick={writable ? onReleaseControl : onTakeControl}
          disabled={!canControl}
          aria-busy={busy || undefined}
          title={!canControl ? t("controlNeedsLiveFrame") : undefined}
          className={`flex h-8 flex-none items-center gap-2 rounded-full border px-3.5 text-xs font-semibold transition-transform active:scale-95 disabled:cursor-not-allowed disabled:opacity-50 ${
            writable
              ? "border-line-strong bg-transparent text-body-ink"
              : "border-brand-soft-border bg-brand-soft text-brand"
          }`}
        >
          <Keyboard className="h-3.5 w-3.5" aria-hidden />
          {writable ? t("releaseControl") : t("takeControl")}
        </button>
        {error ? <span className="w-full truncate text-[11px] text-danger">{error}</span> : null}
        <p className="w-full text-[10.5px] text-muted">{t("nativeEnterException")}</p>
        <p className="w-full text-[10.5px] text-muted">{t("modeChangeScrollback")}</p>
      </div>
      {body}
    </div>
  );
}
