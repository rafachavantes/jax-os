"use client";

import { Eye, Keyboard } from "lucide-react";
import { useTranslations } from "next-intl";
import type { TtydConfig } from "@/lib/api";
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

type Props = {
  session: string | null;
  config: TtydConfig | null;
  configLoading: boolean;
  mode: "ro" | "rw";
  busy: boolean;
  error: string | null;
  onTakeControl: () => void;
  onReleaseControl: () => void;
};

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

export function TerminalPanel({
  session,
  config,
  configLoading,
  mode,
  busy,
  error,
  onTakeControl,
  onReleaseControl,
}: Props) {
  const t = useTranslations("sessions");
  const writable = mode === "rw";

  let body: React.ReactNode;
  if (configLoading) {
    body = <div className="m-5 h-6 animate-pulse rounded bg-surface-2" />;
  } else if (!config?.configured) {
    body = <div className="p-5"><SourceWarning label={t("notConfigured")} /></div>;
  } else if (!config.reachable) {
    body = <div className="p-5"><SourceWarning label={t("unreachable")} /></div>;
  } else if (!session) {
    body = (
      <div className="flex flex-1 items-start p-5 font-mono text-[12.5px] text-muted">
        {t("selectHint")}
        <span className="ml-1 inline-block h-[15px] w-2 bg-brand [animation:jax-blink_1.1s_infinite]" />
      </div>
    );
  } else {
    const base = writable ? config.rwUrl : config.roUrl;
    body = (
      <iframe
        key={`${mode}-${session}`}
        src={`${base}/?arg=${encodeURIComponent(session)}`}
        title={`tmux: ${session}`}
        className="h-full w-full flex-1 border-0"
      />
    );
  }

  return (
    <div className="flex h-full flex-col overflow-hidden rounded-lg border border-line bg-surface-inset">
      <div className="flex h-[46px] flex-none items-center gap-3 border-b border-line bg-surface px-4">
        <div className="flex gap-[7px]">
          <span className="h-[11px] w-[11px] rounded-full bg-danger" />
          <span className="h-[11px] w-[11px] rounded-full bg-warning" />
          <span className="h-[11px] w-[11px] rounded-full bg-success" />
        </div>
        <span className="truncate font-mono text-xs text-body-ink">
          {session ? `tmux · ${session}` : "tmux"}
        </span>
        <span
          className={`flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[11px] ${
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
        {error ? <span className="truncate text-[11px] text-danger">{error}</span> : null}
        <button
          onClick={writable ? onReleaseControl : onTakeControl}
          disabled={busy || !session || !config?.configured || !config.reachable}
          aria-busy={busy || undefined}
          title={
            !session
              ? t("controlNoSession")
              : !config?.configured || !config.reachable
                ? t("controlGatewayDown")
                : undefined
          }
          className={`ml-auto flex h-8 flex-none items-center gap-2 rounded-full border px-3.5 text-xs font-semibold transition-transform active:scale-95 disabled:cursor-not-allowed disabled:opacity-50 ${
            writable
              ? "border-line-strong bg-transparent text-body-ink"
              : "border-brand-soft-border bg-brand-soft text-brand"
          }`}
        >
          <Keyboard className="h-3.5 w-3.5" />
          {writable ? t("releaseControl") : t("takeControl")}
        </button>
      </div>
      {body}
    </div>
  );
}
