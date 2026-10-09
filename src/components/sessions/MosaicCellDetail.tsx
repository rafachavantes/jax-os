"use client";

import { useEffect, useRef, useState } from "react";
import { X } from "lucide-react";
import { useTranslations } from "next-intl";
import { mosaicPaneAction, postWorkflowAction, type MissionCard, type MissionPane } from "@/lib/mission";
import { LIMITS } from "@/lib/workflow";
import { MosaicAnswerOptions, MosaicQuestionText } from "./Mosaic";

// Direct-tested handlers (round-2 F4 pattern) — the ACTUAL POST call, independent of the
// component's own dialog/render lifecycle.
export async function takeControl(
  session: string, setBusy: (b: boolean) => void, setMode: (m: "ro" | "rw") => void, setError: (e: string | null) => void,
) {
  setBusy(true);
  setError(null);
  const result = await postWorkflowAction<unknown>("/api/tmux/take-control", { session });
  setBusy(false);
  if (result.ok) setMode("rw"); else setError(result.error);
}

export async function releaseControl(
  session: string, setBusy: (b: boolean) => void, setMode: (m: "ro" | "rw") => void, setError: (e: string | null) => void,
) {
  setBusy(true);
  await postWorkflowAction<unknown>("/api/tmux/release-control", { session });
  setBusy(false);
  setMode("ro");
  setError(null);
}

export type MosaicCellDetailProps = {
  card: MissionCard;
  pane: MissionPane;
  // Round-3 F3: the caller's already-confirmed non-null tmux session (from `findMobilePane`,
  // Task 3) — a required `string`, never `pane.tmuxSession!`. The type system, not an assertion,
  // is what makes the null case unrepresentable here.
  session: string;
  lines: string[] | null;
  busy: boolean;
  onAction: (url: string, body: unknown) => void;
  onClose: () => void;
};

// Decision 20: a bare <dialog>, sized full-viewport via explicit inline overrides — the
// element's own top-layer gives Escape-to-close and back-gesture-close for free (platform
// feature); only its default centered/fit-content SIZING needed overriding.
export function MosaicCellDetail({ card, pane, session, lines, busy, onAction, onClose }: MosaicCellDetailProps) {
  const t = useTranslations("sessions.mosaic");
  const tSessions = useTranslations("sessions");
  const tMission = useTranslations("mission");
  const ref = useRef<HTMLDialogElement>(null);
  const [mode, setMode] = useState<"ro" | "rw">("ro");
  const [controlBusy, setControlBusy] = useState(false);
  const [controlError, setControlError] = useState<string | null>(null);
  const [reply, setReply] = useState("");
  const action = mosaicPaneAction(pane);
  const replyEventId = pane.pendingQuestion?.eventId ?? pane.capsuleEventId;
  const validReply = reply.trim().length > 0 && reply.length <= LIMITS.freeformReply;

  useEffect(() => {
    if (ref.current && typeof ref.current.showModal === "function") ref.current.showModal();
  }, []);

  return (
    <dialog
      ref={ref}
      onClose={onClose}
      onCancel={(e) => { if (busy || controlBusy) e.preventDefault(); }}
      className="m-0 h-[100dvh] max-h-none w-[100dvw] max-w-none rounded-none border-0 bg-surface p-0 backdrop:bg-base/60"
      style={{ inset: 0 }}
    >
      <div
        className="flex h-full flex-col"
        style={{
          paddingTop: "env(safe-area-inset-top)", paddingBottom: "env(safe-area-inset-bottom)",
          paddingLeft: "env(safe-area-inset-left)", paddingRight: "env(safe-area-inset-right)",
        }}
      >
        <div className="flex flex-none items-center justify-between gap-2 border-b border-line px-3 py-2">
          <span className="min-w-0 truncate text-[13px] font-semibold text-ink">{card.name} · {pane.pane}</span>
          <button type="button" aria-label={t("close")} onClick={onClose}
            className="flex h-11 w-11 flex-none items-center justify-center rounded-md text-body-ink hover:bg-surface-2">
            <X className="h-5 w-5" aria-hidden />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto jax-scroll px-3 py-3">
          <pre className="whitespace-pre-wrap break-all font-mono text-[12px] text-body-ink">
            {lines && lines.length > 0 ? lines.join("\n") : t("noOutput")}
          </pre>
        </div>
        {action.kind !== "none" ? (
          <div className="flex flex-none flex-col gap-2 border-t border-line px-3 py-3">
            {action.kind === "options" ? (
              <>
                <MosaicQuestionText pendingQuestion={action.pendingQuestion} />
                <MosaicAnswerOptions pendingQuestion={action.pendingQuestion} busy={busy} onAction={onAction} size="mobile" />
              </>
            ) : action.kind === "respond" ? (
              <div className="flex flex-col gap-2">
                <button type="button" disabled={controlBusy || mode === "rw"}
                  className="min-h-11 rounded-md border border-line-strong px-3 text-[12.5px] font-semibold text-body-ink disabled:cursor-not-allowed disabled:opacity-50"
                  onClick={() => void takeControl(session, setControlBusy, setMode, setControlError)}>
                  {controlBusy ? tSessions("takingControl") : tSessions("takeControl")}
                </button>
                <div className="flex items-center gap-2">
                  <input
                    type="text" disabled={mode !== "rw" || busy} value={reply} maxLength={LIMITS.freeformReply}
                    onChange={(e) => setReply(e.target.value)} placeholder={tMission("actions.freeformPlaceholder")}
                    className="min-h-11 flex-1 rounded-md border border-line bg-surface px-2.5 text-[13px] disabled:cursor-not-allowed disabled:opacity-50"
                  />
                  <button type="button" disabled={mode !== "rw" || busy || !validReply || replyEventId === null}
                    className="min-h-11 rounded-md border border-brand bg-brand-soft px-3 text-[12.5px] font-semibold text-brand disabled:cursor-not-allowed disabled:opacity-50"
                    onClick={() => { if (replyEventId !== null) { onAction("/api/workflow/answer", { event_id: replyEventId, reply }); setReply(""); } }}>
                    {tMission("actions.submit")}
                  </button>
                </div>
                {mode === "rw" ? (
                  <button type="button" disabled={controlBusy}
                    className="min-h-11 self-start rounded-md border border-line px-3 text-[12px] font-medium text-body-ink disabled:cursor-not-allowed disabled:opacity-50"
                    onClick={() => void releaseControl(session, setControlBusy, setMode, setControlError)}>
                    {controlBusy ? tSessions("releasingControl") : tSessions("releaseControl")}
                  </button>
                ) : null}
                {controlError ? <span className="text-[11px] text-danger">{controlError}</span> : null}
              </div>
            ) : action.kind === "open-pane" ? (
              <span className="text-[12.5px] text-muted">{tMission("inbox.blocked")}</span>
            ) : null}
          </div>
        ) : null}
      </div>
    </dialog>
  );
}
