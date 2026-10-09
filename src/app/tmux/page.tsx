"use client";

import { useQuery } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import { useEffect, useReducer, useRef, useState } from "react";
import { fetchEnvelope, type Envelope, type TtydConfig } from "@/lib/api";
import { RelativeTime } from "@/components/RelativeTime";
import type { Agent } from "@/server/collectors/tmux";
import { useMission } from "@/lib/useMission";
import { buildMission, postWorkflowAction, type Mission } from "@/lib/mission";
import { findMobilePane } from "@/lib/mosaicView";
import { pillReadinessOf } from "@/components/Topbar";
import type { ProjectScan } from "@/server/collectors/projects";
import type { HubEnvelopeData } from "@/server/db/workflows";
import { SessionCard } from "@/components/sessions/SessionCard";
import { controlStateReducer, TerminalPanel, captureControlTicket, controlTicketCurrent } from "@/components/sessions/TerminalPanel";
import { SourceWarning } from "@/components/mission/SourceWarning";
import { Mosaic } from "@/components/sessions/Mosaic";
import { MosaicCellDetail } from "@/components/sessions/MosaicCellDetail";

type PanesTailData = { pane: string; lines: string[] | null }[];

export default function SessionsPage() {
  const t = useTranslations("sessions");
  const tm = useTranslations("sessions.mosaic");
  const tTitle = useTranslations("title");

  const [view, setView] = useState<"mosaic" | "detail">("mosaic");
  const [mobilePane, setMobilePane] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [control, dispatch] = useReducer(controlStateReducer, { mode: "ro", busy: false, error: null });

  // Decision 13/18 (round-1 F4): the mosaic's own polls, gated off while the desktop detail view
  // is open — `agents` is the mirror gate, on only while detail needs its session list.
  const projects = useMission<ProjectScan>("projects", 10_000, view === "mosaic");
  const hub = useMission<HubEnvelopeData>("hub", 10_000, view === "mosaic");
  const agents = useMission<Agent[]>("agents", 5_000, view === "detail");
  const panesTail = useQuery({
    queryKey: ["tmux", "panes-tail"],
    queryFn: ({ signal }) => fetchEnvelope<PanesTailData>("/api/tmux/panes-tail", signal),
    refetchInterval: 3_000,
    enabled: view === "mosaic",
  });
  // Round-3 F6: desktop-only, same as `agents` — gated off while the mosaic is the active view.
  const config = useQuery({
    queryKey: ["tmux", "config"],
    queryFn: ({ signal }) => fetchEnvelope<TtydConfig>("/api/tmux/config", signal),
    refetchInterval: 30_000,
    enabled: view === "detail",
  });

  const mission: Mission = buildMission(projects.data, hub.data, undefined, Date.now());
  const readiness = pillReadinessOf(projects.data, hub.data);
  const panesTailData = panesTail.data?.ok ? panesTail.data.data : null;

  const sessions = agents.data?.ok ? agents.data.data : null;
  const active = (sessions?.find((s) => s.session === selected) ?? sessions?.[0])?.session ?? null;
  const ttyd = config.data?.ok ? config.data.data : null;

  const activeRef = useRef(active);
  activeRef.current = active;
  const selectionToken = useRef(0);
  const busyRef = useRef(false);

  useEffect(() => {
    selectionToken.current += 1;
    dispatch({ type: "reset" });
  }, [active]);

  function select(session: string) {
    setSelected(session);
    selectionToken.current += 1;
    dispatch({ type: "reset" });
  }

  // Decision 13/13a/round-2 F1: desktop cells only ever call this with a real tmuxSession
  // (Mosaic never wires onOpenDetail for a null-session pane).
  function openDetail(tmuxSession: string) {
    select(tmuxSession);
    setView("detail");
  }

  async function post(path: string, session: string): Promise<Envelope<never>> {
    return fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session }) })
      .then((r) => r.json())
      .catch((e) => ({ ok: false as const, error: String(e) }));
  }

  async function takeControl() {
    if (busyRef.current) return;
    const ticket = captureControlTicket(active, selectionToken.current);
    if (!ticket) return;
    busyRef.current = true;
    dispatch({ type: "request", action: "take" });
    try {
      const res = await post("/api/tmux/take-control", ticket.target);
      dispatch({ type: "settle", action: "take", ok: res.ok, error: res.ok ? null : res.error, current: controlTicketCurrent(activeRef.current, selectionToken.current, ticket) });
    } finally { busyRef.current = false; }
  }

  async function releaseControl() {
    if (busyRef.current) return;
    const ticket = captureControlTicket(active, selectionToken.current);
    if (!ticket) return;
    busyRef.current = true;
    dispatch({ type: "request", action: "release" });
    try {
      const res = await post("/api/tmux/release-control", ticket.target);
      dispatch({ type: "settle", action: "release", ok: res.ok, error: res.ok ? null : res.error, current: controlTicketCurrent(activeRef.current, selectionToken.current, ticket) });
    } finally { busyRef.current = false; }
  }

  // Decision 17a/17b: the mosaic's own answer/respond actions post through here.
  async function runAction(url: string, body: unknown) {
    setBusy(true);
    await postWorkflowAction(url, body);
    setBusy(false);
  }

  const mobileTarget = findMobilePane(mission.model.cards, mobilePane);

  // Round-3 F3: `findMobilePane` only matches a non-null `tmuxSession` — if a poll refresh drops
  // it to null (or the pane disappears) while its dialog is open, `mobileTarget` goes null on
  // this render but `mobilePane` (the selected id) is still set. Clear it so the dialog stays
  // closed instead of trying to reopen for a stale id on some later render.
  // Plan correction: this effect (and `mobileTarget` above it) MUST run before the `detail` early
  // return below — a hook after a conditional return breaks the rules of hooks the moment
  // `openDetail` flips `view` (React: "Rendered fewer hooks than expected").
  useEffect(() => {
    if (mobilePane && !mobileTarget) setMobilePane(null);
  }, [mobilePane, mobileTarget]);

  if (view === "detail") {
    let list: React.ReactNode;
    if (agents.isError && !sessions) {
      list = <SourceWarning label={t("unavailable")} detail={agents.error instanceof Error ? agents.error.message : undefined} />;
    } else if (!sessions) {
      list = <div className="h-24 animate-pulse rounded-lg border border-line bg-surface" />;
    } else if (sessions.length === 0) {
      list = <p className="rounded-lg border border-line bg-surface p-4 text-sm text-muted">{t("empty")}</p>;
    } else {
      list = sessions.map((a) => (
        <SessionCard key={a.session} agent={a} selected={a.session === active} onSelect={() => select(a.session)} />
      ));
    }
    return (
      <div className="flex h-full flex-col gap-3 [animation:jax-rise_.4s_ease]">
        <h1 className="sr-only">{tTitle("sessions")}</h1>
        <button type="button" onClick={() => setView("mosaic")} className="self-start text-[12.5px] font-semibold text-body-ink">
          ← {tm("back")}
        </button>
        <div className="grid min-h-0 flex-1 gap-4 lg:grid-cols-[300px_1fr]">
          <div className="flex flex-col gap-[11px] overflow-y-auto jax-scroll pr-1">
            <span className="px-0.5 text-[13px] font-bold text-ink">{t("listTitle", { count: sessions?.length ?? 0 })}</span>
            {agents.isError && sessions ? (
              <SourceWarning label={t("unavailable")} aside={agents.dataUpdatedAt ? <RelativeTime epochMs={agents.dataUpdatedAt} /> : undefined} />
            ) : null}
            {list}
          </div>
          <TerminalPanel
            session={active} config={ttyd} configLoading={config.isLoading}
            mode={control.mode} busy={control.busy} error={control.error}
            onTakeControl={takeControl} onReleaseControl={releaseControl}
          />
        </div>
      </div>
    );
  }

  const mobileLines = mobilePane ? (panesTailData?.find((p) => p.pane === mobilePane)?.lines ?? null) : null;

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto jax-scroll [animation:jax-rise_.4s_ease]">
      <h1 className="sr-only">{tTitle("sessions")}</h1>
      <Mosaic
        readiness={readiness} cards={mission.model.cards} panesTail={panesTailData}
        busy={busy} onAction={runAction} onOpenDetail={openDetail} onOpenMobile={setMobilePane}
      />
      {mobileTarget ? (
        <MosaicCellDetail
          card={mobileTarget.card} pane={mobileTarget.pane} session={mobileTarget.session} lines={mobileLines}
          busy={busy} onAction={runAction} onClose={() => setMobilePane(null)}
        />
      ) : null}
    </div>
  );
}
