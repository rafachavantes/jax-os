"use client";

import { useState } from "react";
import { useQuery, useQueryClient, type QueryClient } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import type { Envelope } from "@/lib/api";
import { FORWARDABLE_EVENT_TYPES } from "@/lib/workflow";
import type { EventType } from "@/lib/workflow";

type Settings = { archiveAfterDays: number };
type AfkSettings = { enabled: boolean; forwardTypes: EventType[] };

export const AFK_QUERY_KEY = ["mission", "afk"];

// Exported (plan review round 1 F2): the network call on its own, so a test can assert its
// exact `fetch` args directly — the static-render convention has no simulated click to drive
// the component's own `save()` handler.
export async function postArchiveAfterDays(days: number): Promise<Envelope<Settings>> {
  return fetch("/api/mission/settings", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ archiveAfterDays: days }),
  }).then((r) => r.json()).catch((e) => ({ ok: false as const, error: String(e) }));
}

// MOA-487 §5 Decision 6/7 (plan review round 2 F2): mirrors postPrefsPatch's own optimistic-cache
// pattern exactly (src/lib/prefsAction.ts) — snapshot the afk query cache, patch forwardTypes
// optimistically so the checklist re-renders this same tick, POST, invalidate on success, roll
// back the pre-patch snapshot on failure. Resolves a plain boolean, not the Envelope — the
// component only needs a pass/fail flag for its one useState<boolean> (round 2 F3).
export type PostForwardTypesFn = (url: string, body: unknown) => Promise<Envelope<AfkSettings>>;

// review 3c7b4f9f1da2 F1: the injected `post` can reject (network failure) or throw during JSON
// parsing, not just resolve `{ok:false}` — wrap the call so any of those roll back the optimistic
// patch and resolve `false` the same way a server-reported failure does.
export async function postForwardTypes(
  queryClient: QueryClient,
  next: EventType[],
  post: PostForwardTypesFn = (url, body) =>
    fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then((r) => r.json()),
): Promise<boolean> {
  const snapshot = queryClient.getQueryData<Envelope<AfkSettings>>(AFK_QUERY_KEY);
  if (snapshot?.ok) {
    queryClient.setQueryData(AFK_QUERY_KEY, { ok: true, data: { ...snapshot.data, forwardTypes: next } });
  }
  try {
    const result = await post("/api/workflow/afk", { forwardTypes: next });
    if (!result.ok) throw new Error(result.error);
  } catch {
    if (snapshot) queryClient.setQueryData(AFK_QUERY_KEY, snapshot);
    return false;
  }
  void queryClient.invalidateQueries({ queryKey: AFK_QUERY_KEY });
  return true;
}

// Pure add/remove — exported so the checkbox wiring is testable without a simulated DOM event
// (this codebase's component tests render static markup only, no jsdom/RTL).
export function computeForwardTypesToggle(current: EventType[], type: EventType, checked: boolean): EventType[] {
  if (checked) return current.includes(type) ? current : [...current, type];
  return current.filter((t) => t !== type);
}

// Exported (review 3c7b4f9f1da2 F3): the checkbox's own onChange wiring, pulled out so a test can
// invoke it directly and assert the POST body, the optimistic patch, and rollback on both
// `{ok:false}` and a rejected `post` — a static-markup render can't simulate a click to reach it.
export async function handleForwardTypeToggle(
  deps: { queryClient: QueryClient; current: EventType[]; type: EventType; checked: boolean; post?: PostForwardTypesFn },
  setSaving: (v: boolean) => void,
  setFailed: (v: boolean) => void,
): Promise<void> {
  const next = computeForwardTypesToggle(deps.current, deps.type, deps.checked);
  setSaving(true);
  try {
    setFailed(!(await postForwardTypes(deps.queryClient, next, deps.post)));
  } finally {
    setSaving(false);
  }
}

// Exported (plan review round 2 F3): isolates the failed-save caption's conditional render so a
// renderToStaticMarkup test can force both branches directly — the flag itself lives in the
// parent's own useState<boolean>, set from postForwardTypes' resolved boolean.
export function ForwardTypesFailedCaption({ failed, t }: { failed: boolean; t: (key: string) => string }) {
  return failed ? <span className="text-[11px] text-danger">{t("forwardTypesSaveFailed")}</span> : null;
}

// MOA-487 §5 Decision 5: label key per forwardable type. FORWARDABLE_EVENT_TYPES stays the single
// source of which SIX types exist and their render order — this map only adds the copy key.
const FORWARD_TYPE_LABEL_KEYS: Partial<Record<EventType, string>> = {
  question: "forwardTypeQuestion",
  "attention-needed": "forwardTypeAttentionNeeded",
  "mission-finished": "forwardTypeMissionFinished",
  "run-finished": "forwardTypeRunFinished",
  "merge-approved": "forwardTypeMergeApproved",
  "mission-status-updated": "forwardTypeMissionStatusUpdated",
};

export function MissionSettingsSection({ initialForwardSaving = false }: { initialForwardSaving?: boolean }) {
  const t = useTranslations("tools");
  const qc = useQueryClient();
  const [draft, setDraft] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [failed, setFailed] = useState(false);

  const query = useQuery<Envelope<Settings>>({
    queryKey: ["mission", "settings"],
    queryFn: async () => {
      const res = await fetch("/api/mission/settings");
      if (!res.ok) throw new Error(`GET /api/mission/settings: ${res.status}`);
      return res.json();
    },
    refetchInterval: 30_000,
  });
  const current = query.data?.ok ? query.data.data.archiveAfterDays : null;
  const value = draft ?? (current !== null ? String(current) : "");
  const parsed = Number(value);
  const valid = Number.isInteger(parsed) && parsed >= 1 && parsed <= 30;

  async function save() {
    if (!valid || pending) return;
    setPending(true);
    setFailed(false);
    const res = await postArchiveAfterDays(parsed);
    if (res.ok) {
      qc.setQueryData(["mission", "settings"], res);
      setDraft(null);
    } else {
      setFailed(true);
    }
    setPending(false);
  }

  const afkQuery = useQuery<Envelope<AfkSettings>>({
    queryKey: AFK_QUERY_KEY,
    queryFn: async () => {
      const res = await fetch("/api/workflow/afk");
      if (!res.ok) throw new Error(`GET /api/workflow/afk: ${res.status}`);
      return res.json();
    },
    refetchInterval: 30_000,
  });
  // No component-local draft (plan review round 2 F2): the checklist reads straight off the
  // query cache, which postForwardTypes patches optimistically and rolls back on failure.
  const forwardTypes = afkQuery.data?.ok ? afkQuery.data.data.forwardTypes : null;
  const [forwardFailed, setForwardFailed] = useState(false);
  // review 3c7b4f9f1da2 F2: disables the whole checklist for the duration of a save so overlapping
  // toggles can't commit out of order (each POST writes the complete array).
  const [forwardSaving, setForwardSaving] = useState(initialForwardSaving);

  async function toggleForwardType(type: EventType, checked: boolean) {
    if (forwardTypes === null) return;
    await handleForwardTypeToggle({ queryClient: qc, current: forwardTypes, type, checked }, setForwardSaving, setForwardFailed);
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-2">
        <span className="text-[11px] font-semibold uppercase tracking-[.08em] text-muted">{t("missionArchiveLabel")}</span>
        <div className="flex items-center gap-2">
          <input type="number" min={1} max={30} value={value} disabled={query.isError || current === null}
            onChange={(e) => setDraft(e.target.value)} className="w-20 rounded-md border border-line bg-surface px-2 py-1 text-[12.5px]" />
          <button type="button" disabled={!valid || pending} onClick={() => void save()}
            className="rounded-md bg-brand px-3 py-1.5 text-[12.5px] font-medium text-on-brand disabled:opacity-60">
            {t("save")}
          </button>
        </div>
        <span className="text-[11px] text-muted">{t("missionArchiveHint")}</span>
        {failed ? <span className="text-[11px] text-danger">{t("missionArchiveFailed")}</span> : null}
      </div>
      <div className="flex flex-col gap-2">
        <span className="text-[11px] font-semibold uppercase tracking-[.08em] text-muted">{t("forwardTypesTitle")}</span>
        <div className="flex flex-col gap-1">
          {FORWARDABLE_EVENT_TYPES.map((type) => {
            const labelKey = FORWARD_TYPE_LABEL_KEYS[type];
            if (!labelKey) return null;
            return (
              <label key={type} className="flex items-center gap-2 text-[12.5px]">
                <input type="checkbox" data-forward-type={type} checked={forwardTypes?.includes(type) ?? false}
                  disabled={afkQuery.isError || forwardTypes === null || forwardSaving}
                  onChange={(e) => void toggleForwardType(type, e.target.checked)} />
                {t(labelKey)}
              </label>
            );
          })}
        </div>
        <ForwardTypesFailedCaption failed={forwardFailed} t={t} />
      </div>
    </div>
  );
}
