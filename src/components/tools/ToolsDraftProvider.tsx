"use client";

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import {
  BOOTSTRAP_DRAFT,
  isSettingsDraftDirty,
  parseEditorRevision,
  saveIsStale,
  settingsWriteKind,
  type ConnectionChoice,
  type CredentialRef,
  type EditorRevision,
  type SettingsDraft,
  type SettingsWriteKind,
} from "@/lib/agent-settings";

export type CredentialDialogRequest = {
  providerId: string;
  current: CredentialRef | null;
  adapter?: string | null;
  invoker?: HTMLElement | null;
  expected?: EditorRevision;
};

type ApplyPayload = {
  draft?: SettingsDraft;
  revision?: EditorRevision;
  connections?: ConnectionChoice[];
  settingsPresent?: boolean;
};

export type ToolsDraftApi = {
  dirty: boolean;
  conflict: boolean;
  draft: SettingsDraft;
  baseline: SettingsDraft;
  revision: EditorRevision | null;
  connections: ConnectionChoice[];
  settingsPresent: boolean;
  busy: boolean;
  resetEpoch: number;
  setBusy: (busy: boolean) => void;
  setDraft: (draft: SettingsDraft) => void;
  discard: () => void;
  applyServer: (payload: ApplyPayload) => void;
  markSaved: (draft: SettingsDraft, revision: EditorRevision) => void;
  credential: CredentialDialogRequest | null;
  openCredential: (req: CredentialDialogRequest) => void;
  closeCredential: () => void;
};

const Ctx = createContext<ToolsDraftApi | null>(null);
const noop = () => {};

const FALLBACK: ToolsDraftApi = {
  dirty: false,
  conflict: false,
  draft: BOOTSTRAP_DRAFT,
  baseline: BOOTSTRAP_DRAFT,
  revision: null,
  connections: [],
  settingsPresent: false,
  busy: false,
  resetEpoch: 0,
  setBusy: noop,
  setDraft: noop,
  discard: noop,
  applyServer: noop,
  markSaved: noop,
  credential: null,
  openCredential: noop,
  closeCredential: noop,
};

export function useToolsDraft(): ToolsDraftApi {
  return useContext(Ctx) ?? FALLBACK;
}

export async function postJson(url: string, body: unknown): Promise<unknown> {
  try {
    return await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => r.json());
  } catch {
    return null;
  }
}

export function writeKindText(
  t: (key: string, params?: Record<string, string>) => string,
  kind: SettingsWriteKind,
  error?: string,
): string {
  if (kind === "ok") return t("saved");
  if (kind === "activation-pending") return t("activationPending");
  if (kind === "applied-unrecorded") return t("writeAppliedUnrecorded");
  if (kind === "unconfirmed") return t("writeUnconfirmed");
  return t("writeRefused", { error: error ?? "unavailable" });
}

export function classifyWrite(res: unknown): { kind: SettingsWriteKind; error?: string; revision?: EditorRevision } {
  const kind = settingsWriteKind(res);
  const env = res && typeof res === "object" && !Array.isArray(res) ? (res as Record<string, unknown>) : null;
  const error = env && typeof env.error === "string" ? env.error : undefined;
  const data = env && env.data && typeof env.data === "object" ? (env.data as Record<string, unknown>) : null;
  const revision = parseEditorRevision(data?.editor_revision) ?? undefined;
  return { kind, error, revision };
}

export function ToolsDraftProvider({ children }: { children: ReactNode }) {
  const [draft, setDraft] = useState(BOOTSTRAP_DRAFT);
  const [baseline, setBaseline] = useState(BOOTSTRAP_DRAFT);
  const [revision, setRevision] = useState<EditorRevision | null>(null);
  const [connections, setConnections] = useState<ConnectionChoice[]>([]);
  const [settingsPresent, setSettingsPresent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [resetEpoch, setResetEpoch] = useState(0);
  const [conflict, setConflict] = useState(false);
  const [credential, setCredential] = useState<CredentialDialogRequest | null>(null);
  const dirty = isSettingsDraftDirty(baseline, draft);

  useEffect(() => {
    if (!dirty && !credential) return;
    const onUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", onUnload);
    return () => window.removeEventListener("beforeunload", onUnload);
  }, [dirty, credential]);

  const value: ToolsDraftApi = {
    dirty,
    conflict,
    draft,
    baseline,
    revision,
    connections,
    settingsPresent,
    busy,
    resetEpoch,
    setBusy,
    setDraft,
    discard: () => {
      setDraft(baseline);
      setConflict(false);
      setResetEpoch((n) => n + 1);
    },
    applyServer: (payload) => {
      if (payload.connections) setConnections(payload.connections);
      if (payload.settingsPresent !== undefined) setSettingsPresent(payload.settingsPresent);
      if (payload.revision) {
        if (dirty && revision && saveIsStale(revision, payload.revision)) {
          setConflict(true);
          return;
        }
        setRevision(payload.revision);
      }
      if (payload.draft && !dirty) {
        // A genuinely different clean baseline invalidates local editor state;
        // an unchanged poll must not.
        if (isSettingsDraftDirty(baseline, payload.draft)) setResetEpoch((n) => n + 1);
        setDraft(payload.draft);
        setBaseline(payload.draft);
        setConflict(false);
      }
    },
    markSaved: (next, rev) => {
      setDraft(next);
      setBaseline(next);
      setRevision(rev);
      setSettingsPresent(true);
      setConflict(false);
      setResetEpoch((n) => n + 1);
    },
    credential,
    openCredential: setCredential,
    closeCredential: () => setCredential(null),
  };

  return (
    <Ctx.Provider value={value}>{children}</Ctx.Provider>
  );
}
