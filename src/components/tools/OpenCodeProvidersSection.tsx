"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useLocale, useTranslations } from "next-intl";
import { useEffect, useRef, useState } from "react";
import { SourceWarning } from "@/components/mission/SourceWarning";
import { fetchEnvelope } from "@/lib/api";
import {
  parseEditorRevision,
  referencedProfiles,
  type AgentSettings,
  type ConnectionChoice,
  type EditorRevision,
  type ModelChoice,
} from "@/lib/agent-settings";
import { ProviderConnectionDialog } from "./ProviderConnectionDialog";
import { ProviderModelDialog } from "./ProviderModelDialog";
import { ProviderRemovalDialog, type RemovalTarget } from "./ProviderRemovalDialog";
import { classifyWrite, postJson, useToolsDraft, writeKindText } from "./ToolsDraftProvider";
import css from "./tools.module.css";

const GHOST =
  "rounded-md border border-line px-3 py-1.5 text-[12.5px] text-body-ink hover:bg-surface-2 disabled:opacity-60";
const BTN =
  "rounded-md bg-brand px-3 py-1.5 text-[12.5px] font-medium text-on-brand hover:bg-brand/90 disabled:opacity-60";

type Payload = { editor_revision?: EditorRevision; connections: ConnectionChoice[] };

function asConnections(value: unknown): ConnectionChoice[] {
  return Array.isArray(value) ? (value as ConnectionChoice[]) : [];
}

function healthClass(health: ConnectionChoice["health"]): string {
  if (health === "registered") return "text-success";
  if (health === "invalid") return "text-danger";
  if (health === "missing" || health === "sync-pending") return "text-warning";
  if (health === "oauth-managed") return "text-info";
  return "text-muted";
}

// Truthful effort summary: list the supported efforts and say when the
// provider default is also allowed; a missing list is unknown, not zero.
function modelEffortText(m: ModelChoice, t: (key: string) => string): string {
  const list = m.efforts.join(", ");
  if (m.efforts.length && m.no_effort) return `${t("modelEfforts")}: ${list} · ${t("effortDefault")}`;
  if (m.efforts.length) return `${t("modelEfforts")}: ${list}`;
  if (m.no_effort) return `${t("modelEffort")}: ${t("effortDefault")}`;
  return `${t("modelEffort")}: ${t("modelEffortUnknown")}`;
}

export function OpenCodeProvidersSection({
  initialConnectionId,
  onOpenAgents,
}: {
  initialConnectionId?: string | null;
  onOpenAgents?: () => void;
} = {}) {
  const t = useTranslations("tools");
  const locale = useLocale();
  const nf = new Intl.NumberFormat(locale);
  const qc = useQueryClient();
  const tools = useToolsDraft();
  const applyRef = useRef(tools.applyServer);
  applyRef.current = tools.applyServer;
  const [query, setQuery] = useState("");
  const [modelQuery, setModelQuery] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [connectionDialog, setConnectionDialog] = useState<{ open: boolean; edit: ConnectionChoice | null; invoker: HTMLButtonElement } | null>(null);
  const [modelFor, setModelFor] = useState<{ connection: string; model: ModelChoice | null; invoker: HTMLButtonElement } | null>(null);
  const [removal, setRemoval] = useState<RemovalTarget | null>(null);
  const [removalPending, setRemovalPending] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const q = useQuery({
    queryKey: ["opencode-providers"],
    queryFn: ({ signal }) => fetchEnvelope<Payload>("/api/opencode-providers", signal),
    refetchInterval: 10_000,
  });
  useEffect(() => {
    if (!q.data?.ok) return;
    applyRef.current({
      revision: parseEditorRevision(q.data.data.editor_revision) ?? undefined,
      connections: asConnections(q.data.data.connections),
    });
  }, [q.data]);

  useEffect(() => {
    if (initialConnectionId) setSelectedId(initialConnectionId);
  }, [initialConnectionId]);

  const rows = tools.connections.filter((c) => {
    const qv = query.trim().toLowerCase();
    if (!qv) return true;
    return c.id.toLowerCase().includes(qv) || c.label.toLowerCase().includes(qv);
  });
  const selected = rows.find((c) => c.id === selectedId) ?? rows[0];

  async function mutate(body: Record<string, unknown>) {
    if (!tools.revision || tools.busy) return;
    const res = await postJson("/api/opencode-providers", { ...body, expected: tools.revision });
    const classified = classifyWrite(res);
    if (classified.kind === "ok") {
      setStatus(t("saved"));
      void qc.invalidateQueries({ queryKey: ["opencode-providers"] });
      void qc.invalidateQueries({ queryKey: ["agent-settings"] });
      return;
    }
    setStatus(writeKindText(t, classified.kind, classified.error));
  }

  function draftReferences(connectionId: string, modelId?: string): string[] {
    return referencedProfiles({ builders: tools.draft.builders } as AgentSettings, connectionId, modelId);
  }

  function confirmRemoval() {
    if (!removal) return;
    setRemovalPending(true);
    const body = removal.kind === "connection"
      ? { action: "remove-connection", connection: removal.id }
      : { action: "remove-model", connection: removal.id, model: removal.model };
    void mutate(body).finally(() => {
      setRemovalPending(false);
      setRemoval(null);
    });
  }

  if (q.isError && !q.data) {
    return <SourceWarning label={t("providersUnavailable")} detail={q.error instanceof Error ? q.error.message : undefined} />;
  }
  if (!q.data) return <div className="h-6 animate-pulse rounded bg-surface-2" />;
  if (!q.data.ok) return <SourceWarning label={t("providersUnavailable")} />;

  const models = selected
    ? selected.models.filter((m) => {
        const qv = modelQuery.trim().toLowerCase();
        if (!qv) return true;
        return m.id.toLowerCase().includes(qv) || m.label.toLowerCase().includes(qv);
      })
    : [];
  const usedBy = selected
    ? [...new Set([...selected.used_by, ...draftReferences(selected.id)])]
    : [];
  // OAuth connections are read-only here: hide their mutation controls.
  // Unsafe/ambiguous native connections keep the controls but disabled, and
  // the shared draft busy state blocks provider mutations during Agents Save.
  const oauth = selected?.auth === "oauth";
  const mutateBlocked = selected?.editable === false || tools.busy;

  return (
    <div className="flex flex-col gap-4">
      <div className={css.providersHead}>
        <div>
          <h2>{t("providersHeading")}</h2>
          <p>{t("providersSubtitle")}</p>
        </div>
        <button type="button" className={BTN} onClick={(event) => setConnectionDialog({ open: true, edit: null, invoker: event.currentTarget })}>
          {t("addConnection")}
        </button>
      </div>
      <p className="text-[12px] text-muted">{t("incompleteHint")}</p>
      <div className={css.providersToolbar}>
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label={t("searchConnections")}
          placeholder={t("searchConnections")}
          className="h-8 w-full rounded-md border border-line bg-surface-2 px-3 text-[13px] text-ink outline-none placeholder:text-muted focus:border-line-strong"
        />
        <span className="text-[11px] text-muted">
          {t("connectionsCount", { count: String(tools.connections.length) })}
        </span>
      </div>
      {status ? <p role="status" className="text-xs text-muted">{status}</p> : null}
      <div className={css.providersLayout}>
        <ul className={`${css.card} ${css.providerList}`} aria-label={t("searchConnections")}>
          {rows.map((c) => (
            <li key={c.id}>
              <button
                type="button"
                onClick={() => setSelectedId(c.id)}
                aria-pressed={selected?.id === c.id}
                className={`${css.providerConn} ${selected?.id === c.id ? css.providerConnSelected : ""}`}
              >
                <strong className="text-[13px] text-ink">{c.label || c.id}</strong>
                <span>
                  {c.id} · {c.models.length} {t("models").toLowerCase()}
                </span>
                <span className={healthClass(c.health)}>
                  {c.health === "oauth-managed" ? t("oauthManaged") : t(`health.${c.health}`)}
                </span>
                {c.pending ? <span className="text-[11px] text-warning">{t("pendingBadge")}</span> : null}
              </button>
            </li>
          ))}
        </ul>
        {selected ? (
          <section aria-label={t("selectedConnection")} className={`${css.card} ${css.providerDetail} flex min-w-0 flex-col gap-3`}>
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <h2 className="text-[17px] font-semibold text-ink">{selected.label || selected.id}</h2>
                <p className="font-mono text-[12px] text-muted">{selected.id}</p>
              </div>
              {onOpenAgents ? (
                <button type="button" className={GHOST} onClick={onOpenAgents}>{t("viewAgents")}</button>
              ) : null}
            </div>
            <dl className={css.providerMeta}>
              <dt>{t("providerAdapter")}</dt>
              <dd>{selected.adapter ?? t("none")}</dd>
              <dt>{t("providerBaseUrl")}</dt>
              <dd>{selected.base_url ?? t("providerManagedUrl")}</dd>
              <dt>{t("providerCredential")}</dt>
              <dd>
                {selected.credential?.kind === "env"
                  ? t("credentialEnv", { env: selected.credential.env })
                  : selected.credential?.kind === "native"
                    ? t("credentialNative")
                    : t("none")}
                {" · "}
                <span className={healthClass(selected.health)}>
                  {selected.health === "oauth-managed" ? t("oauthManaged") : t(`health.${selected.health}`)}
                </span>
              </dd>
              <dt>{t("usedBy")}</dt>
              <dd>{usedBy.join(", ") || t("none")}</dd>
            </dl>
            {selected.reason ? <p className="text-[12px] text-muted">{selected.reason}</p> : null}
            {selected.adapter === "openai-compatible" ? (
              <p className="rounded-md border border-warning bg-warning-soft px-3 py-2 text-xs text-warning">{t("customEndpointWarn")}</p>
            ) : null}
            {oauth ? (
              <p className="rounded-md border border-info bg-info-soft px-3 py-2 text-xs text-info">{t("oauthReadOnly")}</p>
            ) : null}
            {!oauth ? (
              <div className={css.providerActions}>
                <button
                  type="button"
                  className={GHOST}
                  disabled={mutateBlocked}
                  onClick={(event) => {
                    if (mutateBlocked) return;
                    setConnectionDialog({ open: true, edit: selected, invoker: event.currentTarget });
                  }}
                >
                  {t("editConnection")}
                </button>
                <button
                  type="button"
                  className={GHOST}
                  disabled={mutateBlocked}
                  onClick={(e) => {
                    if (mutateBlocked) return;
                    tools.openCredential({
                      providerId: selected.id,
                      current: selected.credential,
                      adapter: selected.adapter,
                      invoker: e.currentTarget,
                    });
                  }}
                >
                  {t("editCredential")}
                </button>
                <button
                  type="button"
                  className={GHOST}
                  disabled={mutateBlocked}
                  onClick={() => {
                    if (mutateBlocked) return;
                    setRemoval({ kind: "connection", id: selected.id });
                  }}
                >
                  {t("removeConnection")}
                </button>
              </div>
            ) : null}
            <div className={css.modelHead}>
              <span className="text-[11px] font-semibold uppercase tracking-wide text-muted">{t("models")}</span>
              {!oauth ? (
                <button
                  type="button"
                  className={GHOST}
                  disabled={mutateBlocked}
                  onClick={(event) => {
                    if (mutateBlocked) return;
                    setModelFor({ connection: selected.id, model: null, invoker: event.currentTarget });
                  }}
                >
                  {t("addModel")}
                </button>
              ) : null}
            </div>
            <input
              type="search"
              value={modelQuery}
              onChange={(e) => setModelQuery(e.target.value)}
              aria-label={t("searchModels")}
              placeholder={t("searchModels")}
              className="h-8 w-full max-w-sm rounded-md border border-line bg-surface-2 px-3 text-[13px] text-ink outline-none placeholder:text-muted focus:border-line-strong"
            />
            {models.length === 0 ? (
              <p className="text-[12px] text-muted">{t("noModels")}</p>
            ) : (
              <ul className={css.providerModels}>
                {models.map((m) => (
                  <li key={m.id} className={css.providerModel}>
                    <div>
                      <strong className="block truncate text-[13px] text-ink">{m.label || m.id}</strong>
                      <span className="block text-[11px] text-muted">{m.id}</span>
                      <span className="block text-[11px] text-muted">
                        {t("modelContext")}: {m.context != null ? nf.format(m.context) : t("modelContextUnknown")}
                        {" · "}
                        {modelEffortText(m, t)}
                      </span>
                    </div>
                    <span className={css.providerModelButtons}>
                      {!oauth ? (
                        <>
                          <button
                            type="button"
                            className={GHOST}
                            disabled={mutateBlocked}
                            onClick={(event) => {
                              if (mutateBlocked) return;
                              setModelFor({ connection: selected.id, model: m, invoker: event.currentTarget });
                            }}
                          >
                            {t("editModel")}
                          </button>
                          <button
                            type="button"
                            className={GHOST}
                            disabled={mutateBlocked}
                            onClick={() => {
                              if (mutateBlocked) return;
                              setRemoval({ kind: "model", id: selected.id, model: m.id, catalog: m.origin === "catalog" });
                            }}
                          >
                            {t("removeModel")}
                          </button>
                        </>
                      ) : null}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </section>
        ) : null}
      </div>
      {connectionDialog ? (
        <ProviderConnectionDialog
          connection={connectionDialog.edit}
          expected={tools.revision}
          invoker={connectionDialog.invoker}
          onClose={() => setConnectionDialog(null)}
        />
      ) : null}
      {modelFor ? (
        <ProviderModelDialog
          connection={modelFor.connection}
          model={modelFor.model}
          models={tools.connections.find((c) => c.id === modelFor.connection)?.models}
          expected={tools.revision}
          invoker={modelFor.invoker}
          onClose={() => setModelFor(null)}
        />
      ) : null}
      {removal ? (
        <ProviderRemovalDialog
          target={removal}
          references={[
            ...new Set([
              ...(tools.connections.find((c) => c.id === removal.id)?.used_by ?? []),
              ...(removal.kind === "connection"
                ? draftReferences(removal.id)
                : draftReferences(removal.id, removal.model)),
            ]),
          ]}
          pending={removalPending}
          onConfirm={confirmRemoval}
          onClose={() => setRemoval(null)}
        />
      ) : null}
    </div>
  );
}
