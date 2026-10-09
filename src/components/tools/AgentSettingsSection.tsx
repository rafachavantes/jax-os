"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import { useEffect, useRef, useState } from "react";
import { SourceWarning } from "@/components/mission/SourceWarning";
import { fetchEnvelope } from "@/lib/api";
import {
  CLAUDE_REVIEWER_EFFORTS,
  CODEX_REVIEWER_EFFORTS,
  activationSaveEnabled,
  configuredModels,
  draftIsValid,
  parseEditorRevision,
  previewFingerprint,
  selectableConnections,
  type AgentSettings,
  type BuilderProfile,
  type ConnectionChoice,
  type EditorRevision,
  type ProfileName,
  type SettingsDraft,
} from "@/lib/agent-settings";
import { agentsOf, useGeneralSettings, type Agents } from "@/lib/settingsQuery";
import { OpenRouterRouting } from "./OpenRouterRouting";
import { classifyWrite, postJson, useToolsDraft, writeKindText } from "./ToolsDraftProvider";
import css from "./tools.module.css";

const BTN =
  "rounded-md bg-brand px-3 py-1.5 text-[12.5px] font-medium text-on-brand hover:bg-brand/90 disabled:opacity-60";
const GHOST =
  "rounded-md border border-line px-3 py-1.5 text-[12.5px] text-body-ink hover:bg-surface-2 disabled:opacity-60";

type Payload = {
  editor_revision?: EditorRevision;
  settings: AgentSettings | null;
  draft: SettingsDraft;
  connections: ConnectionChoice[];
};

function isPayload(value: unknown): value is Payload {
  if (!value || typeof value !== "object") return false;
  const row = value as Partial<Payload>;
  return !!row.draft && typeof row.draft === "object" && "reviewers" in row.draft && "builders" in row.draft;
}

function asConnections(value: unknown): ConnectionChoice[] {
  return Array.isArray(value) ? (value as ConnectionChoice[]) : [];
}

export function AgentSettingsSection({
  active,
  onManageConnection,
}: {
  active: boolean;
  onManageConnection?: (id: string) => void;
}) {
  const t = useTranslations("tools");
  const qc = useQueryClient();
  const tools = useToolsDraft();
  const applyRef = useRef(tools.applyServer);
  applyRef.current = tools.applyServer;
  const submitRef = useRef(false);
  const [status, setStatus] = useState<{ tone: "success" | "error" | "pending"; text: string } | null>(null);
  const [acknowledgedFor, setAcknowledgedFor] = useState<string | null>(null);
  const acknowledgementFingerprint = previewFingerprint(tools.draft, tools.revision);
  const acknowledged = acknowledgedFor !== null && acknowledgedFor === acknowledgementFingerprint;
  const q = useQuery({
    queryKey: ["agent-settings"],
    queryFn: ({ signal }) => fetchEnvelope<Payload>("/api/agent-settings", signal),
    refetchInterval: 10_000,
  });
  const agents = agentsOf(useGeneralSettings().data);

  useEffect(() => {
    if (!q.data?.ok || !isPayload(q.data.data)) return;
    applyRef.current({
      draft: q.data.data.draft,
      revision: parseEditorRevision(q.data.data.editor_revision) ?? undefined,
      connections: asConnections(q.data.data.connections),
      settingsPresent: q.data.data.settings != null,
    });
  }, [q.data]);

  const firstActivation = !tools.settingsPresent;
  const valid = draftIsValid(tools.draft, tools.connections, agents.opencode);
  const saveEnabled = activationSaveEnabled({
    firstActivation,
    dirty: tools.dirty,
    revision: tools.revision,
    conflict: tools.conflict,
    valid,
    busy: tools.busy,
    acknowledged,
  });

  async function save() {
    if (submitRef.current || tools.busy || !tools.revision) return;
    submitRef.current = true;
    tools.setBusy(true);
    const submitted = structuredClone(tools.draft);
    const expected = tools.revision;
    setStatus({ tone: "pending", text: t("saving") });
    try {
      const result = await postJson("/api/agent-settings", {
        action: "save",
        expected,
        reviewers: submitted.reviewers,
        builders: submitted.builders,
      });
      const classified = classifyWrite(result);
      if (classified.kind === "ok" && classified.revision) {
        tools.markSaved(submitted, classified.revision);
        setAcknowledgedFor(null);
        setStatus({ tone: "success", text: t("saved") });
        void qc.invalidateQueries({ queryKey: ["agent-settings"] });
        void qc.invalidateQueries({ queryKey: ["opencode-providers"] });
        return;
      }
      if (classified.kind === "ok") {
        setStatus({ tone: "error", text: t("writeUnconfirmed") });
        return;
      }
      setStatus({ tone: "error", text: writeKindText(t, classified.kind, classified.error) });
    } finally {
      tools.setBusy(false);
      submitRef.current = false;
    }
  }

  async function reconcile() {
    if (submitRef.current || tools.busy || !tools.revision) return;
    submitRef.current = true;
    tools.setBusy(true);
    setStatus({ tone: "pending", text: t("saving") });
    try {
      const result = await postJson("/api/agent-settings", { action: "reconcile", expected: tools.revision });
      const classified = classifyWrite(result);
      if (classified.kind === "ok") {
        setStatus({ tone: "success", text: t("saved") });
        void qc.invalidateQueries({ queryKey: ["agent-settings"] });
        void qc.invalidateQueries({ queryKey: ["opencode-providers"] });
        return;
      }
      setStatus({ tone: "error", text: writeKindText(t, classified.kind, classified.error) });
    } finally {
      tools.setBusy(false);
      submitRef.current = false;
    }
  }

  if (q.isError && !q.data) {
    return <SourceWarning label={t("settingsUnavailable")} detail={q.error instanceof Error ? q.error.message : undefined} />;
  }
  if (!q.data) return <div className="h-6 animate-pulse rounded bg-surface-2" />;
  if (!q.data.ok || !isPayload(q.data.data)) return <SourceWarning label={t("settingsUnavailable")} />;

  return (
    <div className="flex flex-col">
      {firstActivation ? (
        <p className="rounded-md border border-warning bg-warning-soft px-3.5 py-2 text-xs text-warning">{t("activationNote")}</p>
      ) : null}
      {tools.conflict ? (
        <p className="rounded-md border border-warning bg-warning-soft px-3.5 py-2 text-xs text-warning">{t("conflictBanner")}</p>
      ) : null}
      <fieldset disabled={tools.busy} className="contents">
        {agents.opencode ? (
          <>
            <div className={css.sectionHeading}>
              <div>
                <h2>{t("builderHeading")}</h2>
                <p>{t("builderSubtitle")}</p>
              </div>
              <span className={css.tag}>
                <code>opencode-builder</code>
              </span>
            </div>
            <div className={css.builderCards}>
              {(["default", "fallback"] as const).map((name) => (
                <ProfileCard key={name} name={name} active={active} onManageConnection={onManageConnection} />
              ))}
            </div>
            <p className={css.flowNote}>{t("builderFlowNote")}</p>
          </>
        ) : (
          <p className={`${css.hint} mb-3.5`}>{t("builderOff")}</p>
        )}
        <div className={css.sectionHeading}>
          <div>
            <h2>{t("reviewersHeading")}</h2>
            <p>{t("reviewersSubtitle")}</p>
          </div>
          <span className={css.tag}>{t("reviewersFixed")}</span>
        </div>
        <Reviewers agents={agents} />
        <p className={`${css.hint} mb-3.5`}>{t("reviewersModelHint")}</p>
      </fieldset>
      {firstActivation ? (
        <label className={css.checkLine}>
          <input
            type="checkbox"
            checked={acknowledged}
            onChange={(e) => setAcknowledgedFor(e.target.checked ? acknowledgementFingerprint : null)}
          />
          {t("activationAcknowledge")}
        </label>
      ) : null}
      {status ? (
        <p
          role="status"
          className={`rounded-md border px-3.5 py-2 text-xs ${
            status.tone === "error"
              ? "border-danger bg-danger-soft text-danger"
              : status.tone === "success"
                ? "border-success bg-success-soft text-success"
                : "border-line bg-surface-2 text-muted"
          }`}
        >
          {status.text}
        </p>
      ) : null}
      <div className={css.savebar}>
        <p className={`${css.saveState} ${tools.dirty ? css.dirty : ""}`}>
          {tools.dirty ? t("dirtyState") : t("cleanState")}
        </p>
        <div className={css.actionRow}>
          {tools.conflict ? (
            <button type="button" data-testid="agent-reconcile" className={GHOST} disabled={!tools.revision || tools.busy} onClick={() => void reconcile()}>
              {t("reconcile")}
            </button>
          ) : null}
          <button type="button" data-testid="agent-discard" className={GHOST} disabled={!tools.dirty || tools.busy} onClick={() => tools.discard()}>
            {t("discard")}
          </button>
          <button type="button" data-testid="agent-save" className={BTN} disabled={!saveEnabled} onClick={() => void save()}>
            {t("save")}
          </button>
        </div>
      </div>
    </div>
  );
}

function ProfileCard({
  name,
  active,
  onManageConnection,
}: {
  name: ProfileName;
  active: boolean;
  onManageConnection?: (id: string) => void;
}) {
  const t = useTranslations("tools");
  const { draft, setDraft, connections, resetEpoch } = useToolsDraft();
  const profile = draft.builders[name];
  const choices = selectableConnections(connections);
  const selected = connections.find((c) => c.id === profile.connection);
  const models = configuredModels(selected);
  const model = models.find((m) => m.id === profile.model);
  const isOpenRouter = selected?.adapter === "openrouter";
  const listId = `models-${name}`;
  const unknownModel = !!profile.model && !model;
  const savedConnectionUnavailable = !!profile.connection && !choices.some((c) => c.id === profile.connection);

  function patch(next: Partial<BuilderProfile>) {
    setDraft({
      ...draft,
      builders: { ...draft.builders, [name]: { ...profile, ...next } },
    });
  }

  return (
    <article className={`${css.card} ${css.builderCard}`}>
      <div className={css.cardHeading}>
        <div>
          <h3 className="text-[14px] font-bold text-ink">
            {name === "default" ? t("profileDefault") : t("profileFallback")}
          </h3>
          <p>{name === "default" ? t("profileDefaultHint") : t("profileFallbackHint")}</p>
        </div>
        <span className={`${css.tag} ${name === "default" ? css.brand : ""}`}>
          {name === "default" ? t("profilePrimary") : t("profileAlternative")}
        </span>
      </div>
      <div className={css.formGrid}>
        <label className={`${css.field} ${css.full}`}>
          {t("connection")}
          <select
            value={choices.some((c) => c.id === profile.connection) ? profile.connection : ""}
            onChange={(e) => {
              const c = connections.find((row) => row.id === e.target.value);
              patch({
                connection: e.target.value,
                model: "",
                effort: null,
                credential: c?.credential ?? profile.credential,
                routing: c?.adapter === "openrouter" ? profile.routing ?? { sort: "price", allow_fallbacks: true } : null,
              });
            }}
          >
            <option value="">{t("selectConnection")}</option>
            {savedConnectionUnavailable ? (
              <option value={profile.connection}>{profile.connection} · {t("unavailableShort")}</option>
            ) : null}
            {choices.map((c) => (
              <option key={c.id} value={c.id}>
                {c.label || c.id}
                {c.health === "oauth-managed" ? ` · ${t("oauthManaged")}` : ""}
              </option>
            ))}
          </select>
          <span className={css.hint}>{t("connectionHint")}</span>
        </label>
        <label className={`${css.field} ${css.full}`}>
          {t("model")}
          {isOpenRouter ? (
            <input
              type="text"
              autoComplete="off"
              value={profile.model}
              placeholder={t("modelOpenRouterPlaceholder")}
              aria-label={t("model")}
              onChange={(event) => {
                const value = event.target.value;
                const next = models.find((m) => m.id === value);
                const keepEffort = profile.effort && next && next.efforts.includes(profile.effort) ? profile.effort : null;
                patch({ model: value, effort: next?.no_effort ? null : keepEffort });
              }}
            />
          ) : (
            <input
              type="search"
              autoComplete="off"
              list={listId}
              value={profile.model}
              placeholder={t("modelSearchPlaceholder")}
              aria-label={t("model")}
              onChange={(event) => {
                const value = event.target.value;
                const next = models.find((m) => m.id === value);
                const keepEffort = profile.effort && next && next.efforts.includes(profile.effort) ? profile.effort : null;
                patch({ model: value, effort: next?.no_effort ? null : keepEffort });
              }}
            />
          )}
          {!isOpenRouter ? (
            <datalist id={listId}>
              {models.map((m) => (
                <option key={m.id} value={m.id}>{m.label || m.id}</option>
              ))}
            </datalist>
          ) : null}
          <span className={css.hint}>{t("modelHint", { count: String(models.length) })}</span>
        </label>
        {unknownModel ? (
          <p className={`${css.field} ${css.full} text-warning`}>
            {t("modelUnavailable")}{" "}
            <button type="button" className={css.quietButton} onClick={() => onManageConnection?.(profile.connection)}>
              {t("manageConnection")}
            </button>
          </p>
        ) : null}
        {selected?.pending ? (
          <p className={`${css.field} ${css.full} text-warning`}>{t("pendingSelect")}</p>
        ) : null}
        <label className={`${css.field} ${css.full}`}>
          {t("effort")} <span className={css.optional}>{t("optional")}</span>
          <select
            value={model?.no_effort ? "" : profile.effort ?? ""}
            disabled={!model || model.no_effort}
            onChange={(e) => patch({ effort: e.target.value || null })}
          >
            <option value="">{t("effortDefault")}</option>
            {!model?.no_effort
              ? (model?.efforts ?? []).map((effort) => (
                  <option key={effort} value={effort}>{effort}</option>
                ))
              : null}
          </select>
          <span className={css.hint}>
            {model && model.efforts.length ? t("effortHint") : t("effortNoneHint")}
          </span>
        </label>
      </div>
      <button type="button" className={`${GHOST} mt-2.5`} onClick={() => onManageConnection?.(profile.connection)}>
        {t("manageConnection")} →
      </button>
      {isOpenRouter ? (
        <OpenRouterRouting
          key={`${profile.connection}:${profile.model}:${resetEpoch}`}
          connection={profile.connection}
          model={profile.model}
          routing={profile.routing}
          onChange={(routing) => patch({ routing })}
        />
      ) : (
        <p className={css.directNote}>{t("directNote")}</p>
      )}
    </article>
  );
}

function Reviewers({ agents }: { agents: Agents }) {
  const t = useTranslations("tools");
  const { draft, setDraft } = useToolsDraft();
  const leads = (["claude", "codex"] as const).filter((lead) => agents[lead]);
  if (leads.length === 0) return <p className={`${css.hint} mb-3.5`}>{t("reviewersNone")}</p>;
  return (
    <section className={`${css.card} ${css.reviewers}`}>
      {leads.map((lead) => {
        const opposite = lead === "claude" ? "codex" : "claude";
        // MOA-504 D10: the opposite agent reviews when it is on; otherwise the lead reviews itself ("fallback").
        const reviewer = agents[opposite] ? opposite : lead;
        const fallback = reviewer === lead;
        const leadLabel = lead === "claude" ? "Claude" : "Codex";
        const reviewerLabel = reviewer === "claude" ? "Claude" : "Codex";
        const efforts = reviewer === "claude" ? CLAUDE_REVIEWER_EFFORTS : CODEX_REVIEWER_EFFORTS;
        const spec = draft.reviewers[reviewer];
        return (
          <div key={lead} className={css.reviewerRow}>
            <div>
              <h3>
                {t.rich("reviewerLead", {
                  lead: leadLabel,
                  reviewer: reviewerLabel,
                  arrow: (text) => <span className={css.reviewerArrow}>{text}</span>,
                  strong: (text) => <strong>{text}</strong>,
                })}
                {fallback ? <span className={css.tag} data-testid="reviewer-fallback">{t("reviewerFallback")}</span> : null}
              </h3>
              <small>{t("reviewerRuntime", { reviewer: reviewerLabel })}</small>
            </div>
            <label className={css.field}>
              {t("reviewerModel", { reviewer: reviewerLabel })}
              <input
                value={spec.model}
                aria-label={t("reviewerModel", { reviewer: reviewerLabel })}
                onChange={(e) =>
                  setDraft({
                    ...draft,
                    reviewers: { ...draft.reviewers, [reviewer]: { ...spec, model: e.target.value } },
                  })
                }
              />
            </label>
            <label className={css.field}>
              {t("effort")}
              <select
                value={spec.effort}
                aria-label={`${reviewer} ${t("effort")}`}
                onChange={(e) =>
                  setDraft({
                    ...draft,
                    reviewers: { ...draft.reviewers, [reviewer]: { ...spec, effort: e.target.value } },
                  })
                }
              >
                {efforts.map((effort) => (
                  <option key={effort} value={effort}>{effort}</option>
                ))}
              </select>
            </label>
          </div>
        );
      })}
    </section>
  );
}
