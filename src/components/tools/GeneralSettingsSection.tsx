"use client";

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import { postGeneralSettings, SETTINGS_QUERY_KEY, useGeneralSettings, type GeneralSettings } from "../../lib/settingsQuery";
import { Switch } from "../Switch";
import { SourceWarning } from "../mission/SourceWarning";
import css from "./tools.module.css";

// Same shape as server/settings.ts's defaultSettings() — kept local (not imported) so this
// client component never pulls in the node:fs-touching server module (F1: settings-malformed
// renders the form backed by these, so saving with no further edits still repairs the file via
// PUT's merge-onto-defaults).
const DEFAULT_SETTINGS: GeneralSettings = {
  ownerName: "", locale: "en-US", reposRoot: "", vaultPath: null,
  monitoredUnits: { user: [], system: [] },
  integrations: {
    linear: false, ttyd: false, webhook: false, hermesTokens: false,
    agents: { claude: false, codex: false, opencode: false },
    classifier: false, github: false, vault: false,
  },
};

type Integrations = GeneralSettings["integrations"];

// MOA-504 D5: one switch per agent. `credentialKey` is the read-only meter status field of the
// settings route (its API field names did not change: the OpenCode GO field is still `opencodeGo`).
const AGENT_ROWS: { flag: string; labelKey: string; agent: keyof Integrations["agents"]; credentialKey: "claude" | "codex" | "opencodeGo" }[] = [
  { flag: "agentClaude", labelKey: "general.agentClaudeLabel", agent: "claude", credentialKey: "claude" },
  { flag: "agentCodex", labelKey: "general.agentCodexLabel", agent: "codex", credentialKey: "codex" },
  { flag: "agentOpenCode", labelKey: "general.agentOpenCodeLabel", agent: "opencode", credentialKey: "opencodeGo" },
];

const INTEGRATION_ROWS: {
  flag: string;
  labelKey: string;
  get: (i: Integrations) => boolean;
  set: (i: Integrations, v: boolean) => Integrations;
}[] = [
  { flag: "linear", labelKey: "general.linearLabel", get: (i) => i.linear, set: (i, v) => ({ ...i, linear: v }) },
  { flag: "ttyd", labelKey: "general.ttydLabel", get: (i) => i.ttyd, set: (i, v) => ({ ...i, ttyd: v }) },
  { flag: "hermesTokens", labelKey: "general.hermesTokensLabel", get: (i) => i.hermesTokens, set: (i, v) => ({ ...i, hermesTokens: v }) },
  { flag: "classifier", labelKey: "general.classifierLabel", get: (i) => i.classifier, set: (i, v) => ({ ...i, classifier: v }) },
  { flag: "github", labelKey: "general.githubLabel", get: (i) => i.github, set: (i, v) => ({ ...i, github: v }) },
  { flag: "vault", labelKey: "general.vaultLabel", get: (i) => i.vault, set: (i, v) => ({ ...i, vault: v }) },
];

export function GeneralSettingsSection() {
  const t = useTranslations("tools");
  const qc = useQueryClient();
  const query = useGeneralSettings();
  const [draft, setDraft] = useState<GeneralSettings | null>(null);
  const [pending, setPending] = useState(false);
  const [failed, setFailed] = useState(false);
  const server = query.data?.ok ? query.data.data : null;
  // Review 776d86ce35fa F1: settings-unreadable is an OS-level read failure (no bytes to trust,
  // the PUT route refuses it too) — a plain refusal, no form. settings-malformed IS a file the
  // owner can repair from here: the form renders backed by defaults, and saving with no further
  // edits is enough (PUT merges the (empty) patch onto its own schema defaults).
  const malformed = query.data !== undefined && !query.data.ok && query.data.error === "settings-malformed";
  const unreadable = query.data !== undefined && !query.data.ok && query.data.error === "settings-unreadable";
  const current = draft ?? server ?? (malformed ? DEFAULT_SETTINGS : null);
  // Read-only, never draftable: comes from the route's env-var check, not from settings.json.
  const webhookConfigured = query.data?.ok ? query.data.webhookConfigured : false;
  const envFileState = query.data?.ok ? query.data.envFileState : "missing";
  const credentialsConfigured = query.data?.ok ? query.data.credentialsConfigured : undefined;

  async function save() {
    if (!current || pending) return;
    setPending(true);
    setFailed(false);
    const res = await postGeneralSettings(current);
    setPending(false);
    if (res.ok) {
      qc.setQueryData(SETTINGS_QUERY_KEY, res);
      setDraft(null);
    } else {
      setFailed(true);
    }
  }

  if (unreadable) return <SourceWarning label={t("general.settingsUnreadable")} />;
  if (!current) return null; // loading — same quiet-until-loaded convention as other sections
  return (
    <div className={`${css.card} p-5`}>
      {malformed ? (
        <SourceWarning label={t("general.settingsMalformed")} aside={t("general.settingsMalformedRepair")} />
      ) : null}
      <div className={`${css.sectionHeading} ${css.sectionHeadingGap}`}><h2>{t("general.ownerHeading")}</h2></div>
      <div className={css.formGrid}>
        <label className={css.field}>
          {t("general.ownerLabel")}
          <input name="ownerName" value={current.ownerName} onChange={(e) => setDraft({ ...current, ownerName: e.target.value })} />
        </label>
        <label className={css.field}>
          {t("general.localeLabel")}
          <select name="locale" value={current.locale} onChange={(e) => setDraft({ ...current, locale: e.target.value as GeneralSettings["locale"] })}>
            <option value="en-US">English</option>
            <option value="pt-BR">Português</option>
          </select>
        </label>
      </div>
      <div className={`${css.sectionHeading} ${css.sectionHeadingGap}`}><h2>{t("general.pathsHeading")}</h2></div>
      <div className={css.formGrid}>
        <label className={`${css.field} ${css.full}`}>
          {t("general.reposRootLabel")}
          <input name="reposRoot" value={current.reposRoot} onChange={(e) => setDraft({ ...current, reposRoot: e.target.value })} />
        </label>
        <label className={`${css.field} ${css.full}`}>
          {t("general.vaultPathLabel")} <span className={css.optional}>{t("optional")}</span>
          <input name="vaultPath" value={current.vaultPath ?? ""} onChange={(e) => setDraft({ ...current, vaultPath: e.target.value || null })} />
        </label>
      </div>
      <div className={`${css.sectionHeading} ${css.sectionHeadingGap}`}><h2>{t("general.monitoringHeading")}</h2></div>
      {/* monitoredUnits.user/.system: one textarea each, newline-delimited — simplest working
          input for a small string list (ponytail: no drag/drop chip widget for two short lists). */}
      <div className={css.formGrid}>
        <label className={`${css.field} ${css.full}`}>
          {t("general.monitoredUserLabel")}
          <textarea name="monitoredUnitsUser" value={current.monitoredUnits.user.join("\n")}
            onChange={(e) => setDraft({ ...current, monitoredUnits: { ...current.monitoredUnits, user: e.target.value.split("\n").map((s) => s.trim()).filter(Boolean) } })} />
        </label>
        <label className={`${css.field} ${css.full}`}>
          {t("general.monitoredSystemLabel")}
          <textarea name="monitoredUnitsSystem" value={current.monitoredUnits.system.join("\n")}
            onChange={(e) => setDraft({ ...current, monitoredUnits: { ...current.monitoredUnits, system: e.target.value.split("\n").map((s) => s.trim()).filter(Boolean) } })} />
        </label>
      </div>
      <div className={`${css.sectionHeading} ${css.sectionHeadingGap}`}><h2>{t("general.notificationsHeading")}</h2></div>
      {/* Spec's Notifications row: the editable integrations.webhook toggle (same as every other
          integration flag below) PLUS a read-only configured/missing status for the two env vars
          the webhook actually needs. No URL/secret field — those live in .env, owned by MOA-498. */}
      <div className={css.integrationRow} data-integration="webhook">
        <span>
          {t("general.webhookLabel")}{" "}
          <span data-webhook-status={webhookConfigured ? "configured" : "missing"} className={css.optional}>
            {webhookConfigured ? t("general.webhookConfiguredStatus") : t("general.webhookMissingStatus")}
          </span>{" "}
          <span data-env-file-status={envFileState} className={css.optional}>
            {t(`general.envFileStatus.${envFileState}`)}
          </span>
        </span>
        <Switch checked={current.integrations.webhook} aria-label={t("general.webhookLabel")}
          onChange={() => setDraft({ ...current, integrations: { ...current.integrations, webhook: !current.integrations.webhook } })} />
      </div>
      <div className={`${css.sectionHeading} ${css.sectionHeadingGap}`}><h2>{t("general.agentsHeading")}</h2></div>
      <p className={css.hint}>{t("general.agentsHint")}</p>
      {AGENT_ROWS.map((row) => {
        const on = current.integrations.agents[row.agent];
        const metered = credentialsConfigured?.[row.credentialKey];
        return (
          <div key={row.flag} className={css.integrationRow} data-integration={row.flag}>
            <span>
              {t(row.labelKey)}{" "}
              <span data-credential-status={metered ? "configured" : "missing"} className={css.optional}>
                {metered ? t("general.meterConfiguredStatus") : t("general.meterMissingStatus")}
              </span>
            </span>
            <Switch checked={on} aria-label={t(row.labelKey)}
              onChange={() => setDraft({ ...current, integrations: { ...current.integrations, agents: { ...current.integrations.agents, [row.agent]: !on } } })} />
          </div>
        );
      })}
      <div className={`${css.sectionHeading} ${css.sectionHeadingGap}`}><h2>{t("general.integrationsHeading")}</h2></div>
      {INTEGRATION_ROWS.map((row) => (
        <div key={row.flag} className={css.integrationRow} data-integration={row.flag}>
          <span>{t(row.labelKey)}</span>
          <Switch checked={row.get(current.integrations)} aria-label={t(row.labelKey)}
            onChange={() => setDraft({ ...current, integrations: row.set(current.integrations, !row.get(current.integrations)) })} />
        </div>
      ))}
      <div className={css.savebar}>
        <span className={`${css.saveState} ${draft ? css.dirty : ""}`}>{draft ? t("dirtyState") : t("cleanState")}</span>
        <div className={css.actionRow}>
          <button type="button" disabled={(!draft && !malformed) || pending} onClick={() => void save()}
            className="rounded-md bg-brand px-3 py-1.5 text-[12.5px] font-medium text-on-brand disabled:opacity-60">
            {pending ? t("saving") : t("save")}
          </button>
        </div>
      </div>
      {failed ? <p className="text-[11.5px] text-danger">{t("general.saveFailed")}</p> : null}
    </div>
  );
}
