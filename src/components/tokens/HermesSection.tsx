"use client";

import { useRef } from "react";
import { useLocale, useTranslations } from "next-intl";
import { SourceWarning } from "@/components/mission/SourceWarning";
import { RelativeTime } from "@/components/RelativeTime";
import { retainOkMeta } from "@/lib/api";
import { useGeneralSettings } from "@/lib/settingsQuery";
import { useTokensApi } from "@/lib/useTokensApi";
import type { HermesMeta, PromptBudget, SkillActivity } from "@/server/collectors/hermes-meta";

const SKILL_CAP = 15;

const fmtKb = (bytes: number, locale: string) =>
  `${new Intl.NumberFormat(locale, { maximumFractionDigits: 1 }).format(bytes / 1024)} kB`;

function promptGuidance(t: ReturnType<typeof useTranslations>, error: string): string {
  if (error === "timeout") return t("promptTimeout");
  if (error === "invalid-output") return t("promptInvalid");
  return t("promptUnavailable");
}

export function HermesSection() {
  const t = useTranslations("tokens.hermes");
  const locale = useLocale();
  const { data: settings } = useGeneralSettings();
  const hermesTokensEnabled = settings?.ok === true && settings.data.integrations.hermesTokens;
  const { data, isError, dataUpdatedAt } = useTokensApi<HermesMeta>("hermes", 300_000, hermesTokensEnabled);
  const promptHold = useRef<{ data: PromptBudget; updatedAt: number } | undefined>(undefined);
  const skillsHold = useRef<{ data: SkillActivity[]; updatedAt: number } | undefined>(undefined);
  // MOA-496: the whole section is gone when hermesTokens is off (spec per-integration table).
  if (!hermesTokensEnabled) return null;
  if (isError && !data?.ok) {
    return (
      <SourceWarning
        label={t("promptUnavailable")}
        aside={dataUpdatedAt ? <RelativeTime epochMs={dataUpdatedAt} /> : undefined}
      />
    );
  }
  const meta = data?.ok ? data.data : undefined;
  if (!meta) return null;
  const now = dataUpdatedAt || Date.now();
  const prompt = retainOkMeta(promptHold.current, meta.prompt, now);
  const skills = retainOkMeta(skillsHold.current, meta.skills, now);
  if (prompt.data && !prompt.error) promptHold.current = { data: prompt.data, updatedAt: prompt.updatedAt ?? now };
  if (skills.data && !skills.error) skillsHold.current = { data: skills.data, updatedAt: skills.updatedAt ?? now };

  return (
    <>
      {isError ? (
        // transport failure with a retained envelope: TanStack keeps the last
        // successful response, so the sections below still show previous data
        // under this stale warning with the last-success time.
        <SourceWarning
          label={t("promptUnavailable")}
          aside={dataUpdatedAt ? <RelativeTime epochMs={dataUpdatedAt} /> : undefined}
        />
      ) : null}
      <div className="grid gap-4 lg:grid-cols-2">
      {/* Prompt budget */}
      <div className="flex flex-col gap-4 rounded-lg border border-line bg-surface px-5 py-[18px]">
        <div className="flex items-baseline gap-2">
          <span className="text-sm font-bold text-ink">{t("promptTitle")}</span>
          <span className="text-xs text-muted">— {t("perTurn")}</span>
          {prompt.data ? (
            <span className="ml-auto font-mono text-[12.5px] text-ink">{fmtKb(prompt.data.totalBytes, locale)}</span>
          ) : null}
        </div>
        {prompt.error ? (
          <SourceWarning
            label={promptGuidance(t, prompt.error)}
            aside={prompt.updatedAt ? <RelativeTime epochMs={prompt.updatedAt} /> : undefined}
          />
        ) : null}
        {prompt.data ? (
          (() => {
            const budget = prompt.data;
            const total = Math.max(1, budget.totalBytes);
            return budget.components.map((c) => (
              <div key={c.key} className="flex flex-col gap-[7px]">
                <div className="flex items-center gap-2">
                  <span className="text-[12.5px] font-medium text-body-ink">
                    {c.key === "toolSchemas"
                      ? `${t(`component.${c.key}`)} (${t("tools", { count: budget.toolCount })})`
                      : t(`component.${c.key}`)}
                  </span>
                  <span className="ml-auto font-mono text-[11.5px] text-muted">{fmtKb(c.bytes, locale)}</span>
                </div>
                <div className="h-[7px] overflow-hidden rounded-full bg-surface-inset">
                  <div className="h-full rounded-full bg-accent" style={{ width: `${(c.bytes / total) * 100}%` }} />
                </div>
              </div>
            ));
          })()
        ) : null}
      </div>

      {/* Skills activity */}
      <div className="flex flex-col gap-3 rounded-lg border border-line bg-surface px-5 py-[18px]">
        <div className="flex items-baseline gap-2">
          <span className="text-sm font-bold text-ink">{t("skillsTitle")}</span>
          <span className="text-xs text-muted">— {t("skillsHint")}</span>
        </div>
        {skills.error ? (
          <SourceWarning
            label={t("skillsUnavailable")}
            aside={skills.updatedAt ? <RelativeTime epochMs={skills.updatedAt} /> : undefined}
          />
        ) : null}
        {skills.data ? (
          <>
            {skills.data.slice(0, SKILL_CAP).map((s) => (
              <div
                key={s.name}
                className={`flex items-center gap-3 rounded-md border border-line px-3 py-[9px] ${s.flagged ? "bg-warning-soft" : "bg-surface-2"}`}
              >
                <span className="font-mono text-[12.5px] text-body-ink">{s.name}</span>
                {s.flagged ? (
                  <span className="rounded-full bg-warning-soft px-2 py-0.5 text-[10px] font-bold uppercase tracking-[.05em] text-warning">
                    {t("lowUse")}
                  </span>
                ) : null}
                <span className="ml-auto text-[11.5px] text-muted">{t("uses", { count: s.useCount })}</span>
                {s.lastUsedAt ? (
                  <span className="text-[11px] text-muted">
                    <RelativeTime epochMs={Date.parse(s.lastUsedAt)} />
                  </span>
                ) : null}
              </div>
            ))}
            {skills.data.length > SKILL_CAP ? (
              <span className="text-xs text-muted">{t("more", { count: skills.data.length - SKILL_CAP })}</span>
            ) : null}
          </>
        ) : null}
      </div>
    </div>
    </>
  );
}
