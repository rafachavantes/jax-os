"use client";

import { useTranslations } from "next-intl";
import { useRef, useState } from "react";
import { compactRouting, QUANTIZATIONS, type Percentile, type PercentileKey, type Routing } from "@/lib/agent-settings";
import { canCompare, singlePercentileKey } from "@/lib/openrouter-comparison";
import { EndpointComparisonDialog, type EndpointComparisonHandle } from "./EndpointComparisonDialog";
import { useToolsDraft } from "./ToolsDraftProvider";
import css from "./tools.module.css";

const KEYS: PercentileKey[] = ["p50", "p75", "p90", "p99"];

export function parseList(text: string): string[] {
  return text.split(",").map((s) => s.trim()).filter(Boolean);
}

export function toggleQuantization(current: string[] | undefined, value: string, checked: boolean): string[] {
  const set = new Set(current ?? []);
  if (checked) set.add(value);
  else set.delete(value);
  return QUANTIZATIONS.filter((q) => set.has(q));
}

export type StatisticView = {
  displayed: PercentileKey | "";
  mixed: boolean;
  commonKey: PercentileKey | null;
  throughputKey: PercentileKey | null;
  latencyKey: PercentileKey | null;
};

export function routingStatistic(routing: Routing | null, selected: PercentileKey): StatisticView {
  const throughputGoal = routing?.preferred_min_throughput;
  const latencyGoal = routing?.preferred_max_latency;
  const throughputKey = singlePercentileKey(throughputGoal);
  const latencyKey = singlePercentileKey(latencyGoal);
  const mixed = (!!throughputGoal && throughputKey === null) || (!!latencyGoal && latencyKey === null);
  const commonKey = throughputKey !== null && throughputKey === latencyKey ? throughputKey : null;
  const noGoals = !throughputGoal && !latencyGoal;
  const displayed = noGoals
    ? selected
    : mixed
      ? ""
      : commonKey ?? (throughputKey !== null && latencyKey !== null ? "" : throughputKey ?? latencyKey ?? selected);
  return { displayed, mixed, commonKey, throughputKey, latencyKey };
}

export function migrateStatistic(routing: Routing | null, key: PercentileKey): Partial<Routing> | null {
  const { throughputKey, latencyKey } = routingStatistic(routing, key);
  const next: Partial<Routing> = {};
  const throughputGoal = routing?.preferred_min_throughput;
  const latencyGoal = routing?.preferred_max_latency;
  if (throughputKey !== null && throughputGoal) next.preferred_min_throughput = { [key]: throughputGoal[throughputKey] };
  if (latencyKey !== null && latencyGoal) next.preferred_max_latency = { [key]: latencyGoal[latencyKey] };
  return Object.keys(next).length ? next : null;
}

export function OpenRouterRouting({
  connection,
  model,
  routing,
  onChange,
}: {
  connection: string;
  model: string;
  routing: Routing | null;
  onChange: (routing: Routing | null) => void;
}) {
  const t = useTranslations("tools");
  const { connections } = useToolsDraft();
  const dialogRef = useRef<EndpointComparisonHandle>(null);
  const current = routing ?? { sort: "price" as const, allow_fallbacks: true };
  // The selected statistic is local UI intent: it survives while the serialized
  // routing stays clean and is reset by the caller's connection/model/epoch key.
  const [selectedPercentile, setSelectedPercentile] = useState<PercentileKey>(
    () => routingStatistic(routing, "p90").displayed || "p90",
  );
  // Keep the raw text so a delimiter under the cursor is never rewritten.
  const [onlyText, setOnlyText] = useState(() => (current.only ?? []).join(", "));
  const [ignoreText, setIgnoreText] = useState(() => (current.ignore ?? []).join(", "));
  const compareEnabled = canCompare(connections.find((row) => row.id === connection), model);
  const view = routingStatistic(current, selectedPercentile);

  function patch(next: Partial<Routing>) {
    onChange(compactRouting({ ...current, ...next }));
  }

  function chooseStatistic(key: PercentileKey) {
    setSelectedPercentile(key);
    // A bare statistic choice never invents a numeric goal or dirties routing.
    const next = migrateStatistic(current, key);
    if (next) patch(next);
  }

  function setGoal(kind: "throughput" | "latency", raw: string) {
    const key = (kind === "throughput" ? view.throughputKey : view.latencyKey) ?? view.commonKey ?? selectedPercentile;
    if (raw === "") {
      patch(kind === "throughput" ? { preferred_min_throughput: undefined } : { preferred_max_latency: undefined });
      return;
    }
    const value = Number(raw);
    if (!Number.isFinite(value) || value <= 0) return;
    const next = { [key]: value } as Percentile;
    patch(kind === "throughput" ? { preferred_min_throughput: next } : { preferred_max_latency: next });
  }

  return (
    <div className={css.routing}>
      <p className={css.routingTitle}>{t("routingTitle")}</p>
      <div className={css.routingStat}>
        <label className={css.field}>
          {t("routing.statistic")}
          <select value={view.displayed} onChange={(e) => chooseStatistic(e.target.value as PercentileKey)}>
            <option value="" disabled>{view.mixed ? t("routing.mixed") : t("routing.statisticNone")}</option>
            {KEYS.map((key) => (
              <option key={key} value={key}>{key}</option>
            ))}
          </select>
        </label>
        <span className={css.tag}>{t("routing.sort")}</span>
      </div>
      {view.mixed ? (
        <p className={css.hint}>
          {t("routing.mixedNote", {
            throughput: JSON.stringify(current.preferred_min_throughput ?? {}),
            latency: JSON.stringify(current.preferred_max_latency ?? {}),
          })}
        </p>
      ) : null}
      <div className={css.metricsInputs}>
        <label className={css.field}>
          {t("routing.throughput")}
          <input
            type="number"
            min={0}
            step="any"
            value={view.throughputKey && current.preferred_min_throughput ? String(current.preferred_min_throughput[view.throughputKey] ?? "") : ""}
            placeholder={t("routing.noPreference")}
            onChange={(e) => setGoal("throughput", e.target.value)}
          />
          <span className={css.hint}>{t("routing.throughputUnit")}</span>
        </label>
        <label className={css.field}>
          {t("routing.latency")}
          <input
            type="number"
            min={0}
            step="any"
            value={view.latencyKey && current.preferred_max_latency ? String(current.preferred_max_latency[view.latencyKey] ?? "") : ""}
            placeholder={t("routing.noPreference")}
            onChange={(e) => setGoal("latency", e.target.value)}
          />
          <span className={css.hint}>{t("routing.latencyUnit")}</span>
        </label>
      </div>
      <p className={css.hint}>{t("routing.softGoal")}</p>
      <details className={css.routingAdvanced}>
        <summary>{t("routing.advanced")}</summary>
        <div className={css.metricsInputs}>
          <label className={css.field}>
            {t("routing.maxPrompt")}
            <input
              type="number"
              min={0}
              step="any"
              value={current.max_price?.prompt ?? ""}
              onChange={(e) => patch({ max_price: { ...current.max_price, prompt: e.target.value === "" ? undefined : Number(e.target.value) } })}
            />
            <span className={css.hint}>{t("routing.priceUnit")}</span>
          </label>
          <label className={css.field}>
            {t("routing.maxCompletion")}
            <input
              type="number"
              min={0}
              step="any"
              value={current.max_price?.completion ?? ""}
              onChange={(e) => patch({ max_price: { ...current.max_price, completion: e.target.value === "" ? undefined : Number(e.target.value) } })}
            />
            <span className={css.hint}>{t("routing.priceUnit")}</span>
          </label>
        </div>
        <label className={css.field}>
          {t("routing.only")}
          <input
            value={onlyText}
            placeholder={t("routing.allEndpoints")}
            onChange={(e) => {
              setOnlyText(e.target.value);
              patch({ only: parseList(e.target.value) });
            }}
          />
        </label>
        <label className={css.field}>
          {t("routing.ignore")}
          <input
            value={ignoreText}
            placeholder={t("routing.noneExcluded")}
            onChange={(e) => {
              setIgnoreText(e.target.value);
              patch({ ignore: parseList(e.target.value) });
            }}
          />
        </label>
        <div className={css.field}>{t("routing.quantizations")}</div>
        <div className={css.checkGrid} role="group" aria-label={t("routing.quantizations")}>
          {QUANTIZATIONS.map((q) => (
            <label key={q} className={css.checkGridItem}>
              <input
                type="checkbox"
                value={q}
                checked={(current.quantizations ?? []).includes(q)}
                onChange={(e) => patch({ quantizations: toggleQuantization(current.quantizations, q, e.target.checked) })}
              />
              {q}
            </label>
          ))}
        </div>
        <p className={css.hint}>{t("routing.quantizationsHint")}</p>
        <p className={css.hint}>{t("routing.restrictionNote")}</p>
      </details>
      <label className={css.checkLine}>
        <input
          type="checkbox"
          checked={current.allow_fallbacks}
          onChange={(e) => patch({ allow_fallbacks: e.target.checked })}
        />
        {t("routing.endpointFallback")}
      </label>
      <button
        type="button"
        className={css.compareButton}
        disabled={!compareEnabled}
        onClick={() => dialogRef.current?.open(false)}
      >
        {t("routing.compare")}
      </button>
      {!compareEnabled ? <p className={css.hint}>{t("routing.compareDisabled")}</p> : null}
      <EndpointComparisonDialog ref={dialogRef} connection={connection} model={model} routing={current} />
    </div>
  );
}
