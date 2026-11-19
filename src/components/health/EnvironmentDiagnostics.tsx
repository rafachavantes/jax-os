"use client";

// M5 Task 4: four independent Environment polls. A slow package or workload probe can never
// delay components/resources; every section renders its own warning while siblings stay usable.
import { useQuery } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import { fetchEnvelope } from "@/lib/api";
import { RelativeTime } from "@/components/RelativeTime";
import { SourceWarning } from "@/components/mission/SourceWarning";
import type {
  ComponentState, ComponentsSection, PackagesSection, ResourcesSection,
} from "@/server/collectors/environment";
import type { WorkloadsSection } from "@/server/collectors/workloads";

const POLL_MS = { components: 60_000, resources: 10_000, workloads: 15_000, packages: 60_000 } as const;

type Sections = {
  components: ComponentsSection;
  resources: ResourcesSection;
  workloads: WorkloadsSection;
  packages: PackagesSection;
};

function useEnvironmentSection<K extends keyof Sections>(section: K) {
  return useQuery({
    queryKey: ["environment", section],
    queryFn: ({ signal }) => fetchEnvelope<Sections[K]>(`/api/environment?section=${section}`, signal),
    refetchInterval: POLL_MS[section],
  });
}

const STATE_CLASS: Record<ComponentState, string> = {
  healthy: "border-success bg-success-soft text-success",
  "needs-configuration": "border-warning bg-warning-soft text-warning",
  failed: "border-danger bg-danger-soft text-danger",
  "disabled-default": "border-line bg-surface-2 text-muted",
  "disabled-explicit": "border-line bg-surface-2 text-muted",
  unavailable: "border-warning bg-warning-soft text-warning",
  unknown: "border-line bg-surface-2 text-muted",
  "not-applicable": "border-line bg-surface-2 text-muted",
};

// Decimal GB is reserved for filesystem capacity; process RSS is binary MiB/GiB.
const GB = (value: number): string => `${value.toFixed(1)} GB`;
const MiB = (value: number): string => (value >= 1024 ? `${(value / 1024).toFixed(1)} GiB` : `${value.toFixed(0)} MiB`);

function Skeleton() {
  return <span className="block h-6 w-64 animate-pulse rounded bg-surface-2" />;
}

function Panel({ id, title, children }: { id?: string; title: string; children: React.ReactNode }) {
  return (
    <section id={id} className="scroll-mt-24 rounded-lg border border-line bg-surface px-5 py-[18px]">
      <h2 className="mb-3 text-sm font-bold text-ink">{title}</h2>
      {children}
    </section>
  );
}

function Components() {
  const t = useTranslations("health.environment");
  const q = useEnvironmentSection("components");
  const data = q.data?.ok ? q.data.data : null;
  return (
    <Panel id="components" title={t("sections.components")}>
      {q.isError ? <SourceWarning label={t("unavailable")} /> : !data || !Array.isArray(data.components) ? (
        <Skeleton />
      ) : (
        <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
          {data.components.map((component) => (
            <div key={component.id} data-testid={`component-${component.id}`} className="flex min-w-0 flex-col gap-1 rounded-md border border-line-subtle px-3 py-2">
              <div className="flex min-w-0 flex-wrap items-center justify-between gap-2">
                <span className="min-w-0 [overflow-wrap:anywhere] text-[12.5px] font-semibold text-body-ink">
                  {t(`componentsList.${component.id}`)}
                </span>
                <span className={`flex-none rounded-full border px-2 py-0.5 text-[10.5px] font-semibold ${STATE_CLASS[component.state]}`}>
                  {t(`states.${component.state}`)}
                </span>
              </div>
              {component.detail ? (
                <span className="text-[11px] text-muted">{t(`details.${component.detail}`)}</span>
              ) : null}
              <span className="text-[11px] text-muted">
                {t("observed")} <RelativeTime epochMs={Date.parse(component.observedAt)} />
                {component.lastGoodAt ? <> · {t("lastGood")} <RelativeTime epochMs={Date.parse(component.lastGoodAt)} /></> : null}
              </span>
              <span className="text-[11px] text-muted">{t(`nextStep.${component.state}`)}</span>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}

function Resources() {
  const t = useTranslations("health.environment");
  const q = useEnvironmentSection("resources");
  const data = q.data?.ok ? q.data.data : null;
  const tmp = data?.tmp ?? null;
  return (
    <Panel id="tmp" title={t("sections.resources")}>
      {q.isError ? <SourceWarning label={t("unavailable")} /> : !data ? <Skeleton /> : (
        <div className="flex flex-col gap-2">
          {data.state !== "ok" || !tmp ? (
            <SourceWarning
              label={t("resources.unavailable")}
              detail={data.warning ? t(`resources.warnings.${data.warning}`) : undefined}
            />
          ) : null}
          {data.stale ? (
            <span className="text-[11px] text-warning">
              {t("resources.stale")}
              {data.lastGoodAt ? <> · <RelativeTime epochMs={Date.parse(data.lastGoodAt)} /></> : null}
            </span>
          ) : null}
          {tmp ? (
            <div data-testid="tmp-capacity" className="flex flex-col gap-1 rounded-md border border-line-subtle px-3 py-2">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <span className="font-mono text-[12px] text-body-ink">/tmp · {tmp.fsType}</span>
                <span className="font-display text-[18px] font-black text-ink">
                  {GB(tmp.usedGb)} / {GB(tmp.totalGb)}
                </span>
              </div>
              <span className="text-[11px] text-muted">{t("resources.available", { value: GB(tmp.availableGb) })}</span>
              <span className="text-[11px] text-muted">
                {tmp.tmpfs ? t("resources.tmpfs") : tmp.sharedRoot ? t("resources.sharedRoot") : t("resources.distinct")}
              </span>
              <span className="text-[11px] text-muted">
                {t("observed")} <RelativeTime epochMs={Date.parse(data.stale && data.lastGoodAt ? data.lastGoodAt : data.observedAt)} />
              </span>
            </div>
          ) : null}
        </div>
      )}
    </Panel>
  );
}

function Workloads() {
  const t = useTranslations("health.environment");
  const q = useEnvironmentSection("workloads");
  const data = q.data?.ok ? q.data.data : null;
  const unavailable = data?.state === "unavailable";
  return (
    <Panel id="workloads" title={t("sections.workloads")}>
      {q.isError ? <SourceWarning label={t("unavailable")} /> : !data || !Array.isArray(data.groups) ? <Skeleton /> : (
        <div className="flex flex-col gap-2">
          {data.warning === "tmux-unavailable" ? <SourceWarning label={t("workloads.tmuxUnavailable")} /> : null}
          {data.warning === "tmux-unverified" ? <SourceWarning label={t("workloads.tmuxUnverified")} /> : null}
          {unavailable ? <SourceWarning label={t("workloads.unavailable")} /> : null}
          {!unavailable && (data.state === "partial" || data.truncated) ? (
            <SourceWarning label={t("partial")} detail={t("workloads.coverage", { known: data.coverage.ownershipKnown, unknown: data.coverage.ownershipUnknown })} />
          ) : null}
          {!unavailable ? (
            <span className="text-[11px] text-muted">
              {t("workloads.summary", { count: data.totalProcesses })} · {t("workloads.approxRss")}
            </span>
          ) : null}
          {!unavailable && data.groups.length === 0 ? (
            <span className="text-[12.5px] text-muted">{t("workloads.empty")}</span>
          ) : null}
          {data.groups.length > 0 ? (
            data.groups.map((group) => (
              <details key={group.key} data-testid={`workload-${group.family}`} className="rounded-md border border-line-subtle px-3 py-2">
                <summary className="flex cursor-pointer list-none flex-wrap items-center gap-x-3 gap-y-1 text-[12px] text-body-ink">
                  <span className="min-w-0 font-mono font-semibold [overflow-wrap:anywhere]">{group.family}</span>
                  <span>{t("workloads.processes", { count: group.count })}</span>
                  <span className="text-muted">
                    {group.rssMb !== null ? t("workloads.rss", { value: MiB(group.rssMb) }) : t("shortUnavailable")}
                  </span>
                  <span className={group.owner.kind === "pane" ? "text-success" : "text-muted"}>
                    {group.owner.kind === "pane"
                      ? t("workloads.paneOwner", { pane: group.owner.pane })
                      : `${t("workloads.unknownOwner")} · ${t(`workloads.ownerReasons.${group.owner.reason}`)}`}
                  </span>
                  {group.purpose !== null ? <span className="text-muted">{t("workloads.purposePane")}</span> : null}
                  {group.maxAgeSeconds !== null ? <span className="text-muted">{t("workloads.maxAge", { value: group.maxAgeSeconds })}</span> : null}
                  {group.project ? <span className="text-muted">{t("workloads.lastKnown", { project: group.project })}</span> : null}
                  {group.contextObservedAt ? (
                    <span className="text-muted">{t("workloads.lastKnownAt")} <RelativeTime epochMs={Date.parse(group.contextObservedAt)} /></span>
                  ) : null}
                  {group.zombies > 0 ? <span className="text-warning">{t("workloads.zombies", { count: group.zombies })}</span> : null}
                </summary>
                <div className="mt-2 flex flex-col gap-1 border-t border-line-subtle pt-2">
                  {group.purpose === null ? <span className="text-[11px] text-muted">{t("workloads.purposeUnknown")}</span> : null}
                  {group.pids.map((proc) => (
                    <div key={proc.pid} className="flex flex-wrap items-center gap-2 font-mono text-[11px] text-muted">
                      <span className="text-body-ink">{proc.pid}</span>
                      <span>{proc.state}</span>
                      <span>{t("workloads.age", { value: proc.ageSeconds })}</span>
                      <span>{proc.rssMb !== null ? MiB(proc.rssMb) : t("shortUnavailable")}</span>
                      {proc.parentCommand !== null ? (
                        <span>{t("workloads.parent", { command: proc.parentCommand, pid: proc.ppid })}</span>
                      ) : null}
                    </div>
                  ))}
                  {group.detailsTruncated ? (
                    <span className="text-[11px] text-muted">{t("workloads.remainder", { count: group.count - group.pids.length })}</span>
                  ) : null}
                </div>
              </details>
            ))
          ) : null}
          {!unavailable ? <span className="text-[10.5px] italic text-muted">{t("workloads.zombieNote")}</span> : null}
          <span className="text-[11px] text-muted">
            {t("observed")} <RelativeTime epochMs={Date.parse(data.observedAt)} />
          </span>
        </div>
      )}
    </Panel>
  );
}

function Packages() {
  const t = useTranslations("health.environment");
  const q = useEnvironmentSection("packages");
  const data = q.data?.ok ? q.data.data : null;
  return (
    <Panel id="packages" title={t("sections.packages")}>
      {q.isError ? <SourceWarning label={t("unavailable")} /> : !data || !Array.isArray(data.candidates) ? <Skeleton /> : (
        <div className="flex flex-col gap-2">
          {data.warnings.map((warning) => (
            <SourceWarning
              key={warning}
              label={t(`packages.warnings.${warning}`)}
              detail={data.lastGoodAt ? t("lastGood") : undefined}
            />
          ))}
          {data.state !== "unavailable" ? (
            <span className="text-[11px] text-muted">
              {data.counts === null
                ? t("packages.totalUnknown")
                : data.counts.selectedNow === null || data.counts.held === null || data.counts.deferred === null
                  ? t("packages.countsUnknown", { total: data.counts.total })
                  : t("packages.counts", {
                    total: data.counts.total, ready: data.counts.selectedNow,
                    held: data.counts.held, deferred: data.counts.deferred,
                  })}
              {data.counts !== null && data.counts.remainder > 0
                ? ` · ${t("packages.remainder", { count: data.counts.remainder })}`
                : ""}
            </span>
          ) : null}
          {data.state !== "unavailable" && data.candidates.length === 0 ? (
            <span className="text-[12.5px] text-muted">{t("packages.empty")}</span>
          ) : null}
          {data.candidates.length > 0 ? (
            <div className="flex flex-col gap-1">
              {data.candidates.map((candidate) => (
                <div key={candidate.key} data-testid={`package-${candidate.key}`} className="flex min-w-0 flex-wrap items-center gap-2 text-[12px] text-body-ink">
                  <span className="min-w-0 flex-1 [overflow-wrap:anywhere] font-mono">
                    {candidate.name} · {candidate.arch} · {candidate.fromVersion} → {candidate.toVersion}
                  </span>
                  {candidate.security ? (
                    <span className="rounded-full bg-danger-soft px-1.5 text-[10px] font-bold text-danger">{t("packages.security")}</span>
                  ) : null}
                  <span className="rounded-full border border-line px-1.5 text-[10px] font-semibold text-muted">
                    {t(`packages.reasons.${candidate.reason}`)}
                  </span>
                </div>
              ))}
              {data.truncated ? <span className="text-[11px] text-muted">{t("packages.truncated")}</span> : null}
            </div>
          ) : null}
          <span className="text-[11px] text-muted">
            {t("observed")} <RelativeTime epochMs={Date.parse(data.observedAt)} />
            {data.lastGoodAt ? <> · {t("lastGood")} <RelativeTime epochMs={Date.parse(data.lastGoodAt)} /></> : null}
          </span>
          <span className="text-[10.5px] italic text-muted">{t("packages.manualStep")}</span>
        </div>
      )}
    </Panel>
  );
}

export function EnvironmentDiagnostics() {
  return (
    <div className="flex flex-col gap-4">
      <Components />
      <Resources />
      <Workloads />
      <Packages />
    </div>
  );
}
