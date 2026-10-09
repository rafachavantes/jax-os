"use client";

import { Timer } from "lucide-react";
import { useTranslations } from "next-intl";
import type { Agent } from "@/server/collectors/tmux";
import { RelativeTime } from "@/components/RelativeTime";

type Props = { agent: Agent; selected: boolean; onSelect: () => void };

export function SessionCard({ agent, selected, onSelect }: Props) {
  const t = useTranslations("sessions");
  return (
    <button
      onClick={onSelect}
      className={`flex w-full flex-col gap-2 rounded-lg border p-3 text-left transition-colors ${
        selected
          ? "border-brand-soft-border bg-brand-soft"
          : "border-line bg-surface hover:bg-surface-2"
      }`}
    >
      <div className="flex w-full items-center gap-2">
        <span
          className={`h-2 w-2 flex-none rounded-full ${
            agent.project ? "bg-accent [animation:jax-pulse_1.6s_infinite]" : "bg-muted"
          }`}
        />
        <span className="truncate font-mono text-[13px] font-semibold text-ink">
          {agent.session}
        </span>
        <span
          className={`ml-auto flex-none rounded-full px-2 py-0.5 text-[10px] font-bold ${
            agent.project ? "bg-accent-soft text-accent" : "bg-surface-3 text-muted"
          }`}
        >
          {agent.project ?? t("unmanaged")}
        </span>
      </div>
      <span className="w-full truncate font-mono text-[10.5px] text-muted">{agent.path}</span>
      <div className="flex w-full items-center gap-3 text-[11px] text-muted">
        <span className="flex items-center gap-1">
          <Timer className="h-3 w-3" />
          <RelativeTime epochMs={agent.createdEpoch * 1000} />
        </span>
        <span>{t("windows", { count: agent.windows })}</span>
        <span
          className={`ml-auto rounded-full px-2 py-0.5 text-[10px] font-semibold ${
            agent.attached ? "bg-success-soft text-success" : "bg-surface-3 text-muted"
          }`}
        >
          {agent.attached ? t("attached") : t("detached")}
        </span>
      </div>
    </button>
  );
}
