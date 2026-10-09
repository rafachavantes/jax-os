"use client";

import { useTranslations } from "next-intl";
import { SourceWarning } from "@/components/mission/SourceWarning";
import { agentsBannerKind } from "@/lib/settingsBanner";
import { useGeneralSettings } from "@/lib/settingsQuery";

// MOA-504 D7: client-side on purpose. Layouts do not re-render on client navigation, and this must
// update the moment the owner saves (the shared ["settings"] query is refreshed by that save).
export function AgentsBanner() {
  const t = useTranslations("settings");
  const kind = agentsBannerKind(useGeneralSettings().data);
  if (!kind) return null;
  return (
    <div className="px-4 pt-3 pb-[22px]">
      <SourceWarning label={t(kind === "none" ? "noAgents" : "noReviewer")} />
    </div>
  );
}
