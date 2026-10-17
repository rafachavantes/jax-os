"use client";

import { useTranslations } from "next-intl";
import { useRef, useState } from "react";
import { nextToolsTab, resolveToolsTab, visibleToolsTabs, type ToolsTab } from "@/lib/agent-settings";
import { agentsOf, useGeneralSettings } from "@/lib/settingsQuery";
import { GeneralSettingsSection } from "../../components/tools/GeneralSettingsSection";
import { AgentSettingsSection } from "@/components/tools/AgentSettingsSection";
import { MissionSettingsSection } from "@/components/tools/MissionSettingsSection";
import { OpenCodeProvidersSection } from "@/components/tools/OpenCodeProvidersSection";
import { RulesSection } from "@/components/rules/RulesSection";
import { InventorySection } from "@/components/tools/InventorySection";
import css from "@/components/tools/tools.module.css";

export default function SettingsPage() {
  const t = useTranslations("tools");
  const tTitle = useTranslations("title");
  const [selected, setTab] = useState<ToolsTab>("general");
  // MOA-504 D10: fail open while the settings are unknown; a selected tab that disappears falls back to General.
  const tabs = visibleToolsTabs(agentsOf(useGeneralSettings().data));
  const tab = resolveToolsTab(selected, tabs);
  if (tab !== selected) setTab(tab); // reset the stored selection too, so a returning tab does not snap back
  const [providerConnection, setProviderConnection] = useState<string | null>(null);
  const tabRefs = useRef<Partial<Record<ToolsTab, HTMLButtonElement | null>>>({});

  return (
    <div className="mx-auto flex w-full max-w-[1320px] flex-col gap-[22px] [animation:jax-rise_.4s_ease]">
      <h1 className="sr-only">{tTitle("tools")}</h1>
      <div className={css.pageHeading}>
        <div>
          <h1 className="text-[27px] tracking-[-0.7px] text-ink">{t("heading")}</h1>
          <p className="mt-[5px] text-[13px] text-muted">{t("headingSubtitle")}</p>
        </div>
      </div>
      <div role="tablist" aria-label={t("tabsLabel")} className={css.tabs}>
        {tabs.map((id) => {
          const selected = tab === id;
          return (
            <button
              key={id}
              ref={(el) => {
                tabRefs.current[id] = el;
              }}
              type="button"
              role="tab"
              id={`tools-tab-${id}`}
              aria-controls={`tools-panel-${id}`}
              aria-selected={selected}
              tabIndex={selected ? 0 : -1}
              onClick={() => setTab(id)}
              onKeyDown={(e) => {
                const next = nextToolsTab(tab, e.key, tabs);
                if (next === tab) return;
                e.preventDefault();
                setTab(next);
                tabRefs.current[next]?.focus();
              }}
              className={css.tab}
            >
              <svg
                viewBox="0 0 24 24"
                aria-hidden
                fill="none"
                stroke="currentColor"
                strokeWidth={1.6}
                strokeLinecap="round"
                strokeLinejoin="round"
                className="h-[18px] w-[18px]"
              >
                <use href={`#tool-icon-${id}`} />
              </svg>
              {t(`tabs.${id}`)}
            </button>
          );
        })}
      </div>
      <ToolTabIcons />

      <div
        role="tabpanel"
        id="tools-panel-general"
        aria-labelledby="tools-tab-general"
        hidden={tab !== "general"}
      >
        <GeneralSettingsSection />
      </div>
      <div
        role="tabpanel"
        id="tools-panel-agents"
        aria-labelledby="tools-tab-agents"
        hidden={tab !== "agents"}
      >
        <AgentSettingsSection
          active={tab === "agents"}
          onManageConnection={(id) => {
            setProviderConnection(id);
            setTab("providers");
          }}
        />
        <MissionSettingsSection />
      </div>
      {tabs.includes("providers") ? (
        <div
          role="tabpanel"
          id="tools-panel-providers"
          aria-labelledby="tools-tab-providers"
          hidden={tab !== "providers"}
        >
          <OpenCodeProvidersSection
            initialConnectionId={providerConnection}
            onOpenAgents={() => setTab("agents")}
          />
        </div>
      ) : null}
      <div
        role="tabpanel"
        id="tools-panel-rules"
        aria-labelledby="tools-tab-rules"
        hidden={tab !== "rules"}
      >
        <RulesSection />
      </div>
      <div
        role="tabpanel"
        id="tools-panel-inventory"
        aria-labelledby="tools-tab-inventory"
        hidden={tab !== "inventory"}
        className="flex flex-col gap-[22px]"
      >
        <InventorySection />
      </div>
    </div>
  );
}

function ToolTabIcons() {
  return (
    <svg aria-hidden className="hidden">
      <symbol id="tool-icon-general" viewBox="0 0 24 24">
        <path d="M4 7h10M18 7h2M4 17h2M10 17h10M14 4v6M8 14v6" />
      </symbol>
      <symbol id="tool-icon-agents" viewBox="0 0 24 24">
        <circle cx="9" cy="8" r="3" />
        <path d="M3 20a6 6 0 0 1 12 0M16 5a3 3 0 0 1 0 6M21 20a6 6 0 0 0-5-5.9" />
      </symbol>
      <symbol id="tool-icon-providers" viewBox="0 0 24 24">
        <rect x="3" y="3" width="7" height="7" rx="1" />
        <rect x="14" y="3" width="7" height="7" rx="1" />
        <rect x="3" y="14" width="7" height="7" rx="1" />
        <rect x="14" y="14" width="7" height="7" rx="1" />
      </symbol>
      <symbol id="tool-icon-rules" viewBox="0 0 24 24">
        <path d="M6 3h12v18H6zM9 7h6M9 11h6M9 15h3" />
      </symbol>
      <symbol id="tool-icon-inventory" viewBox="0 0 24 24">
        <path d="M12 3 3 8v8l9 5 9-5V8zM3 8l9 5 9-5M12 13v8" />
      </symbol>
    </svg>
  );
}
