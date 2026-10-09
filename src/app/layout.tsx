import "./globals.css";
import type { Metadata } from "next";
import { NextIntlClientProvider } from "next-intl";
import { getLocale, getTranslations } from "next-intl/server";
import { cookies } from "next/headers";
import { readGeneralSettings } from "@/server/settings";
import { bannerLabelKey } from "@/lib/settingsBanner";
import { SourceWarning } from "@/components/mission/SourceWarning";
import type { Theme } from "@/components/ThemeToggle";
import type { SidebarMode } from "@/lib/nav";
import { fontVariables } from "./fonts";
import { Providers } from "./Providers";
import { Shell } from "@/components/Shell";
import { AgentsBanner } from "@/components/AgentsBanner";

export const metadata: Metadata = {
  title: "Jax OS",
  description: "Self-hosted work cockpit. You watch; your agents do the work.",
};

const VALID_SIDEBAR: SidebarMode[] = ["expanded", "collapsed", "hidden"];

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const locale = await getLocale();
  const t = await getTranslations();
  const bannerKey = bannerLabelKey(readGeneralSettings());
  const cookieStore = await cookies();
  const theme: Theme = cookieStore.get("theme")?.value === "light" ? "light" : "dark";
  const rawSidebar = cookieStore.get("sidebar")?.value;
  const sidebar: SidebarMode = VALID_SIDEBAR.includes(rawSidebar as SidebarMode)
    ? (rawSidebar as SidebarMode)
    : "expanded";

  return (
    <html lang={locale} data-theme={theme} className={fontVariables}>
      <body>
        <NextIntlClientProvider>
          <Providers>
            <Shell initialSidebar={sidebar} initialTheme={theme}>
              {bannerKey ? (
                <div className="px-4 pt-3 pb-[22px]">
                  <SourceWarning label={t(bannerKey)} />
                </div>
              ) : null}
              <AgentsBanner />
              {children}
            </Shell>
          </Providers>
        </NextIntlClientProvider>
      </body>
    </html>
  );
}
