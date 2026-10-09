import { getRequestConfig } from "next-intl/server";
import { cookies } from "next/headers";
import { readGeneralSettings } from "../server/settings";

export const LOCALES = ["pt-BR", "en-US"] as const;
export type Locale = (typeof LOCALES)[number];

// Decision 12: a valid cookie always wins; otherwise settings.locale if it's one of LOCALES;
// otherwise en-US (497A's own schema default). Exported so it's unit-testable with no
// next/headers mocking — the async wrapper below is thin glue only.
export function resolveLocale(cookieValue: string | undefined, settingsLocale: string | undefined): Locale {
  if (LOCALES.includes(cookieValue as Locale)) return cookieValue as Locale;
  if (LOCALES.includes(settingsLocale as Locale)) return settingsLocale as Locale;
  return "en-US";
}

export default getRequestConfig(async () => {
  const store = await cookies();
  const fromCookie = store.get("locale")?.value;
  const settings = readGeneralSettings();
  const locale = resolveLocale(fromCookie, settings.ok ? settings.data.locale : undefined);
  return {
    locale,
    messages: (await import(`../../messages/${locale}.json`)).default,
  };
});
