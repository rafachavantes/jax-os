"use client";

import { useLocale } from "next-intl";
import { useRouter } from "next/navigation";

const LOCALES = [
  { value: "pt-BR", label: "PT" },
  { value: "en-US", label: "EN" },
] as const;

export function LanguageSelector() {
  const locale = useLocale();
  const router = useRouter();

  function setLocale(value: string) {
    document.cookie = `locale=${value};path=/;max-age=31536000;SameSite=Lax`;
    router.refresh();
  }

  return (
    <div className="flex h-9 items-center rounded-full border border-line bg-surface-2 p-1">
      {LOCALES.map((l) => (
        <button
          key={l.value}
          onClick={() => setLocale(l.value)}
          className={`flex h-7 items-center rounded-full px-2.5 text-xs font-semibold transition-colors ${
            locale === l.value ? "bg-brand-soft text-brand" : "text-muted hover:text-ink"
          }`}
        >
          {l.label}
        </button>
      ))}
    </div>
  );
}
