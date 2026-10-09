"use client";

import { Moon, Sun } from "lucide-react";
import { useTranslations } from "next-intl";
import { useState } from "react";

export type Theme = "dark" | "light";

export function ThemeToggle({ initial }: { initial: Theme }) {
  const [theme, setTheme] = useState<Theme>(initial);
  const t = useTranslations("shell");

  function toggle() {
    const next: Theme = theme === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.dataset.theme = next;
    document.cookie = `theme=${next};path=/;max-age=31536000;SameSite=Lax`;
  }

  return (
    <button
      onClick={toggle}
      aria-label={t("toggleTheme")}
      className="flex h-9 w-9 items-center justify-center rounded-full border border-line bg-surface-2 text-body-ink transition-transform active:scale-95"
    >
      {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </button>
  );
}
