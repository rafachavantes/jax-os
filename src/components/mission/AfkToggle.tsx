"use client";

import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import { TriangleAlert } from "lucide-react";
import type { Envelope } from "@/lib/api";
import { Switch } from "@/components/Switch";

type AfkState = { enabled: boolean };

// The one write on the whole dashboard (spec §2.4). POST /api/workflow/afk already audits its
// previous value as `workflow-afk-toggle` (setAfk, src/server/db/workflows.ts) — nothing here
// duplicates that.
export function AfkToggle() {
  const t = useTranslations("mission.afk");
  const queryClient = useQueryClient();
  const [pending, setPending] = useState(false);
  const [toggleFailed, setToggleFailed] = useState(false);

  const query = useQuery<Envelope<AfkState>>({
    queryKey: ["workflow", "afk"],
    queryFn: async () => {
      const res = await fetch("/api/workflow/afk");
      // Cold review round 3, Finding 9: a non-2xx response used to fall straight into `.json()`
      // as if an error page were the envelope. Throwing here is what makes `query.isError` a
      // reliable signal below, instead of leaving TanStack Query holding a garbage "success".
      if (!res.ok) throw new Error(`GET /api/workflow/afk: ${res.status}`);
      return res.json();
    },
    refetchInterval: 10_000,
  });
  // `undefined` (GET still in flight), `query.isError` (GET failed at the HTTP/network level),
  // and `{ok:false}` (GET succeeded but the route itself reported failure) used to collapse into
  // the same `enabled === null`, so a failed source read looked identical to "loading" (hard rule
  // 3, cold review Finding 7, sharpened by round 3 Finding 9) — `isError` must win here too, not
  // just the `{ok:false}` envelope case.
  const sourceFailed = query.isError || (query.data !== undefined && !query.data.ok);
  const enabled = query.data?.ok ? query.data.data.enabled : null;

  async function toggle() {
    if (enabled === null || pending) return;
    setPending(true);
    setToggleFailed(false);
    const res: Envelope<AfkState> = await fetch("/api/workflow/afk", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: !enabled }),
    })
      .then(async (r) => {
        // Finding 9 (round 3): the POST path had the identical bug as the GET above — a non-2xx
        // response was parsed as if it were the envelope instead of being treated as a failure.
        if (!r.ok) throw new Error(`POST /api/workflow/afk: ${r.status}`);
        return r.json();
      })
      .catch((e) => ({ ok: false as const, error: String(e) }));
    if (res.ok) {
      queryClient.setQueryData(["workflow", "afk"], res);
    } else {
      // Finding 7: this branch used to be missing entirely — a POST failure just left the
      // switch showing its old value with no sign anything went wrong.
      setToggleFailed(true);
    }
    setPending(false);
  }

  const failed = sourceFailed || toggleFailed;
  const statusLabel = sourceFailed ? t("unavailable") : toggleFailed ? t("toggleFailed") : enabled === null ? t("loading") : null; // on/off: no caption, the switch says it (tooltip keeps t("on")/t("off"))

  return (
    <div className="flex items-center gap-2">
      <span className="text-[11px] font-semibold uppercase tracking-[.08em] text-muted">{t("label")}</span>
      <Switch
        checked={enabled === true}
        disabled={enabled === null || pending || sourceFailed}
        onChange={toggle}
        title={sourceFailed ? t("unavailable") : enabled === true ? t("on") : t("off")}
        aria-label={t("label")}
      />
      {failed ? <TriangleAlert className="h-3.5 w-3.5 flex-none text-warning" aria-hidden="true" /> : null}
      {statusLabel === null ? null : <span
        role="status"
        aria-live="polite"
        className={`text-[11px] font-bold ${failed ? "text-warning" : enabled ? "text-brand" : "text-muted"}`}
      >
        {statusLabel}
      </span>}
    </div>
  );
}
