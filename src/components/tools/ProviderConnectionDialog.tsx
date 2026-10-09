"use client";

import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import type { ConnectionChoice, EditorRevision } from "@/lib/agent-settings";
import { classifyWrite, postJson, writeKindText } from "./ToolsDraftProvider";
import css from "./tools.module.css";

const FIELD =
  "w-full rounded-md border border-line bg-surface-2 px-2 py-1.5 text-[13px] text-ink outline-none focus:border-line-strong disabled:opacity-60";
const BTN =
  "rounded-md bg-brand px-3 py-1.5 text-[12.5px] font-medium text-on-brand hover:bg-brand/90 disabled:opacity-60";
const GHOST =
  "rounded-md border border-line px-3 py-1.5 text-[12.5px] text-body-ink hover:bg-surface-2 disabled:opacity-60";
const ADAPTERS = ["xai", "openrouter", "openai-compatible"] as const;
const CONNECTION_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;

export function ProviderConnectionDialog({
  connection,
  expected,
  invoker,
  onClose,
}: {
  connection?: ConnectionChoice | null;
  expected: EditorRevision | null;
  invoker?: HTMLButtonElement;
  onClose: () => void;
}) {
  const t = useTranslations("tools");
  const qc = useQueryClient();
  const dialogRef = useRef<HTMLDialogElement>(null);
  const editing = connection != null;
  const locked = editing && connection.editable === false;
  const [id, setId] = useState(connection?.id ?? "");
  const [adapter, setAdapter] = useState<(typeof ADAPTERS)[number]>(
    (connection?.adapter as (typeof ADAPTERS)[number]) ?? "xai",
  );
  const [label, setLabel] = useState(connection?.label ?? "");
  const [baseUrl, setBaseUrl] = useState(connection?.base_url ?? "");
  const [pending, setPending] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const gateRef = useRef(false);

  useEffect(() => {
    const dlg = dialogRef.current;
    if (dlg && typeof dlg.showModal === "function") dlg.showModal();
    return () => {
      if (invoker?.isConnected) invoker.focus();
    };
  }, [invoker]);

  const busy = pending;

  function close() {
    if (busy) return;
    onClose();
  }

  async function submit() {
    if (gateRef.current || busy) return;
    if (!expected) return;
    if (!CONNECTION_RE.test(id)) {
      setStatus(t("invalidConnectionId"));
      return;
    }
    if (adapter === "openai-compatible" && (!baseUrl.startsWith("https://") || /[?#]/.test(baseUrl))) {
      setStatus(t("invalidBaseUrl"));
      return;
    }
    gateRef.current = true;
    setPending(true);
    try {
      const payload: { id: string; adapter: string; label?: string; base_url?: string } = { id, adapter };
      if (label) payload.label = label;
      if (adapter === "openai-compatible" && baseUrl) payload.base_url = baseUrl;
      const res = await postJson("/api/opencode-providers", {
        action: "save-connection",
        expected,
        connection: payload,
      });
      const classified = classifyWrite(res);
      if (classified.kind !== "ok") {
        setStatus(writeKindText(t, classified.kind, classified.error));
        return;
      }
      void qc.invalidateQueries({ queryKey: ["opencode-providers"] });
      void qc.invalidateQueries({ queryKey: ["agent-settings"] });
      onClose();
    } finally {
      gateRef.current = false;
      setPending(false);
    }
  }

  return (
    <dialog
      ref={dialogRef}
      aria-label={editing ? t("editConnection") : t("addConnection")}
      onClose={close}
      onCancel={(e) => {
        e.preventDefault();
        close();
      }}
      className={`${css.providerDialog} rounded-lg border border-line bg-surface`}
    >
      <form
        className={css.providerForm}
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
      >
        <div className={css.dialogHeader}>
          <h2>{editing ? t("editConnection") : t("addConnection")}</h2>
          <button type="button" className={css.dialogClose} disabled={busy} onClick={close}>
            {t("dialogClose")}
          </button>
        </div>
        <label className="flex flex-col gap-1 text-[12px] text-muted">
          {t("connectionId")}
          <input
            autoFocus={!editing}
            readOnly={editing}
            value={id}
            onChange={(e) => setId(e.target.value)}
            disabled={busy}
            className={FIELD}
          />
          <span className="text-[11px] text-muted">{editing ? t("identityReadonly") : t("connectionIdHint")}</span>
        </label>
        <label className="flex flex-col gap-1 text-[12px] text-muted">
          {t("adapter")}
          <select
            value={adapter}
            disabled={locked || busy}
            onChange={(e) => setAdapter(e.target.value as (typeof ADAPTERS)[number])}
            className={FIELD}
          >
            {ADAPTERS.map((a) => (
              <option key={a} value={a}>{t(`adapters.${a}`)}</option>
            ))}
          </select>
          {locked ? <span className="text-[11px] text-warning">{t("adapterReadonly")}</span> : null}
        </label>
        <label className="flex flex-col gap-1 text-[12px] text-muted">
          {t("label")}
          <input value={label} onChange={(e) => setLabel(e.target.value)} disabled={busy} className={FIELD} />
        </label>
        {adapter === "openai-compatible" ? (
          <>
            <p className="rounded-md border border-warning bg-warning-soft px-3 py-2 text-xs text-warning">{t("customEndpointWarn")}</p>
            <label className="flex flex-col gap-1 text-[12px] text-muted">
              {t("baseUrl")}
              <input value={baseUrl} disabled={locked || busy} onChange={(e) => setBaseUrl(e.target.value)} className={FIELD} />
            </label>
            <p className="text-[11px] text-muted">{t("httpsOnly")}</p>
          </>
        ) : null}
        {status ? <p role="status" className="text-xs text-muted">{status}</p> : null}
        <div className={css.dialogFooter}>
          <button type="button" disabled={busy} className={GHOST} onClick={close}>{t("cancel")}</button>
          <button type="submit" disabled={busy || !expected} className={BTN}>{busy ? t("saving") : t("save")}</button>
        </div>
      </form>
    </dialog>
  );
}
