"use client";

import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import { fetchEnvelope } from "@/lib/api";
import { isValidCredentialEnvName, type CredentialRef, type EditorRevision } from "@/lib/agent-settings";
import { classifyWrite, postJson, useToolsDraft, writeKindText, type CredentialDialogRequest } from "./ToolsDraftProvider";
import css from "./tools.module.css";

const FIELD =
  "w-full rounded-md border border-line bg-surface-2 px-2 py-1.5 text-[13px] text-ink outline-none focus:border-line-strong disabled:opacity-60";
const BTN =
  "rounded-md bg-brand px-3 py-1.5 text-[12.5px] font-medium text-on-brand hover:bg-brand/90 disabled:opacity-60";
const GHOST =
  "rounded-md border border-line px-3 py-1.5 text-[12.5px] text-body-ink hover:bg-surface-2 disabled:opacity-60";

type EnvStatus = "configured" | "missing" | "unknown";

export function ToolsCredentialHost() {
  const { credential, revision, closeCredential } = useToolsDraft();
  if (!credential) return null;
  return <ProviderCredentialDialog request={credential} expected={revision} onClose={closeCredential} />;
}

export function ProviderCredentialDialog({
  request,
  expected,
  onClose,
}: {
  request: CredentialDialogRequest;
  expected: EditorRevision | null;
  onClose: () => void;
}) {
  const t = useTranslations("tools");
  const qc = useQueryClient();
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [kind, setKind] = useState<"native" | "env">(request.current?.kind === "env" ? "env" : "native");
  const [env, setEnv] = useState(request.current?.kind === "env" ? request.current.env : "");
  const [pending, setPending] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [envStatus, setEnvStatus] = useState<EnvStatus>("unknown");
  const attempted = useRef(false);

  useEffect(() => {
    const dlg = dialogRef.current;
    if (dlg && typeof dlg.showModal === "function") dlg.showModal();
    const invoker = request.invoker;
    return () => {
      invoker?.focus?.();
    };
  }, [request.invoker]);

  // Live status for whatever env name is currently typed — a valid name always re-checks,
  // never a stale answer for a since-edited name.
  useEffect(() => {
    if (kind !== "env" || !isValidCredentialEnvName(env)) {
      setEnvStatus("unknown");
      return;
    }
    let cancelled = false;
    fetchEnvelope<{ state: "configured" | "missing" }>(`/api/opencode-providers?credential-env=${encodeURIComponent(env)}`)
      .then((res) => {
        if (!cancelled) setEnvStatus(res.data.state);
      })
      .catch(() => {
        if (!cancelled) setEnvStatus("unknown");
      });
    return () => {
      cancelled = true;
    };
  }, [kind, env]);

  function close() {
    if (pending) return;
    onClose();
  }

  function currentBinding(): CredentialRef | null {
    if (kind === "native") return { kind: "native" };
    return isValidCredentialEnvName(env) ? { kind: "env", env } : null;
  }

  async function submit() {
    const target = request.expected ?? expected;
    if (pending || !target || attempted.current) return;
    const credential = currentBinding();
    if (!credential) {
      setStatus(t("invalidEnvName"));
      return;
    }
    attempted.current = true;
    setPending(true);
    try {
      const res = await postJson("/api/opencode-providers", {
        action: "bind-credential",
        expected: target,
        connection: request.providerId,
        credential,
      });
      const classified = classifyWrite(res);
      setStatus(writeKindText(t, classified.kind, classified.error));
      if (classified.kind === "ok") {
        void qc.invalidateQueries({ queryKey: ["opencode-providers"] });
        void qc.invalidateQueries({ queryKey: ["agent-settings"] });
        onClose();
        return;
      }
      attempted.current = false;
    } finally {
      setPending(false);
    }
  }

  const envValid = kind === "native" || isValidCredentialEnvName(env);

  return (
    <dialog
      ref={dialogRef}
      aria-label={t("editCredential")}
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
          <h2>{t("editCredential")}</h2>
          <button type="button" className={css.dialogClose} disabled={pending} onClick={close}>
            {t("dialogClose")}
          </button>
        </div>
        {request.adapter === "openai-compatible" ? (
          <p className="rounded-md border border-warning bg-warning-soft px-3 py-2 text-xs text-warning">{t("customEndpointWarn")}</p>
        ) : null}
        <label className="flex flex-col gap-1 text-[12px] text-muted">
          {t("credentialKind")}
          <select value={kind} disabled={pending} onChange={(e) => setKind(e.target.value as "native" | "env")} className={FIELD}>
            <option value="native">{t("credentialNative")}</option>
            <option value="env">{t("envKindOption")}</option>
          </select>
        </label>
        {kind === "env" ? (
          <label className="flex flex-col gap-1 text-[12px] text-muted">
            {t("credentialEnvName")}
            <input
              autoFocus
              value={env}
              disabled={pending}
              onChange={(e) => setEnv(e.target.value.toUpperCase())}
              className={FIELD}
              autoComplete="off"
            />
            {env && !isValidCredentialEnvName(env) ? (
              <span className="text-[11px] text-warning">{t("invalidEnvName")}</span>
            ) : env ? (
              <span className={envStatus === "configured" ? "text-[11px] text-success" : envStatus === "missing" ? "text-[11px] text-warning" : "text-[11px] text-muted"}>
                {envStatus === "configured" ? t("envConfigured") : envStatus === "missing" ? t("envMissing") : ""}
              </span>
            ) : null}
          </label>
        ) : null}
        {status ? <p role="status" className="text-xs text-muted">{status}</p> : null}
        <div className={css.dialogFooter}>
          <button type="button" disabled={pending} className={GHOST} onClick={close}>
            {t("cancel")}
          </button>
          <button type="submit" disabled={pending || !expected || !envValid} className={BTN}>
            {pending ? t("saving") : t("save")}
          </button>
        </div>
      </form>
    </dialog>
  );
}
