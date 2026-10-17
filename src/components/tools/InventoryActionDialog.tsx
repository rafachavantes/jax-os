"use client";

import { useEffect, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import type { InventoryItem, InventoryPreview } from "@/server/collectors/inventory"; // type-only
import css from "./tools.module.css";

export function requestInventoryClose(pending: boolean, close: () => void) {
  if (!pending) close();
}

export function InventoryActionDialog({
  item,
  preview,
  status,
  error,
  pending,
  stale,
  reconcileAvailable,
  onApply,
  onRecheck,
  onReconcile,
  onClose,
}: {
  item: InventoryItem;
  preview: InventoryPreview;
  status: { tone: "pending" | "success" | "error"; text: string } | null;
  error: string | null;
  pending: boolean;
  stale?: boolean;
  reconcileAvailable?: boolean;
  onApply: () => void;
  onRecheck: () => void;
  onReconcile?: () => void;
  onClose: () => void;
}) {
  const t = useTranslations("tools.inv");
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [confirmed, setConfirmed] = useState(false);
  const busy = pending || status?.tone === "pending";

  useEffect(() => {
    const dlg = dialogRef.current;
    if (dlg && typeof dlg.showModal === "function") dlg.showModal();
  }, []);

  return (
    <dialog
      ref={dialogRef}
      aria-label={t("dialogTitle")}
      onClose={onClose}
      onCancel={(e) => {
        e.preventDefault();
        requestInventoryClose(busy, onClose);
      }}
      className={css.invDialog}
    >
      <div className={css.invDialogBody}>
        <div className={css.dialogHeader}>
          <div>
            <h2>{t("dialogTitle")}</h2>
            <p>{item.name}</p>
          </div>
          <button type="button" onClick={onClose} disabled={busy} className={css.dialogClose}>
            {t("close")}
          </button>
        </div>
        <dl className={css.invMeta}>
          <dt>{t("scope")}</dt>
          <dd>
            {item.executor} · {item.scope}
          </dd>
          <dt>{t("effect")}</dt>
          <dd>{t(`effectText.${preview.change}`, { name: item.name })}</dd>
          <dt>{t("targets")}</dt>
          <dd>{preview.targets.join(", ") || "—"}</dd>
          {preview.recovery ? (
            <>
              <dt>{t("recovery")}</dt>
              <dd>{preview.recovery}</dd>
            </>
          ) : null}
        </dl>
        {preview.requiresNewSession ? <p className={css.hint}>{t("newSession")}</p> : null}
        <label className={css.checkLine}>
          <input type="checkbox" checked={confirmed} disabled={busy} onChange={(e) => setConfirmed(e.target.checked)} />
          <span>{t("confirm")}</span>
        </label>
        {error ? (
          <p role="status" aria-live="polite" className={css.statusError}>
            {t("applyRefused", { error })}
          </p>
        ) : null}
        {status ? (
          <p
            role="status"
            aria-live="polite"
            className={status.tone === "error" ? css.statusError : status.tone === "success" ? css.statusSuccess : css.statusPending}
          >
            {status.text}
          </p>
        ) : null}
        <div className={css.dialogFooter}>
          <button type="button" className={css.compareButton} onClick={onRecheck} disabled={busy}>
            {busy ? t("rechecking") : t("recheck")}
          </button>
          {reconcileAvailable && onReconcile ? (
            <button type="button" className={css.compareButton} onClick={onReconcile} disabled={busy}>
              {busy ? t("reconciling") : t("reconcile")}
            </button>
          ) : null}
          <button type="button" className={css.primaryButton} onClick={onApply} disabled={busy || stale || !confirmed}>
            {busy ? t("applying") : t("apply")}
          </button>
        </div>
      </div>
    </dialog>
  );
}
