"use client";

import { useEffect, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import css from "./tools.module.css";

// Settings-only recovery confirmation (F4): uses the preview-reconcile safe
// fields, never a live item/action preview, so it also works for an item that
// is already removed from discovery.
export function InventoryReconcileDialog({
  name,
  pending,
  stale,
  error,
  onConfirm,
  onCancel,
}: {
  name: string;
  pending: boolean;
  stale?: boolean;
  error: string | null;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const t = useTranslations("tools.inv");
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [confirmed, setConfirmed] = useState(false);

  useEffect(() => {
    const dlg = dialogRef.current;
    if (dlg && typeof dlg.showModal === "function") dlg.showModal();
  }, []);

  return (
    <dialog
      ref={dialogRef}
      aria-label={t("reconcileTitle")}
      onClose={onCancel}
      onCancel={(e) => {
        e.preventDefault();
        if (!pending) onCancel();
      }}
      className={css.invDialog}
    >
      <div className={css.invDialogBody}>
        <div className={css.dialogHeader}>
          <div>
            <h2>{t("reconcileTitle")}</h2>
            <p>{name}</p>
          </div>
          <button type="button" onClick={onCancel} disabled={pending} className={css.dialogClose}>
            {t("close")}
          </button>
        </div>
        <p className={css.hint}>{t("reconcileBody")}</p>
        <dl className={css.invMeta}>
          <dt>{t("effect")}</dt>
          <dd>{t("effectText.settings", { name })}</dd>
        </dl>
        {error ? (
          <p role="status" aria-live="polite" className={css.statusError}>
            {t("applyRefused", { error })}
          </p>
        ) : null}
        <label className={css.checkLine}>
          <input type="checkbox" checked={confirmed} disabled={pending} onChange={(e) => setConfirmed(e.target.checked)} />
          <span>{t("reconcileConfirm")}</span>
        </label>
        <div className={css.dialogFooter}>
          <button type="button" className={css.compareButton} onClick={onCancel} disabled={pending}>
            {t("cancel")}
          </button>
          <button type="button" className={css.primaryButton} onClick={onConfirm} disabled={pending || stale || !confirmed}>
            {pending ? t("reconciling") : t("reconcile")}
          </button>
        </div>
      </div>
    </dialog>
  );
}
