"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { diffLines } from "@/lib/rulesDiff";
import { ModalDialog } from "../ModalDialog";
import css from "./rules.module.css";

export function RuleDiffModal({
  appLabel,
  expected,
  disk,
  missing,
  status,
  stale = false,
  path,
  applyLabel,
  onApply,
  onClose,
}: {
  appLabel: string;
  expected: string;
  disk: string; // "" when the file is missing — diffed against the empty document (spec §6.3)
  missing: boolean; // status === "missing" — an existing zero-byte file is NOT missing, so disk === "" alone can't tell
  // section-owned settlement for THIS operation (op-matched by the parent)
  status: { tone: "pending" | "success" | "error"; text: string } | null;
  // a newer polling revision invalidated this fixed preview snapshot
  stale?: boolean;
  path?: string;
  applyLabel?: string;
  onApply: () => Promise<unknown>; // section-owned settlement; the modal only drives the pending apply
  onClose: () => void;
}) {
  const t = useTranslations("rules");
  const [applying, setApplying] = useState(false);
  const [confirmed, setConfirmed] = useState(false);
  const lines = diffLines(disk, expected);

  async function handleApply() {
    if (applying || !confirmed || stale) return;
    setApplying(true);
    try {
      await onApply();
    } finally {
      setApplying(false);
    }
  }

  return (
    <ModalDialog
      pending={applying}
      onClose={onClose}
      aria-label={appLabel}
      className={css.diffDialog}
    >
      <div className={css.dialogBody}>
        <div className={css.dialogTitle}>
          <h3>{appLabel}</h3>
          <button type="button" onClick={onClose} disabled={applying} className={css.dialogClose}>
            {t("cancel")}
          </button>
        </div>
        {path ? <p className={css.hint}>{t("previewNote", { path })}</p> : null}
        {missing ? <p className={css.hint}>{t("diffMissing")}</p> : null}
        {/* side-by-side (spec §6.3): two aligned columns over the same LCS sequence. */}
        <div className={css.diffGrid}>
          <div className={css.diffCol}>
            <span className={css.diffLabel}>{t("diffDisk")}</span>
            {lines.map((l, i) => (
              <div key={i} className={`${css.diffLine}${l.type === "remove" ? ` ${css.diffRemove}` : ""}`}>
                {l.type !== "add" ? l.text || "\u00a0" : "\u00a0"}
              </div>
            ))}
          </div>
          <div className={css.diffCol}>
            <span className={css.diffLabel}>{t("diffExpected")}</span>
            {lines.map((l, i) => (
              <div key={i} className={`${css.diffLine}${l.type === "add" ? ` ${css.diffAdd}` : ""}`}>
                {l.type !== "remove" ? l.text || "\u00a0" : "\u00a0"}
              </div>
            ))}
          </div>
        </div>
        <label className={css.confirmLine}>
          <input
            type="checkbox"
            checked={confirmed}
            disabled={stale}
            onChange={(e) => setConfirmed(e.target.checked)}
          />
          <span>{t("confirmApply", { app: appLabel })}</span>
        </label>
        {stale ? (
          <p role="status" aria-live="polite" className={css.statusError}>
            {t("previewStale")}
          </p>
        ) : null}
        {status ? (
          <p
            role="status"
            aria-live="polite"
            className={
              status.tone === "error" ? css.statusError : status.tone === "success" ? css.statusSuccess : css.statusPending
            }
          >
            {status.text}
          </p>
        ) : null}
        <div className={css.dialogFooter}>
          <button
            type="button"
            onClick={handleApply}
            disabled={applying || !confirmed || stale}
            className={css.primaryButton}
          >
            {applying ? t("applying") : (applyLabel ?? t("apply"))}
          </button>
        </div>
      </div>
    </ModalDialog>
  );
}
