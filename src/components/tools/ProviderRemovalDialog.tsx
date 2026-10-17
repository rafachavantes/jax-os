"use client";

import { useTranslations } from "next-intl";
import { GHOST } from "./ui";
import { ModalDialog } from "../ModalDialog";
import css from "./tools.module.css";

const BTN =
  "rounded-md bg-danger px-3 py-1.5 text-[12.5px] font-medium text-on-brand hover:bg-danger/90 disabled:opacity-60";

export type RemovalTarget =
  | { kind: "connection"; id: string }
  | { kind: "model"; id: string; model: string; catalog: boolean };

export function ProviderRemovalDialog({
  target,
  references,
  pending,
  onConfirm,
  onClose,
}: {
  target: RemovalTarget;
  references: string[];
  pending: boolean;
  onConfirm: () => void;
  onClose: () => void;
}) {
  const t = useTranslations("tools");

  const title = target.kind === "connection"
    ? t("removeConnection")
    : (target.catalog ? t("removeModelCatalog") : t("removeModel"));
  const body = target.kind === "connection"
    ? t("removeConnectionConfirm", { id: target.id })
    : (target.catalog ? t("removeModelCatalogConfirm", { id: target.id, model: target.model }) : t("removeModelConfirm", { id: target.id, model: target.model }));

  return (
    <ModalDialog
      pending={pending}
      onClose={onClose}
      aria-label={title}
      className={`${css.providerDialog} rounded-lg border border-line bg-surface`}
    >
      <div className="flex flex-col gap-3 p-4">
        <h3 className="text-[14px] font-bold text-ink">{title}</h3>
        <p className="text-[12.5px] text-body-ink">{body}</p>
        {references.length ? (
          <p className="rounded-md border border-warning bg-warning-soft px-3 py-2 text-xs text-warning">
            {t("removeBlocked", { refs: references.join(", "), id: target.kind === "model" ? target.model : target.id })}
          </p>
        ) : null}
        <div className="flex justify-end gap-2">
          <button type="button" disabled={pending} className={GHOST} onClick={onClose}>
            {t("cancel")}
          </button>
          <button type="button" disabled={pending || references.length > 0} className={BTN} onClick={onConfirm}>
            {pending ? t("saving") : t("remove")}
          </button>
        </div>
      </div>
    </ModalDialog>
  );
}
