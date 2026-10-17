"use client";

import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import { validateModelInput, type EditorRevision, type ModelChoice } from "@/lib/agent-settings";
import { classifyWrite, postJson, writeKindText } from "./ToolsDraftProvider";
import css from "./tools.module.css";

const FIELD =
  "w-full rounded-md border border-line bg-surface-2 px-2 py-1.5 text-[13px] text-ink outline-none focus:border-line-strong disabled:opacity-60";
const BTN =
  "rounded-md bg-brand px-3 py-1.5 text-[12.5px] font-medium text-on-brand hover:bg-brand/90 disabled:opacity-60";
const GHOST =
  "rounded-md border border-line px-3 py-1.5 text-[12.5px] text-body-ink hover:bg-surface-2 disabled:opacity-60";

export function resolveEffortSource(
  editing: boolean,
  model: ModelChoice | null | undefined,
  models: ModelChoice[] | undefined,
  id: string,
): ModelChoice | null {
  if (editing) return model ?? null;
  return id ? models?.find((m) => m.id === id) ?? null : null;
}

export function ProviderModelDialog({
  connection,
  model,
  models,
  expected,
  invoker,
  onClose,
}: {
  connection: string;
  model?: ModelChoice | null;
  models?: ModelChoice[];
  expected: EditorRevision | null;
  invoker?: HTMLButtonElement;
  onClose: () => void;
}) {
  const t = useTranslations("tools");
  const qc = useQueryClient();
  const dialogRef = useRef<HTMLDialogElement>(null);
  const editing = model != null;
  const [id, setId] = useState(model?.id ?? "");
  const [label, setLabel] = useState(model?.label ?? "");
  const [context, setContext] = useState(model?.context != null ? String(model.context) : "");
  const [effortsText, setEffortsText] = useState((model?.efforts ?? []).join(", "));
  const [pending, setPending] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  // In create mode the proven representation comes from the configured model
  // that matches the typed ID; a genuinely uncatalogued ID has no template.
  const source = resolveEffortSource(editing, model, models, id);
  const effortEditable = source != null && source.effort_template !== null;
  const showEffortUnsupported = !effortEditable && (editing || id !== "");
  const matchEfforts = (editing ? null : source)?.efforts.join(",") ?? "";

  useEffect(() => {
    const dlg = dialogRef.current;
    if (dlg && typeof dlg.showModal === "function") dlg.showModal();
    return () => {
      if (invoker?.isConnected) invoker.focus();
    };
  }, [invoker]);

  useEffect(() => {
    if (editing) return;
    setEffortsText(matchEfforts);
  }, [editing, id, matchEfforts]);

  function close() {
    if (!pending) onClose();
  }

  async function submit() {
    if (pending || !expected || !id || id.startsWith("jaxflow-builder-") || /[\r\n\0]/.test(id)) return;
    const payload: Record<string, unknown> = { id };
    if (label !== (editing ? model.label : "")) {
      if (label) payload.label = label;
    }
    const currentContext = editing && model.context != null ? String(model.context) : "";
    if (context !== currentContext && context !== "") payload.context = Number(context);
    if (effortEditable) {
      const list = effortsText.split(",").map((s) => s.trim()).filter(Boolean);
      const current = source?.efforts ?? [];
      if (list.join(",") !== current.join(",")) payload.efforts = list;
    }
    if (!validateModelInput(payload)) {
      setStatus(t("modelRequires"));
      return;
    }
    setPending(true);
    const res = await postJson("/api/opencode-providers", {
      action: "save-model",
      expected,
      connection,
      model: payload,
    });
    const classified = classifyWrite(res);
    if (classified.kind === "ok") {
      void qc.invalidateQueries({ queryKey: ["opencode-providers"] });
      void qc.invalidateQueries({ queryKey: ["agent-settings"] });
      onClose();
    } else {
      setStatus(writeKindText(t, classified.kind, classified.error));
      setPending(false);
    }
  }

  return (
    <dialog
      ref={dialogRef}
      aria-label={editing ? t("editModel") : t("addModel")}
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
          <h2>{editing ? t("editModel") : t("addModel")}</h2>
          <button type="button" className={css.dialogClose} disabled={pending} onClick={close}>
            {t("dialogClose")}
          </button>
        </div>
        <label className="flex flex-col gap-1 text-[12px] text-muted">
          {t("modelId")}
          <input
            autoFocus={!editing}
            readOnly={editing}
            value={id}
            onChange={(e) => setId(e.target.value)}
            className={FIELD}
          />
        </label>
        <label className="flex flex-col gap-1 text-[12px] text-muted">
          {t("label")}
          <input value={label} onChange={(e) => setLabel(e.target.value)} className={FIELD} />
        </label>
        <details open={!editing}>
          <summary>{t("modelMetadataSummary")}</summary>
          <label className="flex flex-col gap-1 text-[12px] text-muted">
            {t("contextTokens")}
            <input
              type="number"
              min={1}
              step={1}
              value={context}
              onChange={(e) => setContext(e.target.value)}
              className={FIELD}
            />
            <span className="text-[11px] text-muted">{editing ? t("contextKeepHint") : t("contextHint")}</span>
          </label>
          <label className="flex flex-col gap-1 text-[12px] text-muted">
            {t("effortValues")}
            <input
              value={effortsText}
              disabled={!effortEditable}
              onChange={(e) => setEffortsText(e.target.value)}
              className={FIELD}
              placeholder="high, medium"
            />
          </label>
        </details>
        {showEffortUnsupported ? (
          <p className="rounded-md border border-warning bg-warning-soft px-3 py-2 text-xs text-warning">
            {t("effortUnsupported")}
          </p>
        ) : null}
        {editing ? <p className="text-[11px] text-muted">{t("outputInternal")}</p> : null}
        {status ? <p role="status" className="text-xs text-danger">{status}</p> : null}
        <div className={css.dialogFooter}>
          <button type="button" disabled={pending} className={GHOST} onClick={close}>
            {t("cancel")}
          </button>
          <button type="submit" disabled={pending || !expected} className={BTN}>
            {pending ? t("saving") : t("save")}
          </button>
        </div>
      </form>
    </dialog>
  );
}
