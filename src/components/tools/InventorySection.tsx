"use client";

import { useQuery } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import { useRef, useState } from "react";
import { fetchEnvelope } from "@/lib/api";
import { SourceWarning } from "@/components/mission/SourceWarning";
import { RelativeTime } from "@/components/RelativeTime";
import { deriveInventoryView, labelKey, stageKey } from "@/lib/tools";
import type {
  InventoryChange,
  InventoryData,
  InventoryExecutor,
  InventoryItem,
  InventoryKind,
  InventoryPreview,
} from "@/server/collectors/inventory"; // type-only
import { InventoryActionDialog } from "./InventoryActionDialog";
import { InventoryReconcileDialog } from "./InventoryReconcileDialog";
import css from "./tools.module.css";

type PostResult<T> = { ok: true; data: T } | { ok: false; error: string; code?: string; effect?: string };

async function postInventory<T>(body: unknown): Promise<PostResult<T>> {
  try {
    const parsed = (await fetch("/api/tools/inventory", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => r.json())) as Record<string, unknown>;
    if (parsed && typeof parsed === "object" && parsed.ok === true) return { ok: true, data: parsed.data as T };
    return {
      ok: false,
      error: typeof parsed?.error === "string" ? (parsed.error as string) : "unavailable",
      code: typeof parsed?.code === "string" ? (parsed.code as string) : undefined,
      effect: typeof parsed?.effect === "string" ? (parsed.effect as string) : undefined,
    };
  } catch {
    return { ok: false, error: "unavailable" };
  }
}

const CHANGES: readonly InventoryChange[] = ["disable", "enable", "remove-registration", "remove-installation"];

export function InventorySection() {
  const t = useTranslations("tools.inv");
  const q = useQuery({
    queryKey: ["tools-inventory"],
    queryFn: ({ signal }) => fetchEnvelope<InventoryData>("/api/tools", signal),
    refetchInterval: 60_000,
  });
  const [query, setQuery] = useState("");
  const [executor, setExecutor] = useState<"all" | InventoryExecutor>("all");
  const [kind, setKind] = useState<"all" | InventoryKind>("all");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [action, setAction] = useState<{ item: InventoryItem; change: InventoryChange; preview: InventoryPreview } | null>(null);
  const [pending, setPending] = useState(false);
  const [status, setStatus] = useState<{ tone: "pending" | "success" | "error"; text: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeOperation, setActiveOperation] = useState<{ operationId: string; stage: string; itemId: string } | null>(null);
  const [reconcileConfirm, setReconcileConfirm] = useState<{
    operationId: string;
    name: string;
    revision: string;
  } | null>(null);
  const operationId = useRef<string | null>(null);
  // A mutable ref closes duplicate-submission races before awaiting the
  // network; `pending` state alone is async and can be observed stale.
  const inFlight = useRef(false);
  const sourceStale = useRef(false);

  const data = q.data?.ok ? q.data.data : null;
  const failed = q.isError && !data;
  const stale = q.isError && !!data;
  sourceStale.current = stale;
  const view = deriveInventoryView(data, { query, executor, kind });
  const selected = view.rows.find((row) => row.id === selectedId) ?? null;
  // Served reload recovery: the unresolved operation bound to the selected item.
  const selectedOperation = selected?.operation ?? null;

  function itemByRegistration(registration: string, executor: InventoryExecutor): InventoryItem | null {
    if (!data) return null;
    // Scoped to the child's own executor: a same-registration plugin in another
    // executor must never be selected as the parent.
    const bucket = data.executors[executor];
    if (!bucket.ok) return null;
    return bucket.items.find((candidate) => candidate.registration === registration) ?? null;
  }

  // Real parent navigation: reveal the parent (clearing any filter hiding it)
  // and select its stable ID.
  function openParent(registration: string, executor: InventoryExecutor) {
    const parent = itemByRegistration(registration, executor);
    if (!parent) return;
    setQuery("");
    setExecutor("all");
    setKind("all");
    setSelectedId(parent.id);
  }

  function refetch() {
    void q.refetch();
  }

  async function openAction(item: InventoryItem, change: InventoryChange) {
    if (stale) return;
    setError(null);
    setStatus({ tone: "pending", text: t("previewFailed") });
    const out = await postInventory<InventoryPreview>({ action: "preview", itemId: item.id, change });
    if (!out.ok || sourceStale.current) {
      setStatus({ tone: "error", text: !out.ok ? out.error : t("previewFailed") });
      return;
    }
    setStatus(null);
    setAction({ item, change, preview: out.data });
  }

  async function apply() {
    if (!action || stale || pending || inFlight.current) return;
    // An uncertain earlier operation must be read back, never re-minted.
    if (activeOperation?.itemId === action.item.id) {
      await recheck(activeOperation.operationId);
      return;
    }
    inFlight.current = true;
    const id = typeof crypto !== "undefined" && crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}`;
    operationId.current = id;
    setPending(true);
    setStatus({ tone: "pending", text: t("applying") });
    const out = await postInventory<{ effect?: string; stage?: string }>({
      action: "apply",
      itemId: action.item.id,
      change: action.change,
      expectedRevision: action.preview.revision,
      operationId: id,
    });
    setPending(false);
    inFlight.current = false;
    if (!out.ok) {
      if (out.effect !== "not-applied") {
        setActiveOperation({ operationId: id, stage: "unconfirmed", itemId: action.item.id });
      } else {
        setActiveOperation(null);
      }
      setStatus({ tone: "error", text: t("applyRefused", { error: out.error }) });
      setError(out.error);
      return;
    }
    // A successful HTTP envelope is NOT proof the native operation is fully
    // applied: a native-only (OpenCode) publication still needs its settings
    // repair, exposed as a distinct confirmed Reconcile.
    if (out.data?.effect === "activation-pending") {
      setActiveOperation({ operationId: id, stage: "activation-pending", itemId: action.item.id });
      setStatus({ tone: "error", text: t("stage.activation-pending") });
      refetch();
      return;
    }
    setActiveOperation(null);
    setStatus({ tone: "success", text: t("applied") });
    setAction(null);
    refetch();
  }

  async function recheck(operationIdOverride?: string) {
    const id = operationIdOverride ?? activeOperation?.operationId ?? operationId.current ?? selectedOperation?.operationId;
    if (!id) {
      setStatus({ tone: "error", text: t("notApplied") });
      return;
    }
    if (inFlight.current) return;
    inFlight.current = true;
    setPending(true);
    const out = await postInventory<{ stage: string; itemId?: string }>({ action: "recheck", operationId: id });
    setPending(false);
    inFlight.current = false;
    if (!out.ok) {
      setError(out.error);
      return;
    }
    const stage = out.data.stage;
    const operationItemId = out.data.itemId ?? (activeOperation?.operationId === id
      ? activeOperation.itemId : operationId.current === id ? action?.item.id ?? "" : "");
    setStatus({
      tone: stage === "applied" ? "success" : "error",
      text: t(`stage.${stageKey(stage)}`),
    });
    if (stage === "applied" || stage === "not-applied") {
      setActiveOperation(null);
      setAction(null);
      refetch();
    } else {
      setActiveOperation({ operationId: id, stage, itemId: operationItemId });
    }
  }

  // Confirmed settings-only recovery: preview-reconcile's safe confirmation
  // first, then an explicit Confirm. Never a blind native retry, and never a
  // live item preview (which cannot exist for a removed item).
  async function openReconcile(opId: string, name: string) {
    if (stale || inFlight.current) return;
    inFlight.current = true;
    setPending(true);
    const out = await postInventory<{ revision: string }>({ action: "preview-reconcile", operationId: opId });
    setPending(false);
    inFlight.current = false;
    if (!out.ok || sourceStale.current) {
      if (!out.ok) setError(out.error);
      setStatus({ tone: "error", text: t("previewReconcileFailed") });
      return;
    }
    setError(null);
    setReconcileConfirm({ operationId: opId, name, revision: out.data.revision });
  }

  async function confirmReconcile() {
    if (stale || !reconcileConfirm || inFlight.current) return;
    inFlight.current = true;
    setPending(true);
    const opId = reconcileConfirm.operationId;
    const out = await postInventory({ action: "reconcile", operationId: opId, expectedRevision: reconcileConfirm.revision });
    setPending(false);
    inFlight.current = false;
    if (!out.ok) {
      setError(out.error);
      return;
    }
    setReconcileConfirm(null);
    setActiveOperation(null);
    setStatus({ tone: "success", text: t("reconcileDone") });
    refetch();
    // refresh the persisted status after the readback
    await recheck(opId);
  }

  if (failed) {
    return (
      <section className="flex flex-col gap-2">
        <h2 className="text-[17px] font-semibold text-ink">{t("heading")}</h2>
        <SourceWarning label={t("unavailable")} detail={q.error instanceof Error ? q.error.message : undefined} />
      </section>
    );
  }
  if (!data) return <div className="h-16 animate-pulse rounded bg-surface-2" />;

  return (
    <section className={css.invSection}>
      <div className={css.sectionHeading}>
        <div>
          <h2>{t("heading")}</h2>
          <p>{t("subtitle")}</p>
        </div>
        <span className={css.tag}>{t("tagBadge")}</span>
      </div>

      <div className={css.invToolbar}>
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label={t("search")}
          placeholder={t("search")}
        />
        <label className={css.field}>
          {t("executor")}
          <select value={executor} onChange={(e) => setExecutor(e.target.value as "all" | InventoryExecutor)}>
            <option value="all">{t("executorAll")}</option>
            <option value="claude">Claude</option>
            <option value="codex">Codex</option>
            <option value="opencode">OpenCode</option>
          </select>
        </label>
        <label className={css.field}>
          {t("kind")}
          <select value={kind} onChange={(e) => setKind(e.target.value as "all" | InventoryKind)}>
            <option value="all">{t("kindAll")}</option>
            <option value="plugin">{t("kindPlugin")}</option>
            <option value="skill">{t("kindSkill")}</option>
            <option value="agent">{t("kindAgent")}</option>
          </select>
        </label>
        <span role="status" className={css.hint}>
          {t("count", { filtered: view.filtered, known: view.known })}
        </span>
      </div>

      {view.partial ? <SourceWarning label={t("partial")} /> : null}
      {stale ? <SourceWarning label={t("stale")} /> : null}
      {view.empty ? <p className={css.hint}>{t("empty")}</p> : null}
      {/* Operation feedback stays visible after the dialog closes. */}
      {status ? (
        <p
          role="status"
          aria-live="polite"
          className={status.tone === "error" ? css.statusError : status.tone === "success" ? css.statusSuccess : css.statusPending}
        >
          {status.text}
        </p>
      ) : null}

      <div className={css.invWorkspace}>
        <div className={`${css.card} ${css.invList}`}>
          <div className={css.invListHead}>
            <span>{t("listHead")}</span>
            <span>{t("executor")}</span>
            <span>{t("state.enabled")}</span>
          </div>
          {view.rows.map((row) => (
            <button
              key={row.id}
              type="button"
              aria-pressed={row.id === selectedId}
              onClick={() => setSelectedId(row.id)}
              className={`${css.invRow}${row.id === selectedId ? ` ${css.invRowSelected}` : ""}`}
            >
              <span>
                <strong>{row.name}</strong>
                <small>{t(labelKey("originLabel", row.origin))}</small>
              </span>
              <span>{row.executor}</span>
              <span className={css.invState} data-state={row.state}>
                {t(`state.${row.state}`)}
              </span>
            </button>
          ))}
        </div>

        <aside className={`${css.card} ${css.invDetail}`} aria-label={t("heading")}>
          {!selected ? (
            <p className={css.hint}>{t("selectItem")}</p>
          ) : (
            <>
              <h3>{selected.name}</h3>
              <dl className={css.invMeta}>
                <dt>{t("kind")}</dt>
                <dd>{t(labelKey("kindLabel", selected.kind))}</dd>
                {selected.description ? (
                  <>
                    <dt>{t("descriptionLabel")}</dt>
                    <dd>{selected.description}</dd>
                  </>
                ) : null}
                <dt>{t("scope")}</dt>
                <dd>
                  {selected.executor} · {selected.scope}
                </dd>
                <dt>{t("origin")}</dt>
                <dd>{t(labelKey("originLabel", selected.origin))}</dd>
                <dt>{t("pathLabel")}</dt>
                <dd>{selected.path}</dd>
                {selected.parent ? (
                  <>
                    <dt>{t("parent")}</dt>
                    <dd>
                      <button
                        type="button"
                        className={css.compareButton}
                        title={t("parentHint")}
                        onClick={() => openParent(selected.parent as string, selected.executor)}
                      >
                        {selected.parent}
                      </button>
                    </dd>
                  </>
                ) : null}
              </dl>
              {/* Local subagent definitions stay read-only. */}
              <div className={css.actionRow}>
                {CHANGES.map((change) => {
                  const cap = selected.capabilities[change];
                  if (!cap) return null;
                  return (
                    <button
                      key={change}
                      type="button"
                      disabled={!cap.available || selected.kind === "agent" || stale}
                      title={cap.reason ? t(`unavailableReason.${cap.reason}`) : undefined}
                      onClick={() => openAction(selected, change)}
                      className={css.compareButton}
                    >
                      {t(`action.${change}`)}
                    </button>
                  );
                })}
              </div>
              {selectedOperation ? (
                <div className={css.actionRow}>
                  <p role="status" aria-live="polite" className={css.hint}>
                    {t("pendingStatus")} · {t(`stage.${stageKey(selectedOperation.stage)}`)}
                  </p>
                  <button type="button" className={css.compareButton} disabled={pending || stale} onClick={() => recheck(selectedOperation.operationId)}>
                    {t("recheck")}
                  </button>
                  {selectedOperation.stage === "activation-pending" ? (
                    <button
                      type="button"
                      className={css.compareButton}
                      disabled={pending || stale}
                      onClick={() => openReconcile(selectedOperation.operationId, selected.name)}
                    >
                      {t("reconcile")}
                    </button>
                  ) : null}
                </div>
              ) : null}
            </>
          )}
        </aside>
      </div>

      {data.recovery && data.recovery.length > 0 ? (
        <div className={`${css.card} ${css.invRecovery}`}>
          <h3>{t("recoveryHeading")}</h3>
          <p className={css.hint}>{t("recoveryBody")}</p>
          {data.recovery.map((row) => (
            <div key={row.operationId} className={css.invRow}>
              <span>
                <strong>{t(`action.${row.change as InventoryChange}`)}</strong>
                <small>{row.unavailable ? t("sourceUnavailable") : t("removed")}</small>
              </span>
              <span className={css.invState} data-state="pending">
                {t(`stage.${stageKey(row.stage)}`)}
              </span>
              <span>
                <button type="button" className={css.compareButton} disabled={pending || stale} onClick={() => recheck(row.operationId)}>
                  {t("recheck")}
                </button>
                {row.stage === "activation-pending" ? (
                  <button
                    type="button"
                    className={css.compareButton}
                    disabled={pending || stale}
                    onClick={() => openReconcile(row.operationId, `${row.itemId.slice(0, 12)}…`)}
                  >
                    {t("reconcile")}
                  </button>
                ) : null}
              </span>
            </div>
          ))}
        </div>
      ) : null}

      {action ? (
        <InventoryActionDialog
          item={action.item}
          preview={action.preview}
          pending={pending}
          status={status}
          error={error}
          stale={stale}
          reconcileAvailable={activeOperation?.itemId === action.item.id && activeOperation.stage === "activation-pending"}
          onApply={apply}
          onRecheck={() => recheck()}
          onReconcile={
            activeOperation?.itemId === action.item.id
              ? () => openReconcile(activeOperation.operationId, action.item.name)
              : undefined
          }
          onClose={() => {
            if (!pending) setAction(null);
          }}
        />
      ) : null}

      {reconcileConfirm ? (
        <InventoryReconcileDialog
          name={reconcileConfirm.name}
          pending={pending}
          stale={stale}
          error={error}
          onConfirm={confirmReconcile}
          onCancel={() => {
            if (!pending) setReconcileConfirm(null);
          }}
        />
      ) : null}
    </section>
  );
}
