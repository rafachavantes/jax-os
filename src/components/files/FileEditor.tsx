"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import type { BeforeMount, Monaco, OnMount } from "@monaco-editor/react";
import dynamic from "next/dynamic";
import { useLocale, useTranslations } from "next-intl";
import { ChevronLeft } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { Envelope } from "@/lib/api";
import type { FileRead, Root } from "@/server/collectors/files";
import { tabKey, type EditPatch, type EditState } from "@/lib/filesState";
import { breadcrumbSegments } from "@/lib/breadcrumb";
import { Breadcrumb } from "@/components/files/Breadcrumb";
import { saveEdit } from "@/lib/fileSave";
import { reloadEdit, reloadReadResult } from "@/lib/fileReload";
import { SourceWarning } from "@/components/mission/SourceWarning";
import { Markdown } from "@/components/kanban/Markdown";
import { CsvTable, parseCsv } from "@/components/files/CsvTable";
import { useFilesWorkspace } from "@/components/files/FilesWorkspaceProvider";
import { repoOf, shouldRevealLine } from "@/lib/filesWorkspace";
// side-effect: points @monaco-editor/react at /monaco-vs/vs (self-hosted, no CDN)
import "@/components/files/monacoLoader";

export function disposeUnattachedModels(
  getModel: (path: string) => { dispose: () => void } | null,
  retired: readonly string[],
  attached: string | null,
) {
  for (const path of retired) {
    if (path === attached) continue;
    getModel(path)?.dispose();
  }
}

const Editor = dynamic(() => import("@monaco-editor/react"), { ssr: false });
const DiffEditor = dynamic(() => import("@monaco-editor/react").then((mod) => mod.DiffEditor), { ssr: false });

// file extension → monaco language id (only ids our bundle ships; toml has no
// native monaco language → ini gives sane key=value highlighting). Default plaintext.
const extToLang: Record<string, string> = {
  ts: "typescript",
  tsx: "typescript",
  js: "javascript",
  jsx: "javascript",
  json: "json",
  md: "markdown",
  markdown: "markdown",
  css: "css",
  scss: "scss",
  html: "html",
  htm: "html",
  py: "python",
  sh: "shell",
  yaml: "yaml",
  yml: "yaml",
  toml: "ini",
};

function langOf(rel: string): string {
  return extToLang[rel.split(".").pop()?.toLowerCase() ?? ""] ?? "plaintext";
}

// ponytail: client-safe mirror of the server image gate (files.ts helpers pull
// node:fs) — same precedent as UPLOAD_CAP in FileTree.
const IMAGE_EXTS = new Set(["png", "jpg", "jpeg", "gif", "webp", "svg"]);
const IMAGE_CAP = 5 * 1024 * 1024;
function isImage(rel: string): boolean {
  return IMAGE_EXTS.has(rel.split(".").pop()?.toLowerCase() ?? "");
}

// Client-safe mirrors of Part 1's server caps — same reasoning as IMAGE_CAP: files.ts imports node:fs and must never be imported from client code.
const READ_CAP = 2 * 1024 * 1024;
const RAW_PDF_CAP = 20 * 1024 * 1024;
const RAW_MEDIA_CAP = 64 * 1024 * 1024;
const AUDIO_EXTS = new Set(["mp3", "wav", "ogg", "m4a", "aac"]);
const VIDEO_EXTS = new Set(["mp4", "webm", "mov"]);
const CSV_ROW_CAP = 5000;

export type ViewerKind = "markdown" | "html" | "pdf" | "image" | "csv" | "csvTooBig" | "audio" | "video" | "text" | "placeholder";

// Full spec §8 decision table as one pure, fully-tested function (spec §11's "viewerFor" bullet); consumed below only for the NEW kinds this task adds — see this task's "ponytail scope note".
export function viewerFor(ext: string, size: number, meta?: { rows: number }): ViewerKind {
  const e = ext.toLowerCase();
  if (e === "md" || e === "markdown") return size <= READ_CAP ? "markdown" : "placeholder";
  if (e === "html" || e === "htm") return size <= READ_CAP ? "html" : "placeholder";
  if (e === "pdf") return size <= RAW_PDF_CAP ? "pdf" : "placeholder";
  if (IMAGE_EXTS.has(e)) return size <= IMAGE_CAP ? "image" : "placeholder";
  if (e === "csv" || e === "tsv") {
    if (size > READ_CAP) return "placeholder";
    return meta && meta.rows > CSV_ROW_CAP ? "csvTooBig" : "csv";
  }
  if (AUDIO_EXTS.has(e)) return size <= RAW_MEDIA_CAP ? "audio" : "placeholder";
  if (VIDEO_EXTS.has(e)) return size <= RAW_MEDIA_CAP ? "video" : "placeholder";
  return size <= READ_CAP ? "text" : "placeholder";
}

// Extracted, not inline in JSX (which this repo has no DOM harness to test) — round 1 F4.
export function csvDelimiterFor(ext: string): string {
  return ext.toLowerCase() === "tsv" ? "\t" : ",";
}

const PREVIEW_CSP = "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; img-src data:; style-src 'unsafe-inline'; font-src data:\">";

// HTML preview (spec §5 item 10): the user's markup is never parsed/regex-matched —
// it goes into the body of OUR OWN document shell, whose head holds only the CSP meta.
// Regexing raw HTML for a <head> tag lets a fake one inside a comment (`<!-- <head> -->`)
// steal the injection, leaving the real document uncontrolled — an empty sandboxed
// srcdoc iframe still can't fetch a live URL for exfiltration without a CSP.
// Browsers hoist/ignore any nested html/head/body the user's markup contains, so its
// own <style>/<body> content still renders; only our head (and its CSP) governs the page.
export function wrapHtmlForPreview(html: string): string {
  return `<!DOCTYPE html><html><head>${PREVIEW_CSP}</head><body>${html}</body></html>`;
}

export function fmtSize(n: number, locale: string): string {
  const nf = new Intl.NumberFormat(locale, { maximumFractionDigits: 1 });
  if (n >= 1024 * 1024) return `${nf.format(n / 1048576)} MB`;
  if (n >= 1024) return `${nf.format(n / 1024)} KB`;
  return `${new Intl.NumberFormat(locale).format(n)} B`;
}

// Resolve a DS custom property (var-chain / rgba / oklch) to an opaque #rrggbb.
// A throwaway probe lets the browser normalize any color form to rgb()/rgba(),
// which we hex-encode — Monaco's defineTheme rejects raw var()/rgba strings.
function resolveHex(varName: string): string {
  if (typeof document === "undefined") return "#000000";
  const probe = document.createElement("span");
  probe.style.color = getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
  document.body.appendChild(probe);
  const rgb = getComputedStyle(probe).color.match(/\d+/g);
  probe.remove();
  if (!rgb) return "#000000";
  return "#" + rgb.slice(0, 3).map((n) => (+n).toString(16).padStart(2, "0")).join("");
}

// One Monaco theme, "jax", recomputed from the live DS values. `base` gives sane
// syntax colors, so we override only chrome (bg/fg/gutter) from resolved tokens —
// the single sanctioned place Monaco reads concrete colors. Re-defining in place
// on a theme flip is enough; the observer re-runs this on data-theme change.
// ponytail: 3 overrides, not the full Monaco palette — `base` carries the rest.
function applyMonacoTheme(monaco: Monaco) {
  const light = document.documentElement.dataset.theme === "light";
  monaco.editor.defineTheme("jax", {
    base: light ? "vs" : "vs-dark",
    inherit: true,
    rules: [],
    colors: {
      "editor.background": resolveHex("--surface"),
      "editor.foreground": resolveHex("--body-ink"),
      "editorLineNumber.foreground": resolveHex("--text-muted"),
    },
  });
  monaco.editor.setTheme("jax");
}

// Poll-driven staleness decision (pure, unit-tested directly): a background
// refetch's hash differing from the edit's own baseline means disk changed.
// Clean buffer -> silently adopt the new content as the new baseline. Dirty
// buffer -> surface staleness without touching the user's unsaved edits.
export function pollOutcome(
  diskHash: string,
  edit: EditState,
): { action: "none" } | { action: "adopt" } | { action: "stale" } {
  if (diskHash === edit.baseHash) return { action: "none" };
  if (edit.content === edit.savedContent) return { action: "adopt" };
  return { action: "stale" };
}

// Applies pollOutcome's decision via `patch` — the WIRING is under test here,
// not just the bare decision (F3). The "adopt" patch mirrors reloadEdit's own
// successful-reload shape (fileReload.ts), clearing justSaved/reloadKept too (F4).
export function applyPollOutcome(
  disk: { hash: string; content: string },
  edit: EditState,
  patch: (patch: EditPatch) => void,
): { action: "none" } | { action: "adopt" } | { action: "stale" } {
  const outcome = pollOutcome(disk.hash, edit);
  if (outcome.action === "adopt") {
    patch({
      content: disk.content, savedContent: disk.content, baseHash: disk.hash,
      stale: false, justSaved: false, reloadKept: false,
    });
  } else if (outcome.action === "stale" && !edit.stale) {
    patch({ stale: true });
  }
  return outcome;
}

// Called from save()'s patch callback: a successful save (justSaved with a
// fresh baseHash) writes the read query cache to the just-saved content/hash,
// so a poll landing right after a save compares against saved state instead
// of the pre-save disk snapshot (diff review 13b2b3d5fb4c F1). No-ops for
// every other save outcome (stale/unconfirmed/writeErr/auditWarning).
export function syncReadCacheAfterSave(
  patch: EditPatch,
  content: string,
  setReadCache: (envelope: Envelope<FileRead>) => void,
): void {
  if (!patch.justSaved || !patch.baseHash) return;
  setReadCache({
    ok: true,
    data: { binary: false, content, hash: patch.baseHash, size: new TextEncoder().encode(content).length },
  });
}

// Keys are "root:rel" (same shape as tabKey, filesState.ts:42-44; rel === ""
// is the root itself). Root plus every ancestor directory of `rel`, including
// `rel` itself when it names a directory (not the file at a breadcrumb's end).
export function ancestorKeys(root: Root, rel: string, targetIsDir: boolean): string[] {
  const keys = [`${root}:`];
  if (rel === "") return keys;
  const parts = rel.split("/");
  const upTo = targetIsDir ? parts.length : parts.length - 1;
  let acc = "";
  for (let i = 0; i < upTo; i++) {
    acc = acc ? `${acc}/${parts[i]}` : parts[i];
    keys.push(`${root}:${acc}`);
  }
  return keys;
}

export type ChipInfo = { kind: "stale" | "text"; tone: "warning" | "danger" | "muted"; text: string };

// Collapses the nine independent save/reload states into ONE prioritized
// selection over the SAME EditState fields (cold review round 2 decision §5
// item 4) — exactly one is shown at a time. "stale" is actionable (opens the
// diff panel); every other state is a plain text chip.
export function statusChip(m: EditState | undefined, dirty: boolean, t: (key: string) => string): ChipInfo | null {
  if (!m) return null;
  if (m.stale) return { kind: "stale", tone: "warning", text: t("staleReload") };
  if (m.unconfirmed) return { kind: "text", tone: "warning", text: t("checkCurrentState") };
  if (m.auditWarning) return { kind: "text", tone: "warning", text: t("savedAuditPending") };
  if (m.auditUnavailable) return { kind: "text", tone: "danger", text: t("auditUnavailable") };
  if (m.writeErr) return { kind: "text", tone: "danger", text: t("writeError") };
  if (m.reloadKept) return { kind: "text", tone: "warning", text: t("reloadKept") };
  if (m.reloading) return { kind: "text", tone: "muted", text: t("reloading") };
  if (m.saving) return { kind: "text", tone: "muted", text: t("saving") };
  if (m.justSaved && !dirty) return { kind: "text", tone: "muted", text: t("saved") };
  return null;
}

function chipToneClass(tone: "warning" | "danger" | "muted"): string {
  return tone === "danger" ? "text-[11px] text-danger" : tone === "warning" ? "text-[11px] text-warning" : "text-[11px] text-muted";
}

type FileRef = { root: Root; rel: string };

type Props = {
  file: FileRef | null;
  onClose?: () => void;
};

export function FileEditor({ file, onClose }: Props) {
  const t = useTranslations("files");
  const locale = useLocale();
  const ws = useFilesWorkspace();
  const key = file ? tabKey(file) : null;
  const edit = key ? ws.getEdit(key) : undefined;
  const m = edit;
  const pending = useRef(new Set<symbol>());
  const locked = !!file && (ws.isLocked(file.root, file.rel) || !!edit?.pathLocked);

  const isMd = !!file && langOf(file.rel) === "markdown";
  const isHtml = !!file && langOf(file.rel) === "html";
  const [preview, setPreview] = useState(isMd || isHtml);
  const [imgError, setImgError] = useState(false);
  useEffect(() => setPreview(isMd || isHtml), [key, isMd, isHtml]);
  useEffect(() => setImgError(false), [key, file?.root, file?.rel, edit?.baseHash]);
  const [diffOpen, setDiffOpen] = useState(false);
  useEffect(() => setDiffOpen(false), [key]);
  useEffect(() => { if (!m?.stale) setDiffOpen(false); }, [m?.stale]);

  const qc = useQueryClient();
  const q = useQuery<Envelope<FileRead>>({
    queryKey: ["files", "read", file?.root, file?.rel],
    queryFn: async () =>
      (
        await fetch(`/api/files/read?root=${file!.root}&rel=${encodeURIComponent(file!.rel)}`)
      ).json(),
    enabled: !!file,
    staleTime: 30_000,
    refetchInterval: 5000,
  });
  const read = q.data?.ok ? q.data.data : null;
  const failed = q.isError || (q.data && !q.data.ok);

  // Seed page edit state when a TEXT read resolves and no edit exists yet.
  // NEVER via defaultValue — it only applies at model creation; reads are async.
  useEffect(() => {
    if (!key || !read || read.binary || edit) return;
    ws.onSeed(key, read.content, read.hash);
  }, [key, read, edit, ws.onSeed]);

  // Background poll: applyPollOutcome owns both the decision and the exact
  // patch shape for each branch (the seed effect above handles "no edit yet").
  useEffect(() => {
    if (!key || !read || read.binary || !edit) return;
    applyPollOutcome({ hash: read.hash, content: read.content }, edit, (patch) => ws.onEdit(key, edit.id, patch));
  }, [key, read, edit, ws.onEdit]);

  const dirty = !!edit && edit.content !== edit.savedContent;

  async function save() {
    if (!file || !key || !edit || locked) return;
    const submitted = edit;
    const capturedKey = key;
    const readKey = ["files", "read", file.root, file.rel];
    await saveEdit(submitted, pending.current, async () => {
      return fetch("/api/files/write", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ root: file.root, rel: file.rel,
          content: submitted.content, baseHash: submitted.baseHash }),
      }).then((r) => r.json());
    }, (patch) => {
      syncReadCacheAfterSave(patch, submitted.content, (envelope) => qc.setQueryData<Envelope<FileRead>>(readKey, envelope));
      ws.onEdit(capturedKey, submitted.id, patch);
    });
  }

  async function reload() {
    if (!file || !key || !edit || locked) return;
    const capturedKey = key;
    const capturedQ = q;
    await reloadEdit({
      getCurrent: () => ws.getEdit(capturedKey),
      read: async () => reloadReadResult(await capturedQ.refetch()),
      patch: (id, patch) => ws.onEdit(capturedKey, id, patch),
      pending: pending.current,
    });
  }

  function overwriteStale() {
    if (!file || !key || !edit || locked || !read || read.binary) return;
    ws.onEdit(key, edit.id, { baseHash: read.hash, stale: false });
    setDiffOpen(false);
  }

  // Ctrl/Cmd+S: the command is registered once on mount; a ref keeps it
  // pointing at the latest save closure (file/edit change every render).
  const saveRef = useRef<() => void>(() => {});
  saveRef.current = () => void save();

  // ThemeToggle flips data-theme client-side with no event/context, so watch the
  // attribute directly and recompute the Monaco theme on flip. Observer lives on
  // the component (guarded so it's created once) and disconnected on unmount.
  const editorRef = useRef<Parameters<OnMount>[0] | null>(null);
  const [mountTick, setMountTick] = useState(0);
  const observerRef = useRef<MutationObserver | null>(null);
  const beforeMount: BeforeMount = (monaco) => applyMonacoTheme(monaco);
  const monacoRef = useRef<Monaco | null>(null);
  const onMount: OnMount = (editor, monaco) => {
    monacoRef.current = monaco;
    editorRef.current = editor;
    setMountTick((t) => t + 1); // retriggers the reveal effect below once the editor is ready
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => saveRef.current());
    if (observerRef.current) return;
    const obs = new MutationObserver(() => applyMonacoTheme(monaco));
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    observerRef.current = obs;
  };
  useEffect(() => {
    const monaco = monacoRef.current;
    if (!monaco) return;
    const attached = file ? `${file.root}/${file.rel}` : null;
    disposeUnattachedModels(
      (path) => monaco.editor.getModel(monaco.Uri.parse(path)),
      ws.retiredModels,
      attached,
    );
  }, [file, ws.retiredModels]);
  // Round-1 F1: reveal + position the editor on a pending content-search line jump, then consume
  // it (one-shot, mirrors FileTree.tsx:314-321's tree-reveal consumption exactly). `edit` truthy is
  // this file's own existing proxy for "the read has resolved and the model has real content" — the
  // same condition the seed/poll effects above already gate on, reused here rather than introducing
  // a second "is the model ready" signal. `mountTick` retries once the (possibly still-loading)
  // Monaco editor instance actually mounts.
  useEffect(() => {
    const request = ws.lineRevealRequest;
    if (!shouldRevealLine(request, key) || !edit) return;
    const editor = editorRef.current;
    if (!editor) return;
    editor.revealLineInCenter(request!.line);
    editor.setPosition({ lineNumber: request!.line, column: 1 });
    ws.consumeLineReveal(request!.version);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- mirrors FileTree.tsx:314-321's own
    // reveal effect exactly: `ws.consumeLineReveal`/`editorRef` are stable-enough call targets, not
    // triggers; only a NEW request (or the editor/model becoming ready) should re-fire this.
  }, [ws.lineRevealRequest, key, edit, mountTick]);
  useEffect(() => () => observerRef.current?.disconnect(), []);

  if (!file || !key) {
    return (
      <div className="flex flex-1 items-start p-5 font-mono text-[12.5px] text-muted">
        {t("selectHint")}
        <span className="ml-1 inline-block h-[15px] w-2 bg-brand [animation:jax-blink_1.1s_infinite]" />
      </div>
    );
  }

  const isSvg = file.rel.split(".").pop()?.toLowerCase() === "svg";
  const showImage = !!read?.binary && isImage(file.rel) && read.size <= IMAGE_CAP;
  const ext = file.rel.split(".").pop()?.toLowerCase() ?? "";
  const isCsv = ext === "csv" || ext === "tsv";
  const csvDelimiter = csvDelimiterFor(ext);
  const csvRows = isCsv && read && !read.binary ? Math.max(parseCsv(read.content, csvDelimiter).length - 1, 0) : 0; // parseCsv, not a line-break count (round 1 F3)
  // ONE viewerFor call feeds every branch below, html included, unlike the old per-branch calls that never checked "placeholder" (round 1 F3).
  const kind = read ? viewerFor(ext, read.size, isCsv ? { rows: csvRows } : undefined) : null;
  const rawSrc = `/api/files/raw?root=${file.root}&rel=${encodeURIComponent(file.rel)}${
    edit?.baseHash || read?.hash ? `&h=${edit?.baseHash || read?.hash}` : ""
  }`;
  const downloadHref = `/api/files/download?root=${file.root}&rel=${encodeURIComponent(file.rel)}`;
  // pdf/audio/video are picked from `kind` alone (F4) — a small text-classified file (e.g. an
  // ASCII-only PDF) still needs its raw viewer even though it never enters the `read.binary` branch
  // below.
  const mediaKind = kind === "pdf" || kind === "audio" || kind === "video" ? kind : null;
  // overlay renders ON TOP of the always-mounted Editor; "placeholder" is its own value
  // (never folded into "binary"/"csv") so a capped file never reaches a viewer /raw refuses (F3).
  const overlay: "loading" | "error" | "image" | "binary" | "pdf" | "audio" | "video" | "csv" | "csvTooBig" | "placeholder" | null = q.isLoading
    ? "loading"
    : failed
      ? "error"
      : mediaKind
        ? mediaKind
        : read?.binary
          ? showImage ? "image" : kind === "placeholder" ? "placeholder" : "binary"
          : isCsv
            ? (kind === "placeholder" ? "placeholder" : kind === "csvTooBig" ? "csvTooBig" : "csv")
            : kind === "placeholder" ? "placeholder" : null; // round 2 MEDIUM: every non-binary, non-CSV kind (html, markdown, generic text) maps through kind, not just isHtml

  return (
    <>
      <div className="flex h-[46px] flex-none items-center gap-3 border-b border-line bg-surface px-4">
        {onClose ? (
          <>
            <button type="button" onClick={onClose} aria-label={t("backToFolders")} className="flex-none rounded p-1 text-muted transition hover:bg-surface-3 hover:text-body-ink">
              <ChevronLeft className="h-4 w-4" />
            </button>
            <span className="min-w-0 flex-1 truncate text-[13px] font-semibold text-ink">{file.rel.split("/").pop()}</span>
          </>
        ) : (
          <Breadcrumb rootLabel={file.root} ariaLabel={t("breadcrumbLabel")} segments={breadcrumbSegments(file.rel, false)}
            onRootClick={() => {
              // Reveals the FILE'S OWN repo root, not literally "" — for a repo-scoped file "" would
              // mean the global `repos` root (all 69 repos). A vault file's OWN root IS rel === ""
              // (vault has no name prefix) — requestReveal's auto-scope-switch (Task 5, round-1 F3)
              // still switches scope to vault for it, since it keys off `root`, not just `rel`.
              const scope = repoOf(file);
              ws.requestReveal(file.root, scope.kind === "repo" ? scope.name : "", true);
            }}
            onSegmentClick={(rel, isDir) => ws.requestReveal(file.root, rel, isDir)} />
        )}
        <div className="ml-auto flex items-center gap-3">
        {overlay === null ? (
          <>
            {isMd || isHtml || (isSvg && overlay === null) ? (
              <button
                type="button"
                onClick={() => setPreview((p) => !p)}
                className="rounded-md border border-line px-2.5 py-1 text-[11px] font-medium text-body-ink transition-transform hover:bg-surface-2 active:scale-95"
              >
                {t(preview ? (isHtml ? "htmlCode" : "edit") : "preview")}
              </button>
            ) : null}
            {(() => {
              const chip = statusChip(m, dirty, t);
              if (!chip) return null;
              if (chip.kind === "stale") {
                return (
                  <button
                    onClick={() => setDiffOpen(true)}
                    disabled={m?.saving || m?.reloading || locked}
                    className="rounded-md border border-warning px-2.5 py-1 text-[11px] font-medium text-warning transition-transform active:scale-95 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {chip.text}
                  </button>
                );
              }
              return (
                <span aria-live="polite" className={chipToneClass(chip.tone)}>
                  {chip.text}
                </span>
              );
            })()}
            <button
              type="button"
              onClick={() => void save()}
              disabled={!dirty || m?.saving || m?.reloading || locked}
              className="rounded-md bg-brand px-3 py-1.5 text-[12px] font-semibold text-on-brand transition-transform active:scale-95 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {t("save")}
            </button>
          </>
        ) : null}
          {locked ? (
            <span aria-disabled="true" title={t("downloadSavedHint")} className="cursor-not-allowed text-[11px] font-medium text-muted opacity-50">
              {t("download")}
            </span>
          ) : (
            <a
              href={downloadHref}
              download
              title={t("downloadSavedHint")}
              className="rounded-md border border-line px-2.5 py-1 text-[11px] font-medium text-body-ink transition-transform hover:bg-surface-2 active:scale-95"
            >
              {t("download")}
            </a>
          )}
        </div>
      </div>
      <div className="relative min-h-0 flex-1">
        <Editor
          height="100%"
          // path is Uri.parse()d by monaco — use "root/rel" (plain path form),
          // NOT tabKey's "root:rel" (the colon would parse as a URI scheme)
          path={`${file.root}/${file.rel}`}
          language={langOf(file.rel)}
          value={edit?.content ?? ""}
          theme="jax"
          beforeMount={beforeMount}
          onMount={onMount}
            onChange={(v) => {
            if (!edit || locked) return;
            ws.onEdit(key, edit.id, { content: v ?? "", justSaved: false });
            ws.onUserEdit();
          }}
          options={{
            minimap: { enabled: false },
            scrollBeyondLastLine: false,
            readOnly: locked,
          }}
        />
        {isMd && preview && edit && overlay === null ? (
          <div className="absolute inset-0 overflow-auto bg-base p-4">
            <Markdown>{edit.content}</Markdown>
          </div>
        ) : null}
        {isHtml && preview && edit && overlay === null ? (
          <iframe
            title={file.rel}
            sandbox=""
            srcDoc={wrapHtmlForPreview(edit.content)}
            className="absolute inset-0 h-full w-full border-0 bg-base"
          />
        ) : null}
        {isSvg && preview && overlay === null ? (
          <div className="absolute inset-0 overflow-auto bg-base p-4">
            {dirty ? (
              <p aria-live="polite" className="mb-2 text-[12px] text-muted">{t("svgDirtyPreview")}</p>
            ) : null}
            {imgError ? (
              <p aria-live="polite" className="text-[12px] text-danger">{t("imageLoadError")}</p>
            ) : (
              // eslint-disable-next-line @next/next/no-img-element -- loopback raw endpoint, never DOM-injected SVG
              <img src={rawSrc} alt={file.rel} onError={() => setImgError(true)} className="max-w-full" />
            )}
          </div>
        ) : null}
        {diffOpen && m?.stale && overlay === null && read && !read.binary ? (
          <div className="absolute inset-0 z-10 flex flex-col bg-surface-inset">
            <div className="flex flex-none items-center justify-between border-b border-line px-4 py-2">
              <span className="text-[12px] text-body-ink">{t("diffTitle")}</span>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={overwriteStale}
                  className="rounded-md border border-line px-2.5 py-1 text-[11px] font-medium text-body-ink transition-transform hover:bg-surface-2 active:scale-95"
                >
                  {t("diffOverwrite")}
                </button>
                <button
                  type="button"
                  onClick={() => { void reload(); setDiffOpen(false); }}
                  className="rounded-md bg-brand px-2.5 py-1 text-[11px] font-semibold text-on-brand transition-transform active:scale-95"
                >
                  {t("diffReload")}
                </button>
              </div>
            </div>
            <div className="min-h-0 flex-1">
              <DiffEditor
                height="100%"
                language={langOf(file.rel)}
                original={read.content ?? ""}
                modified={edit?.content ?? ""}
                theme="jax"
                beforeMount={beforeMount}
                options={{ readOnly: true, renderSideBySide: true }}
              />
            </div>
          </div>
        ) : null}
        {overlay !== null ? (
          <div className="absolute inset-0 overflow-auto bg-surface-inset">
            {overlay === "loading" ? (
              <div className="p-4">
                <div className="h-4 w-40 animate-pulse rounded bg-surface-2" />
              </div>
            ) : overlay === "error" ? (
              <div className="p-4">
                <SourceWarning
                  label={t("loadError")}
                  detail={q.data && !q.data.ok ? q.data.error : undefined}
                />
              </div>
            ) : overlay === "image" && read ? (
              <div className="flex h-full flex-col items-center justify-center gap-2 p-6">
                {imgError ? (
                  <p aria-live="polite" className="text-[12px] text-danger">{t("imageLoadError")}</p>
                ) : (
                  // eslint-disable-next-line @next/next/no-img-element -- loopback raw endpoint, next/image adds nothing here
                  <img
                    src={rawSrc}
                    alt={file.rel}
                    onError={() => setImgError(true)}
                    className="min-h-0 max-w-full flex-1 object-contain"
                  />
                )}
                <span className="font-mono text-[11px] text-muted">{fmtSize(read.size, locale)}</span>
              </div>
            ) : overlay === "pdf" ? (
              <iframe title={file.rel} src={rawSrc} className="h-full w-full border-0" />
            ) : overlay === "audio" ? (
              <div className="flex h-full items-center justify-center p-6">
                {/* eslint-disable-next-line jsx-a11y/media-has-caption -- source file, no track available */}
                <audio controls src={rawSrc} className="w-full max-w-md" />
              </div>
            ) : overlay === "video" ? (
              <div className="flex h-full items-center justify-center p-6">
                {/* eslint-disable-next-line jsx-a11y/media-has-caption -- source file, no track available */}
                <video controls src={rawSrc} className="max-h-full max-w-full" />
              </div>
            ) : overlay === "csv" && read && !read.binary ? (
              <CsvTable content={read.content} delimiter={csvDelimiter} />
            ) : overlay === "csvTooBig" ? (
              <div className="flex h-full flex-col items-center justify-center gap-1 p-6 text-center">
                <span className="text-[13px] text-body-ink">{t("csvTooManyRows")}</span>
              </div>
            ) : read ? (
              <div className="flex h-full flex-col items-center justify-center gap-1 p-6 text-center">
                <span className="text-[13px] text-body-ink">{t("binaryPlaceholder")}</span>
                <span className="font-mono text-[11px] text-muted">{fmtSize(read.size, locale)}</span>
              </div>
            ) : null}
          </div>
        ) : null}
      </div>
    </>
  );
}
