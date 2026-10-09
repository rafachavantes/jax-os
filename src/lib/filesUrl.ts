import type { Root } from "@/server/collectors/files";

export type FileSelection =
  | { status: "absent" }
  | { status: "valid"; root: Root; rel: string }
  | { status: "invalid" };

function validRel(rel: string): boolean {
  if (rel.includes("\0") || rel.startsWith("/")) return false;
  if (new TextEncoder().encode(rel).length > 4096) return false;
  return rel.split("/").every((segment) => segment !== "" && segment !== "." && segment !== "..");
}

export function parseFileSelection(params: URLSearchParams): FileSelection {
  const root = params.get("root");
  const rel = params.get("rel");
  if (root === null && rel === null) return { status: "absent" };
  if ((root !== "repos" && root !== "vault") || rel === null || !validRel(rel)) return { status: "invalid" };
  return { status: "valid", root, rel };
}

export function preferredActive(
  selection: FileSelection,
  stored: { root: Root; rel: string } | null,
): { status: "invalid" } | { status: "ok"; file: { root: Root; rel: string } | null } {
  if (selection.status === "invalid") return { status: "invalid" };
  if (selection.status === "valid") return { status: "ok", file: { root: selection.root, rel: selection.rel } };
  return { status: "ok", file: stored };
}

export function serializeFileSelection(
  params: URLSearchParams,
  selection: { root: Root; rel: string } | null,
): URLSearchParams {
  const next = new URLSearchParams(params);
  if (!selection) {
    next.delete("root");
    next.delete("rel");
  } else {
    next.set("root", selection.root);
    next.set("rel", selection.rel);
  }
  return next;
}

// Push only for a deliberate (pinned) navigation to a DIFFERENT file; replace for
// a preview click, a same-file pinned-flag promotion, or when nothing is open
// (cold review round 2 F1 — a same-file promotion must not push, or Back breaks).
export function shouldPush(
  prev: { root: Root; rel: string; pinned: boolean } | null,
  next: { root: Root; rel: string; pinned: boolean } | null,
): boolean {
  if (!next || !next.pinned) return false;
  if (prev && prev.root === next.root && prev.rel === next.rel) return false;
  return true;
}

// Read-direction sync (page.tsx's URL->tabs effect): resolves the URL against
// the active tab, calls one action, returns whether it opened a file (caller
// sets urlApplied). Now also requires PINNED to no-op (cold review round 1 F1).
// `lastWritten` is the write-direction effect's last echo (round 2 F1, 011eca3d7f72):
// an exact match means do nothing; a genuine external nav still promotes below.
export function syncFromUrl(
  params: URLSearchParams,
  active: { root: Root; rel: string; pinned: boolean } | null,
  lastWritten: string | null,
  actions: { setUrlError: (invalid: boolean) => void; openFilePinned: (root: Root, rel: string) => void },
): boolean {
  if (lastWritten !== null && lastWritten === params.toString()) return true;
  const preferred = preferredActive(parseFileSelection(params), null);
  if (preferred.status === "invalid") {
    actions.setUrlError(true);
    return false;
  }
  actions.setUrlError(false);
  if (!preferred.file) return false;
  if (active && active.root === preferred.file.root && active.rel === preferred.file.rel && active.pinned) {
    return false;
  }
  actions.openFilePinned(preferred.file.root, preferred.file.rel);
  return true;
}

// Write-direction sync (page.tsx's tabs->URL effect): calls push or replace
// with the resolved href, or makes no call at all when the URL already
// matches the active tab. Returns the query string it wrote (or null for no
// call) so the caller can hand it back to syncFromUrl as `lastWritten`.
export function syncToUrl(
  params: URLSearchParams,
  pathname: string,
  active: { root: Root; rel: string; pinned: boolean } | null,
  prevActive: { root: Root; rel: string; pinned: boolean } | null,
  actions: { push: (href: string) => void; replace: (href: string) => void },
): string | null {
  if (parseFileSelection(params).status === "invalid") return null;
  const next = serializeFileSelection(params, active ? { root: active.root, rel: active.rel } : null);
  if (next.toString() === params.toString()) return null;
  const search = next.toString();
  const href = search ? `${pathname}?${search}` : pathname;
  if (shouldPush(prevActive, active)) actions.push(href);
  else actions.replace(href);
  return search;
}
