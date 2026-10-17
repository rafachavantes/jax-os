import type { EditPatch, EditState } from "./filesState";

export function reloadReadResult(result: {
  isError?: boolean;
  status?: string;
  data?: { ok: boolean; data?: { binary?: boolean; content?: string | null; hash?: string } | null } | undefined;
}): { content: string; hash: string } | null {
  if (result.isError || result.status === "error") return null;
  const payload = result.data;
  if (!payload?.ok || !payload.data || payload.data.binary) return null;
  if (typeof payload.data.content !== "string" || typeof payload.data.hash !== "string") return null;
  return { content: payload.data.content, hash: payload.data.hash };
}

export async function reloadEdit({
  getCurrent,
  read,
  patch,
  pending,
}: {
  getCurrent: () => EditState | undefined;
  read: () => Promise<{ content: string; hash: string } | null>;
  patch: (id: symbol, patch: EditPatch) => void;
  pending: Set<symbol>;
}): Promise<void> {
  const start = getCurrent();
  if (!start || start.saving || start.reloading || pending.has(start.id)) return;
  const capturedId = start.id;
  const capturedRevision = start.revision;
  pending.add(capturedId);
  patch(capturedId, { reloading: true, reloadKept: false, writeErr: false });
  try {
    const fresh = await read();
    const current = getCurrent();
    if (!current || current.id !== capturedId || current.revision !== capturedRevision || current.pathLocked) {
      patch(capturedId, { reloadKept: true });
      return;
    }
    if (!fresh) {
      patch(capturedId, { writeErr: true });
      return;
    }
    patch(capturedId, {
      content: fresh.content,
      savedContent: fresh.content,
      baseHash: fresh.hash,
      stale: false,
      justSaved: false,
      reloadKept: false,
    });
  } catch {
    patch(capturedId, { writeErr: true });
  } finally {
    pending.delete(capturedId);
    patch(capturedId, { reloading: false });
  }
}
