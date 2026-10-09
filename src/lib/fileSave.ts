import type { MutationFailure } from "./mutationOutcome";
import type { EditPatch, EditState } from "./filesState";
export type SaveResult =
  | { ok: true; hash: string }
  | { ok: false; error: string; code?: undefined; effect?: undefined }
  | (MutationFailure & { hash?: string });
export async function saveEdit(
  edit: EditState,
  pending: Set<symbol>,
  write: () => Promise<SaveResult>,
  patch: (value: EditPatch) => void,
): Promise<void> {
  if (edit.content === edit.savedContent || edit.saving || edit.reloading || pending.has(edit.id)) return;
  pending.add(edit.id);
  patch({ saving: true, stale: false, writeErr: false, justSaved: false });
  try {
    const res = await write();
    if (res.ok) {
      patch({
        savedContent: edit.content, baseHash: res.hash, justSaved: true,
        unconfirmed: false, writeErr: false, auditUnavailable: false,
      });
      return;
    }
    if (res.error === "changed on disk") {
      patch({ stale: true });
      return;
    }
    if (res.effect === "applied" && res.hash) {
      patch({ savedContent: edit.content, baseHash: res.hash, justSaved: false, auditWarning: true });
      return;
    }
    if (res.effect === "unconfirmed") {
      patch({ unconfirmed: true, justSaved: false });
      return;
    }
    if (res.effect === "not-applied") {
      patch({ writeErr: true, auditUnavailable: res.code === "audit-unavailable" });
      return;
    }
    patch({ unconfirmed: true, justSaved: false });
  } catch {
    patch({ unconfirmed: true, justSaved: false });
  } finally {
    pending.delete(edit.id);
    patch({ saving: false });
  }
}
