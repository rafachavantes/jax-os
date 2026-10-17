import { getDb } from "../../../../server/db";
import { listLivePanes } from "../../../../server/collectors/workflow-tmux";
import { collectCodexSnapshot, createCodexConnector, type CodexSnapshot } from "../../../../server/collectors/workflow-codex";
import { evaluateStaleness, maybeInsertStalenessAlert } from "../../../../server/db/workflows";
import { collectorResponse, requireSameOrigin } from "../../../../server/api";

export async function handleStalenessCheckPost(req: Request) {
  const originError = requireSameOrigin(req);
  if (originError) return originError;
  return collectorResponse(async () => {
    const db = getDb();
    // tmux and Codex are independent in BOTH directions (C4): a genuine tmux failure is an
    // explicit source warning with an empty snapshot — tmux alert evaluation is skipped without
    // inventing liveness — never a route failure that hides native Codex alerts. The collector's
    // no-server case stays a healthy empty snapshot.
    let live: Awaited<ReturnType<typeof listLivePanes>>;
    let tmuxSourceOk = true;
    try {
      live = await listLivePanes(); // defaults to the collector's own systemExec — Findings 8/29, no execFile here
    } catch {
      live = { paneIds: new Set(), incarnation: null, paneCommands: new Map() };
      tmuxSourceOk = false;
    }
    // Native liveness is independent of tmux; an unavailable Codex source skips only that source
    // and returns a warning, never a false "stuck" alert or an empty healthy evaluation.
    let codex: CodexSnapshot | null = null;
    try {
      codex = await collectCodexSnapshot(createCodexConnector);
    } catch {
      codex = null;
    }
    const now = Date.now();
    const { candidates, nullIncarnation, checked, codexSourceOk } = evaluateStaleness(db, now, live.paneIds, live.incarnation, codex, tmuxSourceOk);
    let alerted = 0;
    for (const c of candidates) if (maybeInsertStalenessAlert(db, c, now)) alerted += 1;
    return { checked, alerted, nullIncarnation, codexSourceOk, tmuxSourceOk }; // `checked` is the evaluation population, not candidates.length (Findings 9/30)
  });
}
